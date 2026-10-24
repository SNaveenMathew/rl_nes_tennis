"""Streamlit UI for reviewing and editing NES Tennis demonstration annotations.

Run from the repository root:
    streamlit run app/annotation_app.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.annotation_store import (  # noqa: E402
    LABEL_OPTIONS,
    REVIEW_STATUSES,
    STATE_EVENT_NAMES,
    STATE_VALUE_FIELDS,
    AnnotationStore,
)
from tennis_rl.actions import ACTION_NAMES  # noqa: E402

DB_PATH = PROJECT_ROOT / "app" / "data" / "annotations.sqlite3"
CACHE_ROOT = PROJECT_ROOT / "app" / ".cache"
TASK_COLUMNS = ["id", "start", "end", "skill", "manager_duration", "confidence", "annotator", "notes"]
STATE_COLUMNS = ["id", "step", "ball_xy", "player_xy", "ball_velocity_xy", "time_to_contact", "events_json"]


def enumerate_points(demo_root: str | Path) -> list[dict[str, Any]]:
    """Discover project-format point NPZ files, including nonstandard outcome folders."""
    root = Path(demo_root).resolve()
    if not root.is_dir():
        return []
    paths = sorted(root.glob("level_*/**/match_*/point_*.npz"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        sidecar = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.is_file():
            try:
                raw = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    meta = raw
            except (OSError, json.JSONDecodeError):
                meta = {}
        try:
            stat = path.stat()
            with np.load(path, allow_pickle=False) as data:
                if "actions" not in data:
                    continue
                length = int(np.asarray(data["actions"]).shape[0])
                level = int(np.asarray(data["level"]).item()) if "level" in data else _level_from_path(path)
                point_index = int(np.asarray(data["point_index"]).item()) if "point_index" in data else _point_from_path(path)
                match_id = _scalar_text(data["match_id"]) if "match_id" in data else path.parent.name
                stored_outcome = _scalar_text(data["outcome"]) if "outcome" in data else path.parent.parent.name
                has_native = "native_frames" in data
                obs_shape = tuple(int(x) for x in np.asarray(data["observations"]).shape) if "observations" in data else ()
            frame_skip = meta.get("frame_skip")
            try:
                frame_skip = int(frame_skip) if frame_skip not in (None, "") else None
            except (TypeError, ValueError):
                frame_skip = None
            rows.append({
                "path": path,
                "rel": rel,
                "length": length,
                "level": int(meta.get("level", level) or level),
                "point_index": int(meta.get("point_index", point_index) or 0),
                "match_id": str(meta.get("match_id", match_id)),
                "outcome": str(meta.get("point_outcome", path.parent.parent.name or stored_outcome)),
                "stored_outcome": stored_outcome,
                "scenario_tag": str(meta.get("scenario_tag", "")),
                "frame_skip": frame_skip,
                "native_frames": bool(meta.get("native_capture", has_native)) and has_native,
                "observation_shape": obs_shape,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def _scalar_text(value: Any) -> str:
    value = np.asarray(value).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _level_from_path(path: Path) -> int:
    try:
        return int(path.parents[2].name.replace("level_", ""))
    except (IndexError, ValueError):
        return 0


def _point_from_path(path: Path) -> int:
    try:
        return int(path.stem.replace("point_", ""))
    except ValueError:
        return -1


def load_point_bundle(point_path: str | Path, mtime_ns: int, size: int) -> dict[str, Any]:
    """Load frames/actions for one point. mtime/size are cache-busting arguments."""
    del mtime_ns, size
    path = Path(point_path)
    with np.load(path, allow_pickle=False) as data:
        actions = np.asarray(data["actions"], dtype=np.int64).copy()
        pulses = np.asarray(data["pulse_frames"], dtype=np.int16).copy() if "pulse_frames" in data else np.full(actions.shape, -1, dtype=np.int16)
        if "native_frames" in data:
            frames = np.asarray(data["native_frames"], dtype=np.uint8).copy()
            native = frames.ndim == 4 and frames.shape[-1] == 3
        elif "observations" in data:
            obs = np.asarray(data["observations"], dtype=np.uint8)
            frames = obs[:, -1].copy()
            native = False
        else:
            raise ValueError(f"No frames or observations are stored in {path}")
    length = min(len(actions), len(frames))
    actions, pulses, frames = actions[:length], pulses[:length], frames[:length]
    if length == 0:
        raise ValueError(f"No observations found in {path}")
    return {"frames": frames, "has_native": native, "actions": actions, "pulses": pulses, "length": length}


def ensure_video(point: dict[str, Any], frames: np.ndarray, frame_skip: int) -> Path:
    """Encode the point's recorded frames as a real, seekable MP4 for browser playback."""
    import imageio_ffmpeg

    CACHE_ROOT.joinpath("videos").mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{point['path']}|{point['mtime_ns']}|{point['size']}|{frames.shape}|{frame_skip}".encode("utf-8")
    ).hexdigest()[:24]
    target = CACHE_ROOT / "videos" / f"{cache_key}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target

    if frames.ndim == 3:
        height, width = frames.shape[1:]
    elif frames.ndim == 4 and frames.shape[-1] == 3:
        height, width = frames.shape[1:3]
    else:
        raise ValueError(f"Unsupported frame array shape {frames.shape}")
    # The emulator canvas is 256x240; both dimensions are even. Legacy grayscale
    # frames are replicated to RGB only for video encoding, never for training.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fps = 60.1 / max(1, int(frame_skip))
    temp_path = target.with_suffix(".building.mp4")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            if frame.ndim == 2:
                rgb = np.repeat(frame[:, :, None], 3, axis=2)
            else:
                rgb = frame
            process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        code = process.wait(timeout=120)
        if code != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace")[-2000:] or f"ffmpeg exited with {code}")
        os.replace(temp_path, target)
    except Exception:
        try:
            process.kill()
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def task_editor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: row.get(k) for k in TASK_COLUMNS} for row in rows]


def state_editor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            "id": row.get("id"),
            "step": row.get("step"),
            "ball_xy": json.dumps(row["ball_xy"]) if row.get("ball_xy") is not None else "",
            "player_xy": json.dumps(row["player_xy"]) if row.get("player_xy") is not None else "",
            "ball_velocity_xy": json.dumps(row["ball_velocity_xy"]) if row.get("ball_velocity_xy") is not None else "",
            "time_to_contact": row.get("time_to_contact"),
            "events_json": json.dumps(row.get("events", {}), sort_keys=True) if row.get("events") else "",
        })
    return output


def _timeline_html(rows: list[dict[str, Any]], length: int) -> str:
    colors = {
        "serve_timing": "#2563eb", "serve_placement_power": "#1d4ed8", "serve_recovery": "#0891b2",
        "receive_position": "#7c3aed", "return_stroke_timing": "#6d28d9", "intercept_track": "#059669",
        "hold_cover": "#0d9488", "recover_reposition": "#65a30d", "approach_net": "#ea580c",
        "retreat_lob": "#c2410c", "stroke_selector": "#9333ea", "stroke_timing": "#be123c",
        "transition_wait_reposition": "#64748b",
    }
    blocks = []
    for row in rows:
        try:
            start, end, label = int(row["start"]), int(row["end"]), str(row["skill"])
        except (TypeError, ValueError, KeyError):
            continue
        if length < 1 or end <= start:
            continue
        left, width = start * 100 / length, max(0.3, (end - start) * 100 / length)
        title = html.escape(f"{start}–{end - 1}: {label}")
        blocks.append(f'<div title="{title}" style="position:absolute;left:{left:.5f}%;width:{width:.5f}%;height:100%;background:{colors.get(label, "#475569")};border-right:1px solid #fff;box-sizing:border-box"></div>')
    return '<div style="height:22px;background:#e2e8f0;border-radius:5px;overflow:hidden;position:relative;margin:6px 0 12px">' + "".join(blocks) + "</div>"


def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Tennis RL Label Studio", page_icon="🎾", layout="wide")
    st.title("🎾 Tennis RL · Label Studio")
    st.caption("Review frame-perfect demonstrations, edit task intervals and state labels, track review status, and export training-compatible JSONL. Raw demonstration files are never modified.")

    with st.sidebar:
        st.header("Project")
        default_root = st.session_state.get("demo_root", str(PROJECT_ROOT / "demonstrations"))
        root_text = st.text_input("Demonstrations directory", value=default_root, help="Directory that contains level_*/outcome/match_*/point_*.npz files.")
        demo_root = Path(root_text).expanduser().resolve()
        st.session_state["demo_root"] = str(demo_root)
        annotator = st.text_input("Annotator ID", value=st.session_state.get("annotator", os.environ.get("USERNAME", "annotator")))
        st.session_state["annotator"] = annotator
        st.caption(f"Local database: `{DB_PATH.relative_to(PROJECT_ROOT)}`")
        st.caption("The SQLite database and generated preview videos are git-ignored.")

    if not demo_root.is_dir():
        st.warning(f"Demonstrations directory not found: {demo_root}")
        st.code("python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --make-sheets")
        st.stop()

    task_jsonl = demo_root / "task_annotations.jsonl"
    state_jsonl = demo_root / "state_annotations.jsonl"
    store = AnnotationStore(DB_PATH)
    bootstrap_counts = store.bootstrap_from_jsonl(demo_root, task_jsonl, state_jsonl)
    points = enumerate_points(demo_root)
    if not points:
        st.warning("No point files found. Expected level_*/outcome/match_*/point_*.npz.")
        st.stop()

    if "bootstrap_notice" not in st.session_state:
        st.session_state["bootstrap_notice"] = bootstrap_counts
    if "bootstrap_notice" in st.session_state:
        count = st.session_state.pop("bootstrap_notice")
        if count.get("task_imported") or count.get("state_imported"):
            st.success(f"Loaded existing labels into the local database: {count.get('task_imported', 0)} task segments, {count.get('state_imported', 0)} state rows.")

    with st.sidebar:
        if st.button("Merge labels from JSONL", help="Adds previously unseen labels from task_annotations.jsonl/state_annotations.jsonl. Existing DB edits are not overwritten."):
            merged_tasks = store.merge_task_jsonl(demo_root, task_jsonl)
            merged_states = store.merge_state_jsonl(demo_root, state_jsonl)
            st.success(f"Merged {merged_tasks} task segments and {merged_states} state rows.")
            st.rerun()
        st.divider()
        st.subheader("Export for training")
        if st.button("Export task labels to task_annotations.jsonl", type="primary"):
            count = store.export_task_jsonl(demo_root, task_jsonl)
            st.success(f"Exported {count} task segments to {task_jsonl}.")
        if st.button("Export state labels to state_annotations.jsonl"):
            count = store.export_state_jsonl(demo_root, state_jsonl)
            st.success(f"Exported {count} state rows to {state_jsonl}.")

    summary = store.point_summary(demo_root)
    statuses = list(REVIEW_STATUSES)
    f1, f2, f3 = st.columns([1.2, 1, 1.4])
    with f1:
        search = st.text_input("Search point / match / tag", placeholder="point_0004, match id, serve-first…")
    with f2:
        status_filter = st.multiselect("Review status", statuses, default=statuses)
    with f3:
        label_filter = st.multiselect("Show points with label", list(LABEL_OPTIONS), default=[])

    filtered: list[dict[str, Any]] = []
    for point in points:
        review = summary.get(point["rel"], {"status": "Not started", "segments": 0, "annotator": "", "notes": ""})
        point["review_status"] = review.get("status", "Not started")
        point["segment_count"] = int(review.get("segments", 0))
        point["review_notes"] = review.get("notes", "")
        if point["review_status"] not in status_filter:
            continue
        haystack = " ".join([point["rel"], point["match_id"], point["scenario_tag"], str(point["point_index"])]).lower()
        if search.strip() and search.strip().lower() not in haystack:
            continue
        if label_filter:
            skills = set(summary.get(point["rel"], {}).get("skills", []))
            if not skills.intersection(label_filter):
                continue
        filtered.append(point)

    total_segments = sum(int(item["segment_count"]) for item in points)
    reviewed_count = sum(1 for item in points if summary.get(item["rel"], {}).get("status") == "Reviewed")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Points in dataset", len(points))
    metric2.metric("Matching filter", len(filtered))
    metric3.metric("Task segments in DB", total_segments)
    metric4.metric("Reviewed points", f"{reviewed_count}/{len(points)}")
    if not filtered:
        st.info("No points match the selected filters.")
        st.stop()

    def point_label(point: dict[str, Any]) -> str:
        tag = f" · {point['scenario_tag']}" if point["scenario_tag"] else ""
        fs = f" · skip={point['frame_skip']}" if point["frame_skip"] is not None else " · skip=unknown"
        return f"{point['rel']} · {point['length']} steps · {point['segment_count']} labels · {point['review_status']}{tag}{fs}"

    path_by_label = {point_label(point): point for point in filtered}
    default_active = st.session_state.get("active_point_path")
    default_index = next((i for i, p in enumerate(filtered) if p["rel"] == default_active), 0)
    selected_label = st.selectbox("Point queue", list(path_by_label), index=default_index, key="point_queue_select")
    point = path_by_label[selected_label]
    rel_path = point["rel"]
    if st.session_state.get("active_point_path") != rel_path:
        st.session_state["active_point_path"] = rel_path
        st.session_state["draft_task_rows"] = task_editor_rows(store.annotations_for_point(demo_root, rel_path))
        st.session_state["draft_state_rows"] = state_editor_rows(store.state_for_point(demo_root, rel_path))
        st.session_state["editor_revision"] = int(st.session_state.get("editor_revision", 0)) + 1
        st.session_state["state_editor_revision"] = int(st.session_state.get("state_editor_revision", 0)) + 1
        st.session_state.pop("preview_step", None)

    @st.cache_resource(max_entries=1, show_spinner=False)
    def cached_bundle(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
        return load_point_bundle(path, mtime_ns, size)

    try:
        with st.spinner("Loading frames and preparing video…"):
            bundle = cached_bundle(str(point["path"]), point["mtime_ns"], point["size"])
            if bundle["length"] != point["length"]:
                point["length"] = bundle["length"]
            frame_skip = point["frame_skip"] or 1
            video_path = ensure_video(point, bundle["frames"], frame_skip)
    except Exception as exc:
        st.error(f"Could not prepare this point's video: {exc}")
        st.info("Install app dependencies with `pip install -r app/requirements.txt` and check that the NPZ is readable.")
        st.stop()

    st.subheader("Recorded point")
    badges = [f"**Path:** `{rel_path}`", f"**Steps:** {point['length']}", f"**Frame skip:** {point['frame_skip'] if point['frame_skip'] else 'unknown (video assumes 1)'}", f"**Frames:** {'native RGB' if bundle['has_native'] else 'stored grayscale observations'}"]
    st.markdown(" &nbsp; · &nbsp; ".join(badges))
    if point["frame_skip"] is None:
        st.warning("This point has no frame_skip in its sidecar. Playback assumes one raw NES frame per step; confirm recording metadata before making timing-sensitive labels.")
    if not bundle["has_native"]:
        st.warning("Native RGB frames are unavailable; this preview uses the stored observation's newest grayscale frame. Lost detail cannot be reconstructed by enlarging it.")

    video_col, preview_col = st.columns([1.2, 1])
    with video_col:
        st.markdown("**Video playback**")
        st.video(str(video_path), autoplay=False, muted=True, loop=False)
        st.caption(f"Browser controls allow play/pause and seeking. Approx. {60.1 / frame_skip:.2f} saved steps per second.")
    with preview_col:
        st.markdown("**Exact step preview**")
        preview_step = st.slider("Inspect saved step", min_value=0, max_value=max(0, bundle["length"] - 1), value=int(st.session_state.get("preview_step", 0)), step=1, key="preview_step")
        frame = bundle["frames"][preview_step]
        if frame.ndim == 2:
            display_image = Image.fromarray(frame, mode="L")
        else:
            display_image = Image.fromarray(frame[:, :, :3], mode="RGB")
        st.image(display_image, caption=f"Step {preview_step} / {bundle['length'] - 1}", width=512)
        action = int(bundle["actions"][preview_step])
        action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"action {action}"
        pulse = int(bundle["pulses"][preview_step])
        st.write(f"**Recorded action:** `{action_name}` · **button pulse frames:** `{pulse if pulse >= 0 else 'unknown'}`")
        start_widget_key = f"start_input_{hashlib.md5(rel_path.encode()).hexdigest()}"
        if st.button("Set new segment start to this step", key=f"set_start_{hashlib.md5(rel_path.encode()).hexdigest()}"):
            # The button occurs before the number_input widget below is instantiated.
            st.session_state[start_widget_key] = preview_step
            st.rerun()

    with st.expander("Label definitions and boundary rules", expanded=False):
        st.markdown("""
- **`serve_timing`**: deliberate preparation/waiting before the serve stroke starts; exclude the serve's execution and flight.
- **`receive_position`**: positioning while the CPU serves; **`return_stroke_timing`**: timing the return of that serve.
- **`intercept_track`**: movement toward a live incoming ball; stop before the shot-timing decision.
- **`stroke_timing`**: the immediate decision window to hit now versus wait; don't extend it across unrelated running/recovery.
- **`hold_cover`**: holding a useful court position while awaiting the opponent's next play; being motionless alone is not enough.
- **`recover_reposition`**: move back to a useful covering position after your own shot; **`serve_recovery`** is the analogous post-serve movement.
- **`approach_net`** / **`retreat_lob`**: clear forward net approach / clear retreat in response to a lob.
- **`transition_wait_reposition`**: between-point / serve-ready setup, not live-rally movement.

Use half-open `[start, end)` ranges. Do not overlap task windows. Leave uncertain frames unlabeled; labels should describe visible behavior, not whether the point was won.
        """)

    st.subheader("Task segments")
    task_rows = st.session_state.get("draft_task_rows", [])
    timeline_input = [{"start": r.get("start"), "end": r.get("end"), "skill": r.get("skill")} for r in task_rows]
    st.markdown(_timeline_html(timeline_input, bundle["length"]), unsafe_allow_html=True)
    if task_rows:
        st.caption("Colored spans are labeled; gaps are intentionally unlabeled. Hover over a span for its interval and label.")
    else:
        st.caption("No task labels yet. Add multiple intervals below, or existing JSONL labels will appear here automatically.")

    editor_key = f"task_editor_{hashlib.md5(rel_path.encode()).hexdigest()}_{st.session_state['editor_revision']}"
    task_df = pd.DataFrame(st.session_state.get("draft_task_rows", []), columns=TASK_COLUMNS)
    task_df = st.data_editor(
        task_df,
        key=editor_key,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_order=["start", "end", "skill", "manager_duration", "confidence", "annotator", "notes"],
        column_config={
            "id": None,
            "start": st.column_config.NumberColumn("Start (inclusive)", min_value=0, max_value=bundle["length"] - 1, step=1, format="%d", required=True, help="First saved step included."),
            "end": st.column_config.NumberColumn("End (exclusive)", min_value=1, max_value=bundle["length"], step=1, format="%d", required=True, help="First saved step excluded."),
            "skill": st.column_config.SelectboxColumn("Task label", options=list(LABEL_OPTIONS), required=True),
            "manager_duration": st.column_config.NumberColumn("Manager duration", min_value=1, max_value=8, step=1, format="%d", help="Optional; must be 1, 2, 4, or 8. Leave blank if unsure."),
            "confidence": st.column_config.NumberColumn("Confidence", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
            "annotator": st.column_config.TextColumn("Annotator"),
            "notes": st.column_config.TextColumn("Notes / rationale", width="large"),
        },
    )
    # Store every in-browser edit in session state before handling action buttons below.
    st.session_state["draft_task_rows"] = task_df.to_dict(orient="records")

    row_col1, row_col2, row_col3 = st.columns([1, 1, 1.6])
    with row_col1:
        start_widget_key = f"start_input_{hashlib.md5(rel_path.encode()).hexdigest()}"
        new_start = st.number_input("New segment start", min_value=0, max_value=bundle["length"] - 1,
                                    value=int(preview_step), step=1, key=start_widget_key)
    with row_col2:
        new_end = st.number_input("New segment end (exclusive)", min_value=1, max_value=bundle["length"],
                                  value=min(bundle["length"], int(new_start) + 1), step=1, key=f"end_input_{hashlib.md5(rel_path.encode()).hexdigest()}")
    with row_col3:
        new_label = st.selectbox("New segment label", list(LABEL_OPTIONS), key=f"new_label_{hashlib.md5(rel_path.encode()).hexdigest()}")

    buttons = st.columns([1, 1, 1.6])
    with buttons[0]:
        if st.button("＋ Add segment", key=f"add_seg_{hashlib.md5(rel_path.encode()).hexdigest()}"):
            draft = st.session_state["draft_task_rows"]
            draft.append({"id": None, "start": int(new_start), "end": int(new_end), "skill": new_label,
                          "manager_duration": None, "confidence": 1.0, "annotator": annotator, "notes": ""})
            st.session_state["draft_task_rows"] = draft
            st.session_state["editor_revision"] += 1
            st.rerun()
    with buttons[1]:
        if st.button("Save edits", type="primary", key=f"save_seg_{hashlib.md5(rel_path.encode()).hexdigest()}"):
            try:
                count = store.save_point_annotations(demo_root, rel_path, st.session_state["draft_task_rows"], bundle["length"], default_annotator=annotator)
                st.session_state["draft_task_rows"] = task_editor_rows(store.annotations_for_point(demo_root, rel_path))
                st.session_state["editor_revision"] += 1
                st.session_state["flash_message"] = f"Saved {count} task segments for {rel_path}."
                st.rerun()
            except (ValueError, sqlite3.Error) as exc:
                st.error(f"Not saved: {exc}")
    with buttons[2]:
        if st.button("Export task labels", key=f"export_one_{hashlib.md5(rel_path.encode()).hexdigest()}"):
            count = store.export_task_jsonl(demo_root, task_jsonl)
            st.session_state["flash_message"] = f"Exported {count} task segments to task_annotations.jsonl."
            st.rerun()

    review = store.review_for_point(demo_root, rel_path)
    st.markdown("**Point review workflow**")
    review_col1, review_col2 = st.columns([1, 2])
    with review_col1:
        st.selectbox("Status", list(REVIEW_STATUSES), index=list(REVIEW_STATUSES).index(review["status"] if review["status"] in REVIEW_STATUSES else "Not started"), key=f"review_status_{hashlib.md5(rel_path.encode()).hexdigest()}")
    with review_col2:
        st.text_input("Review notes", value=review.get("notes", ""), key=f"review_notes_{hashlib.md5(rel_path.encode()).hexdigest()}", placeholder="Ambiguous boundary, needs second review, etc.")
    if st.button("Save point review status", key=f"save_review_{hashlib.md5(rel_path.encode()).hexdigest()}"):
        try:
            store.set_review(demo_root, rel_path, st.session_state.get(f"review_status_{hashlib.md5(rel_path.encode()).hexdigest()}", "In progress"), annotator,
                             st.session_state.get(f"review_notes_{hashlib.md5(rel_path.encode()).hexdigest()}", ""))
            st.session_state["flash_message"] = "Point review status saved."
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if "flash_message" in st.session_state:
        st.success(st.session_state.pop("flash_message"))

    with st.expander("Optional per-frame state / event labels (editable)", expanded=False):
        st.caption("State labels are stored separately from task intervals. Coordinates are normalized x,y in [0,1]; events_json is an object such as {\"agent_contact_window\": 1}. Blank means unknown, not no.")
        state_rows = st.session_state.get("draft_state_rows", [])
        state_key = f"state_editor_{hashlib.md5(rel_path.encode()).hexdigest()}_{st.session_state['state_editor_revision']}"
        state_df = pd.DataFrame(state_rows, columns=STATE_COLUMNS)
        edited_state = st.data_editor(
            state_df,
            key=state_key,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_order=["step", "ball_xy", "player_xy", "ball_velocity_xy", "time_to_contact", "events_json"],
            column_config={
                "id": None,
                "step": st.column_config.NumberColumn("Step", min_value=0, max_value=bundle["length"] - 1, step=1, format="%d", required=True),
                "ball_xy": st.column_config.TextColumn("Ball (x,y) JSON", help='Example: [0.42, 0.67]'),
                "player_xy": st.column_config.TextColumn("Player (x,y) JSON", help='Example: [0.51, 0.83]'),
                "ball_velocity_xy": st.column_config.TextColumn("Ball velocity JSON", help='Signed values, e.g. [-0.1, 0.04]'),
                "time_to_contact": st.column_config.NumberColumn("Time to contact", min_value=0.0, step=0.1),
                "events_json": st.column_config.TextColumn("Events JSON", width="large"),
            },
        )
        st.session_state["draft_state_rows"] = edited_state.to_dict(orient="records")
        st.write("Allowed event fields:", ", ".join(STATE_EVENT_NAMES))
        state_buttons = st.columns(2)
        with state_buttons[0]:
            if st.button("Save state labels", key=f"save_state_{hashlib.md5(rel_path.encode()).hexdigest()}"):
                try:
                    count = store.save_state_for_point(demo_root, rel_path, st.session_state["draft_state_rows"], bundle["length"], annotator)
                    st.session_state["draft_state_rows"] = state_editor_rows(store.state_for_point(demo_root, rel_path))
                    st.session_state["state_editor_revision"] += 1
                    st.session_state["flash_message"] = f"Saved {count} state rows for {rel_path}."
                    st.rerun()
                except (ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
                    st.error(f"State labels not saved: {exc}")
        with state_buttons[1]:
            if st.button("Export state labels", key=f"export_state_{hashlib.md5(rel_path.encode()).hexdigest()}"):
                count = store.export_state_jsonl(demo_root, state_jsonl)
                st.session_state["flash_message"] = f"Exported {count} state rows to state_annotations.jsonl."
                st.rerun()

    with st.expander("Change history / audit log", expanded=False):
        events = store.audit_for_point(demo_root, rel_path)
        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
        else:
            st.caption("No changes recorded yet. Adds, edits, deletes, imports, and review status changes are audited in SQLite.")

    st.divider()
    st.caption("Database is local to this repo under app/data. Export task labels before using the existing training pipeline; it reads demonstrations/task_annotations.jsonl, not SQLite directly.")


if __name__ == "__main__":
    # Delay the Streamlit import at module level so pure helpers remain importable in tests.
    main()
