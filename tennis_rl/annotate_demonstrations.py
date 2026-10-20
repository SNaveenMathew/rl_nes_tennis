"""Review native-resolution NES Tennis demonstrations and add explicit labels.

Examples:
  python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --make-sheets --sample-every 1 --columns 1 --scale 3 --max-cells 4 --review-gifs --gif-scale 2
  python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --interactive --sample-every 1 --columns 1 --scale 3 --max-cells 4 --review-gifs
  python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --interactive --state-labels-only --state-fields ball_xy,player_xy,agent_contact_window,high_floater_lob

Task ranges use half-open [start, end) indices. Native RGB frames are for visual
review/annotation only; the policy is trained on the stored grayscale observations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .demonstrations import _safe_load, iter_point_files
from .hierarchy_schema import DEFAULT_LEAF_TASKS, LEGACY_PHASE_LABELS, phase_for_task, TASK_TO_PHASE

EVENT_FIELDS = (
    "serve_ready", "agent_serve", "cpu_serve", "ball_in_play", "ball_toward_agent",
    "ball_away_from_agent", "opponent_contact_window", "agent_contact_window",
    "high_floater_lob", "point_resolved", "uncertain",
)
COORD_FIELDS = ("ball_xy", "player_xy")
STATE_FIELDS = set(COORD_FIELDS) | {"ball_velocity_xy", "time_to_contact"} | set(EVENT_FIELDS)
DEFAULT_STATE_FIELDS = ["ball_xy", "player_xy", "ball_velocity_xy", "time_to_contact", *EVENT_FIELDS]


def _metadata(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as d:
            for key in ("match_id", "point_index", "level", "outcome", "pulse_frames"):
                if key in d and key != "pulse_frames":
                    val = np.asarray(d[key]).item()
                    out[key] = val.decode("utf-8") if isinstance(val, bytes) else val
            out["has_native_frames"] = "native_frames" in d
    except Exception:
        pass
    side = path.with_suffix(".json")
    if side.exists():
        try:
            obj = json.loads(side.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out.update(obj)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def _load_review_frames(path: Path) -> tuple[list[Image.Image], bool, int]:
    """Return review frames, whether they are true native frames, and frame skip."""
    ep = _safe_load(path, "point")
    if ep is None:
        raise ValueError(f"Could not load point file {path}")
    frame_skip = 2
    meta = _metadata(path)
    try:
        frame_skip = max(1, int(meta.get("frame_skip", 2)))
    except (TypeError, ValueError):
        pass
    with np.load(path, allow_pickle=False) as data:
        native = np.asarray(data["native_frames"], dtype=np.uint8) if "native_frames" in data else None
    native_ok = native is not None and native.shape == (ep.length, 240, 256, 3)
    if native_ok:
        return [Image.fromarray(frame, mode="RGB") for frame in native], True, frame_skip
    # Old points contain only low-resolution grayscale inputs. Upscaling is for
    # display convenience, not recovered detail; mark these sheets accordingly.
    return [Image.fromarray(np.asarray(obs[-1], dtype=np.uint8), mode="L").convert("RGB")
            for obs in ep.observations], False, frame_skip


def _sampled_steps(length: int, sample_every: int) -> list[int]:
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    if length < 1:
        return []
    steps = list(range(0, length, sample_every))
    if steps[-1] != length - 1:
        steps.append(length - 1)
    return steps


def make_contact_sheets(
    path: Path,
    output: Path,
    sample_every: int = 4,
    columns: int = 8,
    scale: int = 1,
    max_cells: int = 64,
) -> list[Path]:
    """Create paginated nearest-neighbor contact sheets with exact step labels."""
    if columns < 1 or scale < 1 or max_cells < 1:
        raise ValueError("columns, scale, and max_cells must be positive")
    frames, has_native, _ = _load_review_frames(path)
    steps = _sampled_steps(len(frames), sample_every)
    if not steps:
        raise ValueError(f"Point file {path} contains no steps")
    page_sets = [steps[i:i + max_cells] for i in range(0, len(steps), max_cells)]
    output.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for page_i, page_steps in enumerate(page_sets):
        page_columns = min(columns, len(page_steps))
        image_w, image_h = frames[page_steps[0]].size
        cell_w, cell_h = image_w * scale + 16, image_h * scale + 38
        rows = max(1, math.ceil(len(page_steps) / page_columns))
        canvas = Image.new("RGB", (page_columns * cell_w, rows * cell_h), "#20242a")
        draw = ImageDraw.Draw(canvas)
        for idx, step in enumerate(page_steps):
            frame = frames[step].resize((image_w * scale, image_h * scale), Image.Resampling.NEAREST)
            x = (idx % page_columns) * cell_w + 8
            y = (idx // page_columns) * cell_h + 5
            canvas.paste(frame, (x, y))
            label = f"step {step}" if has_native else f"step {step}  LEGACY UPSCALE"
            draw.text((x, y + image_h * scale + 6), label, fill="white")
        if len(page_sets) == 1:
            target = output
        else:
            target = output.with_name(f"{output.stem}_page{page_i + 1:03d}{output.suffix or '.png'}")
        canvas.save(target)
        outputs.append(target)
    return outputs


def make_contact_sheet(path: Path, output: Path, sample_every: int = 4, columns: int = 8,
                       scale: int = 1, max_cells: int = 64) -> Path:
    """Compatibility wrapper returning the first page path."""
    return make_contact_sheets(path, output, sample_every, columns, scale, max_cells)[0]


def make_review_gif(path: Path, output: Path, gif_scale: int = 2) -> Path:
    if gif_scale < 1:
        raise ValueError("gif_scale must be positive")
    frames, _has_native, frame_skip = _load_review_frames(path)
    ep = _safe_load(path, "point")
    if ep is None or not frames:
        raise ValueError(f"Could not load frames for {path}")
    first_w, first_h = frames[0].size
    size = (first_w * gif_scale, first_h * gif_scale)
    rendered = [frame.resize(size, Image.Resampling.NEAREST).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
                for frame in frames]
    # NES timing is about 60.1 raw frames/second. GIF viewers usually quantize
    # durations; 20ms is the shortest widely respected delay for skip=1.
    duration_ms = max(20, int(round(1000.0 * frame_skip / 60.1)))
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(output, save_all=True, append_images=rendered[1:], loop=0,
                     duration=duration_ms, optimize=False, disposal=2)
    return output


def _selected_indices(raw: str, n: int) -> list[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(n))
    vals = []
    for p in raw.split(","):
        if p.strip():
            i = int(p.strip())
            if not 0 <= i < n:
                raise ValueError(f"index {i} outside 0..{n-1}")
            vals.append(i)
    return sorted(set(vals))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL {path.name} line {line_no}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")


def _parse_state_fields(raw: str) -> list[str]:
    if not raw.strip() or raw.strip().lower() == "all":
        return DEFAULT_STATE_FIELDS.copy()
    fields = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(fields) - STATE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown state fields: {', '.join(unknown)}. Allowed: {', '.join(sorted(STATE_FIELDS))}")
    return list(dict.fromkeys(fields))


def _parse_xy(raw: str, *, allow_signed: bool = False) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",")]
    if len(values) != 2 or any(not np.isfinite(x) for x in values):
        raise ValueError("expected two finite comma-separated numbers x,y")
    if not allow_signed and any(not 0 <= x <= 1 for x in values):
        raise ValueError("normalized coordinates must both be in [0,1]")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", default="demonstrations")
    parser.add_argument("--make-sheets", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--sheet-out", default="contact_sheets")
    parser.add_argument("--sample-every", type=int, default=4,
                        help="Show every N saved observation steps in sheets; does not alter recording/training timing.")
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--scale", type=int, default=1, help="Nearest-neighbor display magnification for review sheets.")
    parser.add_argument("--max-cells", type=int, default=64, help="Maximum frames per contact-sheet page.")
    parser.add_argument("--review-gifs", action="store_true", help="Create an animated GIF replay for each selected point.")
    parser.add_argument("--gif-scale", type=int, default=2)
    parser.add_argument("--state-labels", action="store_true", help="Also annotate state coordinates and/or events for selected frames.")
    parser.add_argument("--state-labels-only", action="store_true", help="Only write state annotations; do not alter task annotations.")
    parser.add_argument("--state-fields", default="",
                        help="Comma-separated state fields. Default/all asks for coordinates, velocity, contact time, and all listed events.")
    args = parser.parse_args()
    if args.state_labels_only:
        args.state_labels = True
    if args.sample_every < 1 or args.columns < 1 or args.scale < 1 or args.max_cells < 1 or args.gif_scale < 1:
        parser.error("sample-every, columns, scale, max-cells, and gif-scale must be positive")
    try:
        state_fields = _parse_state_fields(args.state_fields) if args.state_labels else []
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(args.demo_root).resolve()
    files = iter_point_files(root)
    if not files:
        raise SystemExit(f"No recorded point .npz files found under {root}")
    point_rows = []
    for i, path in enumerate(files):
        ep = _safe_load(path, "point")
        meta = _metadata(path)
        has_native = bool(meta.get("has_native_frames"))
        point_rows.append((i, path, ep, meta))
        print(f"[{i:04d}] steps={ep.length if ep else '?':>5} outcome={ep.outcome if ep else 'invalid':8} "
              f"nativeRGB={'yes' if has_native else 'NO (legacy upscale)'} tag={meta.get('scenario_tag','-')} path={path.relative_to(root)}")
    sheet_root = Path(args.sheet_out)
    if not sheet_root.is_absolute():
        sheet_root = root / sheet_root

    def create_review_assets(i: int, path: Path, ep) -> Path:
        base = sheet_root / f"{i:04d}_{path.stem}.png"
        sheets = make_contact_sheets(path, base, args.sample_every, args.columns, args.scale, args.max_cells)
        for sheet in sheets:
            print(f"contact sheet: {sheet}")
        if args.review_gifs:
            gif_path = sheet_root / f"{i:04d}_{path.stem}.gif"
            make_review_gif(path, gif_path, args.gif_scale)
            print(f"animated replay: {gif_path}")
        return sheets[0]

    if args.make_sheets:
        for i, path, ep, meta in point_rows:
            if ep is not None:
                create_review_assets(i, path, ep)
    if not args.interactive:
        return 0

    raw = input("Point indices to annotate (comma separated or all): ")
    try:
        chosen = _selected_indices(raw, len(point_rows))
    except ValueError as exc:
        parser.error(str(exc))
    task_path = root / "task_annotations.jsonl"
    task_rows = _read_jsonl(task_path)
    state_path = root / "state_annotations.jsonl"
    state_rows = _read_jsonl(state_path)
    valid_labels = set(DEFAULT_LEAF_TASKS) | set(LEGACY_PHASE_LABELS)

    for i in chosen:
        _, path, ep, meta = point_rows[i]
        if ep is None:
            continue
        rel = path.relative_to(root).as_posix()
        sheet = create_review_assets(i, path, ep)
        try:
            if os.name == "nt":
                os.startfile(str(sheet))  # noqa: S606 - local generated image
        except OSError:
            pass

        if not args.state_labels_only:
            task_rows = [r for r in task_rows if str(r.get("path")) != rel]
            print("Enter task ranges as start,end,task[,manager_duration] (end exclusive); blank line ends this point.")
            print("manager_duration is optional and must be 1, 2, 4, or 8 observation steps; leave blank if unsure.")
            print("Leaf tasks:", ", ".join(DEFAULT_LEAF_TASKS))
            print("Broad labels (phase only): serve, receive, rally/baseline, transition/wait")
            while True:
                raw_line = input("task> ").strip()
                if not raw_line:
                    break
                parts = [x.strip() for x in raw_line.split(",")]
                if len(parts) not in (3, 4):
                    print("Expected start,end,task[,manager_duration], e.g. 40,68,stroke_timing,2")
                    continue
                try:
                    start, end, label = int(parts[0]), int(parts[1]), parts[2]
                    if label not in valid_labels:
                        raise ValueError(f"unknown label {label!r}")
                    if not 0 <= start < end <= ep.length:
                        raise ValueError(f"range must satisfy 0 <= start < end <= {ep.length}")
                    overlaps = any(
                        str(row.get("path")) == rel and int(row.get("start", -1)) < end and start < int(row.get("end", -1))
                        for row in task_rows
                    )
                    if overlaps:
                        raise ValueError("task intervals for one point must not overlap")
                    row: dict[str, Any] = {"path": rel, "start": start, "end": end,
                                           "skill": label, "phase": phase_for_task(label)}
                    if len(parts) == 4 and parts[3]:
                        manager_duration = int(parts[3])
                        if manager_duration not in (1, 2, 4, 8):
                            raise ValueError("manager_duration must be 1, 2, 4, or 8")
                        row["manager_duration"] = manager_duration
                    if label in TASK_TO_PHASE:
                        row["leaf_task"] = label
                    task_rows.append(row)
                except ValueError as exc:
                    print(f"Invalid annotation: {exc}")

        if args.state_labels:
            source_frames = _load_review_frames(path)[1]
            effective_fields = state_fields.copy()
            if not source_frames and any(field in COORD_FIELDS for field in effective_fields):
                print("WARNING: this point has no native RGB frames. Coordinate labels are disabled for this legacy upscaled point.")
                effective_fields = [field for field in effective_fields if field not in COORD_FIELDS]
            print(f"State fields for this point: {', '.join(effective_fields) or '(none)'}")
            print("Coordinates use normalized native-frame positions: (0,0)=top-left, (1,1)=bottom-right; click the sprite center.")
            while True:
                raw_step = input("state step to annotate (blank to finish): ").strip()
                if not raw_step:
                    break
                try:
                    step = int(raw_step)
                    if not 0 <= step < ep.length:
                        raise ValueError(f"step must be in 0..{ep.length - 1}")
                    row: dict[str, Any] = {"path": rel, "step": step}
                    if "ball_xy" in effective_fields:
                        value = input("ball_xy normalized x,y (blank unknown): ").strip()
                        if value:
                            row["ball_xy"] = _parse_xy(value)
                    if "player_xy" in effective_fields:
                        value = input("player_xy normalized x,y (blank unknown): ").strip()
                        if value:
                            row["player_xy"] = _parse_xy(value)
                    if "ball_velocity_xy" in effective_fields:
                        value = input("ball_velocity_xy signed vx,vy per observation (blank unknown): ").strip()
                        if value:
                            row["ball_velocity_xy"] = _parse_xy(value, allow_signed=True)
                    if "time_to_contact" in effective_fields:
                        value = input("time_to_contact in observation intervals (blank unknown): ").strip()
                        if value:
                            time_to_contact = float(value)
                            if not np.isfinite(time_to_contact) or time_to_contact < 0:
                                raise ValueError("time_to_contact must be a finite non-negative number")
                            row["time_to_contact"] = time_to_contact
                    events: dict[str, int] = {}
                    for event in EVENT_FIELDS:
                        if event not in effective_fields:
                            continue
                        answer = input(f"event {event} [1=yes, 0=no, blank=unknown]: ").strip()
                        if answer in ("0", "1"):
                            events[event] = int(answer)
                    if events:
                        row["events"] = events
                    if len(row) > 2:
                        state_rows = [r for r in state_rows if not (str(r.get("path")) == rel and int(r.get("step", -1)) == step)]
                        state_rows.append(row)
                    else:
                        print("No known values entered; no label written for this step.")
                except (ValueError, TypeError) as exc:
                    print(f"Invalid state annotation: {exc}")

    if not args.state_labels_only:
        _write_jsonl(task_path, task_rows)
        print(f"Saved task annotations: {task_path} ({len(task_rows)} rows)")
    if args.state_labels:
        _write_jsonl(state_path, state_rows)
        print(f"Saved state annotations: {state_path} ({len(state_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
