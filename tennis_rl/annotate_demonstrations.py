"""Make contact sheets and interactively annotate task intervals in recorded points.

A task annotation labels the half-open observation/action range [start, end).
Example: 40,68,stroke_timing. Broad labels such as `rally` are allowed for
manager phase supervision but do not train a leaf actor. Use precise leaf labels
only when you can identify the behavior from the contact sheet / replay.

Usage:
  python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --make-sheets
  python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --interactive
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .demonstrations import _safe_load, iter_point_files
from .hierarchy_schema import DEFAULT_LEAF_TASKS, LEGACY_PHASE_LABELS, phase_for_task, TASK_TO_PHASE


def _metadata(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as d:
            for key in ("match_id", "point_index", "level", "outcome", "pulse_frames"):
                if key in d:
                    val = np.asarray(d[key]).item() if key != "pulse_frames" else None
                    if val is not None:
                        out[key] = val.decode("utf-8") if isinstance(val, bytes) else val
    except Exception:
        pass
    side = path.with_suffix(".json")
    if side.exists():
        try:
            obj = json.loads(side.read_text(encoding="utf-8"))
            if isinstance(obj, dict): out.update(obj)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def make_contact_sheet(path: Path, output: Path, sample_every: int = 4, columns: int = 8) -> Path:
    ep = _safe_load(path, "point")
    if ep is None:
        raise ValueError(f"Could not load point file {path}")
    if sample_every < 1 or columns < 1:
        raise ValueError("sample_every and columns must be positive")
    frames = list(range(0, ep.length, sample_every))
    if ep.length and (not frames or frames[-1] != ep.length - 1): frames.append(ep.length - 1)
    cell_w, cell_h = 172, 112
    rows = max(1, math.ceil(len(frames) / columns))
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "#20242a")
    draw = ImageDraw.Draw(canvas)
    for idx, frame_no in enumerate(frames):
        obs = ep.observations[frame_no]
        image_arr = obs[-1] if obs.ndim == 3 else obs
        frame = Image.fromarray(image_arr.astype(np.uint8, copy=False), mode="L").convert("RGB")
        frame = frame.resize((cell_w - 12, cell_h - 30), Image.Resampling.NEAREST)
        x, y = (idx % columns) * cell_w + 6, (idx // columns) * cell_h + 5
        canvas.paste(frame, (x, y))
        draw.text((x, y + cell_h - 28), f"step {frame_no}", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _selected_indices(raw: str, n: int) -> list[int]:
    raw = raw.strip().lower()
    if raw == "all": return list(range(n))
    vals = []
    for p in raw.split(","):
        if p.strip():
            i = int(p.strip())
            if not 0 <= i < n: raise ValueError(f"index {i} outside 0..{n-1}")
            vals.append(i)
    return sorted(set(vals))


def _read_task_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "task_annotations.jsonl"
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", default="demonstrations")
    parser.add_argument("--make-sheets", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--sheet-out", default="contact_sheets")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--state-labels", action="store_true", help="Also interactively annotate coordinates/events for selected frames")
    args = parser.parse_args()
    root = Path(args.demo_root).resolve()
    files = iter_point_files(root)
    if not files:
        raise SystemExit(f"No recorded point .npz files found under {root}")
    point_rows = []
    for i, path in enumerate(files):
        ep = _safe_load(path, "point"); meta = _metadata(path)
        point_rows.append((i, path, ep, meta))
        print(f"[{i:04d}] steps={ep.length if ep else '?':>5} outcome={ep.outcome if ep else 'invalid':8} "
              f"tag={meta.get('scenario_tag','-')} path={path.relative_to(root)}")
    sheet_root = Path(args.sheet_out)
    if not sheet_root.is_absolute(): sheet_root = root / sheet_root
    if args.make_sheets:
        for i, path, ep, meta in point_rows:
            if ep is not None:
                target = make_contact_sheet(path, sheet_root / f"{i:04d}_{path.stem}.png", args.sample_every, args.columns)
                print(f"contact sheet: {target}")
    if not args.interactive:
        return 0
    raw = input("Point indices to annotate (comma separated or all): ")
    try: chosen = _selected_indices(raw, len(point_rows))
    except ValueError as exc: parser.error(str(exc))
    task_path = root / "task_annotations.jsonl"
    task_rows = _read_task_rows(root)
    state_path = root / "state_annotations.jsonl"
    state_rows = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")] if state_path.exists() else []
    valid_labels = set(DEFAULT_LEAF_TASKS) | set(LEGACY_PHASE_LABELS)
    for i in chosen:
        _, path, ep, _meta = point_rows[i]
        if ep is None: continue
        rel = path.relative_to(root).as_posix()
        sheet = make_contact_sheet(path, sheet_root / f"{i:04d}_{path.stem}.png", args.sample_every, args.columns)
        print(f"\nSelected {rel} ({ep.length} steps). Contact sheet: {sheet}")
        try:
            if os.name == "nt": os.startfile(str(sheet))  # noqa: S606 - local generated image
        except OSError:
            pass
        task_rows = [r for r in task_rows if str(r.get("path")) != rel]
        print("Enter task ranges as start,end,task[,manager_duration] (end exclusive); blank line ends this point.")
        print("Optional manager_duration is a deliberate option duration in observation steps: 1, 2, 4, or 8. Leave it blank when unsure.")
        print("Leaf tasks:", ", ".join(DEFAULT_LEAF_TASKS))
        print("Broad labels (phase only): serve, receive, rally/baseline, transition/wait")
        while True:
            raw_line = input("task> ").strip()
            if not raw_line: break
            parts = [x.strip() for x in raw_line.split(",")]
            if len(parts) not in (3, 4):
                print("Expected start,end,task[,manager_duration] e.g. 40,68,stroke_timing,2"); continue
            try:
                start, end, label = int(parts[0]), int(parts[1]), parts[2]
                if label not in valid_labels: raise ValueError(f"unknown label {label!r}")
                if not 0 <= start < end <= ep.length: raise ValueError(f"range must satisfy 0 <= start < end <= {ep.length}")
                row = {"path": rel, "start": start, "end": end, "skill": label, "phase": phase_for_task(label)}
                if len(parts) == 4 and parts[3]:
                    manager_duration = int(parts[3])
                    if manager_duration not in (1, 2, 4, 8):
                        raise ValueError("manager_duration must be 1, 2, 4, or 8")
                    row["manager_duration"] = manager_duration
                if label in TASK_TO_PHASE: row["leaf_task"] = label
                task_rows.append(row)
            except ValueError as exc:
                print(f"Invalid annotation: {exc}")
        if args.state_labels:
            while True:
                raw_step = input("state step to annotate (blank to finish): ").strip()
                if not raw_step: break
                try:
                    step = int(raw_step)
                    if not 0 <= step < ep.length: raise ValueError(f"step must be in 0..{ep.length-1}")
                    row: dict[str, Any] = {"path": rel, "step": step}
                    for key in ("ball_xy", "player_xy"):
                        value = input(f"{key} normalized x,y (blank unknown): ").strip()
                        if value:
                            xy = [float(x.strip()) for x in value.split(",")]
                            if len(xy) != 2 or any(not np.isfinite(x) or not 0 <= x <= 1 for x in xy): raise ValueError(f"{key} must be two values in [0,1]")
                            row[key] = xy
                    value = input("ball_velocity_xy signed vx,vy (blank unknown): ").strip()
                    if value:
                        xy = [float(x.strip()) for x in value.split(",")]
                        if len(xy) != 2 or any(not np.isfinite(x) for x in xy): raise ValueError("velocity must be two finite numbers")
                        row["ball_velocity_xy"] = xy
                    value = input("time_to_contact in observation intervals (blank unknown): ").strip()
                    if value: row["time_to_contact"] = max(0.0, float(value))
                    events = {}
                    for event in ("serve_ready", "agent_serve", "cpu_serve", "ball_in_play", "ball_toward_agent", "ball_away_from_agent", "opponent_contact_window", "agent_contact_window", "high_floater_lob", "point_resolved", "uncertain"):
                        answer = input(f"event {event} [1=yes, 0=no, blank=unknown]: ").strip()
                        if answer in ("0", "1"): events[event] = int(answer)
                    if events: row["events"] = events
                    state_rows = [r for r in state_rows if not (r.get("path") == rel and int(r.get("step", -1)) == step)]
                    state_rows.append(row)
                except (ValueError, TypeError) as exc:
                    print(f"Invalid state annotation: {exc}")
        _write_jsonl(task_path, task_rows)
        if args.state_labels: _write_jsonl(state_path, state_rows)
        print(f"Saved task annotations: {task_path} ({len(task_rows)} rows)")
        if args.state_labels: print(f"Saved state annotations: {state_path} ({len(state_rows)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
