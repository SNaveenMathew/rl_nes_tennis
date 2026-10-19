"""Demonstration dataset auditing, match-safe splits, and truthful segment cataloging.

Usage:
    python -m tennis_rl.dataset stats --demo-root demonstrations
    python -m tennis_rl.dataset extract --demo-root demonstrations --segment-length 16 --stride 8
    python -m tennis_rl.dataset split --demo-root demonstrations --holdout-fraction 0.2

Segment catalogs store metadata only (source point, [start,end), raw reward-to-go,
explicit task label, and match-level split). Frames/actions remain in the original
NPZ files, preventing duplicated multi-GB segment archives and leakage across
train/validation/holdout windows from the same match.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .demonstrations import _safe_load, iter_point_files
from .hierarchy_schema import TASK_TO_PHASE, phase_for_task
from .scenarios import split_for_match


def _metadata(path: Path) -> dict[str, Any]:
    """Read compact NPZ metadata and merge the recorder's JSON sidecar.

    NPZ fields are authoritative for numeric identifiers/outcomes; the sidecar
    carries human tags and recorder details that would otherwise be invisible
    to dataset audits.
    """
    out: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in ("level", "point_index", "match_id", "outcome", "serve_mode", "recorded_game_number"):
                if key in data:
                    value = np.asarray(data[key]).item()
                    out[key] = value.decode() if isinstance(value, bytes) else value
            if "pulse_frames" in data:
                pulses = np.asarray(data["pulse_frames"], dtype=np.int64).reshape(-1)
                actions = np.asarray(data["actions"], dtype=np.int64).reshape(-1) if "actions" in data else np.zeros_like(pulses)
                known = pulses > 0
                shot = (actions % 3) != 0
                out["pulse_labels"] = int((known & shot).sum())
                out["shot_actions"] = int(shot.sum())
                out["pulse_unknown_shot_actions"] = int((shot & ~known).sum())
    except (OSError, ValueError, KeyError):
        pass
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            row = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                for key, value in row.items():
                    out.setdefault(key, value)
        except (OSError, json.JSONDecodeError):
            out["sidecar_error"] = True
    return out


def _task_annotations(root: Path) -> dict[str, list[dict[str, Any]]]:
    f = root / "task_annotations.jsonl"
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not f.exists():
        return out
    for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
            path = (root / str(row["path"])).resolve()
            label = str(row.get("leaf_task") or row.get("skill") or "")
            start, end = int(row["start"]), int(row["end"])
            if not path.is_file() or start < 0 or end <= start:
                raise ValueError("missing source file or invalid interval")
            if phase_for_task(label) is None:
                raise ValueError(f"unknown task label {label!r}")
            row["_resolved_path"] = str(path)
            row["_phase"] = phase_for_task(label)
            row["_leaf_task"] = label if label in TASK_TO_PHASE else ""
            out[str(path)].append(row)
        except Exception as exc:
            raise ValueError(f"Invalid task annotation line {line_no}: {exc}") from exc
    return out


def dataset_stats(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    files = iter_point_files(root)
    counts: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    by_match: dict[str, Counter[str]] = defaultdict(Counter)
    total_frames = 0
    malformed: list[str] = []
    match_paths: set[str] = set()
    scenario_tags: Counter[str] = Counter()
    pulse_coverage = Counter()
    for path in files:
        meta = _metadata(path)
        ep = _safe_load(path, "point")
        if ep is None:
            malformed.append(str(path.relative_to(root)))
            counts["invalid_points"] += 1
            continue
        match_key = f"level_{ep.level}/{ep.match_id or path.parent.name}"
        match_paths.add(match_key)
        counts["points"] += 1
        counts[f"point_outcome_{ep.outcome}"] += 1
        counts["steps"] += ep.length
        total_frames += ep.length
        by_level[f"level_{ep.level}"] += 1
        by_match[match_key][ep.outcome] += 1
        if meta.get("serve_mode"):
            counts[f"serve_mode_{meta['serve_mode']}"] += 1
        tag = str(meta.get("scenario_tag") or "untagged").strip() or "untagged"
        scenario_tags[tag] += 1
        pulse_coverage["shot_actions"] += int(meta.get("shot_actions", 0))
        pulse_coverage["known_pulse_labels"] += int(meta.get("pulse_labels", 0))
        pulse_coverage["unknown_shot_pulse_labels"] += int(meta.get("pulse_unknown_shot_actions", 0))
    annotations = _task_annotations(root)
    annotation_counts = Counter()
    for rows in annotations.values():
        for row in rows:
            label = str(row.get("_leaf_task", "") or "phase_only")
            annotation_counts[f"leaf:{label}" if label != "phase_only" else f"phase:{row['_phase']}"] += 1
    state_annotation_count = 0
    state_file = root / "state_annotations.jsonl"
    if state_file.exists():
        state_annotation_count = sum(1 for x in state_file.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#"))
    return {
        "demo_root": str(root),
        "point_files_seen": len(files),
        "valid_points": int(counts["points"]),
        "invalid_points": int(counts["invalid_points"]),
        "matches": len(match_paths),
        "total_steps": int(counts["steps"]),
        "average_steps_per_point": round(total_frames / max(int(counts["points"]), 1), 3),
        "point_outcomes": {k.removeprefix("point_outcome_"): int(v) for k, v in counts.items() if k.startswith("point_outcome_")},
        "by_level": dict(sorted(by_level.items())),
        "serve_mode_counts": {k.removeprefix("serve_mode_"): int(v) for k, v in counts.items() if k.startswith("serve_mode_")},
        "scenario_tag_counts": dict(sorted(scenario_tags.items())),
        "button_pulse_label_coverage": {
            **{key: int(value) for key, value in pulse_coverage.items()},
            "known_fraction_of_shot_actions": round(pulse_coverage["known_pulse_labels"] / max(pulse_coverage["shot_actions"], 1), 4),
        },
        "task_annotation_segments": int(sum(len(x) for x in annotations.values())),
        "task_annotation_counts": dict(sorted(annotation_counts.items())),
        "state_annotation_frames": int(state_annotation_count),
        "unlabeled_point_files": max(len(files) - len(annotations), 0),
        "malformed_examples": malformed[:25],
    }


def extract_segment_catalog(
    root: str | Path,
    *,
    segment_length: int = 16,
    stride: int = 8,
    holdout_fraction: float = 0.2,
    seed: int = 42,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    if segment_length < 1 or stride < 1:
        raise ValueError("segment_length and stride must be positive")
    if not (0.0 <= holdout_fraction < 0.5):
        raise ValueError("holdout_fraction must be in [0,0.5)")
    task_annotations = _task_annotations(root)
    # A match id always maps to the same split, including all points/windows in it.
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    lengths = Counter()
    match_split: dict[str, str] = {}
    for path in iter_point_files(root):
        ep = _safe_load(path, "point")
        if ep is None:
            continue
        match_id = ep.match_id or path.parent.name
        split = split_for_match(match_id, holdout_fraction)
        match_split[match_id] = split
        rel = path.relative_to(root).as_posix()
        annotations = task_annotations.get(str(path.resolve()), [])
        rtg = np.zeros(ep.length, dtype=np.float32)
        g = 0.0
        for i in range(ep.length - 1, -1, -1):
            g = float(ep.rewards[i]) + 0.99 * g
            rtg[i] = g
        for start in range(0, ep.length - segment_length + 1, stride):
            end = start + segment_length
            covering = [row for row in annotations if int(row["start"]) <= start and int(row["end"]) >= end]
            leaf_task = ""
            phase = ""
            if covering:
                row = covering[0]
                label = str(row.get("leaf_task") or row.get("skill") or "")
                phase = str(row.get("_phase") or phase_for_task(label) or "")
                leaf_task = label if label in TASK_TO_PHASE else ""
            role = "unknown"
            # Scenario role comes from explicit match metadata when present. No inference from pixels/actions.
            meta = _metadata(path)
            serve_mode = str(meta.get("serve_mode", ""))
            if serve_mode in ("serve-first", "serve"):
                role = "serve"
            elif serve_mode in ("receive-first", "return"):
                role = "receive"
            record = {
                "segment_id": f"{match_id}:{ep.point_index}:{start}-{end}",
                "source_path": rel,
                "match_id": match_id,
                "level": ep.level,
                "point_index": ep.point_index,
                "start": start,
                "end": end,
                "length": end - start,
                "split": split,
                "role": role,
                "point_outcome": ep.outcome,
                "phase": phase,
                "leaf_task": leaf_task,
                "label_status": "leaf_labeled" if leaf_task else ("phase_labeled" if phase else "unlabeled"),
                "segment_reward_sum": float(np.sum(ep.rewards[start:end])),
                "reward_to_go_first": float(rtg[start]),
                "reward_to_go_mean": float(np.mean(rtg[start:end])),
            }
            records.append(record)
            counters[f"split:{split}"] += 1
            counters[f"label:{record['label_status']}"] += 1
            counters[f"outcome:{ep.outcome}"] += 1
            counters[f"role:{role}"] += 1
            if leaf_task:
                counters[f"task:{leaf_task}"] += 1
            if phase:
                counters[f"phase:{phase}"] += 1
            lengths[record["length"]] += 1
    out_path = Path(output) if output else root / "segment_catalog.jsonl"
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "demo_root": str(root), "catalog_path": str(out_path),
        "segment_length": segment_length, "stride": stride,
        "segments": len(records), "matches": len(match_split),
        "split_by_match": dict(sorted(Counter(match_split.values()).items())),
        "counts": dict(sorted(counters.items())),
        "length_counts": dict(sorted(lengths.items())),
        "note": "Catalog contains metadata only; source observations/actions remain in the original NPZ files.",
    }
    (out_path.parent / "segment_catalog_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_match_safe_splits(root: str | Path, holdout_fraction: float = 0.2, seed: int = 42) -> dict[str, Any]:
    """Write manifest assigning *whole source matches* to train/validation/holdout."""
    root = Path(root).resolve()
    if not (0.0 < holdout_fraction < 0.5):
        raise ValueError("holdout_fraction must be in (0,0.5)")
    # Existing split_for_match is a deterministic hash split. Seed is carried in the manifest.
    matches: dict[str, list[Path]] = defaultdict(list)
    for path in iter_point_files(root):
        meta = _metadata(path)
        key = str(meta.get("match_id") or path.parent.name)
        matches[key].append(path)
    rows = []
    for match_id, files in sorted(matches.items()):
        rows.append({
            "match_id": match_id,
            "split": split_for_match(match_id, holdout_fraction),
            "point_files": len(files),
            "source_paths": [p.relative_to(root).as_posix() for p in files],
            "seed_metadata": int(seed),
        })
    target = root / "match_splits.jsonl"
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {"path": str(target), "matches": len(rows), "splits": dict(Counter(row["split"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stats", "extract", "split"):
        p = sub.add_parser(name)
        p.add_argument("--demo-root", default="demonstrations")
        if name == "extract":
            p.add_argument("--segment-length", type=int, default=16)
            p.add_argument("--stride", type=int, default=8)
            p.add_argument("--holdout-fraction", type=float, default=0.2)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--output", default=None)
        elif name == "split":
            p.add_argument("--holdout-fraction", type=float, default=0.2)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "stats":
        result = dataset_stats(args.demo_root)
    elif args.command == "extract":
        result = extract_segment_catalog(args.demo_root, segment_length=args.segment_length,
            stride=args.stride, holdout_fraction=args.holdout_fraction, seed=args.seed, output=args.output)
    else:
        result = write_match_safe_splits(args.demo_root, args.holdout_fraction, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
