"""Tag recorded demonstration points with the scenario they started from.

The recorder stores the *match* outcome in each point file's ``outcome`` field,
and the sidecar's ``provisional_point_outcome`` calls any zero-reward segment
"losing".  Neither is a reliable per-point label, so this module rebuilds the
truth from the per-step rewards (point win/loss rewards are +/-, other steps are
0) and from the rules of tennis:

    start score  = replay of earlier points in the same game (see advance_score)
    role         = serve-first -> "serve";  receive-first -> "return"
    point result = sign(sum(rewards)):  >0 win, <0 loss, 0 non-scoring segment

A game whose replay does not finish exactly at game-over is flagged
``consistent=False`` (a missing point file, or a boundary that was not a score
change) and its cells should be treated with suspicion.

Run:  python -m tennis_rl.scenario_index --demo-root demonstrations
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .scenarios import (
    LENGTH_BUCKETS,
    advance_score,
    build_catalog,
    cell_id,
    length_bucket,
)


def _point_result(rewards: np.ndarray) -> str:
    total = float(np.sum(rewards))
    if total > 1e-6:
        return "win"
    if total < -1e-6:
        return "loss"
    return "none"


def _role_from_meta(meta: dict) -> str | None:
    mode = meta.get("serve_mode")
    if mode == "serve-first":
        return "serve"
    if mode == "receive-first":
        return "return"
    return None


def index_demonstrations(demo_root: str | Path, late_steps: int = 45) -> dict:
    root = Path(demo_root)
    by_match: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for p in sorted(root.glob("level_*/*/match_*/point_*.npz")):
        level = int(p.parents[2].name.split("_")[1])
        by_match[(level, p.parent.name)].append(p)

    points: list[dict] = []
    matches: list[dict] = []
    for (level, match_dir), paths in sorted(by_match.items()):
        rows = []
        for path in sorted(paths, key=lambda q: q.name):
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            try:
                with np.load(path, allow_pickle=False) as d:
                    rewards = np.asarray(d["rewards"], dtype=np.float32)
                    steps = int(np.asarray(d["actions"]).shape[0])
                    match_outcome = str(np.asarray(d["outcome"]).item())
                    point_index = int(np.asarray(d["point_index"]).item())
            except (OSError, ValueError, KeyError):
                continue
            rows.append((point_index, path, meta, rewards, steps, match_outcome))
        rows.sort(key=lambda r: r[0])
        if not rows:
            continue

        role = None
        for _, _, meta, *_ in rows:
            role = _role_from_meta(meta)
            if role:
                break
        p, o = 0, 0
        game_finished = False
        ok = role is not None
        seq_indices = [r[0] for r in rows]
        if seq_indices != list(range(seq_indices[0], seq_indices[0] + len(seq_indices))) or seq_indices[0] != 1:
            ok = False  # missing/extra point files

        match_points = []
        for point_index, path, meta, rewards, steps, match_outcome in rows:
            result = _point_result(rewards)
            rec = {
                "file": str(path.relative_to(root)),
                "level": level,
                "match_id": match_dir.replace("match_", "", 1),
                "point_index": point_index,
                "steps": steps,
                "length_bucket": length_bucket(steps),
                "point_result": result,
                "match_outcome": match_outcome,
                "sidecar_outcome": meta.get("provisional_point_outcome"),
                "role": role,
                "start_score": [p, o] if role else None,
                "cell_id": cell_id(role, p, o) if role else None,
                "terminal_window": [max(0, steps - late_steps), steps],
            }
            if result != "none" and role:
                if game_finished:
                    ok = False
                p, o, over, _ = advance_score(p, o, result == "win")
                game_finished = game_finished or over
            match_points.append(rec)
        ok = ok and game_finished
        for rec in match_points:
            rec["consistent"] = bool(ok)
        points.extend(match_points)
        matches.append(
            {
                "level": level,
                "match_id": match_dir.replace("match_", "", 1),
                "role": role,
                "points": len(match_points),
                "consistent": bool(ok),
                "outcome": rows[0][5],
            }
        )

    coverage: dict[str, dict[str, int]] = {c.cell_id: {"win": 0, "loss": 0, "none": 0} for c in build_catalog()}
    length_by_result: dict[str, dict[str, int]] = {r: {b: 0 for b in LENGTH_BUCKETS} for r in ("win", "loss", "none")}
    disagreements = 0
    for r in points:
        if r["cell_id"] and r["consistent"]:
            coverage[r["cell_id"]][r["point_result"]] += 1
        length_by_result[r["point_result"]][r["length_bucket"]] += 1
        side = r["sidecar_outcome"]
        if side and r["point_result"] != "none" and (side == "winning") != (r["point_result"] == "win"):
            disagreements += 1
    mislabeled_by_match = sum(
        1 for r in points
        if r["point_result"] != "none" and (r["match_outcome"] == "winning") != (r["point_result"] == "win")
    )
    return {
        "points": points,
        "matches": matches,
        "coverage": coverage,
        "length_by_result": length_by_result,
        "diagnostics": {
            "points_total": len(points),
            "inconsistent_matches": sum(1 for m in matches if not m["consistent"]),
            "points_whose_match_outcome_differs_from_point_result": mislabeled_by_match,
            "sidecar_vs_reward_disagreements": disagreements,
            "uncovered_cells": [c for c, v in coverage.items() if v["win"] + v["loss"] == 0],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root", default="demonstrations")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    idx = index_demonstrations(args.demo_root)
    out = Path(args.out) if args.out else Path(args.demo_root) / "scenario_index.json"
    out.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    d = idx["diagnostics"]
    print(f"points={d['points_total']} inconsistent_matches={d['inconsistent_matches']}")
    print(f"points whose MATCH outcome != POINT result: {d['points_whose_match_outcome_differs_from_point_result']}")
    print(f"uncovered cells: {len(d['uncovered_cells'])}/36")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
