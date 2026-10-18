"""Re-segment demonstrations recorded with the old point_end-based boundary.

The old recorder cut a "point" every time RAM flag point_end was non-zero (~30 steps around every
real point), so a match holds ~250 one-step fragments and the real rallies carry no reward. The
fragments are still consecutive, so a match can be rebuilt: concatenate them in order and cut after
each non-zero reward.

    python tools/repair_demos.py --root demonstrations            # dry run
    python tools/repair_demos.py --root demonstrations --apply    # rewrite (originals kept in <root>/_pre_repair_backup)

Leading dead-time steps (no-ops after the previous score update) cannot be trimmed here because the
flag was not stored; they are harmless no-op steps. States harvested by the old recorder are NOT
fixed: delete scenario_states and harvest again with the new recorder.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def repair_match(match_dir: Path, backup_root: Path, root: Path, apply: bool) -> str:
    meta_path = match_dir / "match.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("segmentation") == "score-change-v2":
        return "already v2, skipped"
    files = sorted(match_dir.glob("point_*.npz"))
    if not files:
        return "no points"
    parts = [load(f) for f in files]
    parts.sort(key=lambda d: int(d["point_index"]))
    obs = np.concatenate([d["observations"] for d in parts])
    act = np.concatenate([d["actions"] for d in parts])
    rew = np.concatenate([d["rewards"] for d in parts])
    don = np.concatenate([d["dones"] for d in parts])
    cuts = [i + 1 for i in np.flatnonzero(np.abs(rew) > 1e-6)]
    if not cuts:
        return "no rewards found, left alone"
    segs, start = [], 0
    for c in cuts:
        segs.append((start, c)); start = c
    dropped = len(rew) - start                      # trailing steps after the last reward
    summary = f"{len(files)} fragments -> {len(segs)} points ({[round(float(rew[a:b].sum()), 1) for a, b in segs]}), {dropped} trailing steps dropped"
    if not apply:
        return summary
    rel = match_dir.relative_to(root)
    dest = backup_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(match_dir), str(dest))
    match_dir.mkdir(parents=True)
    first = parts[0]
    for n, (a, b) in enumerate(segs, 1):
        np.savez_compressed(
            match_dir / f"point_{n:04d}.npz", observations=obs[a:b], actions=act[a:b], rewards=rew[a:b], dones=don[a:b],
            level=first["level"], outcome=first["outcome"], match_id=first["match_id"], point_index=np.asarray(n, np.int32))
        (match_dir / f"point_{n:04d}.json").write_text(json.dumps({"repaired_from_fragments": True, "point_index": n}), encoding="utf-8")
    meta.update({"segmentation": "score-change-v2", "points": len(segs), "repaired": True})
    (match_dir / "match.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="demonstrations")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    backup = root / "_pre_repair_backup"
    for md in sorted(root.glob("level_*/*/match_*")):
        print(f"{md.relative_to(root)}: {repair_match(md, backup, root, a.apply)}")
    if not a.apply:
        print("dry run only; add --apply to rewrite")


if __name__ == "__main__":
    main()
