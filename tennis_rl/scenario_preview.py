"""Load a harvested state in a *visible* Mesen window so you can check it.

Run this after harvesting, before training, to confirm the "start" snapshots really
are serve-ready and that the cell label (server, score, court side) is right:

    python -m tennis_rl.scenario_preview --cell "return|2-1" --phase start --seconds 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .scenario_sampler import ScenarioLibrary
from .scenarios import load_manifest
from .train import build_workers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--cell", default="", help='e.g. "serve|0-0" or "return|3-3"')
    ap.add_argument("--list", action="store_true", help="Only print which cells have harvested states (no emulator).")
    ap.add_argument("--phase", default="start", choices=("start", "mid", "late"))
    ap.add_argument("--split", default="train", choices=("train", "holdout"))
    ap.add_argument("--noop", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.mesen.workers = 1
    cfg.mesen.headless = False
    state_dir = Path(cfg.scenario.state_dir)
    manifest = state_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest.resolve()} not found: record with --harvest-dir {state_dir} first")
    lib = ScenarioLibrary(load_manifest(manifest, base_dir=state_dir))
    if args.list or not args.cell:
        cov = lib.coverage()
        print(f"{len(cov)} cells have states in {state_dir.resolve()}")
        for cid in sorted(cov):
            print(f"  {cid:14s} train={cov[cid]['train']:3d} holdout={cov[cid]['holdout']:3d}")
        return
    state = lib.pick(args.cell, args.split, args.phase, False, np.random.default_rng(0))
    if state is None:
        raise SystemExit(f"no {args.split}/{args.phase} state for {args.cell}; coverage: {lib.coverage().get(args.cell)}")
    print(f"loading {state.path}  (match {state.match_id}, point {state.point_index}, step {state.step_in_point}, "
          f"human result: {state.demo_outcome})")
    worker = build_workers(cfg, strict_lockstep=True)[0]
    try:
        worker.load_state_from(state.path, args.noop)
        end = time.time() + args.seconds
        while time.time() < end:
            worker.action(0)  # no input: just look at the position
            _, reward, done, info = worker.recv_observation()
            print(f"score {info.get('player_score')}-{info.get('opponent_score')} reward={reward:+.1f}", end="\r")
    finally:
        worker.close(force=True)


if __name__ == "__main__":
    main()
