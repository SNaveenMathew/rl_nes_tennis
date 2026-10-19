"""Replay saved emulator savestates with a hierarchy or forced leaf specialist.

This evaluates *actual emulator states*, not image-only NPZ windows. The scenario
manifest must point to savestates harvested by ``tennis_rl.record --harvest-dir``.
Each listed state is replayed ``--repeats`` times; results are written per trial
(JSONL) plus an aggregate JSON summary. Use the same manifest/repeat seed for
checkpoint comparisons.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config
from .env import ParallelTennis
from .hierarchical import HierarchicalActorCritic, HierarchicalController
from .scenario_sampler import Assignment
from .scenarios import StateRecord, load_manifest
from .staged_train import load_weights, make_model
from .train import build_workers


def _skill_filter(skill: str | None, rec: StateRecord) -> bool:
    if skill is None:
        return True
    role = (rec.cell_id or "").split("|", 1)[0]
    if skill in {"serve_timing", "serve_placement_power"}:
        return role == "serve" and rec.phase == "start"
    if skill == "serve_recovery":
        return role == "serve" and rec.kind == "mid"
    if skill == "receive_position":
        return role == "return" and rec.phase == "start"
    if skill == "return_stroke_timing":
        return role == "return" and rec.kind == "mid"
    if skill in {"intercept_track", "hold_cover", "recover_reposition", "approach_net", "retreat_lob", "stroke_selector", "stroke_timing"}:
        return rec.kind == "mid" or rec.phase in ("mid", "late")
    return True


def _load_checkpoint(model: HierarchicalActorCritic, path: Path, device: torch.device) -> dict[str, Any]:
    payload = load_weights(model, path, device)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scenario.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="scenario_states/manifest.jsonl")
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="holdout")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 means all compatible states")
    parser.add_argument("--skill", choices=["auto"] + list(__import__("tennis_rl.hierarchy_schema", fromlist=["DEFAULT_LEAF_TASKS"]).DEFAULT_LEAF_TASKS), default="auto",
                        help="Force one trained expert; use auto to evaluate learned gates")
    parser.add_argument("--out", default="runs_hierarchical/evaluation.jsonl")
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-max-steps", type=int, default=900)
    parser.add_argument("--noop-frames", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1 or args.point_max_steps < 1 or args.noop_frames < 0:
        parser.error("repeats and point-max-steps must be positive; noop-frames cannot be negative")

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    cfg.mesen.workers = 1  # deterministic scenario order makes comparisons traceable
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(cfg.device)
    model = make_model(cfg).to(device)
    payload = _load_checkpoint(model, Path(args.checkpoint), device)
    model.eval()
    forced_skill = None if args.skill == "auto" else args.skill
    if forced_skill and forced_skill not in model.skill_to_id:
        raise SystemExit(f"Checkpoint/config does not include {forced_skill!r}")

    manifest_path = Path(args.manifest).resolve()
    records = load_manifest(manifest_path, base_dir=manifest_path.parent)
    records = [r for r in records if args.split == "all" or r.split == args.split]
    records = [r for r in records if Path(r.path).is_file() and _skill_filter(forced_skill, r)]
    # One saved state per scenario repeated N times; use stable record order for
    # paired comparisons across checkpoint versions.
    records.sort(key=lambda r: (r.match_id, r.point_index, r.step_in_point, r.path))
    if args.max_scenarios > 0:
        records = records[:args.max_scenarios]
    if not records:
        raise SystemExit(f"No valid scenarios for split={args.split!r}, skill={forced_skill or 'auto'!r}, manifest={manifest_path}. Harvest savestates first.")

    trials: list[dict[str, Any]] = []
    for rec in records:
        for repeat in range(args.repeats):
            trials.append({"record": rec, "repeat": repeat})
    cursor = 0
    def provider(_worker_index: int):
        nonlocal cursor
        if cursor >= len(trials):
            return Assignment(kind="anchor")
        item = trials[cursor]; cursor += 1
        rec = item["record"]
        return Assignment(kind="scenario", cell_id=rec.cell_id,
                          stratum=rec.cell_id, phase=rec.phase,
                          state=rec, noop_frames=args.noop_frames)

    workers = build_workers(cfg)
    env = ParallelTennis(workers, stack_size=cfg.environment.stack_size,
        max_steps=args.point_max_steps, point_episodes=True,
        point_max_steps=args.point_max_steps, scenario_provider=provider,
        segment_mode=False)
    controller = HierarchicalController(model, deterministic=True)
    output_path = Path(args.out).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary).resolve() if args.summary else output_path.with_suffix(".summary.json")
    overall = Counter(); by_scenario: dict[str, Counter] = defaultdict(Counter)
    start = time.perf_counter()
    try:
        obs = env.reset()
        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            episode_step = 0
            current_trial = 0
            current_record = trials[0]["record"]
            current_repeat = trials[0]["repeat"]
            episode_reward = 0.0; episode_steps = 0; faults = 0; restarts = 0
            def finish(outcome: int, truncated: bool, phase: str, cell_id: str, rec: StateRecord, rep: int):
                key = f"{cell_id}|{phase}|{rec.kind}"
                row = {"scenario_id": key, "state_path": rec.path, "match_id": rec.match_id,
                    "point_index": rec.point_index, "step_in_point": rec.step_in_point,
                    "split": rec.split, "forced_skill": forced_skill or "auto", "repeat": rep,
                    "point_result": int(outcome), "point_result_label": "win" if outcome > 0 else ("loss" if outcome < 0 else "unknown"),
                    "truncated": bool(truncated), "episode_reward": float(episode_reward),
                    "episode_steps": int(episode_steps), "service_faults": int(faults), "bridge_restarts": int(restarts)}
                f.write(json.dumps(row, sort_keys=True) + "\n"); f.flush()
                overall["episodes"] += 1; overall["steps"] += episode_steps
                overall["reward_sum"] += episode_reward; overall["faults"] += faults; overall["bridge_restarts"] += restarts
                if outcome > 0: overall["wins"] += 1; by_scenario[key]["wins"] += 1
                elif outcome < 0: overall["losses"] += 1; by_scenario[key]["losses"] += 1
                else: overall["unknown"] += 1; by_scenario[key]["unknown"] += 1
                if truncated: overall["truncated"] += 1; by_scenario[key]["truncated"] += 1
                by_scenario[key]["episodes"] += 1; by_scenario[key]["steps"] += episode_steps
                by_scenario[key]["reward_sum"] += episode_reward; by_scenario[key]["faults"] += faults
            while current_trial < len(trials):
                tensor = torch.from_numpy(obs).to(device)
                with torch.no_grad():
                    actions, pulses = controller.act_with_pulse(tensor, forced_skill=forced_skill)
                batch = env.step(actions.cpu().numpy(), pulse_frames=pulses.cpu().numpy())
                episode_reward += float(batch.rewards[0]); episode_steps += 1
                info = batch.infos[0]
                faults += int(bool(info.get("service_fault"))); restarts += int(bool(info.get("bridge_restart")))
                obs = batch.observations.copy()
                if batch.dones[0]:
                    outcome = int(info.get("point_result", 0))
                    assignment = info.get("assignment")
                    rec = assignment.state if assignment is not None and assignment.state is not None else current_record
                    phase = str(info.get("phase") or rec.phase)
                    scenario_id = str(info.get("cell_id") or rec.cell_id or "unknown")
                    finish(outcome, bool(batch.truncateds[0]) if batch.truncateds is not None else False,
                           phase, scenario_id, rec, current_repeat)
                    # env auto-reset after completion. Reset controller's GRU/options
                    # and consume exactly the observation for the next scenario.
                    obs[0] = env.current_observation(0)
                    controller.reset()
                    current_trial += 1
                    if current_trial >= len(trials):
                        break
                    current_record = trials[current_trial]["record"]
                    current_repeat = trials[current_trial]["repeat"]
                    episode_reward = 0.0; episode_steps = 0; faults = 0; restarts = 0
            elapsed = max(time.perf_counter() - start, 1e-9)
    finally:
        env.close()
    decided = overall["wins"] + overall["losses"]
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_stage": payload.get("stage", "unknown"),
        "forced_skill": forced_skill or "auto", "split": args.split, "manifest": str(manifest_path),
        "scenario_states": len(records), "repeats": args.repeats, "episodes_requested": len(trials),
        "overall": {"episodes": overall["episodes"], "wins": overall["wins"], "losses": overall["losses"],
            "unknown_results": overall["unknown"], "win_rate_decided": overall["wins"] / decided if decided else None,
            "truncated_episodes": overall["truncated"], "mean_episode_reward": overall["reward_sum"] / max(overall["episodes"], 1),
            "mean_episode_steps": overall["steps"] / max(overall["episodes"], 1),
            "service_faults": overall["faults"], "bridge_restarts": overall["bridge_restarts"],
            "elapsed_seconds": elapsed, "episodes_per_second": overall["episodes"] / elapsed},
        "by_scenario": {key: {"episodes": v["episodes"], "wins": v["wins"], "losses": v["losses"],
            "unknown": v["unknown"], "win_rate_decided": v["wins"] / max(v["wins"]+v["losses"], 1),
            "truncated": v["truncated"], "mean_steps": v["steps"] / max(v["episodes"],1),
            "mean_reward": v["reward_sum"] / max(v["episodes"],1), "faults": v["faults"]}
            for key, v in sorted(by_scenario.items())},
        "interpretation": "Use paired comparison with the same state manifest, split, repeat count, seed, and point horizon. Forced-skill results diagnose the leaf; auto results test the learned gates plus leaves."}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2)); print(f"Trial log: {output_path}\nSummary: {summary_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
