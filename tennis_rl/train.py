from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_NAMES
from .config import load_config
from .demonstrations import DemonstrationDataset, summarize as summarize_demos, validate_demo_root_compatibility
from .dqfd import DQFDTrainer, RecurrentDuelingQ, Sequence, pretrain_from_demonstrations
from .env import ParallelTennis
from .mesen_client import MesenWorker
from .model import RecurrentActorCritic
from .ppo import PPOTrainer, Rollout, compute_gae_with_truncation
from .scenario_sampler import SamplerConfig, ScenarioLibrary, ScenarioSampler
from .scenarios import load_manifest
from .segment_replay import DemoSegmentReplayer, SegmentReplayConfig


def _prune_step_checkpoints(save_dir: Path, keep: int, prefix: str = "tennis_step_") -> None:
    save_dir = Path(save_dir)
    if keep < 1:
        raise ValueError("keep must be >= 1")
    files = (
        sorted(
            save_dir.glob(f"{prefix}*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]) if p.stem.rsplit("_", 1)[1].isdigit() else -1,
            reverse=True,
        )
        if save_dir.exists()
        else []
    )
    for stale in files[keep:]:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_workers(cfg, *, strict_lockstep: bool = False):
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{random.getrandbits(40):010x}"
    return [
        MesenWorker(
            executable=cfg.mesen.executable,
            rom=cfg.mesen.rom,
            lua_script=cfg.mesen.lua_script,
            host=cfg.mesen.host,
            port=cfg.mesen.port_start + i,
            width=cfg.mesen.observation_width,
            height=cfg.mesen.observation_height,
            frame_skip=cfg.mesen.frame_skip,
            match_games=cfg.mesen.match_games,
            game_level=cfg.mesen.game_level,
            startup_timeout=cfg.mesen.startup_timeout_s,
            socket_timeout=cfg.mesen.socket_timeout_s,
            point_win=cfg.reward.point_win,
            point_loss=cfg.reward.point_loss,
            game_win=cfg.reward.game_win,
            game_loss=cfg.reward.game_loss,
            match_win=cfg.environment.terminal_match_reward,
            match_loss=cfg.environment.terminal_match_penalty,
            ipc_dir=cfg.mesen.ipc_dir,
            run_id=run_id,
            capture_stderr=cfg.mesen.capture_stderr,
            lockstep=cfg.mesen.lockstep,
            lockstep_timeout_s=cfg.mesen.lockstep_timeout_s,
            strict_lockstep=strict_lockstep,
        )
        for i in range(cfg.mesen.workers)
    ]


def epsilon_by_step(step: int, total: int, start: float, final: float, decay_fraction: float) -> float:
    horizon = max(1, int(total * decay_fraction))
    p = min(1.0, step / horizon)
    return start + (final - start) * p


def ensure_log(run_dir: Path, headers: list[str]):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "training_metrics.csv"
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(headers)
    f.flush()
    return f, writer


def append_seq(buffers, seq_len, obs, action, reward, done, skip=None):
    """Create overlapping recurrent training sequences per worker."""
    for i in range(obs.shape[0]):
        if skip is not None and skip[i]:
            buffers[i].clear()
            continue
        buffers[i].append((obs[i].copy(), int(action[i]), float(reward[i]), bool(done[i])))
        if len(buffers[i]) == seq_len:
            arr = list(buffers[i])
            yield Sequence(
                obs=np.stack([x[0] for x in arr]).astype(np.uint8),
                actions=np.asarray([x[1] for x in arr], dtype=np.int64),
                rewards=np.asarray([x[2] for x in arr], dtype=np.float32),
                dones=np.asarray([x[3] for x in arr], dtype=np.bool_),
                demo=False,
            )
        if done[i]:
            buffers[i].clear()


def collect_segment_rollout(
    model: RecurrentActorCritic,
    env: ParallelTennis,
    trainer: PPOTrainer,
    obs_np: np.ndarray,
    hidden: torch.Tensor,
    cfg,
    sampler: ScenarioSampler | None = None,
):
    """Collect rollouts over parallel segment environments with GAE truncation bootstrapping."""
    N = env.n
    T = cfg.ppo.rollout_steps
    device = trainer.device if trainer is not None else next(model.parameters()).device

    obs_buf = torch.zeros((T, N, env.stack_size, env.height, env.width), dtype=torch.uint8)
    action_buf = torch.zeros((T, N), dtype=torch.long)
    reward_buf = torch.zeros((T, N), dtype=torch.float32)
    done_buf = torch.zeros((T, N), dtype=torch.float32)
    truncated_buf = torch.zeros((T, N), dtype=torch.float32)
    # Value at the actual final observation of a truncated option/segment. The
    # vector environment auto-resets completed workers, so value_buf[t + 1]
    # may belong to a new episode and must never be used for this bootstrap.
    truncated_value_buf = torch.zeros((T, N), dtype=torch.float32)
    value_buf = torch.zeros((T, N), dtype=torch.float32)
    logp_buf = torch.zeros((T, N), dtype=torch.float32)
    hidden_buf = torch.zeros((T, model.gru_layers, N, model.gru_hidden), dtype=torch.float32)

    stats = {
        "episodes": 0,
        "completed_games": 0,
        "wins": 0,
        "losses": 0,
        "service_faults": 0,
        "rally_survivals": 0,
        "late_survivals": 0,
        "late_points": 0,
        "late_wins": 0,
        "mid_points": 0,
        "mid_wins": 0,
        "start_points": 0,
        "start_wins": 0,
    }

    for t in range(T):
        hidden_buf[t] = hidden.detach().cpu()
        obs_buf[t] = torch.from_numpy(obs_np)
        obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True).float() / 255.0

        probe_mask = [a is not None and a.kind == "probe" for a in env.assignments]
        with torch.inference_mode():
            actions, log_probs, values, next_hidden = model.act(obs_t, hidden)
            # Evaluate probes greedily (deterministic)
            if any(probe_mask):
                greedy_act, _, _, _ = model.act(obs_t, hidden, deterministic=True)
                probe_t = torch.tensor(probe_mask, device=device)
                actions = torch.where(probe_t, greedy_act, actions)

        action_np = actions.cpu().numpy().astype(np.int64)
        batch = env.step(action_np)

        action_buf[t] = actions.cpu()
        reward_buf[t] = torch.from_numpy(batch.rewards)
        done_buf[t] = torch.from_numpy(batch.dones.astype(np.float32))
        value_buf[t] = values.cpu()
        logp_buf[t] = log_probs.cpu()
        if batch.truncateds is not None:
            truncated_buf[t] = torch.from_numpy(batch.truncateds.astype(np.float32))
            trunc_mask_np = np.asarray(batch.truncateds, dtype=np.bool_)
            if trunc_mask_np.any():
                # batch.observations preserves the pre-reset final frame stack.
                # next_hidden follows the state in obs_np and is therefore the
                # correct recurrent context for evaluating that final observation.
                trunc_idx = np.flatnonzero(trunc_mask_np)
                terminal_obs = torch.from_numpy(batch.observations[trunc_idx]).to(
                    device, non_blocking=True
                ).float() / 255.0
                terminal_h = next_hidden[:, trunc_idx].contiguous()
                with torch.inference_mode():
                    _, terminal_values, _ = model.forward_sequence(
                        terminal_obs.unsqueeze(0), terminal_h
                    )
                truncated_value_buf[t, trunc_idx] = terminal_values[0].cpu()

        for i, (info, done, r) in enumerate(zip(batch.infos, batch.dones, batch.rewards)):
            if info.get("service_fault"):
                stats["service_faults"] += 1
            if info.get("rally_survival"):
                stats["rally_survivals"] += 1
            if info.get("late_survival"):
                stats["late_survivals"] += 1

            a = info.get("assignment")
            if done:
                stats["episodes"] += 1
                if sampler is not None and a is not None and a.kind in ("scenario", "probe"):
                    result = info.get("point_result", 0)
                    ph = a.phase or "start"
                    if ph == "late":
                        stats["late_points"] += 1
                        if result > 0:
                            stats["late_wins"] += 1
                    elif ph == "mid":
                        stats["mid_points"] += 1
                        if result > 0:
                            stats["mid_wins"] += 1
                    elif ph == "start":
                        stats["start_points"] += 1
                        if result > 0:
                            stats["start_wins"] += 1

                    if result != 0 and not info.get("timeout"):
                        sampler.update(a.cell_id, result > 0, ph, source=a.kind)
                        if result > 0:
                            stats["wins"] += 1
                        else:
                            stats["losses"] += 1
                    elif info.get("service_fault"):
                        sampler.update(a.cell_id, False, ph, source=a.kind)
                        stats["losses"] += 1
                    elif info.get("rally_survival") or info.get("late_survival"):
                        stats["wins"] += 0.5
                else:
                    pg = int(info.get("player_games", 0))
                    og = int(info.get("opponent_games", 0))
                    if pg + og > 0:
                        stats["completed_games"] += 1
                        if pg > og:
                            stats["wins"] += 1
                        elif og > pg:
                            stats["losses"] += 1

        obs_np = batch.observations
        hidden = next_hidden
        if batch.dones.any():
            obs_np = obs_np.copy()
            hidden = hidden.clone()
            for i in np.flatnonzero(batch.dones):
                obs_np[i] = env.current_observation(int(i))
                hidden[:, int(i)] = 0.0

    # Generalized Advantage Estimation (GAE) with truncation bootstrapping
    obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True).float() / 255.0
    with torch.inference_mode():
        _, last_values, _ = model.forward_sequence(obs_t.unsqueeze(0), hidden)
    last_values = last_values[0].cpu()

    advantages, returns = compute_gae_with_truncation(
        reward_buf,
        value_buf,
        done_buf > 0.5,
        truncated_buf > 0.5,
        truncated_value_buf,
        last_values,
        cfg.ppo.gamma,
        cfg.ppo.gae_lambda,
    )

    rollout = Rollout(
        obs=obs_buf,
        actions=action_buf,
        rewards=reward_buf,
        dones=done_buf,
        values=value_buf,
        log_probs=logp_buf,
        hiddens=hidden_buf,
        advantages=advantages,
        returns=returns,
        truncateds=truncated_buf,
    )
    return rollout, obs_np, hidden, stats


def main():
    parser = argparse.ArgumentParser(description="Tennis RL Training")
    parser.add_argument("--config", default="configs/scenario.json")
    parser.add_argument("--algo", default="hierarchical", choices=["hierarchical", "ppo", "dqfd"],
                        help="Train with the learned hierarchy (default), flat recurrent PPO baseline, or DQfD baseline")
    parser.add_argument("--resume", default="", help="Resume model and optimizer state from a compatible trainer checkpoint")
    parser.add_argument("--init-from", default="", help="Initialize weights only from a staged hierarchy checkpoint; optimizer starts fresh")
    parser.add_argument("--level", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of Mesen parallel workers")
    parser.add_argument("--max-updates", type=int, default=None,
                        help="Exit after this many training updates (for testing/benchmarks)")
    parser.add_argument("--pretrain-only", action="store_true",
                        help="Perform only behavioral cloning pretraining on demonstration micro-segments, save checkpoint, and exit")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("Use either --resume (model+optimizer) or --init-from (weights only), not both")

    cfg = load_config(args.config)
    try:
        demo_compat = validate_demo_root_compatibility(
            cfg.ppo.demo_root,
            observation_height=cfg.mesen.observation_height,
            observation_width=cfg.mesen.observation_width,
            frame_skip=cfg.mesen.frame_skip,
            stack_size=cfg.environment.stack_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if demo_compat["point_files"]:
        print(f"Demonstration compatibility: {demo_compat}")
    if args.level is not None:
        cfg.mesen.game_level = args.level
    if args.workers is not None:
        cfg.mesen.workers = args.workers
    if args.algo == "hierarchical":
        from .hierarchical_train import run_hierarchical_training
        run_hierarchical_training(
            cfg, args,
            build_workers=build_workers,
            seed_everything=seed_everything,
            ensure_log=ensure_log,
            prune_checkpoints=_prune_step_checkpoints,
        )
        return
    seed_everything(cfg.seed)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        raise RuntimeError("CUDA is required by the configured training run but is unavailable.")
    device = torch.device(cfg.device)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    # Base RecurrentActorCritic model
    model = RecurrentActorCritic(
        observation_channels=4,
        action_count=cfg.environment.action_count,
        channels=cfg.network.channels,
        embedding_dim=cfg.network.embedding_dim,
        gru_hidden=cfg.network.gru_hidden,
        gru_layers=cfg.network.gru_layers,
        input_size=(cfg.mesen.observation_height, cfg.mesen.observation_width),
    ).to(device)

    # Initialize model dimensions
    with torch.no_grad():
        dummy = torch.zeros((1, cfg.environment.stack_size, cfg.mesen.observation_height, cfg.mesen.observation_width), device=device)
        model.act(dummy, model.initial_hidden(1, device))

    # Load demonstrations
    demo_dataset = DemonstrationDataset(
        cfg.ppo.demo_root,
        window_length=cfg.ppo.demo_window_length,
        return_gamma=cfg.ppo.demo_return_gamma,
        winning_weight=cfg.ppo.demo_winning_weight,
        losing_weight=cfg.ppo.demo_losing_weight,
        outcome_source=cfg.ppo.demo_outcome_source,
    )
    demo_summary = summarize_demos(cfg.ppo.demo_root)
    print(
        f"Demonstrations: points={demo_summary['points']} "
        f"winning={demo_summary['winning_points']} losing={demo_summary['losing_points']} "
        f"steps={demo_summary['steps']}"
    )

    # Initialize micro-segment replayer
    seg_cfg = SegmentReplayConfig(
        seg_len=getattr(cfg.ppo, "seg_replay_seg_len", 16),
        batch_size=getattr(cfg.ppo, "seg_replay_batch_size", 8),
        steps_per_update=getattr(cfg.ppo, "seg_replay_steps_per_update", 4),
        winning_bc_weight=getattr(cfg.ppo, "seg_replay_winning_bc_weight", 2.5),
        losing_bc_weight=getattr(cfg.ppo, "seg_replay_losing_bc_weight", 0.5),
        outcome_coef=getattr(cfg.ppo, "seg_replay_outcome_coef", 0.30),
        learning_rate=getattr(cfg.ppo, "seg_replay_lr", 3e-5),
        refresh_every=getattr(cfg.ppo, "seg_replay_refresh_every", 50),
    )
    seg_replayer: DemoSegmentReplayer | None = None
    if len(demo_dataset) > 0:
        seg_replayer = DemoSegmentReplayer(demo_dataset, model, device, cfg=seg_cfg)
        print(
            f"Segment replayer: {len(seg_replayer._segments)} micro-segments "
            f"(seg_len={seg_cfg.seg_len}, types={list(seg_replayer._by_type.keys())})"
        )

    # Behavioral cloning pretraining only mode (no emulator workers needed)
    if args.pretrain_only:
        if seg_replayer is None:
            raise SystemExit("Cannot run --pretrain-only: no demonstrations found.")
        pretrain_updates = cfg.ppo.demo_pretrain_updates
        print(f"Pretraining policy on demonstration micro-segments ({pretrain_updates} updates)...")
        pm = seg_replayer.update(n_steps=pretrain_updates)
        print(f"Pretrain complete: seg_bc_loss={pm['seg_bc_loss']:.4f}, outcome_loss={pm['seg_outcome_loss']:.4f}")
        save_path = Path(cfg.ppo.save_dir) / "tennis_pretrained.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "model_state_dict": model.state_dict(), "config": cfg.__dict__}, save_path)
        print(f"Pretrained model saved to {save_path}")
        return

    workers = build_workers(cfg)
    sampler = None
    scenario_provider = None
    sc = cfg.scenario
    if sc.enabled:
        state_dir = Path(sc.state_dir)
        manifest_path = state_dir / "manifest.jsonl"
        if manifest_path.exists():
            records = load_manifest(manifest_path, base_dir=state_dir)
            if records:
                library = ScenarioLibrary(records)
                sampler = ScenarioSampler(
                    library,
                    SamplerConfig(
                        anchor_fraction=sc.anchor_fraction,
                        loss_state_fraction=sc.loss_state_fraction,
                        stratum_floor=sc.stratum_floor,
                        retention_boost=sc.retention_boost,
                        noop_min=sc.noop_min,
                        noop_max=sc.noop_max,
                        promote_winrate=getattr(sc, "promote_winrate", 0.45),
                        promote_min_episodes=getattr(sc, "promote_min_episodes", 15),
                        zpd_threshold=getattr(sc, "zpd_threshold", 0.20),
                        zpd_late_bias=getattr(sc, "zpd_late_bias", 0.65),
                    ),
                    natural_counts=Counter(r.cell_id for r in records if r.split == "train"),
                    seed=cfg.seed,
                )
                probe_rng = np.random.default_rng(cfg.seed + 1)
                n_train = sum(1 for r in records if r.split == "train")
                print(
                    f"Scenario training: {n_train} train states / {len(records) - n_train} holdout states, "
                    f"{len(sampler.available)}/36 cells have training states"
                )

                def scenario_provider(i: int):
                    if sc.probe_fraction > 0 and probe_rng.random() < sc.probe_fraction:
                        a = sampler.sample_probe()
                        if a.kind == "probe":
                            return a
                    return sampler.sample()
        elif not args.smoke_test:
            raise SystemExit(
                f"scenario.enabled but no harvested states in {state_dir}/manifest.jsonl "
                f"(harvest with: python -m tennis_rl.record --harvest-dir {state_dir})"
            )

    env = ParallelTennis(
        workers,
        stack_size=cfg.environment.stack_size,
        max_steps=cfg.environment.episode_max_steps,
        point_episodes=sc.enabled and sc.point_episodes,
        point_max_steps=sc.point_max_steps,
        scenario_provider=scenario_provider,
        segment_mode=sc.enabled and getattr(sc, "segment_mode", True),
        late_max_steps=getattr(sc, "late_max_steps", 40),
        mid_max_steps=getattr(sc, "mid_max_steps", 50),
        start_max_steps=getattr(sc, "start_max_steps", 25),
        serve_stall_penalty=getattr(sc, "serve_stall_penalty", -1.0),
        rally_progress_reward=getattr(sc, "rally_progress_reward", 0.20),
        late_progress_reward=getattr(sc, "late_progress_reward", 0.10),
    )

    if args.smoke_test:
        try:
            obs = env.reset()
            print(f"Initial observation: shape={obs.shape} dtype={obs.dtype}")
            for step in range(16):
                batch = env.step(np.zeros(env.n, dtype=np.int64))
                phases = [getattr(a, "phase", "none") if a is not None else "anchor" for a in env.assignments]
                cells = [getattr(a, "cell_id", "anchor") if a is not None else "anchor" for a in env.assignments]
                print(f"step {step:02d}: obs={batch.observations.shape} rew={batch.rewards.tolist()} dones={batch.dones.tolist()} phases={phases} cells={cells}")
        finally:
            env.close()
        return

    # Behavioral cloning warm start on demonstration segments
    if not args.resume and seg_replayer is not None and cfg.ppo.demo_pretrain_updates > 0:
        pretrain_updates = cfg.ppo.demo_pretrain_updates
        print(f"Pretraining policy on demonstration micro-segments ({pretrain_updates} updates)...")
        pm = seg_replayer.update(n_steps=pretrain_updates)
        print(f"Pretrain complete: seg_bc_loss={pm['seg_bc_loss']:.4f}, outcome_loss={pm['seg_outcome_loss']:.4f}")

    run_dir = Path(cfg.ppo.tensorboard_dir) / time.strftime("%Y%m%d-%H%M%S")

    # Primary algorithm: Segment-PPO
    if args.algo == "ppo":
        csv_file, csv_writer = ensure_log(
            run_dir,
            [
                "update", "steps", "fps", "win_rate", "late_win_rate", "mid_win_rate",
                "start_win_rate", "policy_loss", "value_loss", "entropy", "demo_bc_loss",
                "demo_outcome_loss", "motion_loss", "service_faults", "rally_survivals", "episodes"
            ]
        )
        trainer = PPOTrainer(
            model,
            cfg.ppo,
            device,
            str(run_dir),
            demonstration_dataset=demo_dataset,
            segment_replayer=seg_replayer,
        )
        if args.resume:
            extra = trainer.load(args.resume)
            print(f"Resumed PPO from {args.resume}: {extra}")

        print(
            f"Segment-PPO training: target={cfg.ppo.total_steps:,}, workers={env.n}, "
            f"rollout_steps={cfg.ppo.rollout_steps}, device={device}"
        )
        obs = env.reset()
        hidden = model.initial_hidden(env.n, device)
        steps_done = trainer.env_steps
        update = trainer.update_index
        start = time.perf_counter()
        best_probe = -1.0
        scenario_log = open(run_dir / "scenario_report.jsonl", "a", encoding="utf-8") if sampler is not None else None

        try:
            while steps_done < cfg.ppo.total_steps:
                rollout_start = time.perf_counter()
                rollout, obs, hidden, rstats = collect_segment_rollout(
                    model, env, trainer, obs, hidden, cfg, sampler=sampler
                )
                rollout_time = time.perf_counter() - rollout_start

                update_start = time.perf_counter()
                metrics = trainer.update(rollout)
                update_time = time.perf_counter() - update_start

                steps_done += env.n * cfg.ppo.rollout_steps
                trainer.env_steps = steps_done
                update += 1

                elapsed = max(time.perf_counter() - start, 1e-6)
                fps = steps_done / elapsed
                eta = (cfg.ppo.total_steps - steps_done) / max(fps, 1e-6)

                total_sc_points = rstats["late_points"] + rstats["mid_points"] + rstats["start_points"]
                win_rate = (rstats["wins"] / max(1, total_sc_points)) if total_sc_points > 0 else 0.0
                late_wr = (rstats["late_wins"] / max(1, rstats["late_points"])) if rstats["late_points"] > 0 else 0.0
                mid_wr = (rstats["mid_wins"] / max(1, rstats["mid_points"])) if rstats["mid_points"] > 0 else 0.0
                start_wr = (rstats["start_wins"] / max(1, rstats["start_points"])) if rstats["start_points"] > 0 else 0.0

                print(
                    f"update={update:05d} steps={steps_done:,}/{cfg.ppo.total_steps:,} "
                    f"({100 * steps_done / cfg.ppo.total_steps:5.2f}%) fps={fps:4.0f} "
                    f"ETA={int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d} | "
                    f"W-rate={100 * win_rate:4.1f}% [L:{100 * late_wr:3.0f}% M:{100 * mid_wr:3.0f}% S:{100 * start_wr:3.0f}%] | "
                    f"pol={metrics['policy_loss']:.4f} val={metrics['value_loss']:.4f} ent={metrics['entropy']:.3f} | "
                    f"demo_bc={metrics['demo_bc_loss']:.4f} mot={metrics.get('motion_loss', 0.0):.4f} | "
                    f"faults={rstats['service_faults']} rallies={rstats['rally_survivals']}",
                    flush=True,
                )

                csv_writer.writerow([
                    update, steps_done, fps, win_rate, late_wr, mid_wr, start_wr,
                    metrics["policy_loss"], metrics["value_loss"], metrics["entropy"],
                    metrics["demo_bc_loss"], metrics["demo_outcome_loss"],
                    metrics.get("motion_loss", 0.0),
                    rstats["service_faults"], rstats["rally_survivals"], rstats["episodes"]
                ])
                csv_file.flush()

                if sampler is not None and update % max(1, sc.report_every_updates) == 0:
                    rep = sampler.report()
                    pn = sum(n for n, _ in sampler.probe_stats.values())
                    pw = sum(w for _, w in sampler.probe_stats.values())
                    cells = rep["cells"]
                    means = [v["win_slow"] for v in cells.values()]
                    print(
                        f"  scenarios: cells_seen={len(cells)} mean_win={np.mean(means) if means else 0:.2f} "
                        f"forgotten={sampler.forgotten_cells()} holdout_probe={pw}/{pn}",
                        flush=True,
                    )
                    scenario_log.write(
                        json.dumps({"update": update, "steps": steps_done, "probe_wins": pw, "probe_n": pn, **rep}) + "\n"
                    )
                    scenario_log.flush()
                    if pn >= sc.min_probe_episodes and pw / pn > best_probe:
                        best_probe = pw / pn
                        bp = Path(cfg.ppo.save_dir) / "tennis_best_probe.pt"
                        trainer.save(bp, extra={"probe_win_rate": best_probe, "config": cfg.__dict__})
                    sampler.probe_stats.clear()

                if update % max(1, cfg.ppo.checkpoint_every_updates) == 0:
                    ckpt = Path(cfg.ppo.save_dir) / f"tennis_step_{steps_done}.pt"
                    trainer.save(ckpt, extra={"config": cfg.__dict__})
                    _prune_step_checkpoints(Path(cfg.ppo.save_dir), cfg.ppo.max_step_checkpoints)
                    trainer.save(Path(cfg.ppo.save_dir) / "tennis_latest.pt", extra={"config": cfg.__dict__})

                if args.max_updates is not None and update >= args.max_updates:
                    print(f"Reached max_updates ({args.max_updates}), terminating.")
                    trainer.save(Path(cfg.ppo.save_dir) / "tennis_latest.pt", extra={"config": cfg.__dict__})
                    break

        finally:
            csv_file.close()
            if scenario_log is not None:
                scenario_log.close()
            env.close()

    # Alternate algorithm: DQfD
    elif args.algo == "dqfd":
        csv_file, csv_writer = ensure_log(
            run_dir,
            ["update", "steps", "fps", "epsilon", "loss", "td_loss", "margin_loss", "mean_q", "win_rate", "episodes"]
        )
        q_model = RecurrentDuelingQ(model).to(device)
        target = RecurrentDuelingQ(copy.deepcopy(model)).to(device)
        with torch.no_grad():
            dummy = torch.zeros((1, 1, cfg.environment.stack_size, cfg.mesen.observation_height, cfg.mesen.observation_width), device=device)
            q_model(dummy, q_model.initial_hidden(1, device))
            target.load_state_dict(q_model.state_dict())

        trainer = DQFDTrainer(q_model, target, cfg.ppo, device)
        if not args.resume and len(demo_dataset) > 0 and cfg.ppo.demo_pretrain_updates > 0:
            print(f"DQfD demonstration pretrain: updates={cfg.ppo.demo_pretrain_updates}")
            m = pretrain_from_demonstrations(
                trainer, demo_dataset, updates=cfg.ppo.demo_pretrain_updates, batch_size=cfg.ppo.demo_pretrain_batch_size
            )
            print(f"Pretrain complete: loss={m['loss']:.4f}")

        obs = env.reset()
        hidden = q_model.initial_hidden(env.n, device)
        seq_buffers = [deque(maxlen=cfg.ppo.sequence_length) for _ in range(env.n)]
        steps_done = 0
        update = 0
        wins = 0
        episodes = 0
        start = time.perf_counter()

        try:
            while steps_done < cfg.ppo.total_steps:
                for _ in range(cfg.ppo.rollout_steps):
                    epsilon = epsilon_by_step(
                        steps_done, cfg.ppo.total_steps, cfg.ppo.epsilon_start, cfg.ppo.epsilon_final, cfg.ppo.epsilon_decay_fraction
                    )
                    probe_mask = [a is not None and a.kind == "probe" for a in env.assignments]
                    with torch.inference_mode():
                        actions, hidden, q_now = trainer.act(obs, hidden, epsilon)
                        if any(probe_mask):
                            actions = torch.where(torch.tensor(probe_mask, device=actions.device), q_now.argmax(dim=-1), actions)
                    action_np = actions.cpu().numpy().astype(np.int64)
                    batch = env.step(action_np)
                    for seq in append_seq(seq_buffers, cfg.ppo.sequence_length, obs, action_np, batch.rewards, batch.dones, skip=probe_mask):
                        trainer.add_sequence(seq, priority=1.0)
                    for info, done in zip(batch.infos, batch.dones):
                        if done:
                            episodes += 1
                            res = info.get("point_result", 0)
                            if res > 0:
                                wins += 1
                    obs = batch.observations
                    if batch.dones.any():
                        obs = obs.copy()
                        with torch.inference_mode():
                            for i in np.flatnonzero(batch.dones):
                                obs[i] = env.current_observation(int(i))
                                hidden[:, int(i)] = 0.0
                    steps_done += env.n
                    if steps_done >= cfg.ppo.total_steps:
                        break

                metrics_list = []
                for _ in range(max(1, cfg.ppo.updates_per_rollout)):
                    m = trainer.update(cfg.ppo.sequence_batch_size, beta=cfg.ppo.priority_beta)
                    if m is not None:
                        metrics_list.append(m)
                update += 1
                metrics = {k: float(np.mean([x[k] for x in metrics_list])) for k in metrics_list[0]} if metrics_list else {"loss": 0, "td_loss": 0, "margin_loss": 0, "mean_q": 0}
                elapsed = max(time.perf_counter() - start, 1e-6)
                fps = steps_done / elapsed
                win_rate = wins / max(1, episodes)
                print(
                    f"update={update:05d} steps={steps_done:,} fps={fps:.0f} loss={metrics['loss']:.4f} "
                    f"td={metrics['td_loss']:.4f} margin={metrics['margin_loss']:.4f} win={100 * win_rate:.1f}%",
                    flush=True,
                )
                csv_writer.writerow([update, steps_done, fps, epsilon, metrics["loss"], metrics["td_loss"], metrics["margin_loss"], metrics["mean_q"], win_rate, episodes])
                csv_file.flush()
        finally:
            csv_file.close()
            env.close()


if __name__ == "__main__":
    main()
