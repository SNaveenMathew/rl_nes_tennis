from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_NAMES
from .config import Config, load_config
from .env import ParallelTennis
from .mesen_client import MesenWorker
from .model import RecurrentActorCritic
from .ppo import PPOTrainer, Rollout


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _print_rollout_progress(
    *,
    collected_steps: int,
    rollout_steps: int,
    global_steps: int,
    total_target: int,
    elapsed: float,
    rollout_elapsed: float,
) -> None:
    rollout_pct = 100.0 * collected_steps / max(rollout_steps, 1)
    total_pct = 100.0 * global_steps / max(total_target, 1)
    fps = collected_steps / max(rollout_elapsed, 1e-6)
    remaining = max(total_target - global_steps, 0)
    eta = remaining / max(global_steps / max(elapsed, 1e-6), 1e-6) if global_steps else float("inf")
    width = 28
    filled = min(width, int(width * rollout_pct / 100.0))
    bar = "#" * filled + "-" * (width - filled)
    eta_text = _format_duration(eta) if np.isfinite(eta) else "--:--"
    line = (
        f"\rrollout [{bar}] {rollout_pct:6.2f}% | "
        f"steps {global_steps:,}/{total_target:,} ({total_pct:5.2f}%) | "
        f"fps {fps:7.1f} | elapsed {_format_duration(elapsed)} | ETA {eta_text}"
    )
    print(line, end="", flush=True)




def _prune_step_checkpoints(save_dir: Path, keep: int) -> None:
    """Keep only the newest N periodic checkpoints; never remove tennis_latest.pt."""
    if keep < 0:
        raise ValueError("max_step_checkpoints must be >= 0")
    def step_number(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    files = sorted(
        save_dir.glob("tennis_step_*.pt"),
        key=step_number,
        reverse=True,
    )
    for stale in files[keep:]:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

def build_workers(cfg: Config) -> list[MesenWorker]:
    if cfg.mesen.transport != "file_mailbox":
        raise ValueError(f"Unsupported Mesen transport: {cfg.mesen.transport!r}; expected 'file_mailbox'")
    run_id = f"run_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{random.getrandbits(40):010x}"
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
            startup_timeout=cfg.mesen.startup_timeout_s,
            socket_timeout=cfg.mesen.socket_timeout_s,
            point_win=cfg.reward.point_win,
            point_loss=cfg.reward.point_loss,
            game_win=cfg.reward.game_win,
            ipc_dir=cfg.mesen.ipc_dir,
            run_id=run_id,
            capture_stderr=cfg.mesen.capture_stderr,
        )
        for i in range(cfg.mesen.workers)
    ]


def collect_rollout(
    model: RecurrentActorCritic,
    env: ParallelTennis,
    trainer: PPOTrainer,
    obs_np: np.ndarray,
    hidden: torch.Tensor,
    cfg: Config,
) -> tuple[Rollout, np.ndarray, torch.Tensor, dict[str, float]]:
    T = cfg.ppo.rollout_steps
    rollout_started = time.perf_counter()
    last_progress = rollout_started
    N = env.n
    C, H, W = env.observation_shape
    device = trainer.device

    obs_buf = torch.empty((T, N, C, H, W), dtype=torch.uint8)
    action_buf = torch.empty((T, N), dtype=torch.long)
    reward_buf = torch.empty((T, N), dtype=torch.float32)
    done_buf = torch.empty((T, N), dtype=torch.float32)
    value_buf = torch.empty((T, N), dtype=torch.float32)
    logp_buf = torch.empty((T, N), dtype=torch.float32)
    hidden_buf = torch.empty(
        (T, model.gru_layers, N, model.gru_hidden),
        dtype=torch.float32,
    )

    episode_rewards = np.zeros(N, dtype=np.float32)
    completed_rewards: list[float] = []
    completed_games: list[int] = []
    completed_points: list[int] = []

    for t in range(T):
        obs_buf[t] = torch.from_numpy(obs_np)
        hidden_buf[t] = hidden.detach().cpu()

        obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True).float() / 255.0
        with torch.inference_mode():
            action, logp, values, next_hidden = model.act(obs_t, hidden, deterministic=False)

        actions_np = action.cpu().numpy().astype(np.int64)
        batch = env.step(actions_np)

        now = time.perf_counter()
        if (now - last_progress) >= 2.0 or t == T - 1:
            _print_rollout_progress(
                collected_steps=(t + 1) * N,
                rollout_steps=T * N,
                global_steps=trainer.env_steps + (t + 1) * N,
                total_target=cfg.ppo.total_steps,
                elapsed=max(now - getattr(trainer, "run_started", now), 1e-6),
                rollout_elapsed=max(now - rollout_started, 1e-6),
            )
            last_progress = now

        action_buf[t] = action.cpu()
        reward_buf[t] = torch.from_numpy(batch.rewards)
        done_buf[t] = torch.from_numpy(batch.dones.astype(np.float32))
        value_buf[t] = values.cpu()
        logp_buf[t] = logp.cpu()

        episode_rewards += batch.rewards
        for i, info in enumerate(batch.infos):
            if batch.dones[i]:
                completed_rewards.append(float(episode_rewards[i]))
                completed_games.append(int(info.get("player_games", 0)))
                completed_points.append(int(info.get("player_score", 0)))
                episode_rewards[i] = 0.0

        obs_np = batch.observations
        hidden = next_hidden
        if batch.dones.any():
            mask = torch.from_numpy((~batch.dones).astype(np.float32)).to(device)
            hidden = hidden * mask.view(1, N, 1)

    # Bootstrap value for GAE.
    obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True).float() / 255.0
    with torch.inference_mode():
        _, last_values, _ = model.forward_sequence(obs_t.unsqueeze(0), hidden)
    last_values = last_values[0].cpu()

    advantages = torch.zeros_like(reward_buf)
    gae = torch.zeros(N, dtype=torch.float32)
    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else value_buf[t + 1]
        next_nonterminal = 1.0 - done_buf[t]
        delta = reward_buf[t] + cfg.ppo.gamma * next_value * next_nonterminal - value_buf[t]
        gae = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + value_buf

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
    )

    stats = {
        "episodes": float(len(completed_rewards)),
        "episode_reward": float(np.mean(completed_rewards)) if completed_rewards else float("nan"),
        "player_games": float(np.mean(completed_games)) if completed_games else float("nan"),
        "player_score": float(np.mean(completed_points)) if completed_points else float("nan"),
        "rollout_reward": float(reward_buf.mean()),
    }
    return rollout, obs_np, hidden, stats


def smoke_test(cfg: Config) -> None:
    print("Starting Mesen bridge smoke test...")
    workers = build_workers(cfg)
    env = ParallelTennis(workers, stack_size=cfg.environment.stack_size, max_steps=256)
    try:
        obs = env.reset()
        print("Initial observation:", obs.shape, obs.dtype)
        actions = np.zeros(env.n, dtype=np.int64)
        for step in range(16):
            actions[:] = step % len(ACTION_NAMES)
            batch = env.step(actions)
            print(
                f"step={step:03d} shape={batch.observations.shape} "
                f"reward={batch.rewards.tolist()} done={batch.dones.tolist()}"
            )
    finally:
        env.close()
    print("Smoke test completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--resume", default="")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    if not torch.cuda.is_available() and cfg.device == "cuda":
        raise RuntimeError("CUDA is required by the configured training run but is unavailable.")

    if args.smoke_test:
        smoke_test(cfg)
        return

    device = torch.device(cfg.device)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    workers = build_workers(cfg)
    env = ParallelTennis(
        workers,
        stack_size=cfg.environment.stack_size,
        max_steps=cfg.environment.episode_max_steps,
    )

    model = RecurrentActorCritic(
        observation_channels=env.observation_shape[0],
        action_count=cfg.environment.action_count,
        channels=cfg.network.channels,
        embedding_dim=cfg.network.embedding_dim,
        gru_hidden=cfg.network.gru_hidden,
        gru_layers=cfg.network.gru_layers,
    ).to(device)

    # Initialize LazyLinear before optimizer construction / checkpoint load.
    with torch.no_grad():
        dummy = torch.zeros((1, *env.observation_shape), device=device)
        hidden0 = model.initial_hidden(1, device)
        model.act(dummy, hidden0, deterministic=True)

    run_name = time.strftime("%Y%m%d-%H%M%S")
    run_dir = str(Path(cfg.ppo.tensorboard_dir) / run_name)
    trainer = PPOTrainer(model, cfg.ppo, device, run_dir)
    _prune_step_checkpoints(Path(cfg.ppo.save_dir), cfg.ppo.max_step_checkpoints)
    trainer.run_started = time.perf_counter()
    print(
        f"Training started: target={cfg.ppo.total_steps:,} env steps, "
        f"workers={env.n}, rollout={cfg.ppo.rollout_steps} per worker"
    )

    if args.resume:
        extra = trainer.load(args.resume)
        print(f"Resumed from {args.resume}: {extra}")

    obs = env.reset()
    hidden = model.initial_hidden(env.n, device)

    total_target = cfg.ppo.total_steps
    steps_done = trainer.env_steps
    update = trainer.update_index
    start = time.perf_counter()

    try:
        while steps_done < total_target:
            rollout_started = time.perf_counter()
            rollout, obs, hidden, stats = collect_rollout(model, env, trainer, obs, hidden, cfg)
            rollout_time = time.perf_counter() - rollout_started

            update_started = time.perf_counter()
            metrics = trainer.update(rollout)
            update_time = time.perf_counter() - update_started

            steps_done += cfg.ppo.rollout_steps * env.n
            trainer.env_steps = steps_done

            elapsed = time.perf_counter() - start
            fps = steps_done / max(elapsed, 1e-6)
            update_fps = (cfg.ppo.rollout_steps * env.n) / max(rollout_time, 1e-6)
            trainer.log(
                metrics,
                steps_done,
                extra={
                    "rollout_seconds": rollout_time,
                    "ppo_update_seconds": update_time,
                    "env_fps": update_fps,
                    "overall_fps": fps,
                },
            )
            update += 1

            print()
            eta = (total_target - steps_done) / max(fps, 1e-6)
            gpu_mem = 0.0
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(
                f"update={update:05d} | steps={steps_done:,}/{total_target:,} "
                f"({100.0 * steps_done / max(total_target, 1):.2f}%) | "
                f"fps={fps:,.0f} | ETA={_format_duration(eta)} | "
                f"reward={stats['rollout_reward']:.4f} | "
                f"ep_reward={stats['episode_reward']:.3f} | "
                f"policy={metrics['policy_loss']:.4f} | "
                f"value={metrics['value_loss']:.4f} | "
                f"entropy={metrics['entropy']:.4f} | "
                f"rollout={rollout_time:.2f}s | update={update_time:.2f}s | "
                f"GPU_mem={gpu_mem:.2f}GiB",
                flush=True,
            )

            if update % cfg.ppo.checkpoint_every_updates == 0:
                save_dir = Path(cfg.ppo.save_dir)
                trainer.save(
                    save_dir / f"tennis_step_{steps_done}.pt",
                    extra={"config": json.loads(Path(args.config).read_text())},
                )
                _prune_step_checkpoints(save_dir, cfg.ppo.max_step_checkpoints)
                trainer.save(
                    Path(cfg.ppo.save_dir) / "tennis_latest.pt",
                    extra={"config": json.loads(Path(args.config).read_text())},
                )
    finally:
        trainer.save(
            Path(cfg.ppo.save_dir) / "tennis_latest.pt",
            extra={"config": json.loads(Path(args.config).read_text())},
        )
        env.close()
        trainer.writer.close()


if __name__ == "__main__":
    main()
