from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from .config import load_config
from .env import ParallelTennis
from .mesen_client import MesenWorker
from .model import RecurrentActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--checkpoint", default="checkpoints/tennis_latest.pt")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.mesen.transport != "file_mailbox":
        raise ValueError(f"Unsupported Mesen transport: {cfg.mesen.transport!r}; expected 'file_mailbox'")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(cfg.device)

    run_id = f"run_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{random.getrandbits(40):010x}"
    workers = [
        MesenWorker(
            cfg.mesen.executable,
            cfg.mesen.rom,
            cfg.mesen.lua_script,
            cfg.mesen.host,
            cfg.mesen.port_start + i,
            cfg.mesen.observation_width,
            cfg.mesen.observation_height,
            cfg.mesen.frame_skip,
            cfg.mesen.startup_timeout_s,
            cfg.mesen.socket_timeout_s,
            match_games=cfg.mesen.match_games,
            point_win=cfg.reward.point_win,
            point_loss=cfg.reward.point_loss,
            game_win=cfg.reward.game_win,
            ipc_dir=cfg.mesen.ipc_dir,
            run_id=run_id,
        )
        for i in range(min(cfg.mesen.workers, args.episodes))
    ]
    env = ParallelTennis(workers, cfg.environment.stack_size, cfg.environment.episode_max_steps)

    model = RecurrentActorCritic(
        env.observation_shape[0],
        cfg.environment.action_count,
        cfg.network.channels,
        cfg.network.embedding_dim,
        cfg.network.gru_hidden,
        cfg.network.gru_layers,
    ).to(device)
    with torch.no_grad():
        dummy = torch.zeros((1, *env.observation_shape), device=device)
        model.act(dummy, model.initial_hidden(1, device), deterministic=True)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    wins = 0
    rewards: list[float] = []
    completed = 0

    try:
        obs = env.reset()
        hidden = model.initial_hidden(env.n, device)
        current_rewards = np.zeros(env.n, dtype=np.float32)
        while completed < args.episodes:
            obs_t = torch.from_numpy(obs).to(device).float() / 255.0
            with torch.inference_mode():
                actions, _, _, next_hidden = model.act(obs_t, hidden, deterministic=True)
            batch = env.step(actions.cpu().numpy())
            current_rewards += batch.rewards
            obs = batch.observations
            hidden = next_hidden
            if batch.dones.any():
                mask = torch.from_numpy((~batch.dones).astype(np.float32)).to(device)
                hidden = hidden * mask.view(1, env.n, 1)
            for i, done in enumerate(batch.dones.tolist()):
                if done and completed < args.episodes:
                    completed += 1
                    rewards.append(float(current_rewards[i]))
                    if batch.infos[i].get("player_games", 0) >= 6:
                        wins += 1
                    current_rewards[i] = 0
        print(f"episodes={completed}")
        print(f"match_wins={wins}")
        print(f"win_rate={wins / max(completed, 1):.3f}")
        print(f"mean_reward={float(np.mean(rewards)):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
