"""Watch a trained Tennis checkpoint play in a live, visible Mesen window.

This is `evaluate.py`'s loop with two changes: it opens exactly one
non-headless Mesen worker (a real window, roughly real NES speed, instead
of --testrunner) and it prints point-by-point commentary instead of only a
final summary.

Usage:
    python -m tennis_rl.watch
    python -m tennis_rl.watch --checkpoint checkpoints/tennis_step_4800000.pt --episodes 5
    python -m tennis_rl.watch --sample          # stochastic actions instead of greedy
    python -m tennis_rl.watch --level 3         # choose CPU difficulty 1-5 (default: 1)
    python -m tennis_rl.watch --no-autostart    # disable automatic menu navigation

NES Tennis opens on a Singles/Doubles selector, then a difficulty (1-5)
selector. By default this script navigates both screens automatically
so the game starts immediately. Use --level to choose the CPU difficulty.
"""
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--checkpoint", default="checkpoints/tennis_latest.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of taking the argmax (deterministic).")
    parser.add_argument("--level", type=int, default=1, choices=range(1, 6), metavar="{1-5}",
                        help="CPU difficulty level to select on the level-select screen (1=easiest, 5=hardest). Default: 1.")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Disable automatic menu navigation (press Start and choose a level yourself).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(cfg.device)

    run_id = f"watch_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{random.getrandbits(40):010x}"
    worker = MesenWorker(
        cfg.mesen.executable,
        cfg.mesen.rom,
        cfg.mesen.lua_script,
        cfg.mesen.host,
        cfg.mesen.port_start,
        cfg.mesen.observation_width,
        cfg.mesen.observation_height,
        cfg.mesen.frame_skip,
        max(cfg.mesen.startup_timeout_s, 60.0),  # picking a difficulty yourself takes longer than a scripted boot
        cfg.mesen.socket_timeout_s,
        match_games=cfg.mesen.match_games,
        game_level=args.level,
        point_win=cfg.reward.point_win,
        point_loss=cfg.reward.point_loss,
        game_win=cfg.reward.game_win,
        ipc_dir=cfg.mesen.ipc_dir,
        run_id=run_id,
        headless=False,  # real, visible Mesen window
    )
    env = ParallelTennis([worker], cfg.environment.stack_size, cfg.environment.episode_max_steps)

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
        model.act(dummy, model.initial_hidden(1, device), deterministic=True)  # materialize LazyLinear before loading weights
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(update={payload.get('update_index', '?')}, env_steps={payload.get('env_steps', '?'):,})"
        if isinstance(payload.get("env_steps"), int)
        else f"Loaded checkpoint: {args.checkpoint}"
    )

    if not args.no_autostart:
        print(
            f"Auto-boot active: Mesen will boot directly to Singles → Level {args.level} on the court."
        )
    else:
        print(
            "Auto-start suppressed. Navigate the title/difficulty screen manually."
        )

    completed = 0
    try:
        obs = env.reset()
        hidden = model.initial_hidden(1, device)
        episode_reward = 0.0
        last_seen = (0, 0)
        while completed < args.episodes:
            obs_t = torch.from_numpy(obs).to(device).float() / 255.0
            with torch.inference_mode():
                actions, _, _, next_hidden = model.act(obs_t, hidden, deterministic=not args.sample)
            batch = env.step(actions.cpu().numpy())
            hidden = next_hidden
            episode_reward += float(batch.rewards[0])

            info = batch.infos[0]
            seen = (info.get("player_games", 0), info.get("player_score", 0))
            if info.get("point_end") and seen != last_seen:
                print(f"  point ended -> games={seen[0]} score={seen[1]} "
                      f"(reward so far this episode: {episode_reward:+.2f})")
            last_seen = seen

            if batch.dones[0]:
                completed += 1
                won = seen[0] >= cfg.mesen.match_games
                print(f"Episode {completed}/{args.episodes} finished: "
                      f"{'WIN' if won else 'loss'} | games_won={seen[0]} | total_reward={episode_reward:+.2f}")
                episode_reward = 0.0
                hidden = model.initial_hidden(1, device)
                last_seen = (0, 0)
            obs = batch.observations
    finally:
        env.close()


if __name__ == "__main__":
    main()
