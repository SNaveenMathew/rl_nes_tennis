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
from .hierarchical import HierarchicalActorCritic, HierarchicalController
from .inference_compat import validate_checkpoint_runtime
from .rom_patcher import resolve_rom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scenario.json")
    parser.add_argument("--checkpoint", default="checkpoints_hierarchical_native256x240_frame1/hierarchical_latest.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of taking the argmax (deterministic).")
    parser.add_argument("--level", type=int, default=1, choices=range(1, 6), metavar="{1-5}",
                        help="CPU difficulty level to select on the level-select screen (1=easiest, 5=hardest). Default: 1.")
    parser.add_argument("--serve", default=None, choices=["serve-first", "receive-first"],
                        help="Serve mode: who serves game 1 (default from config).")
    parser.add_argument("--sampling", "--episode-mode", choices=["point", "game"], default="game",
                        help="Watching scope: 'point' (watch N individual points) or 'game' (watch N regular games).")
    parser.add_argument("--side", choices=["deuce", "ad", "advantage"], default=None,
                        help="Service side: 'deuce' or 'ad'/'advantage' (only valid when --sampling is 'point').")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Disable automatic menu navigation (press Start and choose a level yourself).")
    args = parser.parse_args()
    if args.sampling == "game" and args.side in {"ad", "advantage"}:
        parser.error("--side (ad/advantage) is only supported when --sampling is 'point'; a regular game always starts on the deuce side.")

    cfg = load_config(args.config)
    if args.level is not None:
        cfg.mesen.game_level = args.level
    if args.serve is not None:
        cfg.mesen.serve_mode = args.serve
    if args.side is not None:
        cfg.mesen.serve_side = args.side
    elif args.sampling == "game":
        cfg.mesen.serve_side = "deuce"
    cfg.mesen.rom = str(resolve_rom(
        cfg.mesen.rom,
        serve_mode=cfg.mesen.serve_mode,
        level=cfg.mesen.game_level,
        side=cfg.mesen.serve_side,
        sampling=args.sampling,
    ))
    print(f"Watch live | Sampling: {args.sampling} | Serve: {cfg.mesen.serve_mode} ({cfg.mesen.serve_side}) | Level: {cfg.mesen.game_level}")
    print(f"ROM: {cfg.mesen.rom}")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(cfg.device)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hierarchical = payload.get("format") == "tennis_rl_hierarchical_v2"
    pulse_durations = validate_checkpoint_runtime(payload, cfg, hierarchical=hierarchical)

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
        game_level=cfg.mesen.game_level,
        serve_mode=cfg.mesen.serve_mode,
        serve_side=cfg.mesen.serve_side,
        point_win=cfg.reward.point_win,
        point_loss=cfg.reward.point_loss,
        game_win=cfg.reward.game_win,
        game_loss=cfg.reward.game_loss,
        ipc_dir=cfg.mesen.ipc_dir,
        run_id=run_id,
        headless=False,  # real, visible Mesen window
        lockstep=cfg.mesen.lockstep,
        lockstep_timeout_s=cfg.mesen.lockstep_timeout_s,
        strict_lockstep=True,
    )
    is_point = args.sampling == "point"
    max_steps = getattr(cfg.scenario, "point_max_steps", 900) if is_point else cfg.environment.episode_max_steps
    env = ParallelTennis([worker], cfg.environment.stack_size, max_steps, point_episodes=is_point, point_max_steps=max_steps)

    controller = None
    if hierarchical:
        model = HierarchicalActorCritic(
            observation_channels=env.observation_shape[0],
            action_count=cfg.environment.action_count,
            channels=cfg.network.channels,
            embedding_dim=cfg.network.embedding_dim,
            hidden_dim=cfg.hierarchical.hidden_dim,
            skills=payload.get("skills", cfg.hierarchical.skills),
            durations=payload.get("durations", cfg.hierarchical.durations),
            input_size=(cfg.mesen.observation_height, cfg.mesen.observation_width),
            button_pulse_durations=pulse_durations or cfg.hierarchical.button_pulse_durations,
        ).to(device)
        model.load_state_dict(payload["model"])
        controller = HierarchicalController(model, deterministic=True)
    else:
        model = RecurrentActorCritic(
            env.observation_shape[0],
            cfg.environment.action_count,
            cfg.network.channels,
            cfg.network.embedding_dim,
            cfg.network.gru_hidden,
            cfg.network.gru_layers,
            input_size=(cfg.mesen.observation_height, cfg.mesen.observation_width),
        ).to(device)
        with torch.no_grad():
            dummy = torch.zeros((1, *env.observation_shape), device=device)
            model.act(dummy, model.initial_hidden(1, device), deterministic=True)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        # The only permitted missing parameter when loading older flat checkpoints
        # is the newly introduced movement x shot interaction head (zero-init).
        bad_missing = [k for k in missing if not k.startswith("joint_interaction_head.")]
        if bad_missing or unexpected:
            raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={unexpected}")
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

    score_labels = {0: "0", 1: "15", 2: "30", 3: "40", 4: "Ad"}
    completed = 0
    try:
        obs = env.reset()
        hidden = model.initial_hidden(1, device)
        episode_reward = 0.0
        last_seen = (0, 0, 0, 0)
        while completed < args.episodes:
            obs_t = torch.from_numpy(obs).to(device).float() / 255.0
            with torch.inference_mode():
                pulse_frames = None
                if controller is not None:
                    controller.deterministic = not args.sample
                    actions, pulse_frames = controller.act_with_pulse(obs_t)
                    next_hidden = None
                else:
                    actions, _, _, next_hidden = model.act(obs_t, hidden, deterministic=not args.sample)
            pulse_np = pulse_frames.cpu().numpy() if pulse_frames is not None else None
            batch = env.step(actions.cpu().numpy(), pulse_frames=pulse_np)
            hidden = next_hidden if next_hidden is not None else hidden
            episode_reward += float(batch.rewards[0])

            info = batch.infos[0]
            p_games = int(info.get("player_games", 0))
            p_score = int(info.get("player_score", 0))
            o_games = int(info.get("opponent_games", 0))
            o_score = int(info.get("opponent_score", 0))
            seen = (p_games, p_score, o_games, o_score)
            if seen != last_seen and seen != (0, 0, 0, 0):
                p_text = score_labels.get(p_score, str(p_score))
                o_text = score_labels.get(o_score, str(o_score))
                print(f"  point ended -> Games [P: {p_games} - O: {o_games}] | "
                      f"Score [P: {p_text} - O: {o_text}] (reward so far: {episode_reward:+.2f})")
            last_seen = seen

            if batch.dones[0]:
                completed += 1
                won = p_games > o_games
                print(f"Episode {completed}/{args.episodes} finished: "
                      f"{'WIN' if won else 'LOSS'} | Games: {p_games}-{o_games} | "
                      f"total_reward={episode_reward:+.2f}")
                episode_reward = 0.0
                if controller is not None:
                    controller.reset(torch.ones(1, dtype=torch.bool, device=device))
                else:
                    hidden = model.initial_hidden(1, device)
                last_seen = (0, 0, 0, 0)
            obs = batch.observations.copy()
            if batch.dones.any():
                obs[0] = env.current_observation(0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
