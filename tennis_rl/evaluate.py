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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario.json")
    parser.add_argument("--checkpoint", default="checkpoints_hierarchical_native256x240_frame1/hierarchical_latest.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--level", type=int, default=None, choices=range(1, 6), metavar="{1-5}",
                        help="CPU difficulty level (1-5, default from config).")
    parser.add_argument("--serve", default=None, choices=["serve-first", "receive-first"],
                        help="Serve mode: who serves game 1 (default from config).")
    parser.add_argument("--sampling", "--episode-mode", choices=["point", "game"], default="game",
                        help="Evaluation sampling / episode scope: 'point' (evaluate N individual points) or 'game' (evaluate N regular games).")
    parser.add_argument("--side", choices=["deuce", "ad", "advantage"], default=None,
                        help="Service side: 'deuce' or 'ad'/'advantage' (only valid when --sampling is 'point').")
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
    print(f"Evaluation | Sampling: {args.sampling} | Serve: {cfg.mesen.serve_mode} ({cfg.mesen.serve_side}) | Level: {cfg.mesen.game_level}")
    print(f"ROM: {cfg.mesen.rom}")
    if cfg.mesen.transport != "file_mailbox":
        raise ValueError(f"Unsupported Mesen transport: {cfg.mesen.transport!r}; expected 'file_mailbox'")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(cfg.device)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hierarchical = payload.get("format") == "tennis_rl_hierarchical_v2"
    pulse_durations = validate_checkpoint_runtime(payload, cfg, hierarchical=hierarchical)

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{random.getrandbits(40):010x}"
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
            game_level=cfg.mesen.game_level,
            serve_mode=cfg.mesen.serve_mode,
            serve_side=cfg.mesen.serve_side,
            point_win=cfg.reward.point_win,
            point_loss=cfg.reward.point_loss,
            game_win=cfg.reward.game_win,
            game_loss=cfg.reward.game_loss,
            ipc_dir=cfg.mesen.ipc_dir,
            run_id=run_id,
            lockstep=cfg.mesen.lockstep,
            lockstep_timeout_s=cfg.mesen.lockstep_timeout_s,
            strict_lockstep=True,
        )
        for i in range(min(cfg.mesen.workers, args.episodes))
    ]
    is_point = args.sampling == "point"
    max_steps = getattr(cfg.scenario, "point_max_steps", 900) if is_point else cfg.environment.episode_max_steps
    env = ParallelTennis(workers, cfg.environment.stack_size, max_steps, point_episodes=is_point, point_max_steps=max_steps)

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

    wins = 0
    rewards: list[float] = []
    player_games_list: list[int] = []
    opponent_games_list: list[int] = []
    completed = 0

    try:
        obs = env.reset()
        hidden = model.initial_hidden(env.n, device)
        current_rewards = np.zeros(env.n, dtype=np.float32)
        while completed < args.episodes:
            obs_t = torch.from_numpy(obs).to(device).float() / 255.0
            with torch.inference_mode():
                pulse_frames = None
                if controller is not None:
                    actions, pulse_frames = controller.act_with_pulse(obs_t)
                    next_hidden = None
                else:
                    actions, _, _, next_hidden = model.act(obs_t, hidden, deterministic=True)
            pulse_np = pulse_frames.cpu().numpy() if pulse_frames is not None else None
            batch = env.step(actions.cpu().numpy(), pulse_frames=pulse_np)
            current_rewards += batch.rewards
            obs = batch.observations.copy()
            hidden = next_hidden if next_hidden is not None else hidden
            if batch.dones.any():
                for i in np.flatnonzero(batch.dones):
                    obs[int(i)] = env.current_observation(int(i))
                if controller is not None:
                    controller.reset(torch.from_numpy(batch.dones).to(device))
                else:
                    mask = torch.from_numpy((~batch.dones).astype(np.float32)).to(device)
                    hidden = hidden * mask.view(1, env.n, 1)
            for i, done in enumerate(batch.dones.tolist()):
                if done and completed < args.episodes:
                    completed += 1
                    rewards.append(float(current_rewards[i]))
                    p_games = int(batch.infos[i].get("player_games", 0))
                    o_games = int(batch.infos[i].get("opponent_games", 0))
                    player_games_list.append(p_games)
                    opponent_games_list.append(o_games)
                    if args.sampling == "point":
                        pt_res = int(batch.infos[i].get("point_result", 0))
                        is_win = (pt_res > 0) if pt_res != 0 else (current_rewards[i] > 0)
                        if is_win:
                            wins += 1
                    else:
                        if p_games > o_games or (p_games == o_games and current_rewards[i] > 0):
                            wins += 1
                    current_rewards[i] = 0
        unit = "points" if args.sampling == "point" else "games"
        print(f"evaluation_mode={args.sampling}")
        print(f"episodes={completed} {unit}")
        print(f"wins={wins}")
        print(f"win_rate={wins / max(completed, 1):.3f}")
        if args.sampling == "game":
            print(f"mean_player_games={float(np.mean(player_games_list)):.2f}")
            print(f"mean_opponent_games={float(np.mean(opponent_games_list)):.2f}")
        print(f"mean_reward={float(np.mean(rewards)):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
