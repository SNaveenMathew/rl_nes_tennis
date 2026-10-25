from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from .actions import ACTIONS, ACTION_NAMES, encode_action
from .config import load_config, validate_frame_perfect_config
from .mesen_client import BridgeError, MesenWorker
from .rom_patcher import resolve_rom
from .scenarios import StateRecord, advance_score, cell_id, split_for_match


def _action_from_buttons(buttons: dict[str, bool]) -> int:
    move: list[str] = []
    if buttons.get("left"):
        move.append("left")
    if buttons.get("right"):
        move.append("right")
    if buttons.get("up"):
        move.append("up")
    if buttons.get("down"):
        move.append("down")
    if len(move) == 2 and set(move) in ({"left", "up"}, {"right", "up"}, {"left", "down"}, {"right", "down"}):
        movement_name = "+".join(move)
    elif len(move) == 1:
        movement_name = move[0]
    else:
        movement_name = ""
    shot = "a" if buttons.get("a") else ("b" if buttons.get("b") else "")
    if movement_name:
        movement = next(i for i, names in enumerate(("", "left", "right", "up", "down", "left+up", "right+up", "left+down", "right+down")) if names == movement_name)
    else:
        movement = 0
    shot_idx = ("", "a", "b").index(shot)
    return encode_action(movement, shot_idx)


def _level_outcome_dir(root: Path, level: int, outcome: str, match_id: str) -> Path:
    path = root / f"level_{level}" / outcome / f"match_{match_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_point(
    path: Path,
    observations: list[np.ndarray] | np.ndarray,
    actions: list[int] | np.ndarray,
    rewards: list[float] | np.ndarray,
    dones: list[bool] | np.ndarray,
    *,
    level: int,
    outcome: str,
    match_id: str,
    point_index: int,
    metadata: dict[str, object],
    pulse_frames: list[int] | np.ndarray | None = None,
    native_frames: list[np.ndarray] | np.ndarray | None = None,
) -> None:
    """Persist a point plus optional exact 256x240 RGB frames for human review.

    ``observations`` remain compact grayscale policy inputs. Native RGB frames are
    a separate sidecar array aligned one-to-one with observations and are never
    consumed by the policy/network.
    """
    action_array = np.asarray(actions, dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "observations": np.asarray(observations, dtype=np.uint8),
        "actions": action_array,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "pulse_frames": np.asarray(pulse_frames if pulse_frames is not None else [-1] * len(action_array), dtype=np.int16),
        "level": np.asarray(level, dtype=np.int16),
        "outcome": np.asarray(outcome),
        "match_id": np.asarray(match_id),
        "point_index": np.asarray(point_index, dtype=np.int32),
    }
    if native_frames is not None:
        native = np.asarray(native_frames, dtype=np.uint8)
        expected_shape = (len(action_array), 240, 256, 3)
        if native.shape != expected_shape:
            raise ValueError(f"native_frames must have shape {expected_shape}; got {native.shape}")
        arrays["native_frames"] = native
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    serve_mode = getattr(args, "serve", "serve-first")
    if serve_mode not in {"serve-first", "receive-first"}:
        raise ValueError(f"Unsupported serve mode: {serve_mode!r}")
    cfg.mesen.serve_mode = serve_mode
    effective_match_games = cfg.mesen.match_games
    # NES Tennis has player 1 serve the first game and alternates service by game.
    # The patched ROM boots directly into the correct server configuration, so
    # recording starts at game 1 in both modes.
    record_game_number = 1
    if args.level is not None:
        cfg.mesen.game_level = args.level
    if getattr(args, "frame_skip", None) is not None:
        if int(args.frame_skip) < 1:
            raise ValueError("--frame-skip must be a positive number of raw NES frames")
        cfg.mesen.frame_skip = int(args.frame_skip)
    # Resolve and generate the correct patched ROM for this serve_mode + level.
    cfg.mesen.rom = str(resolve_rom(
        cfg.mesen.rom, serve_mode=cfg.mesen.serve_mode, level=cfg.mesen.game_level,
    ))
    print(f"ROM: {cfg.mesen.rom}")
    # CLI overrides must not quietly invalidate the one-input-per-raw-frame
    # demonstration contract enforced by the standard configs.
    validate_frame_perfect_config(cfg, context="recording options")
    output_root = Path(args.output)
    match_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{random.getrandbits(24):06x}"
    worker = MesenWorker(
        cfg.mesen.executable,
        cfg.mesen.rom,
        cfg.mesen.lua_script,
        cfg.mesen.host,
        cfg.mesen.port_start,
        cfg.mesen.observation_width,
        cfg.mesen.observation_height,
        cfg.mesen.frame_skip,
        cfg.mesen.startup_timeout_s,
        cfg.mesen.socket_timeout_s,
        match_games=effective_match_games,
        game_level=cfg.mesen.game_level,
        serve_mode=cfg.mesen.serve_mode,
        point_win=cfg.reward.point_win,
        point_loss=cfg.reward.point_loss,
        game_win=cfg.reward.game_win,
        game_loss=cfg.reward.game_loss,
        match_win=cfg.environment.terminal_match_reward,
        match_loss=cfg.environment.terminal_match_penalty,
        ipc_dir=cfg.mesen.ipc_dir,
        run_id=f"record_{match_id}",
        record_mode=True,
        headless=False,
        capture_native_frames=cfg.mesen.capture_native_frames,
    )

    frames: deque[np.ndarray] = deque(maxlen=cfg.environment.stack_size)
    actions: list[int] = []
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    pulse_frames: list[int] = []
    native_frames: list[np.ndarray] = []
    latest_native_frame: np.ndarray | None = None
    point_index = 0
    prev_p_score = 0
    prev_o_score = 0
    previous_obs: np.ndarray | None = None
    prev_game_total = 0
    prev_p_games = 0
    prev_o_games = 0
    recording_started = True
    recorded_game_outcome: str | None = None
    in_dead_time = False
    points: list[dict[str, object]] = []
    metadata_base = {
        "rom": cfg.mesen.rom,
        "level": cfg.mesen.game_level,
        "frame_skip": cfg.mesen.frame_skip,
        "observation_shape": [cfg.environment.stack_size, cfg.mesen.observation_height, cfg.mesen.observation_width],
        "action_count": len(ACTIONS),
        "action_names": list(ACTION_NAMES),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "segmentation": "score-change-v2",
        "scenario_tag": str(getattr(args, "scenario_tag", "") or ""),
        "records_button_pulse_frames": True,
        "capture_native_frames": bool(cfg.mesen.capture_native_frames),
        "native_frame_shape": [240, 256, 3] if cfg.mesen.capture_native_frames else None,
        "native_frame_format": "RGB" if cfg.mesen.capture_native_frames else None,
        "policy_observation_format": "grayscale_uint8",
    }

    # ---- optional savestate harvesting for point-level scenario training ----
    harvest_dir = Path(args.harvest_dir) if getattr(args, "harvest_dir", "") else None
    h_role = "serve" if serve_mode == "serve-first" else "return"
    h_score = (0, 0)
    h_step = 0
    h_start_taken = False
    h_manifest = None
    if harvest_dir is not None:
        (harvest_dir / "states" / match_id).mkdir(parents=True, exist_ok=True)
        h_manifest = harvest_dir / "manifest.jsonl"
        h_split = split_for_match(match_id, cfg.scenario.holdout_fraction)

    def h_write(row: dict) -> None:
        with h_manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    decision_hz = 60.1 / max(cfg.mesen.frame_skip, 1)
    native_text = "enabled (256x240 RGB per step)" if cfg.mesen.capture_native_frames else "disabled"
    sampling_mode = getattr(args, "sampling", "game")
    target_desc = f"{getattr(args, 'points', 1)} point(s)" if sampling_mode == "point" else "regular game"
    print(f"Recording NES Tennis human demonstration | level={cfg.mesen.game_level}")
    print(f"Policy input: {cfg.mesen.observation_width}x{cfg.mesen.observation_height} grayscale | native frames: {native_text}")
    print(f"Frame skip: {cfg.mesen.frame_skip} raw NES frames/step (~{decision_hz:.1f} decisions/s)")
    print(f"Serve mode: {serve_mode} | Sampling: {sampling_mode} ({target_desc})")
    print(f"Output: {output_root.resolve()}")
    print("Play until target is reached. Close Mesen or Ctrl+C to stop.")
    try:
        # First observation establishes the initial state; action/reward pair starts with the next sample.
        worker.action(0)
        frame, _, _, info = worker.recv_observation()
        for _ in range(cfg.environment.stack_size):
            frames.append(frame)
        previous_obs = np.stack(tuple(frames), axis=0)
        prev_p_score = int(info.get("player_score", 0))
        prev_o_score = int(info.get("opponent_score", 0))
        initial_p_games = int(info.get("player_games", 0))
        initial_o_games = int(info.get("opponent_games", 0))
        prev_p_games = initial_p_games
        prev_o_games = initial_o_games
        prev_game_total = initial_p_games + initial_o_games
        latest_native_frame = info.get("native_frame")
        if cfg.mesen.capture_native_frames and latest_native_frame is None:
            raise BridgeError("Native frame capture was enabled but the Mesen bridge returned no native frame")

        while True:
            worker.action(0)
            action_native_frame = latest_native_frame
            frame, reward, done, info = worker.recv_observation()
            latest_native_frame = info.get("native_frame", latest_native_frame)
            current_stack = np.stack(tuple(frames), axis=0)
            human_action = int(info.get("human_action", 0))
            frames.append(frame)
            next_stack = np.stack(tuple(frames), axis=0)

            p_score = int(info.get("player_score", 0))
            o_score = int(info.get("opponent_score", 0))
            p_games = int(info.get("player_games", 0))
            o_games = int(info.get("opponent_games", 0))
            game_total = p_games + o_games


            # RAM flag 'point_end' is non-zero for the whole dead time around a
            # point (rally over -> score updates -> next serve is ready), roughly
            # 30 observations. It must not be used as a point boundary: doing so
            # cut ~250 one-step "points" per game and labelled the real rally by
            # a reward that had not arrived yet. A point ends when the score changes.
            # Dead time after a score change is not part of the next point.
            flag_set = int(info.get("point_end", 0)) != 0
            if in_dead_time and flag_set and not done:
                prev_p_score, prev_o_score = p_score, o_score
                prev_game_total = game_total
                previous_obs = next_stack
                h_step = 0
                continue
            in_dead_time = False

            if recording_started:
                observations.append(current_stack)
                if cfg.mesen.capture_native_frames:
                    if action_native_frame is None:
                        raise BridgeError("Native frame capture disappeared during recording")
                    native_frames.append(np.asarray(action_native_frame, dtype=np.uint8))
                actions.append(human_action)
                rewards.append(float(reward))
                dones.append(bool(done))
                shot_id = human_action % 3
                if shot_id == 1:
                    pulse_frames.append(int(info.get("a_pulse_frames", -1)))
                elif shot_id == 2:
                    pulse_frames.append(int(info.get("b_pulse_frames", -1)))
                else:
                    pulse_frames.append(-1)

            point_boundary = (
                p_score != prev_p_score
                or o_score != prev_o_score
                or bool(done)
            )
            prev_p_score, prev_o_score = p_score, o_score
            previous_obs = next_stack

            if point_boundary:
                # At a game/match boundary the reward can be game-level rather than a separate point reward.
                point_reward = float(sum(rewards))
                point_outcome = "winning" if point_reward > 1e-6 else ("losing" if point_reward < -1e-6 else "unknown")
                # We do not yet know overall match outcome, so temporarily spool point data.
                if not recording_started:
                    prev_p_score, prev_o_score = p_score, o_score
                    prev_p_games, prev_o_games, prev_game_total = p_games, o_games, game_total
                    continue
                point_index += 1
                print(f"point={point_index:03d} steps={len(actions):4d} outcome={point_outcome} cumulative_reward={sum(rewards):+.2f}", flush=True)
                # Points are held in memory until the recorded game/match outcome is known.
                saved = {
                    "observations": np.asarray(observations, dtype=np.uint8),
                    "actions": np.asarray(actions, dtype=np.int64),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                    "dones": np.asarray(dones, dtype=np.bool_),
                    "pulse_frames": np.asarray(pulse_frames, dtype=np.int16),
                    "native_frames": np.asarray(native_frames, dtype=np.uint8) if cfg.mesen.capture_native_frames else None,
                    "point_index": point_index,
                    "p_score": p_score,
                    "o_score": o_score,
                    "provisional_outcome": point_outcome,
                }
                # Store in a lightweight in-memory list attached to function scope.
                points.append(saved)
                observations, actions, rewards, dones, pulse_frames, native_frames = [], [], [], [], [], []
                in_dead_time = True

            if harvest_dir is not None and recording_started:
                if point_boundary and points:
                    last = points[-1]
                    total = float(np.sum(last["rewards"]))
                    h_write({"finalize": {"match_id": match_id, "point_index": int(last["point_index"]),
                                          "point_steps": int(len(last["actions"])),
                                          "won": None if abs(total) < 1e-6 else bool(total > 0),
                                          "late_steps": args.late_steps}})
                    if abs(total) >= 1e-6:
                        h_score = advance_score(h_score[0], h_score[1], total > 0)[:2]
                    h_step, h_start_taken = 0, False
                else:
                    h_step += 1
                    kind = None
                    if not h_start_taken:
                        if h_step >= args.settle_steps:
                            kind, h_start_taken = "start", True
                    elif args.harvest_every > 0 and h_step % args.harvest_every == 0:
                        kind = "mid"
                    if kind is not None:
                        pidx = point_index + 1  # the point now in progress
                        rel = f"states/{match_id}/p{pidx:04d}_s{h_step:04d}.state"
                        worker.save_state_to(harvest_dir / rel)
                        h_write(StateRecord(path=rel, cell_id=cell_id(h_role, *h_score), kind=kind,
                                            match_id=match_id, level=cfg.mesen.game_level, point_index=pidx,
                                            step_in_point=h_step, split=h_split).to_json())

            game_changed = game_total != prev_game_total
            finish_point_sampling = (sampling_mode == "point" and point_boundary and len(points) >= getattr(args, "points", 1))
            finish_game_sampling = (sampling_mode == "game" and game_changed and recording_started and game_total >= effective_match_games)

            if finish_point_sampling or finish_game_sampling:
                if sampling_mode == "point":
                    pt_wins = sum(1 for p in points if p["provisional_outcome"] == "winning")
                    match_outcome = "winning" if pt_wins >= len(points) - pt_wins else "losing"
                else:
                    match_outcome = "winning" if p_games > prev_p_games else "losing"
                match_dir = _level_outcome_dir(output_root, cfg.mesen.game_level, match_outcome, match_id)
                for p in points:
                    metadata = dict(metadata_base)
                    metadata.update({
                        "match_id": match_id,
                        "point_index": p["point_index"],
                        "serve_mode": serve_mode,
                        "sampling": sampling_mode,
                        "recorded_game_number": record_game_number,
                        "recorded_game_outcome": match_outcome,
                        "player_games_at_end": p_games,
                        "opponent_games_at_end": o_games,
                        "provisional_point_outcome": p["provisional_outcome"],
                    })
                    point_name = match_dir / f"point_{p['point_index']:04d}.npz"
                    save_point(point_name, p["observations"], p["actions"], p["rewards"], p["dones"],
                               level=cfg.mesen.game_level, outcome=match_outcome, match_id=match_id,
                               point_index=p["point_index"], metadata=metadata, pulse_frames=p.get("pulse_frames"),
                               native_frames=p.get("native_frames"))
                summary = {**metadata_base, "match_id": match_id, "serve_mode": serve_mode,
                           "sampling": sampling_mode,
                           "recorded_game_number": record_game_number, "recorded_game_outcome": match_outcome,
                           "player_games_at_end": p_games, "opponent_games_at_end": o_games,
                           "points": len(points)}
                (match_dir / "match.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                label = f"{len(points)} point(s)" if sampling_mode == "point" else f"Game {record_game_number} ({p_games}-{o_games})"
                print(f"RECORDING COMPLETE: {match_outcome} | {label}")
                print(f"Saved {len(points)} point episode(s) under {match_dir}")
                return 0

            prev_p_games, prev_o_games, prev_game_total = p_games, o_games, game_total

            if done:
                final_p_games = p_games
                final_o_games = o_games
                match_outcome = "winning" if final_p_games > final_o_games else "losing"
                match_dir = _level_outcome_dir(output_root, cfg.mesen.game_level, match_outcome, match_id)
                for p in points:
                    metadata = dict(metadata_base)
                    metadata.update({
                        "match_id": match_id,
                        "point_index": p["point_index"],
                        "serve_mode": serve_mode,
                        "sampling": sampling_mode,
                        "match_outcome": match_outcome,
                        "player_games": final_p_games,
                        "opponent_games": final_o_games,
                        "provisional_point_outcome": p["provisional_outcome"],
                    })
                    point_name = match_dir / f"point_{p['point_index']:04d}.npz"
                    save_point(
                        point_name,
                        p["observations"], p["actions"], p["rewards"], p["dones"],
                        level=cfg.mesen.game_level, outcome=match_outcome, match_id=match_id,
                        point_index=p["point_index"], metadata=metadata, pulse_frames=p.get("pulse_frames"),
                        native_frames=p.get("native_frames"),
                    )
                summary = {
                    **metadata_base,
                    "match_id": match_id,
                    "serve_mode": serve_mode,
                    "sampling": sampling_mode,
                    "recorded_game_number": record_game_number,
                    "match_outcome": match_outcome,
                    "player_games": final_p_games,
                    "opponent_games": final_o_games,
                    "points": len(points),
                }
                (match_dir / "match.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"MATCH COMPLETE: {match_outcome} {final_p_games}-{final_o_games}")
                print(f"Saved {len(points)} point episodes under {match_dir}")
                return 0
    except KeyboardInterrupt:
        print("\nRecording stopped before match completion; unsaved partial points were intentionally discarded.")
        return 130
    except BridgeError as exc:
        print(f"Recorder bridge error: {exc}")
        return 2
    finally:
        worker.close(force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a human NES Tennis match into point-level training episodes.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--output", default="demonstrations")
    parser.add_argument("--level", type=int, choices=range(1, 6), default=None,
                        help="CPU opponent level (1-5, default from config).")
    parser.add_argument("--sampling", "--episode-mode", choices=("point", "game"), default="game",
                        help="Recording approach: 'point' (play till end of point, record N points and exit) or 'game' (play a regular game).")
    parser.add_argument("--points", type=int, default=1,
                        help="Target number of points to record when --sampling is 'point' (default: 1).")
    parser.add_argument("--harvest-dir", default="",
                        help="Also save emulator states for point-level scenario training into this directory.")
    parser.add_argument("--settle-steps", type=int, default=4,
                        help="Live-play agent steps (dead time between points is skipped) before the 'start' snapshot is taken; it must still be before the serve.")
    parser.add_argument("--harvest-every", type=int, default=0,
                        help="Also snapshot every N steps inside a point (feeds the 'mid'/'late' reverse-curriculum phases). 0 = off.")
    parser.add_argument("--late-steps", type=int, default=45,
                        help="A snapshot within this many steps of the point's end counts as 'late'.")
    parser.add_argument("--serve", choices=("serve-first", "receive-first"), default="serve-first",
                        help="Serve mode: player 1 serves first, or CPU serves first (player 1 receives).")
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="Raw NES frames per recorded observation. Default: config value. Match this value in train/evaluation config.")
    parser.add_argument("--scenario-tag", default="", help="Optional human-readable scenario label written into each point's metadata.")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
