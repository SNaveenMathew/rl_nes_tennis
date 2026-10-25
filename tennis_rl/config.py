from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MesenConfig:
    executable: str = ""
    rom: str = ""
    lua_script: str = "mesen/bridge.lua"
    workers: int = 8
    host: str = "127.0.0.1"
    port_start: int = 39100
    startup_timeout_s: float = 20.0
    socket_timeout_s: float = 30.0
    # Keep collection, training, evaluation, and inference on the same temporal/spatial contract.
    frame_skip: int = 1
    match_games: int = 6
    game_level: int = 1
    # "serve-first": player serves game 1; "receive-first": CPU serves game 1.
    # resolve_rom() reads this to select and generate the correct patched ROM.
    serve_mode: str = "serve-first"
    # "deuce": start serving on deuce (right) court; "ad": advantage (left) court (point mode only).
    serve_side: str = "deuce"
    observation_width: int = 256
    observation_height: int = 240
    # Native RGB is stored only by the human demonstration recorder, never by RL workers.
    capture_native_frames: bool = True
    save_state_path: str = ""
    auto_reset_to_saved_state: bool = True
    reset_startup_frames: int = 0
    transport: str = "file_mailbox"
    ipc_dir: str = "runs/mesen_ipc"
    capture_stderr: bool = False
    headless: bool = True
    # Block the emulator between observations until the next action arrives
    # (deterministic frame timing). Ignored for human recording.
    lockstep: bool = True
    lockstep_timeout_s: float = 600.0



@dataclass
class EnvConfig:
    action_count: int = 27
    stack_size: int = 4
    episode_max_steps: int = 15000
    terminal_match_reward: float = 10.0
    terminal_match_penalty: float = -10.0
    living_penalty: float = 0.0


@dataclass
class RewardConfig:
    point_win: float = 1.0
    point_loss: float = -1.0
    game_win: float = 3.0
    game_loss: float = -3.0
    game_progress: float = 0.0
    diagnostic_ram: bool = True


@dataclass
class NetworkConfig:
    channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    embedding_dim: int = 256
    gru_hidden: int = 256
    gru_layers: int = 1


@dataclass
class PPOConfig:
    """Configuration for the legacy flat recurrent PPO baseline."""

    rollout_steps: int = 1024
    epochs: int = 4
    sequence_length: int = 64
    sequence_batch_size: int = 16
    learning_rate: float = 2.5e-4
    learning_rate_final: float = 2.5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_clip_range: float = 0.1
    entropy_coef_start: float = 0.020
    entropy_coef_final: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    amp: bool = True
    total_steps: int = 5_000_000
    checkpoint_every_updates: int = 25
    save_dir: str = "checkpoints_native256x240_frame1"
    tensorboard_dir: str = "runs_native256x240_frame1"
    max_step_checkpoints: int = 3

    # Imitation is supplemental supervision; do not infer the correctness of an
    # action merely from whether its point or match was won.
    demo_bc_coef: float = 0.5
    demo_bc_coef_start: float = 0.5
    demo_bc_coef_final: float = 0.15
    demo_bc_decay_steps: int = 1_500_000
    aux_motion_coef: float = 0.0  # disabled: image motion centroid is not a ball label
    demo_outcome_coef: float = 0.25
    demo_window_length: int = 64
    demo_return_gamma: float = 0.99
    demo_batches_per_update: int = 1
    demo_batch_size: int = 4
    demo_pretrain_updates: int = 500
    demo_pretrain_batch_size: int = 4
    demo_pretrain_learning_rate: float = 1e-4
    demo_root: str = "demonstrations"
    demo_outcome_source: str = "point"
    demo_winning_weight: float = 2.0
    demo_losing_weight: float = 0.5

    # Optional DQfD experiments retained for reproducible baseline comparisons.
    demo_replay_capacity: int = 20_000
    online_replay_capacity: int = 50_000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    demo_priority_bonus: float = 2.0
    demo_fraction: float = 0.5
    demo_margin: float = 0.8
    demo_large_margin_coef: float = 1.0
    epsilon_start: float = 0.10
    epsilon_final: float = 0.01
    epsilon_decay_fraction: float = 0.50
    updates_per_rollout: int = 8
    target_update_every: int = 100
    n_step_return: int = 3

    # Demonstration micro-segment replay. Segment labels inferred from action
    # frequency are weak metadata, not ground-truth tennis task labels.
    seg_replay_seg_len: int = 16
    seg_replay_batch_size: int = 8
    seg_replay_steps_per_update: int = 4
    seg_replay_winning_bc_weight: float = 2.5
    seg_replay_losing_bc_weight: float = 0.4
    seg_replay_outcome_coef: float = 0.30
    seg_replay_lr: float = 3e-5
    seg_replay_refresh_every: int = 50


@dataclass
class HierarchicalConfig:
    """Options for the learned manager + specialized expert policies."""

    enabled: bool = True
    # Ordered leaf tasks. Root phase-gate and phase-specific sub-gates are
    # derived from hierarchy_schema.PHASE_TASKS; inference routing is learned.
    skills: list[str] = field(default_factory=lambda: [
        "serve_timing", "serve_placement_power", "serve_recovery",
        "receive_position", "return_stroke_timing", "intercept_track",
        "hold_cover", "recover_reposition", "approach_net", "retreat_lob",
        "stroke_selector", "stroke_timing", "transition_wait_reposition",
    ])
    durations: list[int] = field(default_factory=lambda: [1])
    # Shot button pulse durations are measured in raw NES frames and must be
    # <= mesen.frame_skip. The bridge preserves movement while pulsing A/B.
    button_pulse_durations: list[int] = field(default_factory=lambda: [1])
    hidden_dim: int = 128
    rollout_steps: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    manager_supervision_coef: float = 0.15
    demo_bc_coef: float = 0.5
    perception_coef: float = 0.2
    batch_size: int = 64
    epochs: int = 3
    checkpoint_every_updates: int = 10
    max_step_checkpoints: int = 3
    save_dir: str = "checkpoints_hierarchical_native256x240_frame1"
    log_dir: str = "runs_hierarchical_native256x240_frame1"
    # Inference and training are never allowed to apply heuristic task routing.
    # This flag only enables supervised manager targets from explicit scenario
    # metadata (e.g. the ROM's known server/receiver role at point start).
    use_verified_start_role_labels: bool = True


@dataclass
class SILConfig:
    enabled: bool = False
    # Replay buffer size in chunks (each up to ppo.sequence_length steps).
    # Memory cost is roughly capacity_chunks * sequence_length * stack_size *
    # obs_width * obs_height bytes (uint8 observations) - the default of 256
    # chunks at sequence_length=128 is ~925MB of CPU RAM, not GPU VRAM.
    buffer_capacity_chunks: int = 256
    min_buffer_chunks: int = 32  # don't run SIL updates until the buffer has at least this many
    batch_size: int = 8          # chunks sampled per SIL gradient step
    updates_per_iteration: int = 4  # SIL gradient steps per PPO rollout/update cycle
    policy_coef: float = 1.0
    value_coef: float = 0.5
    # Sample chunks with probability proportional to exp(priority / temperature)
    # instead of uniformly. Uniform sampling meant a rare winning chunk had a
    # real chance of never being drawn at all before eviction (see sil.py) -
    # this is the "prioritize past episodes with a good outcome" lever.
    # Lower temperature = sharper preference for high-return chunks; disable
    # with priority_sampling=False to fall back to the original uniform draw.
    priority_sampling: bool = True
    priority_temperature: float = 5.0


@dataclass
class ScenarioConfig:
    """Point-level training from harvested savestates (see SCENARIOS.md)."""

    enabled: bool = False
    state_dir: str = "scenario_states"
    point_episodes: bool = True       # scenario episodes end when the point ends
    point_max_steps: int = 900        # timeout for one full point (agent steps, fallback)
    segment_mode: bool = True         # True: use short segment horizons by phase
    late_max_steps: int = 40          # Horizon for endgame/resolution segments
    mid_max_steps: int = 50           # Horizon for rally/in-play segments
    start_max_steps: int = 25         # Horizon for serve/receive initiation segments
    serve_stall_penalty: float = -1.0 # Penalty if start segment times out without ball put in play
    rally_progress_reward: float = 0.20 # Reward for sustaining rally through the mid segment
    late_progress_reward: float = 0.10  # Reward for sustaining rally through the late segment
    anchor_fraction: float = 0.15     # share of episodes that are ordinary full games
    probe_fraction: float = 0.10      # share of episodes that start from HOLDOUT states, greedy, not trained on
    noop_min: int = 0
    noop_max: int = 30                # random no-op frames after loading a state
    loss_state_fraction: float = 0.25 # share of states cut from points the human lost
    stratum_floor: float = 0.5        # probability mass reserved as a per-stratum floor
    retention_boost: float = 3.0      # extra weight for strata whose win-rate has dropped
    promote_winrate: float = 0.45     # reverse curriculum winrate required to unlock earlier phase
    promote_min_episodes: int = 15    # minimum episodes required before phase promotion
    zpd_threshold: float = 0.20       # winrate below which struggling cells focus on late phase
    zpd_late_bias: float = 0.65       # probability to assign late phase for struggling cells
    holdout_fraction: float = 0.2     # fraction of source MATCHES held out (used by the harvester)
    min_probe_episodes: int = 20      # probe results needed before a checkpoint is called "best"
    report_every_updates: int = 20


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    mesen: MesenConfig = field(default_factory=MesenConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sil: SILConfig = field(default_factory=SILConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)


def _merge_dataclass(dc: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


FRAME_PERFECT_WIDTH = 256
FRAME_PERFECT_HEIGHT = 240
FRAME_PERFECT_SKIP = 1


def validate_frame_perfect_config(cfg: Config, *, context: str = "configuration") -> None:
    """Reject any runtime contract that would discard spatial or temporal detail.

    One policy step must represent exactly one raw NES frame, and the model input
    must preserve the complete 256x240 raster. Hierarchical option and A/B pulse
    durations must also be one frame so no action is committed for multiple frames.
    """
    actual = (
        int(cfg.mesen.observation_width),
        int(cfg.mesen.observation_height),
        int(cfg.mesen.frame_skip),
    )
    expected = (FRAME_PERFECT_WIDTH, FRAME_PERFECT_HEIGHT, FRAME_PERFECT_SKIP)
    problems: list[str] = []
    if actual != expected:
        problems.append(
            f"expected observation_width/height/frame_skip={expected}, got {actual}"
        )
    hc = getattr(cfg, "hierarchical", None)
    if hc is not None and bool(hc.enabled):
        durations = tuple(int(x) for x in hc.durations)
        pulse_durations = tuple(int(x) for x in hc.button_pulse_durations)
        if durations != (1,):
            problems.append(f"hierarchical.durations must be [1], got {list(durations)}")
        if pulse_durations != (1,):
            problems.append(
                "hierarchical.button_pulse_durations must be [1], "
                f"got {list(pulse_durations)}"
            )
    if problems:
        raise ValueError(
            f"{context} violates the frame-perfect contract: " + "; ".join(problems)
            + ". Use full 256x240 observations, frame_skip=1, and one-frame actions/pulses."
        )


def load_config(path: str | Path) -> Config:
    root = Config()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _merge_dataclass(root, data)
    validate_frame_perfect_config(root, context=f"config {path}")
    return root
