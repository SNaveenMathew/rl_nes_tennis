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
    frame_skip: int = 4
    match_games: int = 6
    game_level: int = 1
    observation_width: int = 84
    observation_height: int = 84
    save_state_path: str = ""
    auto_reset_to_saved_state: bool = True
    reset_startup_frames: int = 0
    transport: str = "file_mailbox"
    ipc_dir: str = "runs/mesen_ipc"
    capture_stderr: bool = False
    headless: bool = True


@dataclass
class EnvConfig:
    action_count: int = 19
    stack_size: int = 4
    episode_max_steps: int = 15000
    terminal_match_reward: float = 25.0
    living_penalty: float = 0.0


@dataclass
class RewardConfig:
    point_win: float = 1.0
    point_loss: float = -1.0
    game_win: float = 5.0
    game_loss: float = -5.0
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
    rollout_steps: int = 1024
    epochs: int = 4
    sequence_length: int = 32
    sequence_batch_size: int = 16
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_clip_range: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    amp: bool = True
    total_steps: int = 5_000_000
    checkpoint_every_updates: int = 25
    save_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    max_step_checkpoints: int = 3


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    mesen: MesenConfig = field(default_factory=MesenConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


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


def load_config(path: str | Path) -> Config:
    root = Config()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _merge_dataclass(root, data)
    return root
