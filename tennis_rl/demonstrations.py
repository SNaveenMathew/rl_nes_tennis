from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .actions import ACTION_NAMES
from .model import RecurrentActorCritic


@dataclass(frozen=True)
class PointEpisode:
    path: Path
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    outcome: str  # "winning" | "losing" | "unknown"
    level: int
    match_id: str
    point_index: int
    match_outcome: str = ""  # outcome of the whole recorded game (legacy label)
    pulse_frames: np.ndarray | None = None  # raw button-hold frames per observation; -1 = unknown

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def point_target(self) -> float:
        return 1.0 if self.outcome == "winning" else (0.0 if self.outcome == "losing" else -1.0)


@dataclass(frozen=True)
class DemoWindow:
    observations: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    outcome_target: np.ndarray
    return_to_go: np.ndarray
    outcome: str


def iter_point_files(root: str | Path, *, level: int | None = None) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    pattern = f"level_{level}" if level is not None else "level_*"
    paths: list[Path] = []
    for level_dir in sorted(root.glob(pattern)):
        for outcome in ("winning", "losing"):
            paths.extend(sorted((level_dir / outcome).glob("match_*/point_*.npz")))
    return paths


def _safe_load(path: Path, outcome_source: str = "point") -> PointEpisode | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.uint8)
            actions = np.asarray(data["actions"], dtype=np.int64)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            dones = np.asarray(data["dones"], dtype=np.bool_)
            level = int(np.asarray(data["level"]).item())
            point_index = int(np.asarray(data["point_index"]).item())
            match_id = str(np.asarray(data["match_id"]).item())
            outcome = str(np.asarray(data["outcome"]).item())
            pulse_frames = np.asarray(data["pulse_frames"], dtype=np.int16) if "pulse_frames" in data else None
        # The NPZ's stored outcome is normally the whole match/game result.
        # For point-level learning, use the point's own reward only. A zero-sum
        # point with no terminal reward is UNKNOWN; never infer the point outcome
        # from its containing match's result.
        if outcome_source not in {"point", "match"}:
            return None
        if observations.ndim != 4 or actions.ndim != 1:
            return None
        if observations.shape[0] != actions.shape[0] or observations.shape[1] != 4:
            return None
        if actions.size == 0:
            return None
        if rewards.shape != actions.shape or dones.shape != actions.shape:
            return None
        if pulse_frames is not None and pulse_frames.shape != actions.shape:
            return None
        if pulse_frames is None:
            pulse_frames = np.full(actions.shape, -1, dtype=np.int16)
        if np.any((actions < 0) | (actions >= len(ACTION_NAMES))):
            return None
        match_outcome = outcome if outcome in {"winning", "losing"} else "unknown"
        if outcome_source == "point":
            total = float(rewards.sum())
            if total > 1e-6:
                outcome = "winning"
            elif total < -1e-6:
                outcome = "losing"
            else:
                outcome = "unknown"
        elif outcome not in {"winning", "losing"}:
            outcome = "unknown"
        return PointEpisode(path, observations, actions, rewards, dones, outcome, level, match_id, point_index, match_outcome, pulse_frames)
    except (OSError, ValueError, KeyError, TypeError):
        return None


class DemonstrationDataset:
    """Point-level demonstrations with outcome-aware partial-window sampling.

    The human action remains the supervised policy target. Winning/losing is
    never used to relabel the human action as correct/incorrect. Instead,
    point outcome supplies (1) a sample weight for imitation, (2) a binary
    auxiliary outcome target, and (3) a discounted return-to-go used to make
    late, outcome-relevant parts of a point higher-value training examples.
    """

    def __init__(self, root: str | Path, winning_weight: float = 2.0, losing_weight: float = 0.5,
                 window_length: int = 64, return_gamma: float = 0.99, outcome_source: str = "point"):
        if outcome_source not in {"point", "match"}:
            raise ValueError("outcome_source must be 'point' or 'match'")
        self.outcome_source = outcome_source
        self.root = Path(root)
        self.winning_weight = float(winning_weight)
        self.losing_weight = float(losing_weight)
        self.window_length = int(window_length)
        self.return_gamma = float(return_gamma)
        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        self.episodes: list[PointEpisode] = []
        self.refresh()

    def refresh(self) -> None:
        episodes: list[PointEpisode] = []
        for path in iter_point_files(self.root):
            ep = _safe_load(path, self.outcome_source)
            if ep is not None:
                episodes.append(ep)
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def winning_points(self) -> int:
        return sum(ep.outcome == "winning" for ep in self.episodes)

    @property
    def losing_points(self) -> int:
        return sum(ep.outcome == "losing" for ep in self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(ep.length for ep in self.episodes)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[PointEpisode]:
        if not self.episodes:
            return []
        weights = np.asarray([
            self.winning_weight if ep.outcome == "winning" else (
                self.losing_weight if ep.outcome == "losing" else 1.0
            )
            for ep in self.episodes
        ], dtype=np.float64)
        weights /= weights.sum()
        idx = rng.choice(len(self.episodes), size=batch_size, replace=True, p=weights)
        return [self.episodes[int(i)] for i in np.asarray(idx).reshape(-1)]

    def sample_windows(self, batch_size: int, rng: np.random.Generator) -> list[DemoWindow]:
        if not self.episodes:
            return []
        chosen = self.sample(batch_size, rng)
        windows: list[DemoWindow] = []
        for ep in chosen:
            n = ep.length
            start = 0 if n <= self.window_length else int(rng.integers(0, n - self.window_length + 1))
            end = min(n, start + self.window_length)
            obs = ep.observations[start:end]
            actions = ep.actions[start:end]
            rewards = ep.rewards[start:end]
            # Compute the actual discounted reward-to-go available within this
            # recorded point window. This preserves the environment's reward
            # definition; it does not invent ball-position or shot events.
            rtg = np.zeros((end - start,), dtype=np.float32)
            g = 0.0
            for i in range(len(rtg) - 1, -1, -1):
                g = float(rewards[i]) + self.return_gamma * g
                rtg[i] = g
            if np.max(np.abs(rtg)) > 1e-6:
                scale = float(np.max(np.abs(rtg)))
                rtg = np.clip(rtg / scale, -1.0, 1.0)
            # Do not fabricate a return when the true point reward is outside
            # this window. The auxiliary outcome label below is separately masked
            # when the point result itself is unknown.
            windows.append(DemoWindow(
                observations=obs,
                actions=actions,
                mask=np.ones((len(actions),), dtype=np.bool_),
                outcome_target=np.full((len(actions),), ep.point_target, dtype=np.float32),
                return_to_go=rtg,
                outcome=ep.outcome,
            ))
        return windows


def collate_point_episodes(episodes: list[PointEpisode], device: torch.device):
    if not episodes:
        raise ValueError("Cannot collate empty demonstration batch")
    windows = [DemoWindow(ep.observations, ep.actions, np.ones(ep.length, dtype=np.bool_),
                          np.full(ep.length, ep.point_target, dtype=np.float32),
                          np.asarray(ep.rewards, dtype=np.float32), ep.outcome) for ep in episodes]
    obs, actions, mask, outcome, rtg, weights = collate_windows(windows, device)
    return obs, actions, mask, weights


def collate_windows(windows: list[DemoWindow], device: torch.device):
    if not windows:
        raise ValueError("Cannot collate empty demonstration batch")
    max_len = max(int(w.actions.shape[0]) for w in windows)
    batch = len(windows)
    obs = torch.zeros((max_len, batch, 4, 84, 84), dtype=torch.uint8, device=device)
    actions = torch.zeros((max_len, batch), dtype=torch.long, device=device)
    mask = torch.zeros((max_len, batch), dtype=torch.bool, device=device)
    outcome = torch.zeros((max_len, batch), dtype=torch.float32, device=device)
    rtg = torch.zeros((max_len, batch), dtype=torch.float32, device=device)
    weights = torch.ones((batch,), dtype=torch.float32, device=device)
    for j, w in enumerate(windows):
        n = int(w.actions.shape[0])
        if w.observations.shape[1:] != (4, 84, 84):
            raise ValueError(f"Unexpected demonstration observation shape: {w.observations.shape}")
        obs[:n, j] = torch.from_numpy(w.observations).to(device)
        actions[:n, j] = torch.from_numpy(w.actions).to(device)
        mask[:n, j] = torch.from_numpy(w.mask)
        outcome[:n, j] = torch.from_numpy(w.outcome_target)
        rtg[:n, j] = torch.from_numpy(w.return_to_go)
        weights[j] = 2.0 if w.outcome == "winning" else 0.5
    return obs, actions, mask, outcome, rtg, weights


def behavior_cloning_loss(model: RecurrentActorCritic, obs: torch.Tensor, actions: torch.Tensor,
                          mask: torch.Tensor, episode_weights: torch.Tensor) -> torch.Tensor:
    hidden = model.initial_hidden(obs.shape[1], obs.device)
    dist, _, _ = model.forward_sequence(obs.float() / 255.0, hidden)
    logp = dist.log_prob(actions)
    weighted = logp * mask.float() * episode_weights.view(1, -1)
    denom = (mask.float() * episode_weights.view(1, -1)).sum().clamp_min(1.0)
    return -weighted.sum() / denom


def partial_supervision_losses(model: RecurrentActorCritic, obs: torch.Tensor, actions: torch.Tensor,
                               mask: torch.Tensor, outcome_target: torch.Tensor,
                               return_to_go: torch.Tensor, episode_weights: torch.Tensor):
    """Policy BC + point-outcome auxiliary BCE + return-aware weighting.

    `return_to_go` is the discounted return from the recorded environment
    rewards within the sampled window. It is used only to modulate the BC
    weight: positive-return decisions are emphasized, while negative-return
    decisions are down-weighted rather than treated as an incorrect action.
    """
    hidden = model.initial_hidden(obs.shape[1], obs.device)
    dist, _, _, outcome_logits = model.forward_sequence(obs.float() / 255.0, hidden, include_aux=True)
    logp = dist.log_prob(actions)
    base = mask.float() * episode_weights.view(1, -1)
    # Smoothly emphasize demonstrator actions associated with positive local
    # return and reduce, but do not reverse, the weight for negative local return.
    local_weight = (1.0 + 0.75 * return_to_go).clamp_min(0.25)
    bc_weight = base * local_weight
    bc_denom = bc_weight.sum().clamp_min(1.0)
    bc_loss = -(logp * bc_weight).sum() / bc_denom
    known_outcome = mask.bool() & (outcome_target >= 0)
    safe_targets = outcome_target.clamp(0.0, 1.0)
    outcome_bce = nn.functional.binary_cross_entropy_with_logits(
        outcome_logits, safe_targets, reduction="none"
    )
    outcome_loss = (outcome_bce * known_outcome.float()).sum() / known_outcome.float().sum().clamp_min(1.0)
    return bc_loss, outcome_loss


def summarize(root: str | Path) -> dict[str, int]:
    ds = DemonstrationDataset(root)
    levels: dict[str, int] = {}
    for ep in ds.episodes:
        key = f"level_{ep.level}_{ep.outcome}"
        levels[key] = levels.get(key, 0) + 1
    return {"points": len(ds), "winning_points": ds.winning_points, "losing_points": ds.losing_points,
            "unknown_points": sum(ep.outcome == "unknown" for ep in ds.episodes),
            "steps": ds.total_steps, **levels}


def pretrain_behavior_cloning(model: RecurrentActorCritic, dataset: DemonstrationDataset, *,
                              updates: int, batch_size: int, learning_rate: float,
                              device: torch.device, seed: int = 42, outcome_coef: float = 0.25) -> dict[str, float]:
    if len(dataset) == 0 or updates <= 0:
        return {"bc_loss": 0.0, "outcome_loss": 0.0, "bc_steps": 0.0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    model.train()
    bc_total = 0.0
    outcome_total = 0.0
    update_count = 0
    for _ in range(updates):
        windows = dataset.sample_windows(batch_size, rng)
        if not windows:
            break
        obs, actions, mask, outcome_target, rtg, weights = collate_windows(windows, device)
        bc_loss, outcome_loss = partial_supervision_losses(model, obs, actions, mask, outcome_target, rtg, weights)
        loss = bc_loss + outcome_coef * outcome_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        bc_total += float(bc_loss.detach())
        outcome_total += float(outcome_loss.detach())
        update_count += 1
    return {"bc_loss": bc_total / max(update_count, 1),
            "outcome_loss": outcome_total / max(update_count, 1),
            "bc_steps": float(update_count)}
