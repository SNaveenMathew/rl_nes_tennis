from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass

from .model import RecurrentActorCritic


@dataclass
class Rollout:
    obs: torch.Tensor          # [T, N, C, H, W], uint8
    actions: torch.Tensor      # [T, N]
    rewards: torch.Tensor      # [T, N]
    dones: torch.Tensor        # [T, N]
    values: torch.Tensor       # [T, N]
    log_probs: torch.Tensor    # [T, N]
    hiddens: torch.Tensor      # [T, L, N, H]
    advantages: torch.Tensor   # [T, N]
    returns: torch.Tensor      # [T, N]


class PPOTrainer:
    def __init__(self, model: RecurrentActorCritic, cfg, device: torch.device, run_dir: str):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            eps=1e-5,
            weight_decay=0.0,
        )
        self.writer = SummaryWriter(run_dir)
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
        self.update_index = 0
        self.env_steps = 0

    def _split_sequences(self, rollout: Rollout) -> list[tuple[torch.Tensor, ...]]:
        T, N = rollout.actions.shape
        L = self.cfg.sequence_length
        if T % L != 0:
            raise ValueError(f"rollout_steps={T} must be divisible by sequence_length={L}")

        sequences = []
        for env_idx in range(N):
            for start in range(0, T, L):
                end = start + L
                h0 = rollout.hiddens[start, :, env_idx, :].contiguous()
                sequences.append((
                    rollout.obs[start:end, env_idx],
                    rollout.actions[start:end, env_idx],
                    rollout.old_log_probs[start:end, env_idx] if hasattr(rollout, "old_log_probs") else rollout.log_probs[start:end, env_idx],
                    rollout.advantages[start:end, env_idx],
                    rollout.returns[start:end, env_idx],
                    rollout.values[start:end, env_idx],
                    h0,
                ))
        return sequences

    def update(self, rollout: Rollout) -> dict[str, float]:
        self.model.train()
        sequences = self._split_sequences(rollout)
        rng = np.random.default_rng(self.update_index + 17)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        minibatches = 0

        for epoch in range(self.cfg.epochs):
            order = rng.permutation(len(sequences))
            for pos in range(0, len(order), self.cfg.sequence_batch_size):
                ids = order[pos:pos + self.cfg.sequence_batch_size]
                if len(ids) == 0:
                    continue
                obs = torch.stack([sequences[i][0] for i in ids], dim=1).to(self.device, non_blocking=True).float() / 255.0
                actions = torch.stack([sequences[i][1] for i in ids], dim=1).to(self.device)
                old_logp = torch.stack([sequences[i][2] for i in ids], dim=1).to(self.device)
                advantages = torch.stack([sequences[i][3] for i in ids], dim=1).to(self.device)
                returns = torch.stack([sequences[i][4] for i in ids], dim=1).to(self.device)
                old_values = torch.stack([sequences[i][5] for i in ids], dim=1).to(self.device)
                h0 = torch.stack([sequences[i][6] for i in ids], dim=1).to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    dist, values, _ = self.model.forward_sequence(obs, h0)
                    new_logp = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_logp - old_logp)
                    unclipped = ratio * advantages
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_range,
                        1.0 + self.cfg.clip_range,
                    ) * advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()

                    value_pred = values
                    if self.cfg.value_clip_range > 0:
                        value_clipped = old_values + torch.clamp(
                            value_pred - old_values,
                            -self.cfg.value_clip_range,
                            self.cfg.value_clip_range,
                        )
                        value_loss_a = (value_pred - returns).pow(2)
                        value_loss_b = (value_clipped - returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_a, value_loss_b).mean()
                    else:
                        value_loss = 0.5 * (value_pred - returns).pow(2).mean()

                    loss = (
                        policy_loss
                        + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_logp - new_logp).mean()).abs()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_range).float().mean()

                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.detach())
                metrics["approx_kl"] += float(approx_kl.detach())
                metrics["clip_frac"] += float(clip_frac.detach())
                minibatches += 1

        if minibatches:
            for key in metrics:
                metrics[key] /= minibatches

        self.update_index += 1
        return metrics

    def log(
        self,
        metrics: dict[str, float],
        env_steps: int,
        extra: dict[str, float] | None = None,
    ) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"ppo/{key}", value, env_steps)
        for key, value in (extra or {}).items():
            self.writer.add_scalar(f"system/{key}", value, env_steps)
        self.writer.add_scalar("system/env_steps", env_steps, env_steps)
        self.writer.add_scalar("system/update", self.update_index, env_steps)
        self.writer.flush()

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "update_index": self.update_index,
            "env_steps": self.env_steps,
            "extra": extra or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scaler.is_enabled() and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.update_index = int(payload.get("update_index", 0))
        self.env_steps = int(payload.get("env_steps", 0))
        return payload.get("extra", {})
