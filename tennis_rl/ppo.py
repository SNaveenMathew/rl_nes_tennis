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

from .model import RecurrentActorCritic, compute_spatial_motion_target


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
    truncateds: torch.Tensor | None = None


def compute_gae_with_truncation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    truncateds: torch.Tensor,
    truncated_values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without bootstrapping across an auto-reset boundary.

    ``truncated_values[t]`` must be the value of the final observation from the
    interrupted episode. It is deliberately distinct from ``values[t + 1]``,
    which may be the first state of a newly reset worker.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape or rewards.shape != truncateds.shape:
        raise ValueError("rewards, values, dones and truncateds must have identical [T,N] shapes")
    if rewards.shape != truncated_values.shape:
        raise ValueError("truncated_values must have the same [T,N] shape as rewards")
    if last_values.shape != rewards.shape[1:]:
        raise ValueError("last_values must have shape [N]")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_values if t == rewards.shape[0] - 1 else values[t + 1]
        is_truncated = truncateds[t].to(dtype=torch.bool)
        next_value = torch.where(is_truncated, truncated_values[t], next_value)
        is_terminated = dones[t].to(dtype=torch.bool) & (~is_truncated)
        bootstrap_mask = (~is_terminated).to(rewards.dtype)
        delta = rewards[t] + float(gamma) * next_value * bootstrap_mask - values[t]
        # No trace crosses a done boundary. For truncation, its terminal-state
        # bootstrap is included in delta, but the reset episode's trace is cut.
        trace_mask = (~dones[t].to(dtype=torch.bool)).to(rewards.dtype)
        gae = delta + float(gamma) * float(gae_lambda) * trace_mask * gae
        advantages[t] = gae
    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, model: RecurrentActorCritic, cfg, device: torch.device, run_dir: str,
                 demonstration_dataset=None, segment_replayer=None):
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
        self.demonstration_dataset = demonstration_dataset
        self.segment_replayer = segment_replayer

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

        # Linear LR annealing: full learning_rate early, decaying to
        # learning_rate_final by total_steps. env_steps/update_index are
        # already checkpointed, so this schedule resumes correctly.
        progress = min(1.0, self.env_steps / max(1, self.cfg.total_steps))
        lr_now = self.cfg.learning_rate + (self.cfg.learning_rate_final - self.cfg.learning_rate) * progress
        for group in self.optimizer.param_groups:
            group["lr"] = lr_now

        # Cosine annealing for demonstration behavioral cloning coefficient:
        # High early to anchor on human shot execution, decaying smoothly to allow
        # PPO exploration and novel winning strategies without forgetting.
        bc_decay_steps = getattr(self.cfg, "demo_bc_decay_steps", 1_500_000)
        bc_start = getattr(self.cfg, "demo_bc_coef_start", self.cfg.demo_bc_coef)
        bc_final = getattr(self.cfg, "demo_bc_coef_final", 0.15)
        bc_progress = min(1.0, self.env_steps / max(1, bc_decay_steps))
        demo_bc_coef_now = bc_final + 0.5 * (bc_start - bc_final) * (1.0 + math.cos(math.pi * bc_progress))

        sequences = self._split_sequences(rollout)
        rng = np.random.default_rng(self.update_index + 17)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "motion_loss": 0.0,
        }
        progress_for_entropy = min(1.0, self.env_steps / max(1, self.cfg.total_steps))
        entropy_coef_now = self.cfg.entropy_coef_start + (self.cfg.entropy_coef_final - self.cfg.entropy_coef_start) * progress_for_entropy
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
                    aux_motion_coef = getattr(self.cfg, "aux_motion_coef", 0.10)
                    if aux_motion_coef > 0.0:
                        dist, values, _, motion_pred = self.model.forward_sequence(obs, h0, include_motion=True)
                        motion_target = compute_spatial_motion_target(obs)
                        motion_loss = torch.nn.functional.smooth_l1_loss(motion_pred, motion_target)
                    else:
                        dist, values, _ = self.model.forward_sequence(obs, h0)
                        motion_loss = torch.zeros((), device=self.device)

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
                        - entropy_coef_now * entropy
                        + aux_motion_coef * motion_loss
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
                metrics["motion_loss"] += float(motion_loss.detach())
                minibatches += 1

        metrics["demo_bc_loss"] = 0.0
        metrics["demo_outcome_loss"] = 0.0
        metrics["demo_used"] = 0.0
        if self.segment_replayer is not None and len(self.segment_replayer._segments) > 0 and demo_bc_coef_now > 0.0:
            demo_batches = max(1, int(self.cfg.demo_batches_per_update))
            demo_loss_sum = 0.0
            demo_outcome_sum = 0.0
            for _ in range(demo_batches):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    demo_bc, demo_outcome = self.segment_replayer.compute_losses(self.model)
                    weighted_demo = demo_bc_coef_now * (
                        demo_bc + self.cfg.demo_outcome_coef * demo_outcome
                    )
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(weighted_demo).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    weighted_demo.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                demo_loss_sum += float(demo_bc.detach())
                demo_outcome_sum += float(demo_outcome.detach())
            metrics["demo_bc_loss"] = demo_loss_sum / max(demo_batches, 1)
            metrics["demo_outcome_loss"] = demo_outcome_sum / max(demo_batches, 1)
            metrics["demo_used"] = 1.0
        elif self.demonstration_dataset is not None and len(self.demonstration_dataset) > 0 and demo_bc_coef_now > 0.0:
            from .demonstrations import collate_windows, partial_supervision_losses
            demo_rng = np.random.default_rng(self.update_index + 10091)
            demo_batches = max(1, int(self.cfg.demo_batches_per_update))
            demo_loss_sum = 0.0
            demo_outcome_sum = 0.0
            for _ in range(demo_batches):
                windows = self.demonstration_dataset.sample_windows(self.cfg.demo_batch_size, demo_rng)
                if not windows:
                    break
                d_obs, d_actions, d_mask, d_outcome, d_rtg, d_weights = collate_windows(windows, self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    demo_bc, demo_outcome = partial_supervision_losses(
                        self.model, d_obs, d_actions, d_mask, d_outcome, d_rtg, d_weights
                    )
                    weighted_demo = demo_bc_coef_now * (
                        demo_bc + self.cfg.demo_outcome_coef * demo_outcome
                    )
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(weighted_demo).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    weighted_demo.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                demo_loss_sum += float(demo_bc.detach())
                demo_outcome_sum += float(demo_outcome.detach())
            metrics["demo_bc_loss"] = demo_loss_sum / max(demo_batches, 1)
            metrics["demo_outcome_loss"] = demo_outcome_sum / max(demo_batches, 1)
            metrics["demo_used"] = 1.0

        if minibatches:
            for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "motion_loss"):
                metrics[key] /= minibatches
        metrics["lr"] = lr_now
        metrics["entropy_coef"] = entropy_coef_now
        metrics["demo_bc_coef"] = demo_bc_coef_now

        self.update_index += 1
        return metrics

    def sil_update(self, buffer, sil_cfg) -> dict[str, float] | None:
        """Self-imitation learning step (see sil.py for the algorithm).

        Runs after the regular PPO update, sampling from the agent's own
        past high-return trajectory chunks. Skipped (returns None) until the
        buffer has at least sil_cfg.min_buffer_chunks stored - early in
        training there's nothing worth imitating yet.

        Processes sampled chunks one at a time rather than batched: chunks
        vary in length (episodes rarely divide evenly into sequence_length),
        so batching requires padding, and CPU benchmarking a padded/masked
        version showed it was consistently *slower* than this simple loop -
        even in the near-zero-padding-waste case - not faster. If SIL's
        overhead needs trimming, sil.updates_per_iteration / sil.batch_size
        in the config are the levers: fewer or smaller passes, same
        per-pass cost, no architecture-dependent gamble.
        """
        if len(buffer) < sil_cfg.min_buffer_chunks:
            return None

        self.model.train()
        rng = np.random.default_rng(self.update_index + 91)
        metrics = {"sil_loss": 0.0, "sil_policy_loss": 0.0, "sil_value_loss": 0.0, "sil_used_frac": 0.0}
        steps_done = 0

        for _ in range(sil_cfg.updates_per_iteration):
            chunks = buffer.sample(sil_cfg.batch_size, rng)
            if not chunks:
                break

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                total_policy_loss = torch.zeros((), device=self.device)
                total_value_loss = torch.zeros((), device=self.device)
                total_used = 0.0
                total_count = 0

                for chunk in chunks:
                    obs = chunk.obs.to(self.device, non_blocking=True).unsqueeze(1).float() / 255.0
                    actions = chunk.actions.to(self.device).unsqueeze(1)
                    returns = chunk.returns.to(self.device).unsqueeze(1)
                    h0 = chunk.h0.to(self.device)

                    dist, values, _ = self.model.forward_sequence(obs, h0)
                    logp = dist.log_prob(actions)

                    advantage = torch.clamp(returns - values.detach(), min=0.0)
                    total_used += float((advantage > 0).float().sum())
                    total_count += advantage.numel()

                    total_policy_loss = total_policy_loss - (logp * advantage).sum()
                    total_value_loss = total_value_loss + 0.5 * torch.clamp(returns - values, min=0.0).pow(2).sum()

                denom = max(total_count, 1)
                loss = (
                    sil_cfg.policy_coef * total_policy_loss / denom
                    + sil_cfg.value_coef * total_value_loss / denom
                )

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

            metrics["sil_loss"] += float(loss.detach())
            metrics["sil_policy_loss"] += float(total_policy_loss.detach()) / denom
            metrics["sil_value_loss"] += float(total_value_loss.detach()) / denom
            metrics["sil_used_frac"] += total_used / denom
            steps_done += 1

        if steps_done:
            for key in metrics:
                metrics[key] /= steps_done
        metrics["sil_buffer_size"] = float(len(buffer))
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
        m_state = payload.get("model") or payload.get("model_state_dict")
        if m_state is not None:
            self.model.load_state_dict(m_state, strict=False)
        if "optimizer" in payload and payload["optimizer"] is not None:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass
        if self.scaler.is_enabled() and payload.get("scaler"):
            try:
                self.scaler.load_state_dict(payload["scaler"])
            except Exception:
                pass
        self.update_index = int(payload.get("update_index", 0))
        self.env_steps = int(payload.get("env_steps", 0))
        return payload.get("extra", {})
