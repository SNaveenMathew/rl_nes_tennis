"""Learned hierarchical policy for NES Tennis.

The high-level manager selects a skill and a temporally extended duration.  Each
skill owns separate actor/critic parameters, while a visual encoder is shared to
keep the model footprint manageable.  No court-position or shot-selection rules
are embedded in inference: only learned distributions route control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .model import VisualEncoder

DEFAULT_SKILLS = (
    "serve", "receive", "baseline", "approach_net", "recover", "shot", "wait"
)
DEFAULT_DURATIONS = (1, 2, 4, 8)


@dataclass
class HierarchicalAction:
    actions: torch.Tensor
    skills: torch.Tensor
    duration_indices: torch.Tensor
    manager_log_probs: torch.Tensor
    expert_log_probs: torch.Tensor
    manager_values: torch.Tensor
    expert_values: torch.Tensor


class HierarchicalActorCritic(nn.Module):
    """Shared perception, learned manager, and independent skill actor-critics.

    Observations follow the repository convention ``[B, 4, H, W]``. The stacked
    frames and explicit frame-difference channels in ``VisualEncoder`` provide
    short-horizon temporal context. Durations mean environment decisions, not
    raw NES frames; the Mesen bridge's frame-skip remains configurable.
    """

    def __init__(
        self,
        observation_channels: int = 4,
        action_count: int = 27,
        channels: list[int] | None = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        skills: list[str] | tuple[str, ...] = DEFAULT_SKILLS,
        durations: list[int] | tuple[int, ...] = DEFAULT_DURATIONS,
        input_size: tuple[int, int] = (84, 84),
    ):
        super().__init__()
        if action_count != 27:
            raise ValueError("NES Tennis hierarchical policy requires the 27-action controller space")
        if len(skills) < 2 or len(set(skills)) != len(skills):
            raise ValueError("skills must contain at least two unique names")
        if not durations or any(int(d) <= 0 for d in durations):
            raise ValueError("durations must contain positive integers")
        if len(set(int(d) for d in durations)) != len(durations):
            raise ValueError("durations must be unique")
        self.skills = tuple(str(s) for s in skills)
        self.durations = tuple(int(d) for d in durations)
        self.skill_to_id = {name: i for i, name in enumerate(self.skills)}
        self.encoder = VisualEncoder(
            observation_channels, channels or [16, 32, 32], embedding_dim, input_size
        )
        self.manager = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.manager_skill_head = nn.Linear(hidden_dim, len(self.skills))
        self.manager_duration_head = nn.Linear(hidden_dim, len(self.durations))
        self.manager_value_head = nn.Linear(hidden_dim, 1)
        # These heads are trained only from explicitly supplied annotations.
        # They are diagnostic/perceptual features, not hand-crafted state rules.
        self.ball_position_head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.player_position_head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.event_head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 4))
        self.experts = nn.ModuleDict({
            name: nn.ModuleDict({
                "trunk": nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                ),
                "action": nn.Linear(hidden_dim, action_count),
                "value": nn.Linear(hidden_dim, 1),
            }) for name in self.skills
        })
        self._init_heads()

    def _init_heads(self) -> None:
        for head in (self.manager_skill_head, self.manager_duration_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.manager_value_head.weight, gain=1.0)
        nn.init.zeros_(self.manager_value_head.bias)
        for expert in self.experts.values():
            nn.init.orthogonal_(expert["action"].weight, gain=0.01)
            nn.init.zeros_(expert["action"].bias)
            nn.init.orthogonal_(expert["value"].weight, gain=1.0)
            nn.init.zeros_(expert["value"].bias)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate manager and all experts for a batch of observations."""
        if observations.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] observations, got {tuple(observations.shape)}")
        features = self.encoder(observations)
        manager_features = self.manager(features)
        logits = []
        values = []
        for skill in self.skills:
            expert_features = self.experts[skill]["trunk"](features)
            logits.append(self.experts[skill]["action"](expert_features))
            values.append(self.experts[skill]["value"](expert_features).squeeze(-1))
        return {
            "features": features,
            "manager_skill_logits": self.manager_skill_head(manager_features),
            "manager_duration_logits": self.manager_duration_head(manager_features),
            "manager_values": self.manager_value_head(manager_features).squeeze(-1),
            "ball_xy": torch.sigmoid(self.ball_position_head(features)),
            "player_xy": torch.sigmoid(self.player_position_head(features)),
            "event_logits": self.event_head(features),
            "expert_logits": torch.stack(logits, dim=1),  # [B, num_skills, 27]
            "expert_values": torch.stack(values, dim=1),  # [B, num_skills]
        }

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        active_skills: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> HierarchicalAction:
        """Sample a manager decision and one expert action per environment.

        ``active_skills=None`` asks the manager for a fresh skill. Passing skill
        IDs lets a caller keep an option active for its learned duration. Duration
        is still sampled only from the manager's distribution.
        """
        output = self.forward(observations)
        skill_dist = Categorical(logits=output["manager_skill_logits"])
        duration_dist = Categorical(logits=output["manager_duration_logits"])
        if active_skills is None:
            if deterministic:
                skills = torch.argmax(skill_dist.logits, dim=-1)
                duration_ids = torch.argmax(duration_dist.logits, dim=-1)
            else:
                skills = skill_dist.sample()
                duration_ids = duration_dist.sample()
            manager_log_probs = skill_dist.log_prob(skills) + duration_dist.log_prob(duration_ids)
        else:
            skills = active_skills.long()
            if skills.shape != (observations.shape[0],):
                raise ValueError("active_skills must have shape [B]")
            # When the caller keeps an option active, its manager log-probability
            # is not a new policy decision and is zeroed for clarity.
            duration_ids = torch.zeros_like(skills)
            manager_log_probs = torch.zeros_like(skills, dtype=torch.float32)

        rows = torch.arange(observations.shape[0], device=observations.device)
        selected_logits = output["expert_logits"][rows, skills]
        expert_dist = Categorical(logits=selected_logits)
        actions = torch.argmax(selected_logits, dim=-1) if deterministic else expert_dist.sample()
        expert_log_probs = expert_dist.log_prob(actions)
        expert_values = output["expert_values"][rows, skills]
        return HierarchicalAction(
            actions=actions,
            skills=skills,
            duration_indices=duration_ids,
            manager_log_probs=manager_log_probs,
            expert_log_probs=expert_log_probs,
            manager_values=output["manager_values"],
            expert_values=expert_values,
        )

    def load_compatible_state_dict(self, state_dict: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Load a hierarchy checkpoint and surface missing/unexpected parameter names."""
        result = self.load_state_dict(state_dict, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)


class HierarchicalPPOTrainer:
    """PPO updates for manager options and the active specialized expert.

    `batch` is produced by `hierarchical_train.collect_rollout`. Low-level expert
    samples and semi-Markov manager decisions have independent advantages and
    returns, but share one optimizer so the shared visual representation receives
    both objectives.
    """

    def __init__(self, model: HierarchicalActorCritic, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg.learning_rate), eps=1e-5, weight_decay=0.0
        )
        self.update_index = 0
        self.env_steps = 0

    @staticmethod
    def _categorical_loss(
        dist: Categorical,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = dist.log_prob(actions)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        return -torch.minimum(unclipped, clipped).mean(), dist.entropy().mean(), log_probs

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        manager_obs = batch["manager_obs"].to(self.device)
        manager_skill = batch["manager_skill"].to(self.device).long()
        manager_duration = batch["manager_duration"].to(self.device).long()
        manager_old_logp = batch["manager_old_logp"].to(self.device).float()
        manager_adv = batch["manager_advantages"].to(self.device).float()
        manager_returns = batch["manager_returns"].to(self.device).float()
        manager_targets = batch.get("manager_target_skill")
        if manager_targets is None:
            manager_targets = torch.full_like(manager_skill, -1)
        else:
            manager_targets = manager_targets.to(self.device).long()

        expert_obs = batch["expert_obs"].to(self.device)
        expert_skill = batch["expert_skill"].to(self.device).long()
        expert_action = batch["expert_action"].to(self.device).long()
        expert_old_logp = batch["expert_old_logp"].to(self.device).float()
        expert_adv = batch["expert_advantages"].to(self.device).float()
        expert_returns = batch["expert_returns"].to(self.device).float()

        demo_obs = batch.get("demo_obs")
        if demo_obs is not None:
            demo_obs = demo_obs.to(self.device)
            demo_skill = batch["demo_skill"].to(self.device).long()
            demo_action = batch["demo_action"].to(self.device).long()
            demo_manager_obs = batch["demo_manager_obs"].to(self.device)
            demo_manager_skill = batch["demo_manager_skill"].to(self.device).long()
            demo_manager_duration = batch["demo_manager_duration"].to(self.device).long()
            demo_order = torch.randperm(demo_obs.shape[0], device=self.device)
            demo_manager_order = torch.randperm(demo_manager_obs.shape[0], device=self.device)
        else:
            demo_skill = demo_action = demo_manager_obs = demo_manager_skill = demo_manager_duration = None
            demo_order = demo_manager_order = None
        state_obs = batch.get("state_obs")
        if state_obs is not None:
            state_obs = state_obs.to(self.device)
            state_ball = batch["state_ball_xy"].to(self.device).float()
            state_player = batch["state_player_xy"].to(self.device).float()
            state_events = batch["state_events"].to(self.device).float()
            state_order = torch.randperm(state_obs.shape[0], device=self.device)
        else:
            state_ball = state_player = state_events = state_order = None

        if manager_obs.shape[0] == 0 or expert_obs.shape[0] == 0:
            raise ValueError("Hierarchical update requires manager and expert samples")
        batch_size = max(1, int(self.cfg.batch_size))
        epochs = max(1, int(self.cfg.epochs))
        metrics = {"manager_policy_loss": 0.0, "expert_policy_loss": 0.0,
                   "manager_value_loss": 0.0, "expert_value_loss": 0.0,
                   "manager_entropy": 0.0, "expert_entropy": 0.0,
                   "manager_supervision_loss": 0.0, "demo_bc_loss": 0.0,
                   "perception_loss": 0.0, "loss": 0.0}
        updates = 0
        n_m, n_e = manager_obs.shape[0], expert_obs.shape[0]
        for _ in range(epochs):
            # The dataset sizes differ because manager samples occur at option
            # boundaries while expert samples occur at every environment action.
            m_order = torch.randperm(n_m, device=self.device)
            e_order = torch.randperm(n_e, device=self.device)
            n_batches = max((n_m + batch_size - 1) // batch_size,
                            (n_e + batch_size - 1) // batch_size)
            for k in range(n_batches):
                mi = m_order[(k * batch_size) % n_m:min((k + 1) * batch_size, n_m)]
                ei = e_order[(k * batch_size) % n_e:min((k + 1) * batch_size, n_e)]
                if mi.numel() == 0 or ei.numel() == 0:
                    continue
                mo = self.model(manager_obs[mi])
                m_skill_dist = Categorical(logits=mo["manager_skill_logits"])
                m_dur_dist = Categorical(logits=mo["manager_duration_logits"])
                m_logp = m_skill_dist.log_prob(manager_skill[mi]) + m_dur_dist.log_prob(manager_duration[mi])
                m_adv = manager_adv[mi]
                # Use the joint manager likelihood (skill and duration).
                m_ratio = torch.exp(m_logp - manager_old_logp[mi])
                m_norm_adv = (m_adv - m_adv.mean()) / (m_adv.std(unbiased=False) + 1e-8)
                m_unclipped = m_ratio * m_norm_adv
                m_clipped = torch.clamp(m_ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * m_norm_adv
                manager_policy_loss = -torch.minimum(m_unclipped, m_clipped).mean()
                manager_value_loss = 0.5 * (mo["manager_values"] - manager_returns[mi]).pow(2).mean()
                manager_entropy = m_skill_dist.entropy().mean() + m_dur_dist.entropy().mean()
                valid_targets = manager_targets[mi] >= 0
                if valid_targets.any():
                    manager_supervision_loss = nn.functional.cross_entropy(
                        mo["manager_skill_logits"][valid_targets], manager_targets[mi][valid_targets]
                    )
                else:
                    manager_supervision_loss = torch.zeros((), device=self.device)

                eo = self.model(expert_obs[ei])
                row = torch.arange(ei.numel(), device=self.device)
                e_logits = eo["expert_logits"][row, expert_skill[ei]]
                e_dist = Categorical(logits=e_logits)
                expert_policy_loss, expert_entropy, _ = self._categorical_loss(
                    e_dist, expert_action[ei], expert_old_logp[ei], expert_adv[ei], self.cfg.clip_range
                )
                e_values = eo["expert_values"][row, expert_skill[ei]]
                expert_value_loss = 0.5 * (e_values - expert_returns[ei]).pow(2).mean()
                demo_bc_loss = torch.zeros((), device=self.device)
                if demo_obs is not None and demo_obs.shape[0] > 0:
                    d_start = (k * batch_size) % demo_obs.shape[0]
                    d_ids = demo_order[d_start:min(d_start + batch_size, demo_obs.shape[0])]
                    if d_ids.numel() == 0:
                        d_ids = demo_order[:min(batch_size, demo_obs.shape[0])]
                    d_out = self.model(demo_obs[d_ids])
                    d_rows = torch.arange(d_ids.numel(), device=self.device)
                    d_logits = d_out["expert_logits"][d_rows, demo_skill[d_ids]]
                    demo_action_loss = nn.functional.cross_entropy(d_logits, demo_action[d_ids])

                    dm_start = (k * batch_size) % demo_manager_obs.shape[0]
                    dm_ids = demo_manager_order[dm_start:min(dm_start + batch_size, demo_manager_obs.shape[0])]
                    if dm_ids.numel() == 0:
                        dm_ids = demo_manager_order[:min(batch_size, demo_manager_obs.shape[0])]
                    dm_out = self.model(demo_manager_obs[dm_ids])
                    dm_skill_loss = nn.functional.cross_entropy(
                        dm_out["manager_skill_logits"], demo_manager_skill[dm_ids]
                    )
                    dm_duration_loss = nn.functional.cross_entropy(
                        dm_out["manager_duration_logits"], demo_manager_duration[dm_ids]
                    )
                    demo_bc_loss = demo_action_loss + self.cfg.manager_supervision_coef * (dm_skill_loss + dm_duration_loss)

                perception_loss = torch.zeros((), device=self.device)
                if state_obs is not None and state_obs.shape[0] > 0:
                    st_start = (k * batch_size) % state_obs.shape[0]
                    st_ids = state_order[st_start:min(st_start + batch_size, state_obs.shape[0])]
                    if st_ids.numel() == 0:
                        st_ids = state_order[:min(batch_size, state_obs.shape[0])]
                    st_out = self.model(state_obs[st_ids])
                    ball_y = state_ball[st_ids]
                    player_y = state_player[st_ids]
                    if torch.isfinite(ball_y).all(dim=-1).any():
                        valid = torch.isfinite(ball_y).all(dim=-1)
                        ball_loss = nn.functional.smooth_l1_loss(st_out["ball_xy"][valid], ball_y[valid])
                        perception_loss = perception_loss + ball_loss
                    if torch.isfinite(player_y).all(dim=-1).any():
                        valid = torch.isfinite(player_y).all(dim=-1)
                        player_loss = nn.functional.smooth_l1_loss(st_out["player_xy"][valid], player_y[valid])
                        perception_loss = perception_loss + player_loss
                    event_y = state_events[st_ids]
                    valid_events = event_y >= 0
                    if valid_events.any():
                        event_loss = nn.functional.binary_cross_entropy_with_logits(
                            st_out["event_logits"][valid_events], event_y[valid_events]
                        )
                        perception_loss = perception_loss + event_loss

                loss = (
                    manager_policy_loss + expert_policy_loss
                    + self.cfg.value_coef * (manager_value_loss + expert_value_loss)
                    - self.cfg.entropy_coef * (manager_entropy + expert_entropy)
                    + self.cfg.manager_supervision_coef * manager_supervision_loss
                    + float(getattr(self.cfg, "demo_bc_coef", 0.5)) * demo_bc_loss
                    + float(getattr(self.cfg, "perception_coef", 0.2)) * perception_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                for key, value in (
                    ("manager_policy_loss", manager_policy_loss),
                    ("expert_policy_loss", expert_policy_loss),
                    ("manager_value_loss", manager_value_loss),
                    ("expert_value_loss", expert_value_loss),
                    ("manager_entropy", manager_entropy),
                    ("expert_entropy", expert_entropy),
                    ("manager_supervision_loss", manager_supervision_loss),
                    ("demo_bc_loss", demo_bc_loss),
                    ("perception_loss", perception_loss),
                    ("loss", loss),
                ):
                    metrics[key] += float(value.detach())
                updates += 1
        if updates:
            metrics = {k: v / updates for k, v in metrics.items()}
        metrics["optimizer_steps"] = float(updates)
        metrics["manager_samples"] = float(n_m)
        metrics["expert_samples"] = float(n_e)
        self.update_index += 1
        return metrics

    def save(self, path: str, extra: dict[str, Any] | None = None) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "tennis_rl_hierarchical_v1",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_index": self.update_index,
            "env_steps": self.env_steps,
            "skills": list(self.model.skills),
            "durations": list(self.model.durations),
            "extra": extra or {},
        }, path)

    def load(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format") != "tennis_rl_hierarchical_v1":
            raise ValueError("Checkpoint is not a hierarchical NES Tennis checkpoint")
        if tuple(payload.get("skills", self.model.skills)) != self.model.skills:
            raise ValueError("Checkpoint skill list does not match configured skills")
        if tuple(payload.get("durations", self.model.durations)) != self.model.durations:
            raise ValueError("Checkpoint duration list does not match configured durations")
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        self.update_index = int(payload.get("update_index", 0))
        self.env_steps = int(payload.get("env_steps", 0))
        return payload.get("extra", {})


class HierarchicalController:
    """Stateful inference router that executes learned options for learned durations."""

    def __init__(self, model: HierarchicalActorCritic, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)
        self.active_skills: torch.Tensor | None = None
        self.remaining: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.active_skills is None or self.remaining is None or mask is None:
            self.active_skills = None
            self.remaining = None
            return
        mask = mask.to(device=self.active_skills.device, dtype=torch.bool)
        self.active_skills[mask] = -1
        self.remaining[mask] = 0

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.model(observations)
        n = observations.shape[0]
        device = observations.device
        if self.active_skills is None or self.remaining is None or self.active_skills.shape[0] != n:
            self.active_skills = torch.full((n,), -1, dtype=torch.long, device=device)
            self.remaining = torch.zeros((n,), dtype=torch.long, device=device)
        boundary = (self.active_skills < 0) | (self.remaining <= 0)
        idx = torch.nonzero(boundary, as_tuple=False).squeeze(-1)
        if idx.numel():
            skill_dist = Categorical(logits=output["manager_skill_logits"][idx])
            duration_dist = Categorical(logits=output["manager_duration_logits"][idx])
            if self.deterministic:
                new_skills = torch.argmax(skill_dist.logits, dim=-1)
                duration_ids = torch.argmax(duration_dist.logits, dim=-1)
            else:
                new_skills = skill_dist.sample()
                duration_ids = duration_dist.sample()
            duration_values = torch.as_tensor(self.model.durations, device=device, dtype=torch.long)
            self.active_skills[idx] = new_skills
            self.remaining[idx] = duration_values[duration_ids]
        rows = torch.arange(n, device=device)
        logits = output["expert_logits"][rows, self.active_skills]
        action_dist = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if self.deterministic else action_dist.sample()
        self.remaining -= 1
        return actions
