"""Discriminative hierarchy for NES Tennis.

One recurrent belief encoder feeds a learned broad-phase gate. A phase-specific
learned gate selects a leaf task, and a task-specific actor chooses movement,
conditional shot/button input, and a button pulse duration. The manager's
categorical distribution is the mathematically correct joint distribution
P(phase) * P(task | phase), not a hand-coded inference router.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .hierarchy_schema import PHASES, DEFAULT_LEAF_TASKS, EVENT_NAMES, phase_task_indices
from .model import VisualEncoder

DEFAULT_SKILLS = DEFAULT_LEAF_TASKS
DEFAULT_DURATIONS = (1, 2, 4, 8)
DEFAULT_BUTTON_PULSE_DURATIONS = (1, 2, 4)


@dataclass
class HierarchicalAction:
    actions: torch.Tensor
    skills: torch.Tensor
    duration_indices: torch.Tensor
    manager_log_probs: torch.Tensor
    expert_log_probs: torch.Tensor
    manager_values: torch.Tensor
    expert_values: torch.Tensor
    pulse_duration_indices: torch.Tensor
    pulse_frames: torch.Tensor
    expert_pulse_log_probs: torch.Tensor
    belief_hidden: torch.Tensor


class HierarchicalActorCritic(nn.Module):
    """Recurrent state encoder + phase gate + conditional task gates + leaf actors.

    Observations are ``[B, 4, H, W]`` uint8 or float tensors. ``belief_hidden`` is
    a persistent GRU state shaped ``[layers, B, hidden]``; callers must reset the
    corresponding batch entries only at true episode boundaries. All routing is
    learned. Emulator role/score data is used only as optional training labels.
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
        button_pulse_durations: list[int] | tuple[int, ...] = DEFAULT_BUTTON_PULSE_DURATIONS,
        gru_hidden_dim: int | None = None,
        gru_layers: int = 1,
    ):
        super().__init__()
        if action_count != 27:
            raise ValueError("NES Tennis hierarchy requires the 27-action movement/button space")
        if len(skills) < 2 or len(set(skills)) != len(skills):
            raise ValueError("skills must contain at least two unique names")
        if not durations or any(int(d) <= 0 for d in durations) or len(set(durations)) != len(durations):
            raise ValueError("durations must be unique positive decision counts")
        if not button_pulse_durations or any(int(d) <= 0 for d in button_pulse_durations):
            raise ValueError("button_pulse_durations must be positive NES-frame counts")
        if len(set(button_pulse_durations)) != len(button_pulse_durations):
            raise ValueError("button_pulse_durations must be unique")
        self.skills = tuple(str(s) for s in skills)
        self.durations = tuple(int(d) for d in durations)
        self.button_pulse_durations = tuple(int(d) for d in button_pulse_durations)
        self.skill_to_id = {name: i for i, name in enumerate(self.skills)}
        self.phase_to_id = {name: i for i, name in enumerate(PHASES)}
        self.phase_task_ids = phase_task_indices(self.skills)
        self.skill_phase = tuple(
            next(phase for phase, ids in self.phase_task_ids.items() if i in ids)
            for i in range(len(self.skills))
        )
        # Disabled branches are masked in the root gate, so tiny custom test
        # models may omit a branch while production config uses the full tree.
        self.active_phases = tuple(p for p in PHASES if self.phase_task_ids[p])
        self.encoder = VisualEncoder(
            observation_channels, channels or [16, 32, 32], embedding_dim, input_size
        )
        self.belief_hidden_dim = int(gru_hidden_dim or hidden_dim)
        self.gru_layers = int(gru_layers)
        if self.belief_hidden_dim < 8 or self.gru_layers < 1:
            raise ValueError("GRU hidden size must be >=8 and layers >=1")
        self.belief_gru = nn.GRU(
            embedding_dim, self.belief_hidden_dim, num_layers=self.gru_layers, batch_first=True
        )
        belief_dim = self.belief_hidden_dim

        # Level 1: broad phase gate.
        self.manager = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.phase_head = nn.Linear(hidden_dim, len(PHASES))
        self.manager_duration_head = nn.Linear(hidden_dim, len(self.durations))
        self.manager_value_head = nn.Linear(hidden_dim, 1)

        # Level 2: independent, phase-conditional task gates. Leaf task labels
        # are the outputs of a gate, not hard-coded conditions on screen state.
        self.branch_gates = nn.ModuleDict({
            phase: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, len(self.phase_task_ids[phase])),
            )
            for phase in self.active_phases
        })

        # Learned perception heads; train only from validated annotations.
        self.ball_position_head = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.player_position_head = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.ball_velocity_head = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.time_to_contact_head = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.time_to_contact_bin_head = nn.Linear(belief_dim, 3)
        self.event_head = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, len(EVENT_NAMES)))

        # Level 3: independently parameterized leaf specialists. Movement is
        # selected first; the button logits are conditional on the selected move.
        self.experts = nn.ModuleDict({
            name: nn.ModuleDict({
                "trunk": nn.Sequential(
                    nn.Linear(belief_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                ),
                "movement": nn.Linear(hidden_dim, 9),
                "button_given_movement": nn.Linear(hidden_dim, 9 * 3),
                "pulse_duration": nn.Linear(hidden_dim, len(self.button_pulse_durations)),
                "value": nn.Linear(hidden_dim, 1),
            }) for name in self.skills
        })
        self._init_heads()

    def initial_belief(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(self.gru_layers, int(batch_size), self.belief_hidden_dim, device=device)

    def _init_heads(self) -> None:
        for head in (self.phase_head, self.manager_duration_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.manager_value_head.weight, gain=1.0)
        nn.init.zeros_(self.manager_value_head.bias)
        for gate in self.branch_gates.values():
            last = gate[-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)
        for expert in self.experts.values():
            for key in ("movement", "button_given_movement", "pulse_duration"):
                nn.init.orthogonal_(expert[key].weight, gain=0.01)
                nn.init.zeros_(expert[key].bias)
            nn.init.orthogonal_(expert["value"].weight, gain=1.0)
            nn.init.zeros_(expert["value"].bias)

    def forward(
        self, observations: torch.Tensor, belief_hidden: torch.Tensor | None = None
    ) -> dict[str, Any]:
        if observations.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] observations, got {tuple(observations.shape)}")
        enc = self.encoder(observations)
        if belief_hidden is None:
            belief_hidden = self.initial_belief(observations.shape[0], observations.device)
        recurrent, belief_hidden_out = self.belief_gru(enc.unsqueeze(1), belief_hidden)
        belief = recurrent[:, 0]
        manager_features = self.manager(belief)
        raw_phase_logits = self.phase_head(manager_features)
        phase_logits = raw_phase_logits.clone()
        for phase in PHASES:
            if not self.phase_task_ids[phase]:
                phase_logits[:, self.phase_to_id[phase]] = -1e9
        phase_log_probs = torch.log_softmax(phase_logits, dim=-1)
        branch_logits: dict[str, torch.Tensor] = {}
        joint_skill_log_probs = torch.full(
            (observations.shape[0], len(self.skills)), -1e9,
            dtype=belief.dtype, device=belief.device,
        )
        for phase in self.active_phases:
            logits = self.branch_gates[phase](manager_features)
            branch_logits[phase] = logits
            ids = self.phase_task_ids[phase]
            local_logp = torch.log_softmax(logits, dim=-1)
            for local_idx, global_idx in enumerate(ids):
                joint_skill_log_probs[:, global_idx] = (
                    phase_log_probs[:, self.phase_to_id[phase]] + local_logp[:, local_idx]
                )

        expert_logits: list[torch.Tensor] = []
        expert_pulse_logits: list[torch.Tensor] = []
        expert_values: list[torch.Tensor] = []
        for name in self.skills:
            expert = self.experts[name]
            z = expert["trunk"](belief)
            movement_logp = torch.log_softmax(expert["movement"](z), dim=-1)
            conditional_button_logp = torch.log_softmax(
                expert["button_given_movement"](z).view(-1, 9, 3), dim=-1
            )
            # Joint normalized distribution P(move) P(button | move).
            joint_logp = (movement_logp.unsqueeze(-1) + conditional_button_logp).reshape(-1, 27)
            expert_logits.append(joint_logp)
            expert_pulse_logits.append(expert["pulse_duration"](z))
            expert_values.append(expert["value"](z).squeeze(-1))

        return {
            "belief": belief,
            "belief_hidden_out": belief_hidden_out,
            "features": belief,
            "phase_logits": phase_logits,
            "branch_logits": branch_logits,
            # Log-probability of a leaf = log P(phase) + log P(leaf | phase).
            # Softmaxing these normalized joint log probs reproduces the same
            # categorical distribution and keeps the old trainer API compatible.
            "manager_skill_logits": joint_skill_log_probs,
            "manager_duration_logits": self.manager_duration_head(manager_features),
            "manager_values": self.manager_value_head(manager_features).squeeze(-1),
            "ball_xy": torch.sigmoid(self.ball_position_head(belief)),
            "player_xy": torch.sigmoid(self.player_position_head(belief)),
            "ball_velocity_xy": self.ball_velocity_head(belief),
            "time_to_contact": torch.nn.functional.softplus(self.time_to_contact_head(belief)).squeeze(-1),
            "time_to_contact_logits": self.time_to_contact_bin_head(belief),
            "event_logits": self.event_head(belief),
            "expert_logits": torch.stack(expert_logits, dim=1),
            "expert_pulse_logits": torch.stack(expert_pulse_logits, dim=1),
            "expert_values": torch.stack(expert_values, dim=1),
        }

    def select_leaf_skills(
        self, output: dict[str, Any], *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Traverse root phase -> conditional branch -> leaf skill.

        A flat argmax over the joint leaf distribution is not a valid greedy
        traversal: singleton branches get a larger per-leaf prior than wide
        branches even when the root gate prefers the latter phase. Sampling
        phases then children preserves the exact joint policy distribution;
        deterministic inference uses top-down greedy routing.
        """
        phase_logits = output["phase_logits"]
        phase_dist = Categorical(logits=phase_logits)
        phase_ids = phase_logits.argmax(dim=-1) if deterministic else phase_dist.sample()
        chosen = torch.zeros_like(phase_ids)
        for phase in self.active_phases:
            mask = phase_ids == self.phase_to_id[phase]
            if not bool(mask.any()):
                continue
            branch_logits = output["branch_logits"][phase][mask]
            branch_dist = Categorical(logits=branch_logits)
            local = branch_logits.argmax(dim=-1) if deterministic else branch_dist.sample()
            global_ids = torch.as_tensor(self.phase_task_ids[phase], device=phase_ids.device, dtype=torch.long)
            chosen[mask] = global_ids[local]
        return chosen, phase_ids

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        active_skills: torch.Tensor | None = None,
        deterministic: bool = False,
        belief_hidden: torch.Tensor | None = None,
    ) -> HierarchicalAction:
        output = self.forward(observations, belief_hidden)
        skill_dist = Categorical(logits=output["manager_skill_logits"])
        duration_dist = Categorical(logits=output["manager_duration_logits"])
        if active_skills is None:
            skills, _phase_ids = self.select_leaf_skills(output, deterministic=deterministic)
            if deterministic:
                duration_ids = torch.argmax(duration_dist.logits, dim=-1)
            else:
                duration_ids = duration_dist.sample()
            manager_log_probs = skill_dist.log_prob(skills) + duration_dist.log_prob(duration_ids)
        else:
            skills = active_skills.long()
            if skills.shape != (observations.shape[0],):
                raise ValueError("active_skills must have shape [B]")
            duration_ids = torch.zeros_like(skills)
            manager_log_probs = torch.zeros_like(skills, dtype=torch.float32)
        rows = torch.arange(observations.shape[0], device=observations.device)
        selected_logits = output["expert_logits"][rows, skills]
        action_dist = Categorical(logits=selected_logits)
        actions = torch.argmax(selected_logits, dim=-1) if deterministic else action_dist.sample()
        expert_log_probs = action_dist.log_prob(actions)
        pulse_logits = output["expert_pulse_logits"][rows, skills]
        pulse_dist = Categorical(logits=pulse_logits)
        pulse_ids = torch.argmax(pulse_logits, dim=-1) if deterministic else pulse_dist.sample()
        pulse_frames = torch.as_tensor(self.button_pulse_durations, device=observations.device)[pulse_ids]
        pulse_logp = pulse_dist.log_prob(pulse_ids) * (actions.remainder(3) != 0).float()
        expert_values = output["expert_values"][rows, skills]
        return HierarchicalAction(
            actions=actions, skills=skills, duration_indices=duration_ids,
            manager_log_probs=manager_log_probs, expert_log_probs=expert_log_probs,
            manager_values=output["manager_values"], expert_values=expert_values,
            pulse_duration_indices=pulse_ids, pulse_frames=pulse_frames,
            expert_pulse_log_probs=pulse_logp, belief_hidden=output["belief_hidden_out"],
        )

    def load_compatible_state_dict(self, state_dict: dict[str, Any]) -> tuple[list[str], list[str]]:
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
        """One robust PPO update with optional verified task/state supervision.

        Rollout rows retain the pre-observation GRU state. PPO therefore trains
        the same one-step recurrent transition used during collection, rather
        than accidentally evaluating every observation from a zero hidden state.
        """
        def dev(name: str, default=None):
            value = batch.get(name, default)
            return None if value is None else value.to(self.device)

        manager_obs = dev("manager_obs")
        manager_skill = dev("manager_skill").long()
        manager_duration = dev("manager_duration").long()
        manager_old_logp = dev("manager_old_logp").float()
        manager_adv = dev("manager_advantages").float()
        manager_returns = dev("manager_returns").float()
        manager_targets = dev("manager_target_skill", torch.full_like(manager_skill, -1)).long()
        manager_phase_targets = dev("manager_target_phase", torch.full_like(manager_skill, -1)).long()
        manager_hidden = dev("manager_hidden")

        expert_obs = dev("expert_obs")
        expert_skill = dev("expert_skill").long()
        expert_action = dev("expert_action").long()
        expert_old_logp = dev("expert_old_logp").float()
        expert_adv = dev("expert_advantages").float()
        expert_returns = dev("expert_returns").float()
        expert_hidden = dev("expert_hidden")
        expert_pulse_ids = dev("expert_pulse_duration_ids")
        if expert_pulse_ids is not None:
            expert_pulse_ids = expert_pulse_ids.long()

        demo_obs = dev("demo_obs")
        demo_skill = dev("demo_skill")
        demo_action = dev("demo_action")
        demo_manager_obs = dev("demo_manager_obs")
        demo_manager_skill = dev("demo_manager_skill")
        demo_manager_duration = dev("demo_manager_duration")
        demo_manager_phase = dev("demo_manager_phase")
        demo_pulse_duration_ids = dev("demo_pulse_duration_ids")
        if demo_obs is not None:
            demo_skill = demo_skill.long()
            demo_action = demo_action.long()
            if demo_manager_skill is not None:
                demo_manager_skill = demo_manager_skill.long()
            if demo_manager_duration is not None:
                demo_manager_duration = demo_manager_duration.long()
            demo_order = torch.randperm(demo_obs.shape[0], device=self.device)
            demo_manager_order = (torch.randperm(demo_manager_obs.shape[0], device=self.device)
                                  if demo_manager_obs is not None else None)
        else:
            demo_skill = demo_action = demo_manager_obs = demo_manager_skill = demo_manager_duration = demo_manager_phase = None
            demo_pulse_duration_ids = None
            demo_order = demo_manager_order = None

        state_obs = dev("state_obs")
        if state_obs is not None:
            state_ball = dev("state_ball_xy").float()
            state_player = dev("state_player_xy").float()
            state_velocity = dev("state_ball_velocity_xy")
            state_ttc = dev("state_time_to_contact")
            state_ttc_bin = dev("state_time_to_contact_bin")
            state_events = dev("state_events").float()
            state_order = torch.randperm(state_obs.shape[0], device=self.device)
        else:
            state_ball = state_player = state_velocity = state_ttc = state_ttc_bin = state_events = state_order = None

        if manager_obs.shape[0] == 0 or expert_obs.shape[0] == 0:
            raise ValueError("Hierarchical update requires manager and expert samples")
        batch_size = max(1, int(self.cfg.batch_size))
        epochs = max(1, int(self.cfg.epochs))
        metric_keys = (
            "manager_policy_loss", "expert_policy_loss", "manager_value_loss", "expert_value_loss",
            "manager_entropy", "expert_entropy", "manager_supervision_loss", "phase_supervision_loss",
            "demo_bc_loss", "perception_loss", "loss",
        )
        metrics = {key: 0.0 for key in metric_keys}
        updates = 0
        n_m, n_e = manager_obs.shape[0], expert_obs.shape[0]
        for _ in range(epochs):
            m_order = torch.randperm(n_m, device=self.device)
            e_order = torch.randperm(n_e, device=self.device)
            n_batches = max((n_m + batch_size - 1) // batch_size,
                            (n_e + batch_size - 1) // batch_size)
            for k in range(n_batches):
                mi = m_order[(k * batch_size) % n_m:min((k + 1) * batch_size, n_m)]
                ei = e_order[(k * batch_size) % n_e:min((k + 1) * batch_size, n_e)]
                if mi.numel() == 0 or ei.numel() == 0:
                    continue
                mh = None if manager_hidden is None else manager_hidden[mi].permute(1, 0, 2).contiguous()
                mo = self.model(manager_obs[mi], mh)
                m_skill_dist = Categorical(logits=mo["manager_skill_logits"])
                m_dur_dist = Categorical(logits=mo["manager_duration_logits"])
                m_logp = m_skill_dist.log_prob(manager_skill[mi]) + m_dur_dist.log_prob(manager_duration[mi])
                m_adv = manager_adv[mi]
                m_norm_adv = (m_adv - m_adv.mean()) / (m_adv.std(unbiased=False) + 1e-8)
                m_ratio = torch.exp(m_logp - manager_old_logp[mi])
                m_unclipped = m_ratio * m_norm_adv
                m_clipped = torch.clamp(m_ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * m_norm_adv
                manager_policy_loss = -torch.minimum(m_unclipped, m_clipped).mean()
                manager_value_loss = 0.5 * (mo["manager_values"] - manager_returns[mi]).pow(2).mean()
                manager_entropy = m_skill_dist.entropy().mean() + m_dur_dist.entropy().mean()

                valid_targets = manager_targets[mi] >= 0
                if valid_targets.any():
                    manager_leaf_sup = nn.functional.cross_entropy(
                        mo["manager_skill_logits"][valid_targets], manager_targets[mi][valid_targets]
                    )
                else:
                    manager_leaf_sup = torch.zeros((), device=self.device)
                valid_phase = manager_phase_targets[mi] >= 0
                if valid_phase.any():
                    phase_sup = nn.functional.cross_entropy(
                        mo["phase_logits"][valid_phase], manager_phase_targets[mi][valid_phase]
                    )
                else:
                    phase_sup = torch.zeros((), device=self.device)
                manager_supervision_loss = manager_leaf_sup + phase_sup

                eh = None if expert_hidden is None else expert_hidden[ei].permute(1, 0, 2).contiguous()
                eo = self.model(expert_obs[ei], eh)
                row = torch.arange(ei.numel(), device=self.device)
                e_logits = eo["expert_logits"][row, expert_skill[ei]]
                e_dist = Categorical(logits=e_logits)
                action_logp = e_dist.log_prob(expert_action[ei])
                action_entropy = e_dist.entropy()
                if expert_pulse_ids is not None:
                    pulse_logits = eo["expert_pulse_logits"][row, expert_skill[ei]]
                    pulse_dist = Categorical(logits=pulse_logits)
                    pulse_mask = (expert_action[ei].remainder(3) != 0).float()
                    pulse_logp = pulse_dist.log_prob(expert_pulse_ids[ei]) * pulse_mask
                    current_expert_logp = action_logp + pulse_logp
                    pulse_entropy = (pulse_dist.entropy() * pulse_mask).sum() / pulse_mask.sum().clamp_min(1.0)
                    expert_entropy = action_entropy.mean() + pulse_entropy
                else:
                    current_expert_logp = action_logp
                    expert_entropy = action_entropy.mean()
                adv_norm = (expert_adv[ei] - expert_adv[ei].mean()) / (expert_adv[ei].std(unbiased=False) + 1e-8)
                e_ratio = torch.exp(current_expert_logp - expert_old_logp[ei])
                e_unclipped = e_ratio * adv_norm
                e_clipped = torch.clamp(e_ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * adv_norm
                expert_policy_loss = -torch.minimum(e_unclipped, e_clipped).mean()
                e_values = eo["expert_values"][row, expert_skill[ei]]
                expert_value_loss = 0.5 * (e_values - expert_returns[ei]).pow(2).mean()

                demo_bc_loss = torch.zeros((), device=self.device)
                # Leaf-action supervision and manager supervision are independent:
                # phase-only annotations must still train the gate when no exact
                # leaf labels exist in the demonstration set.
                if demo_obs is not None and demo_obs.shape[0] > 0:
                    d_start = (k * batch_size) % demo_obs.shape[0]
                    d_ids = demo_order[d_start:min(d_start + batch_size, demo_obs.shape[0])]
                    if d_ids.numel() == 0:
                        d_ids = demo_order[:min(batch_size, demo_obs.shape[0])]
                    d_out = self.model(demo_obs[d_ids])
                    valid_demo_leaf = demo_skill[d_ids] >= 0
                    if valid_demo_leaf.any():
                        d_rows = torch.arange(d_ids.numel(), device=self.device)[valid_demo_leaf]
                        d_skill = demo_skill[d_ids][valid_demo_leaf]
                        d_logits = d_out["expert_logits"][d_rows, d_skill]
                        demo_bc_loss = nn.functional.cross_entropy(d_logits, demo_action[d_ids][valid_demo_leaf])
                        if demo_pulse_duration_ids is not None:
                            raw_pulse_ids = demo_pulse_duration_ids[d_ids][valid_demo_leaf]
                            raw_actions = demo_action[d_ids][valid_demo_leaf]
                            valid_pulse = (raw_pulse_ids >= 0) & (raw_actions.remainder(3) != 0)
                            if valid_pulse.any():
                                pulse_logits = d_out["expert_pulse_logits"][d_rows[valid_pulse], d_skill[valid_pulse]]
                                demo_bc_loss = demo_bc_loss + 0.5 * nn.functional.cross_entropy(
                                    pulse_logits, raw_pulse_ids[valid_pulse]
                                )

                if demo_manager_obs is not None and demo_manager_obs.shape[0] > 0:
                    dm_start = (k * batch_size) % demo_manager_obs.shape[0]
                    dm_ids = demo_manager_order[dm_start:min(dm_start + batch_size, demo_manager_obs.shape[0])]
                    if dm_ids.numel() == 0:
                        dm_ids = demo_manager_order[:min(batch_size, demo_manager_obs.shape[0])]
                    dm_out = self.model(demo_manager_obs[dm_ids])
                    dm_skill_loss = torch.zeros((), device=self.device)
                    if demo_manager_skill is not None:
                        dm_leaf_targets = demo_manager_skill[dm_ids].long()
                        dm_leaf_valid = dm_leaf_targets >= 0
                        if dm_leaf_valid.any():
                            dm_skill_loss = nn.functional.cross_entropy(
                                dm_out["manager_skill_logits"][dm_leaf_valid], dm_leaf_targets[dm_leaf_valid]
                            )
                    dm_phase_loss = torch.zeros((), device=self.device)
                    valid_dm_phase = torch.zeros(dm_ids.numel(), dtype=torch.bool, device=self.device)
                    if demo_manager_phase is not None:
                        dm_phase_targets = demo_manager_phase[dm_ids].long()
                        valid_dm_phase = dm_phase_targets >= 0
                        if valid_dm_phase.any():
                            dm_phase_loss = nn.functional.cross_entropy(
                                dm_out["phase_logits"][valid_dm_phase], dm_phase_targets[valid_dm_phase]
                            )
                    dm_duration_loss = torch.zeros((), device=self.device)
                    if demo_manager_duration is not None:
                        dm_duration_targets = demo_manager_duration[dm_ids].long()
                        dm_duration_valid = valid_dm_phase & (dm_duration_targets >= 0)
                        if dm_duration_valid.any():
                            dm_duration_loss = nn.functional.cross_entropy(
                                dm_out["manager_duration_logits"][dm_duration_valid], dm_duration_targets[dm_duration_valid]
                            )
                    demo_bc_loss = demo_bc_loss + float(self.cfg.manager_supervision_coef) * (
                        dm_skill_loss + dm_phase_loss + dm_duration_loss
                    )

                perception_loss = torch.zeros((), device=self.device)
                if state_obs is not None and state_obs.shape[0] > 0:
                    st_start = (k * batch_size) % state_obs.shape[0]
                    st_ids = state_order[st_start:min(st_start + batch_size, state_obs.shape[0])]
                    if st_ids.numel() == 0:
                        st_ids = state_order[:min(batch_size, state_obs.shape[0])]
                    st_out = self.model(state_obs[st_ids])
                    ball_y, player_y = state_ball[st_ids], state_player[st_ids]
                    valid = torch.isfinite(ball_y).all(dim=-1)
                    if valid.any():
                        perception_loss = perception_loss + nn.functional.smooth_l1_loss(st_out["ball_xy"][valid], ball_y[valid])
                    valid = torch.isfinite(player_y).all(dim=-1)
                    if valid.any():
                        perception_loss = perception_loss + nn.functional.smooth_l1_loss(st_out["player_xy"][valid], player_y[valid])
                    if state_velocity is not None:
                        vel_y = state_velocity[st_ids].float()
                        valid = torch.isfinite(vel_y).all(dim=-1)
                        if valid.any():
                            perception_loss = perception_loss + nn.functional.smooth_l1_loss(st_out["ball_velocity_xy"][valid], vel_y[valid])
                    if state_ttc is not None:
                        ttc_y = state_ttc[st_ids].float().reshape(-1)
                        valid = torch.isfinite(ttc_y) & (ttc_y >= 0)
                        if valid.any():
                            perception_loss = perception_loss + nn.functional.smooth_l1_loss(st_out["time_to_contact"][valid], ttc_y[valid])
                    if state_ttc_bin is not None:
                        bins = state_ttc_bin[st_ids].long().reshape(-1)
                        valid = (bins >= 0) & (bins < 3)
                        if valid.any():
                            perception_loss = perception_loss + nn.functional.cross_entropy(st_out["time_to_contact_logits"][valid], bins[valid])
                    event_y = state_events[st_ids]
                    # Annotation schema may grow over time. Missing trailing event
                    # fields are unknown, never implicit negatives.
                    event_logits = st_out["event_logits"]
                    if event_y.shape[-1] < event_logits.shape[-1]:
                        pad = torch.full(
                            (event_y.shape[0], event_logits.shape[-1] - event_y.shape[-1]),
                            -1.0, device=self.device, dtype=event_y.dtype,
                        )
                        event_y = torch.cat((event_y, pad), dim=-1)
                    elif event_y.shape[-1] > event_logits.shape[-1]:
                        raise ValueError("state_events contains more columns than the configured event schema")
                    valid_events = (event_y >= 0) & (event_y <= 1)
                    if valid_events.any():
                        loss_values = nn.functional.binary_cross_entropy_with_logits(
                            event_logits, event_y.clamp(0, 1), reduction="none"
                        )
                        perception_loss = perception_loss + (loss_values * valid_events.float()).sum() / valid_events.float().sum().clamp_min(1.0)

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
                    ("manager_policy_loss", manager_policy_loss), ("expert_policy_loss", expert_policy_loss),
                    ("manager_value_loss", manager_value_loss), ("expert_value_loss", expert_value_loss),
                    ("manager_entropy", manager_entropy), ("expert_entropy", expert_entropy),
                    ("manager_supervision_loss", manager_supervision_loss), ("phase_supervision_loss", phase_sup),
                    ("demo_bc_loss", demo_bc_loss), ("perception_loss", perception_loss), ("loss", loss),
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
            "format": "tennis_rl_hierarchical_v2",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_index": self.update_index,
            "env_steps": self.env_steps,
            "skills": list(self.model.skills),
            "durations": list(self.model.durations),
            "button_pulse_durations": list(self.model.button_pulse_durations),
            "skills_by_phase": {p: [self.model.skills[i] for i in ids] for p, ids in self.model.phase_task_ids.items()},
            "extra": extra or {},
        }, path)

    def load(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format") != "tennis_rl_hierarchical_v2":
            raise ValueError("Checkpoint is not a current hierarchical NES Tennis v2 checkpoint")
        if tuple(payload.get("skills", self.model.skills)) != self.model.skills:
            raise ValueError("Checkpoint skill list does not match configured skills")
        if tuple(payload.get("durations", self.model.durations)) != self.model.durations:
            raise ValueError("Checkpoint duration list does not match configured durations")
        if tuple(payload.get("button_pulse_durations", self.model.button_pulse_durations)) != self.model.button_pulse_durations:
            raise ValueError("Checkpoint button pulse durations do not match configuration")
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        self.update_index = int(payload.get("update_index", 0))
        self.env_steps = int(payload.get("env_steps", 0))
        return payload.get("extra", {})


class HierarchicalController:
    """Stateful inference controller with learned routing and pulse durations."""

    def __init__(self, model: HierarchicalActorCritic, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)
        self.active_skills: torch.Tensor | None = None
        self.remaining: torch.Tensor | None = None
        self.belief_hidden: torch.Tensor | None = None
        self.last_skills: torch.Tensor | None = None
        self.last_durations: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None or self.active_skills is None or self.remaining is None:
            self.active_skills = None
            self.remaining = None
            self.belief_hidden = None
            self.last_skills = None
            self.last_durations = None
            return
        mask = mask.to(device=self.active_skills.device, dtype=torch.bool)
        self.active_skills[mask] = -1
        self.remaining[mask] = 0
        if self.belief_hidden is not None:
            self.belief_hidden[:, mask, :] = 0.0

    @torch.no_grad()
    def act_with_pulse(
        self, observations: torch.Tensor, *, forced_skill: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, device = observations.shape[0], observations.device
        if self.belief_hidden is None or self.belief_hidden.shape[1] != n:
            self.belief_hidden = self.model.initial_belief(n, device)
        output = self.model(observations, self.belief_hidden)
        # The controller's belief advances exactly once per incoming observation.
        self.belief_hidden = output["belief_hidden_out"].detach()
        if self.active_skills is None or self.remaining is None or self.active_skills.shape[0] != n:
            self.active_skills = torch.full((n,), -1, dtype=torch.long, device=device)
            self.remaining = torch.zeros((n,), dtype=torch.long, device=device)
        if forced_skill is not None:
            if forced_skill not in self.model.skill_to_id:
                raise ValueError(f"Unknown forced skill {forced_skill!r}; available: {self.model.skills}")
            self.active_skills.fill_(self.model.skill_to_id[forced_skill])
            self.remaining.fill_(1)
        boundary = (self.active_skills < 0) | (self.remaining <= 0)
        idx = torch.nonzero(boundary, as_tuple=False).squeeze(-1)
        if idx.numel():
            subset = {
                "phase_logits": output["phase_logits"][idx],
                "branch_logits": {key: value[idx] for key, value in output["branch_logits"].items()},
            }
            new_skills, _selected_phase = self.model.select_leaf_skills(subset, deterministic=self.deterministic)
            duration_logits = output["manager_duration_logits"][idx]
            duration_dist = Categorical(logits=duration_logits)
            if self.deterministic:
                duration_ids = duration_logits.argmax(dim=-1)
            else:
                duration_ids = duration_dist.sample()
            duration_values = torch.as_tensor(self.model.durations, device=device, dtype=torch.long)
            self.active_skills[idx] = new_skills
            self.remaining[idx] = duration_values[duration_ids]
            self.last_durations = duration_values[duration_ids].detach()
        rows = torch.arange(n, device=device)
        self.last_skills = self.active_skills.detach().clone()
        action_logits = output["expert_logits"][rows, self.active_skills]
        action_dist = Categorical(logits=action_logits)
        actions = torch.argmax(action_logits, dim=-1) if self.deterministic else action_dist.sample()
        pulse_logits = output["expert_pulse_logits"][rows, self.active_skills]
        pulse_dist = Categorical(logits=pulse_logits)
        pulse_ids = torch.argmax(pulse_logits, dim=-1) if self.deterministic else pulse_dist.sample()
        pulse_choices = torch.as_tensor(self.model.button_pulse_durations, device=device, dtype=torch.long)
        pulse_frames = pulse_choices[pulse_ids]
        self.remaining -= 1
        return actions, pulse_frames

    @torch.no_grad()
    def act(self, observations: torch.Tensor, *, forced_skill: str | None = None) -> torch.Tensor:
        actions, _pulse = self.act_with_pulse(observations, forced_skill=forced_skill)
        return actions
