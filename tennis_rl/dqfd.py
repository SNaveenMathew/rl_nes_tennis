from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
import random
import torch
from torch import nn

from .model import RecurrentActorCritic


@dataclass
class Sequence:
    obs: np.ndarray          # [T, 4, 84, 84], uint8
    actions: np.ndarray      # [T]
    rewards: np.ndarray      # [T]
    dones: np.ndarray        # [T]
    demo: bool = False


class RecurrentDuelingQ(nn.Module):
    """CNN + GRU + dueling discrete Q network for the 27 NES controls."""

    def __init__(self, base: RecurrentActorCritic):
        super().__init__()
        self.encoder = base.encoder
        self.gru = base.gru
        h = base.gru_hidden
        self.gru_hidden = h
        self.gru_layers = base.gru_layers
        self.action_count = 27
        self.value = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.advantage = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, self.action_count))
        nn.init.orthogonal_(self.value[0].weight, gain=1.0)
        nn.init.zeros_(self.value[0].bias)
        nn.init.orthogonal_(self.value[2].weight, gain=1.0)
        nn.init.zeros_(self.value[2].bias)
        nn.init.orthogonal_(self.advantage[0].weight, gain=1.0)
        nn.init.zeros_(self.advantage[0].bias)
        nn.init.orthogonal_(self.advantage[2].weight, gain=0.01)
        nn.init.zeros_(self.advantage[2].bias)

    def initial_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch, self.gru_hidden, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        # obs [T,B,C,H,W]
        t, b = obs.shape[:2]
        z = self.encoder(obs.reshape(t * b, *obs.shape[2:])).reshape(t, b, -1)
        hseq, hn = self.gru(z, hidden)
        v = self.value(hseq)
        a = self.advantage(hseq)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q, hn


class SequenceReplay:
    """Permanent demonstration replay + bounded online replay."""

    def __init__(self, demo_capacity: int, online_capacity: int, alpha: float, demo_priority_bonus: float):
        self.demo_capacity = int(demo_capacity)
        self.online_capacity = int(online_capacity)
        self.alpha = float(alpha)
        self.demo_priority_bonus = float(demo_priority_bonus)
        self.demo: list[tuple[Sequence, float]] = []
        self.online: list[tuple[Sequence, float]] = []

    def __len__(self):
        return len(self.demo) + len(self.online)

    def _append(self, target, capacity, seq, priority):
        if len(target) >= capacity:
            target.pop(0)
        target.append((seq, max(float(priority), 1e-4)))

    def add(self, seq: Sequence, priority: float = 1.0):
        if seq.demo:
            self._append(self.demo, self.demo_capacity, seq, priority + self.demo_priority_bonus)
        else:
            self._append(self.online, self.online_capacity, seq, priority)

    def sample(self, batch_size: int, demo_fraction: float, rng: np.random.Generator):
        if not self.demo and not self.online:
            return None
        requested_demo = min(len(self.demo), int(round(batch_size * demo_fraction)))
        requested_online = min(len(self.online), batch_size - requested_demo)
        # Fill remaining capacity from whichever pool has samples.
        while requested_demo + requested_online < batch_size:
            if len(self.demo) > requested_demo:
                requested_demo += 1
            elif len(self.online) > requested_online:
                requested_online += 1
            else:
                break
        chosen: list[tuple[Sequence, bool, int]] = []
        for pool, count, is_demo in ((self.demo, requested_demo, True), (self.online, requested_online, False)):
            if count <= 0 or not pool:
                continue
            p = np.asarray([priority for _, priority in pool], dtype=np.float64) ** self.alpha
            p /= p.sum()
            idx = rng.choice(len(pool), size=min(count, len(pool)), replace=len(pool) < count, p=p)
            for i in np.asarray(idx).reshape(-1):
                chosen.append((pool[int(i)][0], is_demo, int(i)))
        rng.shuffle(chosen)
        return chosen

    def stats(self):
        return len(self.demo), len(self.online)


class DQFDTrainer:
    def __init__(self, model: RecurrentDuelingQ, target: RecurrentDuelingQ, cfg, device: torch.device):
        self.model = model
        self.target = target
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.target.load_state_dict(model.state_dict())
        self.target.eval()
        self.replay = SequenceReplay(cfg.demo_replay_capacity, cfg.online_replay_capacity, cfg.priority_alpha, cfg.demo_priority_bonus)
        self.rng = np.random.default_rng(42)
        self.update_index = 0
        self.env_steps = 0

    def add_sequence(self, seq: Sequence, priority: float = 1.0):
        self.replay.add(seq, priority)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, hidden: torch.Tensor, epsilon: float):
        obs = torch.from_numpy(obs_np).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.0
        q, next_hidden = self.model(obs, hidden)
        q_now = q[0]
        greedy = q_now.argmax(dim=-1)
        if random.random() < epsilon:
            actions = torch.randint(0, self.model.action_count, greedy.shape, device=self.device)
        else:
            actions = greedy
        return actions, next_hidden, q_now

    def _collate(self, sampled):
        max_t = max(int(s.obs.shape[0]) for s, _, _ in sampled)
        b = len(sampled)
        obs = torch.zeros((max_t, b, 4, 84, 84), dtype=torch.uint8, device=self.device)
        actions = torch.zeros((max_t, b), dtype=torch.long, device=self.device)
        rewards = torch.zeros((max_t, b), dtype=torch.float32, device=self.device)
        dones = torch.ones((max_t, b), dtype=torch.float32, device=self.device)
        mask = torch.zeros((max_t, b), dtype=torch.bool, device=self.device)
        demo_mask = torch.zeros((b,), dtype=torch.bool, device=self.device)
        for j, (s, is_demo, _) in enumerate(sampled):
            t = s.obs.shape[0]
            obs[:t, j] = torch.from_numpy(s.obs).to(self.device)
            actions[:t, j] = torch.from_numpy(s.actions).to(self.device)
            rewards[:t, j] = torch.from_numpy(s.rewards).to(self.device)
            dones[:t, j] = torch.from_numpy(s.dones.astype(np.float32)).to(self.device)
            mask[:t, j] = True
            demo_mask[j] = is_demo
        return obs, actions, rewards, dones, mask, demo_mask

    def update(self, batch_size: int, beta: float = 0.4):
        sampled = self.replay.sample(batch_size, self.cfg.demo_fraction, self.rng)
        if not sampled:
            return None
        obs, actions, rewards, dones, mask, demo_mask = self._collate(sampled)
        h0 = self.model.initial_hidden(obs.shape[1], self.device)
        q, _ = self.model(obs.float() / 255.0, h0)
        with torch.no_grad():
            nq_online, _ = self.model(obs.float() / 255.0, h0)
            next_actions = nq_online.argmax(dim=-1)
            next_q_target, _ = self.target(obs.float() / 255.0, h0)
            next_q = next_q_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)

            n = int(self.cfg.n_step_return)
            target = torch.zeros_like(rewards)
            valid = torch.zeros_like(mask)
            for t in range(obs.shape[0]):
                steps = min(n, obs.shape[0] - t - 1)
                acc = torch.zeros(obs.shape[1], device=self.device)
                discount = 1.0
                terminal = torch.zeros(obs.shape[1], device=self.device)
                for k in range(steps):
                    acc = acc + discount * rewards[t + k]
                    terminal = torch.maximum(terminal, dones[t + k])
                    discount *= self.cfg.gamma
                if t + steps < obs.shape[0]:
                    acc = acc + discount * (1.0 - terminal) * next_q[t + steps]
                    valid[t] = mask[t]
                target[t] = acc
        q_taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        td_abs = (target - q_taken).abs().detach()
        td_loss = nn.functional.smooth_l1_loss(q_taken[valid], target[valid], reduction="mean") if valid.any() else torch.zeros((), device=self.device)

        margin_loss = torch.zeros((), device=self.device)
        demo_steps = demo_mask[None, :].expand_as(mask) & mask
        if demo_steps.any():
            dq = q.permute(1, 0, 2)[demo_steps.permute(1, 0)]
            da = actions.permute(1, 0)[demo_steps.permute(1, 0)]
            margins = torch.full_like(dq, self.cfg.demo_margin)
            margins.scatter_(1, da[:, None], 0.0)
            margin_loss = torch.relu(dq + margins - dq.gather(1, da[:, None])).mean()

        loss = td_loss + self.cfg.demo_large_margin_coef * margin_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        if self.update_index % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.update_index += 1
        demo_fraction = float(demo_mask.float().mean().detach())
        return {
            "loss": float(loss.detach()),
            "td_loss": float(td_loss.detach()),
            "margin_loss": float(margin_loss.detach()),
            "mean_q": float(q_taken[mask].mean().detach()),
            "td_abs": float(td_abs[valid].mean().detach()) if valid.any() else 0.0,
            "demo_frac": demo_fraction,
            "demo_replay": float(len(self.replay.demo)),
            "online_replay": float(len(self.replay.online)),
        }


def _sequence_from_episode(ep, demo: bool, max_len: int | None = None):
    n = ep.length if max_len is None else min(ep.length, int(max_len))
    if n < 2:
        return None
    return Sequence(
        obs=np.asarray(ep.observations[:n], dtype=np.uint8),
        actions=np.asarray(ep.actions[:n], dtype=np.int64),
        rewards=np.asarray(ep.rewards[:n], dtype=np.float32),
        dones=np.asarray(ep.dones[:n], dtype=np.bool_),
        demo=demo,
    )


def pretrain_from_demonstrations(trainer: DQFDTrainer, dataset, *, updates: int, batch_size: int) -> dict[str, float]:
    for ep in dataset.episodes:
        # Keep point sequences intact when possible. Very long examples are
        # split into fixed windows so they remain cheap enough to train.
        seq_len = int(trainer.cfg.sequence_length)
        if ep.length <= seq_len:
            seq = _sequence_from_episode(ep, True)
            if seq is not None:
                trainer.add_sequence(seq, priority=2.0)
        else:
            for start in range(0, ep.length - 1, max(1, seq_len // 2)):
                part = type("Part", (), {
                    "length": min(seq_len, ep.length - start),
                    "observations": ep.observations[start:start+seq_len],
                    "actions": ep.actions[start:start+seq_len],
                    "rewards": ep.rewards[start:start+seq_len],
                    "dones": ep.dones[start:start+seq_len],
                })()
                seq = _sequence_from_episode(part, True)
                if seq is not None:
                    trainer.add_sequence(seq, priority=2.0)
    totals = {"loss": 0.0, "td_loss": 0.0, "margin_loss": 0.0}
    done = 0
    for _ in range(updates):
        m = trainer.update(batch_size)
        if m is None:
            break
        for k in totals:
            totals[k] += m[k]
        done += 1
    for k in totals:
        totals[k] /= max(done, 1)
    totals["updates"] = float(done)
    return totals
