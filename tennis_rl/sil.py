"""Self-Imitation Learning (Oh et al., 2018) replay buffer, adapted for a
recurrent policy.

The core idea: PPO only ever learns from its current rollout batch, so a
rare good trajectory it stumbled into several updates ago has no protected
influence on the current gradient - if recent batches are mostly losses,
nothing stops the policy drifting away from behavior that used to win.
SIL fixes this with a small auxiliary loss, computed on replayed samples
from the agent's own past experience:

    A(s, a) = max(R - V(s), 0)                    # clipped positive advantage
    L_policy = -logpi(a|s) * A(s, a)               # only reinforce when the
    L_value  = 0.5 * A(s, a)^2                     #   actual outcome beat
                                                    #   what the critic expected

R is the true (Monte Carlo) discounted return actually observed after that
state, not a bootstrapped estimate. Because the advantage is clipped at
zero, replaying a mediocre or losing episode contributes ~nothing to the
loss automatically - there's no need to curate what goes into the buffer,
which is why this implementation stores every finished episode rather than
only "good" ones.

Recurrence note: unlike the original (feedforward) SIL paper, this policy
carries a GRU hidden state across time. Each stored chunk also stores the
*actual* hidden state the model had at the start of that chunk during
collection, so replay can resume the recurrence faithfully rather than
starting from a zeroed/re-estimated state. That stored hidden state grows
mildly stale as the policy updates (the same approximation recurrent replay
methods like R2D2 make); it is not recomputed on every replay, since a full
burn-in recompute would cost a stateful forward pass through the entire
episode prefix for every sampled chunk.

This buffer samples with priority proportional to each chunk's stored
return (a softmax over returns, temperature-controlled) rather than
uniformly. Uniform sampling turned out to be a real problem in practice,
not just a theoretical one: with wins occurring in maybe 3-10% of episodes,
a buffer of 256 chunks holds only ~8-26 winning ones, and a uniformly-drawn
batch of 8 has a 40-80% chance of containing zero of them - most SIL
updates had nothing to reinforce even though a win was sitting right there
in the buffer. Priority-weighted sampling (as in the original SIL paper,
which this implementation had simplified away from) fixes that directly:
winning chunks get drawn far more often than their share of the buffer,
regardless of how rare they are.
Not persisted across checkpoints - it repopulates from fresh episodes as
training continues, which is a deliberate simplicity/size trade-off, not
an oversight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SILChunk:
    obs: torch.Tensor      # [T, C, H, W] uint8, cpu
    actions: torch.Tensor  # [T] long, cpu
    returns: torch.Tensor  # [T] float32, cpu - true discounted return-to-go
    h0: torch.Tensor       # [L, 1, H] float32, cpu - hidden state at chunk start
    priority: float        # max return-to-go anywhere in the chunk - used for sampling


class SelfImitationBuffer:
    def __init__(
        self,
        capacity_chunks: int,
        chunk_length: int,
        gamma: float,
        priority_sampling: bool = True,
        priority_temperature: float = 5.0,
    ):
        if capacity_chunks <= 0:
            raise ValueError("capacity_chunks must be > 0")
        if chunk_length <= 0:
            raise ValueError("chunk_length must be > 0")
        self.chunk_length = chunk_length
        self.gamma = gamma
        self.priority_sampling = priority_sampling
        self.priority_temperature = max(priority_temperature, 1e-6)
        self._chunks: deque[SILChunk] = deque(maxlen=capacity_chunks)

        self._n_workers = 0
        self._ep_obs: list[list[np.ndarray]] = []
        self._ep_actions: list[list[int]] = []
        self._ep_rewards: list[list[float]] = []
        self._ep_hidden: list[list[torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def _ensure_workers(self, n: int) -> None:
        if self._n_workers == n:
            return
        self._n_workers = n
        self._ep_obs = [[] for _ in range(n)]
        self._ep_actions = [[] for _ in range(n)]
        self._ep_rewards = [[] for _ in range(n)]
        self._ep_hidden = [[] for _ in range(n)]

    def record_step(
        self,
        obs_np: np.ndarray,      # [N, C, H, W] uint8 - observation BEFORE this step
        hidden: torch.Tensor,    # [L, N, H] - hidden state BEFORE this step
        actions_np: np.ndarray,  # [N]
        rewards: np.ndarray,     # [N]
        dones: np.ndarray,       # [N]
    ) -> None:
        n = obs_np.shape[0]
        self._ensure_workers(n)
        # .clone() is required, not just .detach(): hidden is produced inside
        # collect_rollout's torch.inference_mode() block, and .detach() alone
        # does not strip that taint (verified directly - a plain detach()
        # still raises "Inference tensors cannot be saved for backward" the
        # first time this hidden state is later used in a SIL backward pass).
        hidden_cpu = hidden.detach().to("cpu").clone()
        for i in range(n):
            self._ep_obs[i].append(obs_np[i])
            self._ep_actions[i].append(int(actions_np[i]))
            self._ep_rewards[i].append(float(rewards[i]))
            # hidden_cpu[:, i, :] has shape [L, H] (int index drops dim 1).
            self._ep_hidden[i].append(hidden_cpu[:, i, :])
            if dones[i]:
                self._finish_episode(i)

    def _finish_episode(self, i: int) -> None:
        rewards = self._ep_rewards[i]
        length = len(rewards)
        if length == 0:
            return

        returns = [0.0] * length
        running = 0.0
        for t in range(length - 1, -1, -1):
            running = rewards[t] + self.gamma * running
            returns[t] = running

        obs_seq = self._ep_obs[i]
        action_seq = self._ep_actions[i]
        hidden_seq = self._ep_hidden[i]

        for start in range(0, length, self.chunk_length):
            end = min(start + self.chunk_length, length)
            obs_chunk = torch.from_numpy(np.stack(obs_seq[start:end], axis=0))
            action_chunk = torch.tensor(action_seq[start:end], dtype=torch.long)
            return_chunk = torch.tensor(returns[start:end], dtype=torch.float32)
            h0 = hidden_seq[start].unsqueeze(1).clone()  # [L, H] -> [L, 1, H]
            priority = float(return_chunk.max())
            self._chunks.append(SILChunk(obs_chunk, action_chunk, return_chunk, h0, priority))

        self._ep_obs[i] = []
        self._ep_actions[i] = []
        self._ep_rewards[i] = []
        self._ep_hidden[i] = []

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[SILChunk]:
        n = len(self._chunks)
        if n == 0:
            return []
        chunks_list = list(self._chunks)
        k = min(batch_size, n)

        if not self.priority_sampling:
            idx = rng.integers(0, n, size=k)
            return [chunks_list[j] for j in idx]

        priorities = np.array([c.priority for c in chunks_list], dtype=np.float64)
        # Softmax over priorities, temperature-scaled. Subtracting the max
        # before exponentiating is the standard stabilization: it changes
        # nothing about the resulting probabilities (they're invariant to a
        # constant shift) but keeps exp() in a safe range regardless of how
        # large the raw returns are.
        scaled = (priorities - priorities.max()) / self.priority_temperature
        weights = np.exp(np.clip(scaled, -50.0, 0.0))
        probs = weights / weights.sum()
        idx = rng.choice(n, size=k, replace=True, p=probs)
        return [chunks_list[j] for j in idx]
