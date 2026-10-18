"""Demonstration micro-segment replay for NES Tennis RL.

Problem this solves
-------------------
Training directly on whole tennis points (~60-600 steps) leads to credit-assignment
failure and severe temporal dilution.  The single terminal reward at point-end is
too sparse, causing agents to learn degeneracies like "never serve" or random flailing.

Approach
--------
This module decomposes recorded demonstrations into semantically grounded
**micro-segments** (e.g. 16-32 steps):
1. ``serve``    -- opening serve / toss execution (first steps of point).
2. ``stroke``   -- hitting window with active A/B racket swing.
3. ``approach`` -- directional movement towards intercept position.
4. ``late``     -- endgame window deciding the point (last 30 steps).
5. ``neutral``  -- in-play positioning.

Reward & Return-To-Go Assignment
--------------------------------
Returns-to-go are computed across the whole demonstration point and sliced to
the local segment:
- Winning points: positive RTG discounted from +1.0 throughout the rally.
- Losing points: the terminal blunder receives negative RTG (-1.0), while
  early/mid segments (valid serves and sustained rallies) receive a neutral/positive
  continuation credit (+0.15), ensuring good human play before an unforced error
  is NOT falsely penalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from .demonstrations import DemonstrationDataset, PointEpisode
from .model import RecurrentActorCritic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmentReplayConfig:
    """Hyper-parameters for the micro-segment demonstration replay."""

    # Segment length in agent steps (1 step = frame_skip frames).
    seg_len: int = 16

    # Number of segments to draw per gradient step.
    batch_size: int = 8

    # Number of gradient steps to take in standalone ``update()`` call.
    steps_per_update: int = 4

    # Weight multiplier applied to winning-segment BC loss.
    winning_bc_weight: float = 2.5

    # Weight multiplier applied to losing-segment BC loss (>0, not zero).
    losing_bc_weight: float = 0.5

    # Coefficient for the point-outcome auxiliary BCE head.
    outcome_coef: float = 0.30

    # Return-to-go discount across the point.
    rtg_gamma: float = 0.99

    # RTG modulation strength: BC_weight *= (1 + rtg_coef * rtg).
    rtg_coef: float = 0.75

    # Minimum BC weight after RTG modulation (never zero).
    min_bc_weight: float = 0.20

    # Gradient clip for segment updates.
    max_grad_norm: float = 0.5

    # Learning rate for the standalone optimiser.
    learning_rate: float = 3e-5

    # Rebuild the segment priority table every N outer-loop updates.
    refresh_every: int = 50

    # Priority bonus applied to winning segments during sampling.
    winning_priority_bonus: float = 2.0

    # Minimum priority (all segments always have a chance).
    priority_eps: float = 0.05

    # Segment taxonomy thresholds.
    serve_lead_steps: int = 12       # first N steps of a point -> serve phase
    late_lead_steps: int = 30        # last N steps of a point -> late / endgame
    stroke_btn_thresh: float = 0.15  # >=15% of steps have A or B -> stroke
    approach_move_thresh: float = 0.50 # >=50% steps have movement -> approach


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class _Segment(NamedTuple):
    """One contiguous micro-segment cut from a recorded point."""
    obs: np.ndarray        # [seg_len, 4, 84, 84] uint8
    actions: np.ndarray    # [seg_len]             int64
    rtg: np.ndarray        # [seg_len]             float32  (normalised, -1..1)
    outcome: str           # "winning" | "losing"
    seg_type: str          # "serve" | "stroke" | "approach" | "late" | "neutral"
    priority: float


def _classify_segment(actions: np.ndarray, start: int, point_len: int,
                      cfg: SegmentReplayConfig) -> str:
    """Heuristic segment type from action content and position in point."""
    shots = actions % 3    # 0=none, 1=A, 2=B
    moves = actions // 3   # 0=no_move, 1-8=directional

    btn_frac = float((shots > 0).mean())
    move_frac = float((moves > 0).mean())

    if start < cfg.serve_lead_steps:
        return "serve"
    if start + cfg.seg_len >= point_len - cfg.late_lead_steps:
        return "late"
    if btn_frac >= cfg.stroke_btn_thresh:
        return "stroke"
    if move_frac >= cfg.approach_move_thresh:
        return "approach"
    return "neutral"


def _compute_point_rtg(ep: PointEpisode, gamma: float) -> np.ndarray:
    """Compute point-level return-to-go smoothly discounted from final outcome."""
    n = ep.length
    rewards = ep.rewards
    outcome = ep.outcome
    rtg = np.zeros(n, dtype=np.float32)

    # If the episode has explicit environment terminal reward:
    has_nonzero = np.any(np.abs(rewards) > 1e-4)
    if has_nonzero:
        g = 0.0
        for i in range(n - 1, -1, -1):
            g = float(rewards[i]) + gamma * g
            rtg[i] = g
        scale = float(np.abs(rtg).max())
        if scale > 1e-6:
            rtg = rtg / scale
    else:
        final_val = 1.0 if outcome == "winning" else -1.0
        for i in range(n):
            dist = n - 1 - i
            rtg[i] = (gamma ** dist) * final_val

    # For losing points: the blunder happened at the very end.
    # Earlier rally frames were valid play; reward rally survival instead of penalizing:
    if outcome == "losing":
        late_boundary = max(0, n - 25)
        for i in range(late_boundary):
            rtg[i] = 0.15

    return rtg


def _extract_segments(ep: PointEpisode, cfg: SegmentReplayConfig) -> list[_Segment]:
    """Cut a recorded point into overlapping micro-segments with ground-truth RTG."""
    n = ep.length
    if n < cfg.seg_len:
        return []

    point_rtg = _compute_point_rtg(ep, cfg.rtg_gamma)
    segments: list[_Segment] = []
    stride = max(1, cfg.seg_len // 2)  # 50% overlap

    for start in range(0, n - cfg.seg_len + 1, stride):
        end = start + cfg.seg_len
        obs = ep.observations[start:end]
        acts = ep.actions[start:end]
        rtg = point_rtg[start:end]

        seg_type = _classify_segment(acts, start, n, cfg)

        # Priority calculation:
        # 1. Base priority from outcome (winning gets bonus).
        bonus = cfg.winning_priority_bonus if ep.outcome == "winning" else 1.0
        # 2. Type-specific priority boost: ensure serve and stroke segments are prominent!
        type_boost = 1.0
        if seg_type == "serve":
            type_boost = 1.8  # Always maintain heavy serve supervision
        elif seg_type == "stroke":
            type_boost = 1.5  # Maintain stroke timing supervision
        elif seg_type == "late" and ep.outcome == "winning":
            type_boost = 1.6  # Endgame winners

        rtg_mag = float(np.abs(rtg).mean())
        priority = (bonus * type_boost * (0.5 + 0.5 * rtg_mag)) + cfg.priority_eps

        segments.append(_Segment(
            obs=obs.astype(np.uint8, copy=False),
            actions=acts.astype(np.int64, copy=False),
            rtg=rtg.copy(),
            outcome=ep.outcome,
            seg_type=seg_type,
            priority=priority,
        ))
    return segments


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DemoSegmentReplayer:
    """Micro-segment demonstration replayer and supervised regularizer."""

    def __init__(
        self,
        dataset: DemonstrationDataset,
        model: RecurrentActorCritic,
        device: torch.device,
        cfg: SegmentReplayConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.device = device
        self.cfg = cfg or SegmentReplayConfig()

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=0.0
        )

        self.rng = np.random.default_rng(1337)
        self._segments: list[_Segment] = []
        self._priorities: np.ndarray = np.array([], dtype=np.float64)
        self._by_type: dict[str, list[_Segment]] = {}
        self._outer_update: int = 0

        # Metrics accumulated between calls to ``report()``
        self._bc_total: float = 0.0
        self._outcome_total: float = 0.0
        self._n_updates: int = 0
        self._type_counts: dict[str, int] = {
            "serve": 0, "stroke": 0, "approach": 0, "late": 0, "neutral": 0
        }

        self._rebuild_segments()

    # ------------------------------------------------------------------
    # Segment table management
    # ------------------------------------------------------------------

    def _rebuild_segments(self) -> None:
        """Extract all micro-segments from the current demonstration set."""
        segs: list[_Segment] = []
        by_type: dict[str, list[_Segment]] = {
            "serve": [], "stroke": [], "approach": [], "late": [], "neutral": []
        }
        for ep in self.dataset.episodes:
            extracted = _extract_segments(ep, self.cfg)
            segs.extend(extracted)
            for s in extracted:
                by_type.setdefault(s.seg_type, []).append(s)

        self._segments = segs
        self._by_type = by_type
        if segs:
            p = np.asarray([s.priority for s in segs], dtype=np.float64)
            self._priorities = p / p.sum()
        else:
            self._priorities = np.array([], dtype=np.float64)

    def sample_batch(self, batch_size: int | None = None) -> list[_Segment]:
        """Sample a batch of micro-segments with stratified type balancing."""
        if not self._segments:
            return []
        B = batch_size if batch_size is not None else self.cfg.batch_size
        B = min(B, len(self._segments))

        # Stratified sampling: 25% serve, 35% stroke, 25% late, 15% other
        chosen: list[_Segment] = []
        slots = {
            "serve": max(1, int(round(B * 0.25))),
            "stroke": max(1, int(round(B * 0.35))),
            "late": max(1, int(round(B * 0.25))),
        }
        for stype, count in slots.items():
            pool = self._by_type.get(stype, [])
            if pool:
                take = min(count, len(pool))
                idx = self.rng.choice(len(pool), size=take, replace=len(pool) < take)
                for i in np.asarray(idx).reshape(-1):
                    chosen.append(pool[int(i)])

        # Fill remainder from general priority distribution
        rem = B - len(chosen)
        if rem > 0 and len(self._segments) > 0:
            idx = self.rng.choice(
                len(self._segments),
                size=rem,
                replace=True,
                p=self._priorities,
            )
            for i in np.asarray(idx).reshape(-1):
                chosen.append(self._segments[int(i)])

        self.rng.shuffle(chosen)
        return chosen[:B]

    # ------------------------------------------------------------------
    # Collation helpers
    # ------------------------------------------------------------------

    def collate(self, segs: list[_Segment]):
        """Stack a list of segments into batched tensors [T, B, ...]."""
        T = self.cfg.seg_len
        B = len(segs)
        dev = self.device

        obs = torch.zeros((T, B, 4, 84, 84), dtype=torch.uint8, device=dev)
        actions = torch.zeros((T, B), dtype=torch.long, device=dev)
        outcome_target = torch.zeros((T, B), dtype=torch.float32, device=dev)
        bc_weight = torch.zeros((T, B), dtype=torch.float32, device=dev)

        cfg = self.cfg
        for j, seg in enumerate(segs):
            t = seg.obs.shape[0]
            obs[:t, j] = torch.from_numpy(seg.obs)
            actions[:t, j] = torch.from_numpy(seg.actions)

            rtg_t = torch.from_numpy(seg.rtg[:t]).to(dev)

            ot = 1.0 if seg.outcome == "winning" else 0.0
            outcome_target[:t, j] = ot

            # BC weight = outcome_weight x RTG modulation, clamped to min.
            ow = cfg.winning_bc_weight if seg.outcome == "winning" else cfg.losing_bc_weight
            w = ow * (1.0 + cfg.rtg_coef * rtg_t).clamp_min(cfg.min_bc_weight)
            bc_weight[:t, j] = w

        return obs, actions, outcome_target, bc_weight

    # ------------------------------------------------------------------
    # Differentiable loss computation (for unified training with PPO)
    # ------------------------------------------------------------------

    def compute_losses(
        self,
        model: RecurrentActorCritic,
        segs: list[_Segment] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute BC and auxiliary outcome losses without updating parameters.

        Used by PPOTrainer to integrate demonstration supervision in a unified step.
        """
        if segs is None:
            segs = self.sample_batch()
        if not segs:
            zero = torch.zeros((), device=self.device)
            return zero, zero

        for s in segs:
            self._type_counts[s.seg_type] = self._type_counts.get(s.seg_type, 0) + 1

        obs, actions, outcome_target, bc_weight = self.collate(segs)
        hidden = model.initial_hidden(obs.shape[1], self.device)
        dist, _, _, outcome_logits = model.forward_sequence(
            obs.float() / 255.0, hidden, include_aux=True
        )

        # Factorized policy log-probability (movement + shot)
        logp = dist.log_prob(actions)  # [T, B]
        bc_denom = bc_weight.sum().clamp_min(1.0)
        bc_loss = -(logp * bc_weight).sum() / bc_denom

        outcome_bce = nn.functional.binary_cross_entropy_with_logits(
            outcome_logits, outcome_target, reduction="none"
        )
        outcome_loss = outcome_bce.mean()

        self._bc_total += float(bc_loss.detach())
        self._outcome_total += float(outcome_loss.detach())
        self._n_updates += 1

        return bc_loss, outcome_loss

    # ------------------------------------------------------------------
    # Standalone gradient step
    # ------------------------------------------------------------------

    def _gradient_step(self, segs: list[_Segment]) -> tuple[float, float]:
        """One standalone gradient update on the given segment batch."""
        bc_loss, outcome_loss = self.compute_losses(self.model, segs)
        loss = bc_loss + self.cfg.outcome_coef * outcome_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return float(bc_loss.detach()), float(outcome_loss.detach())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, n_steps: int | None = None) -> dict[str, float]:
        """Run ``n_steps`` standalone gradient steps on demonstration micro-segments."""
        self._outer_update += 1
        if self._outer_update % self.cfg.refresh_every == 0:
            self._rebuild_segments()

        if not self._segments:
            return {"seg_bc_loss": 0.0, "seg_outcome_loss": 0.0,
                    "seg_n_updates": 0, "seg_n_segments": 0}

        steps = n_steps if n_steps is not None else self.cfg.steps_per_update
        self.model.train()
        bc_acc = 0.0
        out_acc = 0.0

        for _ in range(steps):
            segs = self.sample_batch()
            if not segs:
                break
            bc_l, out_l = self._gradient_step(segs)
            bc_acc += bc_l
            out_acc += out_l

        n = max(steps, 1)
        return {
            "seg_bc_loss": bc_acc / n,
            "seg_outcome_loss": out_acc / n,
            "seg_n_updates": steps,
            "seg_n_segments": len(self._segments),
        }

    def report(self) -> dict[str, float]:
        """Return accumulated statistics and reset counters."""
        n = max(self._n_updates, 1)
        out = {
            "seg_bc_loss_mean": self._bc_total / n,
            "seg_outcome_loss_mean": self._outcome_total / n,
            "seg_total_updates": float(self._n_updates),
            "seg_n_segments": float(len(self._segments)),
            **{f"seg_type_{k}": float(v) for k, v in self._type_counts.items()},
        }
        self._bc_total = 0.0
        self._outcome_total = 0.0
        self._n_updates = 0
        self._type_counts = {k: 0 for k in self._type_counts}
        return out
