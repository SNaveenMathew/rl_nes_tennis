"""Faithful micro-segment extraction/replay for NES Tennis demonstrations.

Segments are temporal windows, not guessed tennis tasks. A segment receives a
semantic label only when an explicit annotation covers its whole [start, end)
range. Outcome/return values are computed from recorded environment rewards;
match-level outcomes are never substituted for unknown point results, and lost
points do not receive fabricated positive continuation credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import json

import numpy as np
import torch
from torch import nn

from .demonstrations import DemonstrationDataset, PointEpisode
from .model import RecurrentActorCritic
from .hierarchy_schema import TASK_TO_PHASE, phase_for_task


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

    # Kept only to read older config files. New sampling does not upweight
    # whole points based on outcome or invent task labels from action patterns.
    winning_priority_bonus: float = 1.0
    priority_eps: float = 0.05
    serve_lead_steps: int = 12
    late_lead_steps: int = 30
    stroke_btn_thresh: float = 0.15
    approach_move_thresh: float = 0.50


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class _Segment(NamedTuple):
    """A contiguous slice with truthful reward metadata and optional explicit task label."""
    obs: np.ndarray
    actions: np.ndarray
    rtg: np.ndarray
    outcome: str                    # winning | losing | unknown
    seg_type: str                   # exact annotation task or "unlabeled"
    priority: float
    source_path: str = ""
    start: int = 0
    end: int = 0
    phase: str | None = None


def _classify_segment(
    actions: np.ndarray,
    start: int,
    point_len: int,
    cfg: SegmentReplayConfig,
    annotations: list[dict] | None = None,
    source_path: str | None = None,
) -> str:
    """Return an exact task label only if an annotation covers the whole slice."""
    del actions, point_len  # action frequency and window location are not task labels
    if annotations and source_path:
        for row in annotations:
            try:
                same = Path(str(row["resolved_path"])).resolve() == Path(source_path).resolve()
                if same and int(row["start"]) <= start and int(row["end"]) >= start + int(cfg.seg_len):
                    label = str(row.get("leaf_task") or row.get("skill") or "")
                    if label in TASK_TO_PHASE:
                        return label
            except (KeyError, TypeError, ValueError, OSError):
                continue
    return "unlabeled"


def _compute_point_rtg(ep: PointEpisode, gamma: float) -> np.ndarray:
    """Discount only actual recorded rewards; unknown outcome stays zero-return."""
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0,1]")
    rewards = np.asarray(ep.rewards, dtype=np.float32)
    rtg = np.zeros(ep.length, dtype=np.float32)
    g = 0.0
    for i in range(ep.length - 1, -1, -1):
        g = float(rewards[i]) + gamma * g
        rtg[i] = g
    return rtg


def _read_task_annotations(root: Path) -> dict[str, list[dict]]:
    """Parse explicit sidecars into per-file interval indexes, rejecting malformed rows."""
    path = root / "task_annotations.jsonl"
    result: dict[str, list[dict]] = {}
    if not path.exists():
        return result
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw)
            source = (root / str(row["path"])).resolve()
            start, end = int(row["start"]), int(row["end"])
            label = str(row.get("leaf_task") or row.get("skill") or "")
            if not source.is_file():
                raise ValueError(f"missing source file {source}")
            if start < 0 or end <= start:
                raise ValueError("invalid half-open interval")
            if not label:
                raise ValueError("missing skill/leaf_task")
            result.setdefault(str(source), []).append({
                "resolved_path": str(source), "start": start, "end": end,
                "leaf_task": label, "phase": row.get("phase") or phase_for_task(label),
            })
        except Exception as exc:
            raise ValueError(f"Invalid task annotation line {line_no}: {exc}") from exc
    return result


def _extract_segments(
    ep: PointEpisode,
    cfg: SegmentReplayConfig,
    annotations: list[dict] | None = None,
) -> list[_Segment]:
    """Cut overlapping windows while preserving raw return-to-go and exact labels."""
    n = ep.length
    if n < cfg.seg_len:
        return []
    if cfg.seg_len <= 0:
        raise ValueError("seg_len must be positive")
    point_rtg = _compute_point_rtg(ep, cfg.rtg_gamma)
    segments: list[_Segment] = []
    stride = max(1, cfg.seg_len // 2)
    resolved_path = str(Path(ep.path).resolve())
    annotations = annotations or []
    for start in range(0, n - cfg.seg_len + 1, stride):
        end = start + cfg.seg_len
        seg_type = "unlabeled"
        phase = None
        for row in annotations:
            if row.get("resolved_path") == resolved_path and int(row["start"]) <= start and int(row["end"]) >= end:
                label = str(row.get("leaf_task") or row.get("skill") or "")
                if label in TASK_TO_PHASE:
                    seg_type, phase = label, phase_for_task(label)
                else:
                    # Broad legacy labels may supply a phase, but not a leaf task.
                    phase = phase_for_task(label)
                break
        obs = ep.observations[start:end].astype(np.uint8, copy=False)
        acts = ep.actions[start:end].astype(np.int64, copy=False)
        rtg = point_rtg[start:end].copy()
        # Priority depends on the actual return signal only. Unknown / zero-return
        # segments retain a nonzero uniform chance, not an outcome-based guess.
        priority = float(np.abs(rtg).mean()) + float(cfg.priority_eps)
        segments.append(_Segment(obs, acts, rtg, ep.outcome, seg_type, priority,
                                 resolved_path, start, end, phase))
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
        self._type_counts: dict[str, int] = {"unlabeled": 0}
        self._task_annotations = _read_task_annotations(self.dataset.root)
        self._rebuild_segments()

    # ------------------------------------------------------------------
    # Segment table management
    # ------------------------------------------------------------------

    def _rebuild_segments(self) -> None:
        """Extract all micro-segments from the current demonstration set."""
        segs: list[_Segment] = []
        by_type: dict[str, list[_Segment]] = {"unlabeled": []}
        for ep in self.dataset.episodes:
            rows = self._task_annotations.get(str(Path(ep.path).resolve()), [])
            extracted = _extract_segments(ep, self.cfg, rows)
            segs.extend(extracted)
            for segment in extracted:
                by_type.setdefault(segment.seg_type, []).append(segment)
        self._segments = segs
        self._by_type = by_type
        self._type_counts = {key: len(value) for key, value in sorted(by_type.items())}
        if segs:
            # Small epsilon ensures even zero-return/unknown segments remain sampleable.
            priorities = np.asarray([max(float(s.priority), self.cfg.priority_eps) for s in segs], dtype=np.float64)
            self._priorities = priorities / priorities.sum()
        else:
            self._priorities = np.array([], dtype=np.float64)

    def sample_batch(self, batch_size: int | None = None) -> list[_Segment]:
        """Sample actual labelled tasks where available, otherwise neutral windows.

        Stratification only balances labels that came from explicit annotation
        sidecars. It never synthesizes a task from action-button frequency.
        """
        if not self._segments:
            return []
        size = int(batch_size if batch_size is not None else self.cfg.batch_size)
        if size < 1:
            raise ValueError("batch_size must be positive")
        labelled = {key: value for key, value in self._by_type.items() if key != "unlabeled" and value}
        chosen: list[_Segment] = []
        if labelled:
            # Allocate roughly equal slots across available leaf labels; remainder
            # is sampled by return-weight priority to preserve coverage.
            names = sorted(labelled)
            per = max(1, size // len(names))
            for name in names:
                pool = labelled[name]
                idx = self.rng.choice(len(pool), size=min(per, size), replace=len(pool) < per)
                chosen.extend(pool[int(i)] for i in np.asarray(idx).reshape(-1))
                if len(chosen) >= size:
                    break
        rem = size - len(chosen)
        if rem > 0:
            idx = self.rng.choice(len(self._segments), size=rem, replace=True, p=self._priorities)
            chosen.extend(self._segments[int(i)] for i in np.asarray(idx).reshape(-1))
        self.rng.shuffle(chosen)
        return chosen[:size]

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
        outcome_target = torch.full((T, B), -1.0, dtype=torch.float32, device=dev)
        bc_weight = torch.zeros((T, B), dtype=torch.float32, device=dev)

        cfg = self.cfg
        for j, seg in enumerate(segs):
            t = seg.obs.shape[0]
            obs[:t, j] = torch.from_numpy(seg.obs)
            actions[:t, j] = torch.from_numpy(seg.actions)

            rtg_t = torch.from_numpy(seg.rtg[:t]).to(dev)

            if seg.outcome in {"winning", "losing"}:
                outcome_target[:t, j] = 1.0 if seg.outcome == "winning" else 0.0

            # Human actions in a lost point are not automatically wrong. Weight
            # by the real, bounded local return only; no outcome-derived relabeling.
            w = (1.0 + cfg.rtg_coef * torch.tanh(rtg_t)).clamp(cfg.min_bc_weight, 1.0 + cfg.rtg_coef)
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

        known_outcome = outcome_target >= 0
        outcome_bce = nn.functional.binary_cross_entropy_with_logits(
            outcome_logits, outcome_target.clamp(0.0, 1.0), reduction="none"
        )
        outcome_loss = (outcome_bce * known_outcome.float()).sum() / known_outcome.float().sum().clamp_min(1.0)

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
