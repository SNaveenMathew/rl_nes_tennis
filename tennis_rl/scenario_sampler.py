"""Sampling of point-level start states for training.

Design goals (each maps to a failure mode of naive scenario training):

* **Catastrophic forgetting** - every stratum keeps a probability floor; mastered
  cells are re-sampled on purpose; a cell whose win-rate falls well below its own
  best gets a retention boost; a share of rollouts always starts from a normal
  game start ("anchor") so the natural state distribution is never dropped.
* **Overfitting to a few states** - the NES is deterministic, so a savestate plus
  a fixed action sequence replays identically.  We pick the source match first
  (uniform), then a state, and add random no-op frames before control is handed
  over.  Train/holdout is split by *source match*; holdout states are only used
  by probe workers, whose experience is never added to the replay buffer.
* **Curriculum without tunnel vision** - start states are drawn from late in a
  point first (short horizon, reward close), then earlier phases unlock as the
  cell gets solved (reverse curriculum); earlier phases stay in the mix.

Prioritisation is rank based, in the spirit of Prioritized Level Replay
(Jiang et al., ICML 2021), with a learning-progress term (Graves et al., 2017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .scenarios import PHASES, ScenarioCell, StateRecord, build_catalog

# Hardest-last ordering used for unlocking: late (easy) -> mid -> start (full point).
_PHASE_ORDER = ("late", "mid", "start")


@dataclass
class SamplerConfig:
    anchor_fraction: float = 0.15
    mix_uniform: float = 0.40
    mix_priority: float = 0.40
    mix_natural: float = 0.20
    retention_boost: float = 3.0  # extra weight for strata with forgotten cells
    stratum_floor: float = 0.5  # total probability mass reserved as per-stratum floor
    ema_fast: float = 0.2
    ema_slow: float = 0.03
    priority_temperature: float = 0.5
    staleness_mix: float = 0.2
    learning_progress_coef: float = 1.0
    mastery_winrate: float = 0.8
    mastery_min_episodes: int = 20
    forget_drop: float = 0.15
    forget_min_episodes: int = 10
    promote_winrate: float = 0.45
    promote_min_episodes: int = 15
    zpd_threshold: float = 0.20
    zpd_late_bias: float = 0.65
    loss_state_fraction: float = 0.25
    newest_phase_fraction: float = 0.5
    noop_min: int = 0
    noop_max: int = 30


@dataclass
class Assignment:
    kind: str  # "scenario", "anchor" or "probe"
    cell_id: str | None = None
    stratum: str | None = None
    phase: str | None = None
    state: StateRecord | None = None
    noop_frames: int = 0

    @property
    def demo_outcome(self) -> str:
        return self.state.demo_outcome if self.state else "none"


class ScenarioLibrary:
    """Index of harvested states: cell -> split -> phase -> outcome-group -> match."""

    def __init__(self, records: Iterable[StateRecord]):
        self.records = list(records)
        self._idx: dict[str, dict[str, dict[str, dict[str, dict[str, list[StateRecord]]]]]] = {}
        for r in self.records:
            group = "loss" if r.demo_outcome == "loss" else "win"
            (
                self._idx.setdefault(r.cell_id, {})
                .setdefault(r.split, {})
                .setdefault(r.phase, {})
                .setdefault(group, {})
                .setdefault(r.match_id, [])
                .append(r)
            )

    def count(self, cell_id: str, split: str = "train") -> int:
        total = 0
        for groups in self._idx.get(cell_id, {}).get(split, {}).values():
            for matches in groups.values():
                total += sum(len(v) for v in matches.values())
        return total

    def phases(self, cell_id: str, split: str = "train") -> set[str]:
        return {p for p, g in self._idx.get(cell_id, {}).get(split, {}).items() if any(g.values())}

    def pick(
        self,
        cell_id: str,
        split: str,
        phase: str,
        want_loss: bool,
        rng: np.random.Generator,
    ) -> StateRecord | None:
        groups = self._idx.get(cell_id, {}).get(split, {}).get(phase, {})
        order = ("loss", "win") if want_loss else ("win", "loss")
        for g in order:
            matches = groups.get(g)
            if matches:
                keys = sorted(matches)
                match = keys[int(rng.integers(len(keys)))]  # match first -> no long-match dominance
                states = matches[match]
                return states[int(rng.integers(len(states)))]
        return None

    def coverage(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for cid in self._idx:
            out[cid] = {s: self.count(cid, s) for s in ("train", "holdout")}
        return out


class _CellStats:
    __slots__ = (
        "n", "fast", "slow", "best_slow", "mastered_once", "forgotten", "stage",
        "last_tick", "phase_n", "phase_ema", "n_since_mastery",
    )

    def __init__(self) -> None:
        self.n = 0
        self.fast = 0.5
        self.slow = 0.5
        self.best_slow = 0.0
        self.mastered_once = False
        self.forgotten = False
        self.stage = 0
        self.last_tick = 0
        self.phase_n = {p: 0 for p in _PHASE_ORDER}
        self.phase_ema = {p: 0.0 for p in _PHASE_ORDER}
        self.n_since_mastery = 0


class ScenarioSampler:
    def __init__(
        self,
        library: ScenarioLibrary,
        cfg: SamplerConfig | None = None,
        cells: Iterable[ScenarioCell] | None = None,
        natural_counts: Mapping[str, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or SamplerConfig()
        self.library = library
        self.rng = np.random.default_rng(seed)
        self.cells = {c.cell_id: c for c in (cells if cells is not None else build_catalog())}
        self.stats = {cid: _CellStats() for cid in self.cells}
        self.natural = dict(natural_counts or {})
        self.tick = 0
        self.probe_stats: dict[str, list[int]] = {}
        self._probe_cursor = 0
        self.refresh()

    # -- availability ---------------------------------------------------- #
    def refresh(self) -> None:
        """Recompute which cells/strata have training states (call after harvesting)."""
        self.available = [cid for cid in self.cells if self.library.count(cid, "train") > 0]
        self.strata: dict[str, list[str]] = {}
        for cid in self.available:
            self.strata.setdefault(self.cells[cid].stratum, []).append(cid)

    # -- statistics ------------------------------------------------------ #
    def update(self, cell_id: str, won: bool, phase: str = "start", source: str = "scenario") -> None:
        """Record one finished point.  ``source``: scenario | anchor | probe."""
        if cell_id not in self.cells:
            return
        if source == "probe":
            n_w = self.probe_stats.setdefault(cell_id, [0, 0])
            n_w[0] += 1
            n_w[1] += int(won)
            return
        c, cfg = self.stats[cell_id], self.cfg
        x = 1.0 if won else 0.0
        c.n += 1
        a_f = max(cfg.ema_fast, 1.0 / (c.n + 1))
        a_s = max(cfg.ema_slow, 1.0 / (c.n + 1))
        c.fast += a_f * (x - c.fast)
        c.slow += a_s * (x - c.slow)

        if c.mastered_once:
            c.n_since_mastery += 1
            if c.slow > c.best_slow:
                c.best_slow = c.slow
            drop = c.best_slow - c.slow
            if not c.forgotten and drop > cfg.forget_drop and c.n_since_mastery >= cfg.forget_min_episodes:
                c.forgotten = True
            elif c.forgotten and drop < cfg.forget_drop / 2:
                c.forgotten = False
        elif c.n >= cfg.mastery_min_episodes and c.slow >= cfg.mastery_winrate:
            c.mastered_once = True
            c.best_slow = c.slow
            c.n_since_mastery = 0

        if source == "scenario" and phase in c.phase_n:
            c.phase_n[phase] += 1
            k = c.phase_n[phase]
            c.phase_ema[phase] += max(cfg.ema_slow * 2, 1.0 / k) * (x - c.phase_ema[phase])
            self._maybe_promote(c)

    def _maybe_promote(self, c: _CellStats) -> None:
        cfg = self.cfg
        newest = _PHASE_ORDER[c.stage]
        if (
            c.stage < len(_PHASE_ORDER) - 1
            and c.phase_n[newest] >= cfg.promote_min_episodes
            and c.phase_ema[newest] >= cfg.promote_winrate
        ):
            c.stage += 1

    # -- probabilities --------------------------------------------------- #
    def stratum_probs(self) -> dict[str, float]:
        names = list(self.strata)
        n = len(names)
        if n == 0:
            return {}
        cfg = self.cfg
        uniform = np.full(n, 1.0 / n)

        scores, stale, natural, forgotten = (np.zeros(n) for _ in range(4))
        for i, s in enumerate(names):
            cells = self.strata[s]
            seen = [self.stats[c] for c in cells if self.stats[c].n > 0]
            if not seen:
                scores[i] = 2.0  # optimistic: unexplored strata rank first
            else:
                scores[i] = float(np.mean([(1 - c.slow) + cfg.learning_progress_coef * abs(c.fast - c.slow) for c in seen]))
            stale[i] = float(self.tick - min(self.stats[c].last_tick for c in cells))
            natural[i] = float(sum(self.natural.get(c, 0) for c in cells))
            forgotten[i] = float(np.mean([1.0 if self.stats[c].forgotten else 0.0 for c in cells]))

        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        prio = ranks ** (-1.0 / cfg.priority_temperature)
        prio /= prio.sum()
        if stale.sum() > 0:
            prio = (1 - cfg.staleness_mix) * prio + cfg.staleness_mix * stale / stale.sum()
        natural = natural / natural.sum() if natural.sum() > 0 else uniform

        mix = cfg.mix_uniform + cfg.mix_priority + cfg.mix_natural
        probs = (cfg.mix_uniform * uniform + cfg.mix_priority * prio + cfg.mix_natural * natural) / mix
        # Retention is a multiplier, not a mixture component: a single forgotten
        # stratum must stay boosted relative to the rest (normalising a
        # standalone retention distribution would hide exactly that case).
        probs = probs * (1.0 + cfg.retention_boost * forgotten)
        probs /= probs.sum()
        floor = cfg.stratum_floor / n
        probs = np.maximum(probs, floor)
        probs /= probs.sum()
        return dict(zip(names, probs.tolist()))

    # -- sampling -------------------------------------------------------- #
    def sample(self) -> Assignment:
        cfg = self.cfg
        self.tick += 1
        if not self.strata or self.rng.random() < cfg.anchor_fraction:
            return Assignment(kind="anchor")
        probs = self.stratum_probs()
        names = list(probs)
        stratum = names[int(self.rng.choice(len(names), p=np.fromiter(probs.values(), float)))]
        cells = self.strata[stratum]
        w = np.array(
            [0.5 + (1 - self.stats[c].slow) + (2.0 if self.stats[c].forgotten else 0.0) for c in cells]
        )
        cell_id = cells[int(self.rng.choice(len(cells), p=w / w.sum()))]
        self.stats[cell_id].last_tick = self.tick
        phase = self._pick_phase(cell_id)
        want_loss = self.rng.random() < cfg.loss_state_fraction
        state = self.library.pick(cell_id, "train", phase, want_loss, self.rng)
        noop = int(self.rng.integers(cfg.noop_min, cfg.noop_max + 1))
        return Assignment("scenario", cell_id, stratum, phase, state, noop)

    def _pick_phase(self, cell_id: str) -> str:
        have = self.library.phases(cell_id, "train")
        stage = self.stats[cell_id].stage
        unlocked = [p for p in _PHASE_ORDER[: stage + 1] if p in have]
        if not unlocked:  # nothing unlocked has states yet -> use whatever exists, easiest first
            return next(p for p in _PHASE_ORDER if p in have)
        if len(unlocked) == 1:
            return unlocked[0]

        # Zone of Proximal Development (ZPD):
        # When an agent struggles on a cell (winrate < zpd_threshold) after initial tries,
        # bias towards "late" phase to provide immediate, dense reward signals for basic contact.
        stat = self.stats[cell_id]
        if "late" in unlocked and stat.n >= 5 and stat.slow < getattr(self.cfg, "zpd_threshold", 0.20):
            if self.rng.random() < getattr(self.cfg, "zpd_late_bias", 0.65):
                return "late"

        newest, older = unlocked[-1], unlocked[:-1]
        if self.rng.random() < self.cfg.newest_phase_fraction:
            return newest
        return older[int(self.rng.integers(len(older)))]

    def sample_probe(self) -> Assignment:
        """Round-robin over cells that have holdout states (never trained on)."""
        cands = [c for c in self.cells if self.library.count(c, "holdout") > 0]
        if not cands:
            return Assignment(kind="anchor")
        cell_id = cands[self._probe_cursor % len(cands)]
        self._probe_cursor += 1
        phases = self.library.phases(cell_id, "holdout")
        phase = "start" if "start" in phases else sorted(phases)[0]
        state = self.library.pick(cell_id, "holdout", phase, False, self.rng)
        noop = int(self.rng.integers(self.cfg.noop_min, self.cfg.noop_max + 1))
        return Assignment("probe", cell_id, self.cells[cell_id].stratum, phase, state, noop)

    # -- reporting ------------------------------------------------------- #
    def forgotten_cells(self) -> list[str]:
        return [c for c, s in self.stats.items() if s.forgotten]

    def report(self) -> dict:
        return {
            "stratum_probs": self.stratum_probs(),
            "cells": {
                cid: {
                    "n": s.n,
                    "win_slow": round(s.slow, 3),
                    "win_fast": round(s.fast, 3),
                    "stage": s.stage,
                    "mastered": s.mastered_once,
                    "forgotten": s.forgotten,
                }
                for cid, s in self.stats.items()
                if s.n > 0
            },
            "probe": {
                cid: {"n": n, "win": round(w / max(n, 1), 3)} for cid, (n, w) in self.probe_stats.items()
            },
        }
