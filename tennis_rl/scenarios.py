"""Point-level scenario catalog for NES Tennis.

A *scenario* is a way a single point can begin.  Everything here is derived from
things the bridge can already observe or from the rules of tennis:

* ``role``     - who serves: the agent ("serve") or the CPU ("return").  NES Tennis
                 alternates service by game, the agent serving game 1.
* ``(p, o)``   - point score inside the game, 0..3 = 0/15/30/40, 4 = advantage
                 (the bridge reads these from RAM 0x58 / 0x59).
* ``side``     - which service court the point is played from.  By the rules of
                 tennis it is "deuce" when the points played so far are even and
                 "ad" when odd; with the 0..4 score encoding that is parity of p+o.
* ``pressure`` - what the point means for the score (opening, build-up, game
                 point for/against, deuce, advantage for/against).

That gives 18 reachable score states x 2 roles = 36 *cells*; cells that share
(role, side, pressure) form 20 *strata*.  Nothing below invents ball/contact
labels; the repo has no validated decoder for those.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROLES = ("serve", "return")
SIDES = ("deuce", "ad")
PRESSURES = (
    "opening",
    "build_up",
    "game_point_for",
    "game_point_against",
    "deuce",
    "adv_for",
    "adv_against",
)
# Where inside a point a scenario starts (reverse-curriculum axis).
PHASES = ("late", "mid", "start")
LENGTH_BUCKETS = ("short", "medium", "long")

_POINT_NAMES = ("0", "15", "30", "40", "Ad")


# --------------------------------------------------------------------------- #
# Score machine
# --------------------------------------------------------------------------- #
def reachable_scores() -> tuple[tuple[int, int], ...]:
    """Every (player, opponent) score at which a point can start (18 states)."""
    base = tuple((p, o) for p in range(4) for o in range(4))
    return base + ((4, 3), (3, 4))


def advance_score(p: int, o: int, player_won_point: bool) -> tuple[int, int, bool, bool]:
    """Return (p', o', game_over, player_won_game) after one point."""
    if player_won_point:
        if p == 4 or (p == 3 and o < 3):
            return 0, 0, True, True
        if p == 3 and o == 3:
            return 4, 3, False, False
        if o == 4:
            return 3, 3, False, False
        return p + 1, o, False, False
    if o == 4 or (o == 3 and p < 3):
        return 0, 0, True, False
    if o == 3 and p == 3:
        return 3, 4, False, False
    if p == 4:
        return 3, 3, False, False
    return p, o + 1, False, False


def court_side(p: int, o: int) -> str:
    return "deuce" if (p + o) % 2 == 0 else "ad"


def pressure(p: int, o: int) -> str:
    if (p, o) == (0, 0):
        return "opening"
    if (p, o) == (3, 3):
        return "deuce"
    if (p, o) == (4, 3):
        return "adv_for"
    if (p, o) == (3, 4):
        return "adv_against"
    if p == 3 and o < 3:
        return "game_point_for"
    if o == 3 and p < 3:
        return "game_point_against"
    return "build_up"


def score_label(p: int, o: int) -> str:
    if (p, o) == (3, 3):
        return "deuce"
    if (p, o) == (4, 3):
        return "Ad-in"
    if (p, o) == (3, 4):
        return "Ad-out"
    return f"{_POINT_NAMES[p]}-{_POINT_NAMES[o]}"


def role_for_game(game_number: int) -> str:
    """Game 1 is served by the agent; service alternates every game."""
    return "serve" if game_number % 2 == 1 else "return"


# --------------------------------------------------------------------------- #
# Cells and strata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioCell:
    cell_id: str
    stratum: str
    role: str
    p: int
    o: int
    side: str
    pressure: str
    label: str

    def to_json(self) -> dict:
        return asdict(self)


def cell_id(role: str, p: int, o: int) -> str:
    return f"{role}|{p}-{o}"


def stratum_id(role: str, p: int, o: int) -> str:
    return f"{role}|{court_side(p, o)}|{pressure(p, o)}"


def make_cell(role: str, p: int, o: int) -> ScenarioCell:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if (p, o) not in reachable_scores():
        raise ValueError(f"({p}, {o}) is not a reachable point-start score")
    return ScenarioCell(
        cell_id=cell_id(role, p, o),
        stratum=stratum_id(role, p, o),
        role=role,
        p=p,
        o=o,
        side=court_side(p, o),
        pressure=pressure(p, o),
        label=f"{role} {score_label(p, o)} ({court_side(p, o)} court)",
    )


def build_catalog() -> list[ScenarioCell]:
    return [make_cell(role, p, o) for role in ROLES for (p, o) in reachable_scores()]


def strata(cells: Iterable[ScenarioCell] | None = None) -> dict[str, list[ScenarioCell]]:
    out: dict[str, list[ScenarioCell]] = {}
    for c in cells if cells is not None else build_catalog():
        out.setdefault(c.stratum, []).append(c)
    return out


# --------------------------------------------------------------------------- #
# Phases / buckets
# --------------------------------------------------------------------------- #
def state_phase(step_in_point: int, steps_to_end: int | None, late_steps: int = 45) -> str:
    """Phase of a harvested start state inside its point.

    ``steps_to_end`` is only known once the point has finished; states harvested
    in a point that never completed carry ``None`` and are treated as "mid".
    """
    if step_in_point == 0:
        return "start"
    if steps_to_end is not None and steps_to_end <= late_steps:
        return "late"
    return "mid"


def length_bucket(steps: int, short_max: int = 20, medium_max: int = 75) -> str:
    """Coarse proxy for how the point ended (quick finish vs. long rally)."""
    if steps <= short_max:
        return "short"
    if steps <= medium_max:
        return "medium"
    return "long"


def window_phase(window_start: int, window_len: int, point_len: int, late_steps: int = 45) -> str:
    """Phase label for a demonstration window inside a recorded point."""
    if window_start <= 0:
        return "start"
    if window_start + window_len >= point_len - late_steps // 2:
        return "late"
    return "mid"


# --------------------------------------------------------------------------- #
# Harvested start states
# --------------------------------------------------------------------------- #
@dataclass
class StateRecord:
    """One savestate on disk that a point-level episode can start from."""

    path: str
    cell_id: str
    phase: str = "start"
    kind: str = "start"  # how it was harvested: "start" (serve-ready) or "mid" (in-rally)
    match_id: str = ""
    level: int = 1
    point_index: int = 0
    step_in_point: int = 0
    steps_to_end: int | None = None
    demo_outcome: str = "none"  # "win" / "loss" = how the human's point ended, "none" = unknown
    split: str = "train"  # "train" or "holdout"

    def to_json(self) -> dict:
        return asdict(self)


def split_for_match(match_id: str, holdout_fraction: float = 0.2) -> str:
    """Deterministic train/holdout split keyed on the *source match*.

    Splitting by match (not by state) keeps neighbouring states of one point out
    of both sides, so holdout performance measures generalisation.
    """
    h = zlib.crc32(match_id.encode("utf-8")) % 10_000 / 10_000.0
    return "holdout" if h < holdout_fraction else "train"


def load_manifest(path: str | Path, base_dir: str | Path | None = None) -> list[StateRecord]:
    """Read ``manifest.jsonl`` written by the state harvester.

    Lines are either full state records or ``{"finalize": {...}}`` patches that
    fill in ``steps_to_end`` / ``demo_outcome`` once a point has finished.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: dict[str, StateRecord] = {}
    patches: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "finalize" in row:
            patches.append(row["finalize"])
        else:
            try:
                rec = StateRecord(**row)
            except TypeError:
                continue
            if base_dir is not None and not Path(rec.path).is_absolute():
                rec.path = str(Path(base_dir) / rec.path)
            records[rec.path] = rec
    for patch in patches:
        for rec in records.values():
            if rec.match_id == patch.get("match_id") and rec.point_index == patch.get("point_index"):
                total = int(patch["point_steps"])
                rec.steps_to_end = max(0, total - rec.step_in_point)
                won = patch.get("won")
                rec.demo_outcome = "none" if won is None else ("win" if won else "loss")
    late = int(patches[0].get("late_steps", 45)) if patches else 45
    for rec in records.values():
        rec.phase = state_phase(0 if rec.kind == "start" else max(1, rec.step_in_point), rec.steps_to_end, late)
    return list(records.values())


def write_catalog(path: str | Path, cells: Sequence[ScenarioCell] | None = None) -> dict:
    cells = list(cells) if cells is not None else build_catalog()
    payload = {
        "axes": {
            "roles": ROLES,
            "sides": SIDES,
            "pressures": PRESSURES,
            "phases": PHASES,
            "length_buckets": LENGTH_BUCKETS,
        },
        "counts": {"cells": len(cells), "strata": len(strata(cells))},
        "cells": [c.to_json() for c in cells],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
