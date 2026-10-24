"""Validated task/state annotation loading for hierarchical NES Tennis training.

Task annotations live in ``task_annotations.jsonl`` relative to the demonstration
root. Each row labels a half-open interval of a recorded point. ``skill`` may be
an exact leaf task (action supervision is permitted) or an older broad label
(phase supervision only). Action frequency and point outcome are never used to
invent a task label.

State annotations live in ``state_annotations.jsonl`` with optional normalized
coordinates, velocity, time-to-contact and event labels. Missing labels are
masked in losses rather than treated as negative examples.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hierarchy_schema import EVENT_NAMES, TASK_TO_PHASE, phase_for_task


@dataclass(frozen=True)
class AnnotatedTaskSegment:
    path: Path
    start: int
    end: int
    skill: str
    phase: str | None
    leaf_task: str | None
    observations: np.ndarray
    actions: np.ndarray
    pulse_frames: np.ndarray | None = None
    manager_duration_index: int = -1

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class StateAnnotation:
    observation: np.ndarray
    ball_xy: tuple[float, float] | None
    player_xy: tuple[float, float] | None
    ball_velocity_xy: tuple[float, float] | None
    time_to_contact: float | None
    time_to_contact_bin: int | None
    events: tuple[float, ...]


def _resolve_under(root: Path, raw_path: str) -> Path:
    candidate = (root / raw_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"Annotation path escapes demonstration root: {raw_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.uint8)
        actions = np.asarray(data["actions"], dtype=np.int64)
        pulse_frames = np.asarray(data["pulse_frames"], dtype=np.int16) if "pulse_frames" in data else np.full(actions.shape, -1, dtype=np.int16)
    if observations.ndim != 4 or observations.shape[0] != actions.shape[0]:
        raise ValueError(f"Invalid observation/action arrays in {path}")
    if observations.shape[1] != 4:
        raise ValueError(f"Expected four stacked frames in {path}, got {observations.shape}")
    if np.any((actions < 0) | (actions >= 27)):
        raise ValueError(f"Invalid action ID in {path}")
    if pulse_frames.shape != actions.shape:
        raise ValueError(f"Invalid pulse_frames array in {path}")
    return observations, actions, pulse_frames


def _xy(raw: Any, field: str, line_no: int) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field} at line {line_no} must be [x,y]")
    x, y = float(raw[0]), float(raw[1])
    if not (np.isfinite(x) and np.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"{field} coordinates at line {line_no} must be finite and in [0,1]")
    return x, y


def _optional_nonnegative(raw: Any, field: str, line_no: int) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{field} at line {line_no} must be finite and non-negative")
    return value


class TaskAnnotationDataset:
    """Load and sample only explicit labels; broad labels never supervise leaf actions."""

    def __init__(
        self, root: str | Path, skills: tuple[str, ...] | list[str],
        durations: tuple[int, ...] | list[int],
        button_pulse_durations: tuple[int, ...] | list[int] | None = None,
        split: str = "all", holdout_fraction: float = 0.2,
    ):
        self.root = Path(root).resolve()
        self.skills = tuple(skills)
        self.durations = tuple(int(x) for x in durations)
        self.button_pulse_durations = tuple(int(x) for x in (button_pulse_durations or durations))
        if split not in ("all", "train", "holdout"):
            raise ValueError("split must be all, train, or holdout")
        if not (0.0 <= float(holdout_fraction) < 0.5):
            raise ValueError("holdout_fraction must be in [0, 0.5)")
        self.split = split
        self.holdout_fraction = float(holdout_fraction)
        self.segments: list[AnnotatedTaskSegment] = []
        self.state_annotations: list[StateAnnotation] = []
        self.refresh()

    def _included_path(self, path: Path) -> bool:
        if self.split == "all":
            return True
        # Match identifiers are written by record.py. Legacy demonstrations
        # without one fall back to the point's parent/stem and are reported by
        # dataset stats because true match-safe splitting is then impossible.
        try:
            from .scenarios import split_for_match
            with np.load(path, allow_pickle=False) as data:
                if "match_id" in data:
                    raw = np.asarray(data["match_id"]).item()
                    match_id = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                else:
                    match_id = path.parent.name if path.parent.name.startswith("match_") else path.stem
            assigned = split_for_match(match_id, self.holdout_fraction)
            return assigned == self.split
        except (OSError, ValueError, KeyError):
            return False

    def refresh(self) -> None:
        self.segments.clear()
        self.state_annotations.clear()
        task_file = self.root / "task_annotations.jsonl"
        if task_file.exists():
            for line_no, line in enumerate(task_file.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    row = json.loads(line)
                    path = _resolve_under(self.root, str(row["path"]))
                    observations, actions, pulses = _load_npz(path)
                    if not self._included_path(path):
                        continue
                    start, end = int(row["start"]), int(row["end"])
                    # Prefer leaf_task; skill is retained as a compatibility field.
                    skill = str(row.get("leaf_task", row.get("skill", ""))).strip()
                    if not skill:
                        raise ValueError("expected an explicit 'leaf_task' or 'skill'")
                    if start < 0 or end <= start or end > len(actions):
                        raise ValueError(f"Invalid [start,end) interval at line {line_no}")
                    phase = phase_for_task(skill)
                    if phase is None:
                        raise ValueError(f"Unknown task label {skill!r}; update hierarchy_schema.py")
                    explicit_phase = row.get("phase")
                    if explicit_phase is not None and str(explicit_phase) != phase:
                        raise ValueError(f"phase={explicit_phase!r} conflicts with task {skill!r} (phase {phase!r})")
                    leaf = skill if skill in TASK_TO_PHASE and skill in self.skills else None
                    raw_manager_duration = row.get("manager_duration")
                    if raw_manager_duration is None:
                        manager_duration_index = -1
                    else:
                        try:
                            duration_value = int(raw_manager_duration)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"manager_duration at line {line_no} must be one of {self.durations}") from exc
                        if duration_value not in self.durations:
                            raise ValueError(f"manager_duration at line {line_no} must be one of {self.durations}")
                        manager_duration_index = self.durations.index(duration_value)
                    # A leaf annotation must correspond to an expert configured in this run.
                    if skill in TASK_TO_PHASE and skill not in self.skills:
                        raise ValueError(f"Leaf task {skill!r} is not in configured skills {self.skills}")
                    self.segments.append(AnnotatedTaskSegment(
                        path=path, start=start, end=end, skill=skill, phase=phase, leaf_task=leaf,
                        observations=observations[start:end], actions=actions[start:end], pulse_frames=pulses[start:end],
                        manager_duration_index=manager_duration_index,
                    ))
                except Exception as exc:
                    raise ValueError(f"Invalid task annotation line {line_no}: {exc}") from exc

        state_file = self.root / "state_annotations.jsonl"
        if state_file.exists():
            for line_no, line in enumerate(state_file.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    row = json.loads(line)
                    path = _resolve_under(self.root, str(row["path"]))
                    observations, _, _ = _load_npz(path)
                    if not self._included_path(path):
                        continue
                    step = int(row["step"])
                    if step < 0 or step >= len(observations):
                        raise ValueError(f"step out of range at line {line_no}")
                    ball = _xy(row.get("ball_xy"), "ball_xy", line_no)
                    player = _xy(row.get("player_xy"), "player_xy", line_no)
                    velocity = row.get("ball_velocity_xy", row.get("ball_velocity"))
                    velocity_xy = None
                    if velocity is not None:
                        # Velocity is signed; unlike position, do not normalize/clamp it to [0,1].
                        if not isinstance(velocity, (list, tuple)) or len(velocity) != 2:
                            raise ValueError(f"ball_velocity_xy at line {line_no} must be [vx,vy]")
                        vx, vy = float(velocity[0]), float(velocity[1])
                        if not (np.isfinite(vx) and np.isfinite(vy)):
                            raise ValueError(f"ball_velocity_xy at line {line_no} must be finite")
                        velocity_xy = (vx, vy)
                    ttc = _optional_nonnegative(row.get("time_to_contact"), "time_to_contact", line_no)
                    ttc_bin_raw = row.get("time_to_contact_bin")
                    if ttc_bin_raw is None:
                        ttc_bin = None
                    elif isinstance(ttc_bin_raw, int) and 0 <= ttc_bin_raw <= 2:
                        ttc_bin = int(ttc_bin_raw)
                    else:
                        from .hierarchy_schema import TIME_TO_CONTACT_BINS
                        try:
                            ttc_bin = TIME_TO_CONTACT_BINS.index(str(ttc_bin_raw))
                        except ValueError as exc:
                            raise ValueError(f"time_to_contact_bin at line {line_no} must be urgent/soon/not_imminent or 0..2") from exc
                    raw_events = row.get("events", {}) or {}
                    if not isinstance(raw_events, dict):
                        raise ValueError(f"events at line {line_no} must be an object")
                    unknown = set(raw_events) - set(EVENT_NAMES)
                    if unknown:
                        raise ValueError(f"unknown events at line {line_no}: {sorted(unknown)}")
                    event_values: list[float] = []
                    for name in EVENT_NAMES:
                        value = raw_events.get(name, -1)
                        if value not in (-1, 0, 1, None, False, True):
                            raise ValueError(f"{name} at line {line_no} must be 0, 1 or omitted")
                        event_values.append(-1.0 if value is None else float(value))
                    self.state_annotations.append(StateAnnotation(
                        observations[step].copy(), ball, player, velocity_xy, ttc, ttc_bin, tuple(event_values)
                    ))
                except Exception as exc:
                    raise ValueError(f"Invalid state annotation line {line_no}: {exc}") from exc

    def __len__(self) -> int:
        return len(self.segments)

    def summary(self) -> dict[str, Any]:
        phases = Counter(seg.phase for seg in self.segments if seg.phase)
        leaves = Counter(seg.leaf_task for seg in self.segments if seg.leaf_task)
        return {
            "task_segments": len(self.segments),
            "state_frames": len(self.state_annotations),
            "phase_segments": dict(sorted(phases.items())),
            "leaf_task_segments": dict(sorted(leaves.items())),
            "unlabeled_leaf_segments": sum(seg.leaf_task is None for seg in self.segments),
        }

    def sample_training_batch(
        self, batch_size: int, rng: np.random.Generator,
        selected_skills: set[str] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Sample leaf actions balanced by task and gate labels balanced by phase.

        Leaf imitation samples are drawn only from explicitly labeled segments.
        Broad/legacy phase labels supervise the manager but never fabricate
        actions for an expert. Sampling is balanced across available tasks to
        prevent common baseline segments from dominating rare serve/receive data.
        """
        if not self.segments and not self.state_annotations:
            return None
        n = max(1, int(batch_size))
        eligible_leaves = [
            seg for seg in self.segments if seg.leaf_task is not None
            and (selected_skills is None or seg.leaf_task in selected_skills)
        ]
        by_leaf: dict[str, list[AnnotatedTaskSegment]] = {}
        for segment in eligible_leaves:
            by_leaf.setdefault(str(segment.leaf_task), []).append(segment)
        by_phase: dict[str, list[AnnotatedTaskSegment]] = {}
        for segment in self.segments:
            if segment.phase:
                by_phase.setdefault(str(segment.phase), []).append(segment)

        expert_obs: list[np.ndarray] = []
        expert_actions: list[int] = []
        expert_skills: list[int] = []
        expert_pulse_duration_ids: list[int] = []
        if by_leaf:
            names = sorted(by_leaf)
            for task in (names[int(rng.integers(len(names)))] for _ in range(n)):
                segment = by_leaf[task][int(rng.integers(len(by_leaf[task])))]
                step = int(rng.integers(segment.length))
                action = int(segment.actions[step])
                expert_obs.append(segment.observations[step])
                expert_actions.append(action)
                expert_skills.append(self.skills.index(task))
                raw_pulse = -1 if segment.pulse_frames is None else int(segment.pulse_frames[step])
                if raw_pulse > 0 and action % 3 != 0:
                    pulse_choices = self.button_pulse_durations
                    expert_pulse_duration_ids.append(min(range(len(pulse_choices)), key=lambda j: abs(pulse_choices[j] - raw_pulse)))
                else:
                    expert_pulse_duration_ids.append(-1)

        manager_obs: list[np.ndarray] = []
        manager_skills: list[int] = []
        manager_phases: list[int] = []
        manager_durations: list[int] = []
        if by_phase:
            from .hierarchy_schema import PHASES
            phases = sorted(by_phase)
            for phase in (phases[int(rng.integers(len(phases)))] for _ in range(n)):
                segment = by_phase[phase][int(rng.integers(len(by_phase[phase])))]
                manager_obs.append(segment.observations[0])
                manager_skills.append(self.skills.index(segment.leaf_task) if segment.leaf_task in self.skills else -1)
                manager_phases.append(PHASES.index(phase))
                # Skill-segment length is NOT the same thing as an option's
                # chosen execution duration. Train this head only if explicitly
                # annotated; otherwise PPO learns duration from online returns.
                manager_durations.append(int(segment.manager_duration_index))

        sample_shape = None
        if self.segments:
            sample_shape = tuple(self.segments[0].observations.shape[1:])
        elif self.state_annotations:
            sample_shape = tuple(self.state_annotations[0].observation.shape)
        else:
            sample_shape = (4, 240, 256)
        empty_obs = lambda: torch.empty((0, *sample_shape), dtype=torch.uint8)
        batch: dict[str, torch.Tensor] = {
            "demo_obs": torch.from_numpy(np.stack(expert_obs).astype(np.uint8, copy=False)) if expert_obs else empty_obs(),
            "demo_skill": torch.tensor(expert_skills, dtype=torch.long),
            "demo_action": torch.tensor(expert_actions, dtype=torch.long),
            "demo_pulse_duration_ids": torch.tensor(expert_pulse_duration_ids, dtype=torch.long),
            "demo_manager_obs": torch.from_numpy(np.stack(manager_obs).astype(np.uint8, copy=False)) if manager_obs else empty_obs(),
            "demo_manager_skill": torch.tensor(manager_skills, dtype=torch.long),
            "demo_manager_phase": torch.tensor(manager_phases, dtype=torch.long),
            "demo_manager_duration": torch.tensor(manager_durations, dtype=torch.long),
        }
        if self.state_annotations:
            state_ids = rng.integers(0, len(self.state_annotations), size=n)
            states = [self.state_annotations[int(i)] for i in state_ids]
            batch["state_obs"] = torch.from_numpy(np.stack([x.observation for x in states]).astype(np.uint8, copy=False))
            def xy_array(items, field):
                out = np.full((len(items), 2), np.nan, dtype=np.float32)
                for i, item in enumerate(items):
                    value = getattr(item, field)
                    if value is not None:
                        out[i] = value
                return out
            batch["state_ball_xy"] = torch.from_numpy(xy_array(states, "ball_xy"))
            batch["state_player_xy"] = torch.from_numpy(xy_array(states, "player_xy"))
            batch["state_ball_velocity_xy"] = torch.from_numpy(xy_array(states, "ball_velocity_xy"))
            batch["state_time_to_contact"] = torch.tensor([np.nan if x.time_to_contact is None else x.time_to_contact for x in states], dtype=torch.float32)
            batch["state_time_to_contact_bin"] = torch.tensor([-1 if x.time_to_contact_bin is None else x.time_to_contact_bin for x in states], dtype=torch.long)
            batch["state_events"] = torch.from_numpy(np.asarray([x.events for x in states], dtype=np.float32))
        return batch
