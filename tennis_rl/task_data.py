"""Explicit task/state annotation readers for skill-specific supervision.

No skill, ball-contact, or ball-position labels are inferred from action frequency
or image centroids. Annotation files are optional sidecars under the demonstration
root and only the annotations actually supplied by a user are trained.

Task annotation JSONL, one segment per row:
  {"path":"level_1/winning/match_x/point_0.npz",
   "start":0,"end":12,"skill":"serve"}

State annotation JSONL, one frame per row:
  {"path":"level_1/winning/match_x/point_0.npz","step":7,
   "ball_xy":[0.62,0.31],"player_xy":[0.44,0.83],
   "events":{"serve_ready":0,"ball_in_play":1,"contact_window":0,"point_resolved":0}}
Coordinates are normalized to [0, 1]. Missing targets are masked, never guessed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

EVENT_NAMES = ("serve_ready", "ball_in_play", "contact_window", "point_resolved")


@dataclass(frozen=True)
class AnnotatedTaskSegment:
    path: Path
    start: int
    end: int
    skill: str
    observations: np.ndarray
    actions: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class StateAnnotation:
    observation: np.ndarray
    ball_xy: tuple[float, float] | None
    player_xy: tuple[float, float] | None
    events: tuple[float, ...]


def _resolve_under(root: Path, raw_path: str) -> Path:
    candidate = (root / raw_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"Annotation path escapes demonstration root: {raw_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.uint8)
        actions = np.asarray(data["actions"], dtype=np.int64)
    if observations.ndim != 4 or observations.shape[0] != actions.shape[0]:
        raise ValueError(f"Invalid observation/action arrays in {path}")
    if observations.shape[1] != 4:
        raise ValueError(f"Expected four stacked frames in {path}, got {observations.shape}")
    if np.any((actions < 0) | (actions >= 27)):
        raise ValueError(f"Invalid action ID in {path}")
    return observations, actions


def _xy(raw: Any, field: str, line_no: int) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field} at line {line_no} must be [x,y]")
    x, y = float(raw[0]), float(raw[1])
    if not (np.isfinite(x) and np.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"{field} coordinates at line {line_no} must be finite and in [0,1]")
    return x, y


class TaskAnnotationDataset:
    """Samples only explicitly annotated task segments from recorded demos."""

    def __init__(self, root: str | Path, skills: tuple[str, ...] | list[str], durations: tuple[int, ...] | list[int]):
        self.root = Path(root)
        self.skills = tuple(skills)
        self.durations = tuple(int(x) for x in durations)
        self.segments: list[AnnotatedTaskSegment] = []
        self.state_annotations: list[StateAnnotation] = []
        self.refresh()

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
                    observations, actions = _load_npz(path)
                    start, end = int(row["start"]), int(row["end"])
                    skill = str(row["skill"])
                    if skill not in self.skills:
                        raise ValueError(f"Unknown skill {skill!r} at line {line_no}; expected {self.skills}")
                    if start < 0 or end <= start or end > len(actions):
                        raise ValueError(f"Invalid [start,end) interval at line {line_no}")
                    self.segments.append(AnnotatedTaskSegment(
                        path, start, end, skill, observations[start:end], actions[start:end]
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
                    observations, _ = _load_npz(path)
                    step = int(row["step"])
                    if step < 0 or step >= len(observations):
                        raise ValueError(f"step out of range at line {line_no}")
                    ball = _xy(row.get("ball_xy"), "ball_xy", line_no)
                    player = _xy(row.get("player_xy"), "player_xy", line_no)
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
                        observations[step].copy(), ball, player, tuple(event_values)
                    ))
                except Exception as exc:
                    raise ValueError(f"Invalid state annotation line {line_no}: {exc}") from exc

    def __len__(self) -> int:
        return len(self.segments)

    def sample_training_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor] | None:
        if not self.segments:
            return None
        n = max(1, int(batch_size))
        segment_ids = rng.integers(0, len(self.segments), size=n)
        expert_obs: list[np.ndarray] = []
        expert_actions: list[int] = []
        expert_skills: list[int] = []
        manager_obs: list[np.ndarray] = []
        manager_skills: list[int] = []
        manager_durations: list[int] = []
        for idx in segment_ids:
            segment = self.segments[int(idx)]
            step = int(rng.integers(0, segment.length))
            expert_obs.append(segment.observations[step])
            expert_actions.append(int(segment.actions[step]))
            expert_skills.append(self.skills.index(segment.skill))
            # The manager learns task and duration at annotated segment starts.
            # Each segment contributes one start-state manager supervision sample.
            seg_duration_id = min(
                range(len(self.durations)), key=lambda j: abs(self.durations[j] - segment.length)
            )
            manager_obs.append(segment.observations[0])
            manager_skills.append(self.skills.index(segment.skill))
            manager_durations.append(seg_duration_id)

        batch: dict[str, torch.Tensor] = {
            "demo_obs": torch.from_numpy(np.stack(expert_obs).astype(np.uint8, copy=False)),
            "demo_skill": torch.tensor(expert_skills, dtype=torch.long),
            "demo_action": torch.tensor(expert_actions, dtype=torch.long),
            "demo_manager_obs": torch.from_numpy(np.stack(manager_obs).astype(np.uint8, copy=False)),
            "demo_manager_skill": torch.tensor(manager_skills, dtype=torch.long),
            "demo_manager_duration": torch.tensor(manager_durations, dtype=torch.long),
        }
        if self.state_annotations:
            state_ids = rng.integers(0, len(self.state_annotations), size=n)
            states = [self.state_annotations[int(i)] for i in state_ids]
            batch["state_obs"] = torch.from_numpy(np.stack([x.observation for x in states]).astype(np.uint8, copy=False))
            ball = np.full((n, 2), np.nan, dtype=np.float32)
            player = np.full((n, 2), np.nan, dtype=np.float32)
            events = np.asarray([x.events for x in states], dtype=np.float32)
            for i, item in enumerate(states):
                if item.ball_xy is not None:
                    ball[i] = item.ball_xy
                if item.player_xy is not None:
                    player[i] = item.player_xy
            batch["state_ball_xy"] = torch.from_numpy(ball)
            batch["state_player_xy"] = torch.from_numpy(player)
            batch["state_events"] = torch.from_numpy(events)
        return batch
