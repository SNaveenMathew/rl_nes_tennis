"""Decision taxonomy for the learned NES Tennis hierarchy.

The taxonomy is intentionally factorized: one learned root gate selects a broad
phase, one phase-specific gate selects a leaf task, and a task-specific actor
selects controller input. It does NOT enumerate every Cartesian combination of
score, lane, trajectory and player location as a separate model.
"""
from __future__ import annotations

PHASES = ("serve", "receive", "rally", "transition")

# This ordering is serialized into new checkpoints. Do not reorder silently.
PHASE_TASKS: dict[str, tuple[str, ...]] = {
    "serve": ("serve_timing", "serve_placement_power", "serve_recovery"),
    "receive": ("receive_position", "return_stroke_timing"),
    "rally": (
        "intercept_track", "hold_cover", "recover_reposition", "approach_net",
        "retreat_lob", "stroke_selector", "stroke_timing",
    ),
    "transition": ("transition_wait_reposition",),
}
DEFAULT_LEAF_TASKS = tuple(task for phase in PHASES for task in PHASE_TASKS[phase])
TASK_TO_PHASE = {task: phase for phase, tasks in PHASE_TASKS.items() for task in tasks}

# Only explicit legacy labels are accepted as coarse phase annotations. They do
# not get silently promoted to a leaf-task action label.
LEGACY_PHASE_LABELS = {
    "serve": "serve", "receive": "receive", "return": "receive",
    "baseline": "rally", "approach_net": "rally", "recover": "rally",
    "shot": "rally", "wait": "transition", "transition": "transition",
}

LEGACY_LEAF_ALIASES = {
    "serve_timing": "serve_timing",
    "serve_placement_power": "serve_placement_power",
    "serve_recovery": "serve_recovery",
    "receive_position": "receive_position",
    "return_stroke_timing": "return_stroke_timing",
    "intercept_track": "intercept_track",
    "hold_cover": "hold_cover",
    "recover_reposition": "recover_reposition",
    "approach_net": "approach_net",
    "retreat_lob": "retreat_lob",
    "stroke_selector": "stroke_selector",
    "stroke_timing": "stroke_timing",
    "transition_wait_reposition": "transition_wait_reposition",
}

EVENT_NAMES = (
    "serve_ready", "agent_serve", "cpu_serve", "ball_in_play",
    "ball_toward_agent", "ball_away_from_agent", "opponent_contact_window",
    "agent_contact_window", "high_floater_lob", "point_resolved", "uncertain",
)
TIME_TO_CONTACT_BINS = ("urgent", "soon", "not_imminent")


def phase_for_task(task: str) -> str | None:
    if task in TASK_TO_PHASE:
        return TASK_TO_PHASE[task]
    return LEGACY_PHASE_LABELS.get(task)


def phase_task_indices(tasks: tuple[str, ...] | list[str]) -> dict[str, list[int]]:
    result = {phase: [] for phase in PHASES}
    for i, task in enumerate(tasks):
        phase = phase_for_task(task)
        if phase is None:
            raise ValueError(f"Unknown hierarchy task {task!r}; update hierarchy_schema.py")
        result[phase].append(i)
    return result
