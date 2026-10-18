from __future__ import annotations

# NES Tennis has two independent control dimensions: the D-pad (8 directions
# plus neutral) and the A/B action buttons (none/A/B). Combining them gives the
# complete 9 x 3 controller action set. The official manual documents the
# D-pad, A swing/serve, and B lob controls; contemporary play guides also note
# that up/down while striking changes serve/shot power.
MOVEMENTS: tuple[tuple[str, ...], ...] = (
    (), ("left",), ("right",), ("up",), ("down",),
    ("left", "up"), ("right", "up"), ("left", "down"), ("right", "down"),
)
MOVEMENT_NAMES = (
    "NO_MOVE", "LEFT", "RIGHT", "UP", "DOWN",
    "LEFT+UP", "RIGHT+UP", "LEFT+DOWN", "RIGHT+DOWN",
)
SHOTS: tuple[tuple[str, ...], ...] = ((), ("a",), ("b",))
SHOT_NAMES = ("NONE", "A", "B")

ACTIONS: tuple[tuple[str, ...], ...] = tuple(
    move + shot for move in MOVEMENTS for shot in SHOTS
)
ACTION_NAMES = tuple(
    f"{MOVEMENT_NAMES[m]}+{SHOT_NAMES[s]}"
    for m in range(len(MOVEMENTS))
    for s in range(len(SHOTS))
)

assert len(MOVEMENTS) == 9
assert len(SHOTS) == 3
assert len(ACTIONS) == len(ACTION_NAMES) == 27


def decode_action(action: int) -> tuple[int, int]:
    if not 0 <= int(action) < len(ACTIONS):
        raise ValueError(f"Invalid action {action}")
    return int(action) // 3, int(action) % 3


def encode_action(movement: int, shot: int) -> int:
    if not 0 <= int(movement) < len(MOVEMENTS):
        raise ValueError(f"Invalid movement {movement}")
    if not 0 <= int(shot) < len(SHOTS):
        raise ValueError(f"Invalid shot {shot}")
    return int(movement) * 3 + int(shot)
