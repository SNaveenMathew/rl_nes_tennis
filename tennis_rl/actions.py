from __future__ import annotations

# Action names match the Lua bridge and Mesen NES controller API.
ACTIONS: tuple[tuple[str, ...], ...] = (
    (),
    ("left",),
    ("right",),
    ("up",),
    ("down",),
    ("a",),
    ("b",),
    ("left", "a"),
    ("right", "a"),
    ("up", "a"),
    ("down", "a"),
    ("left", "b"),
    ("right", "b"),
    ("up", "b"),
    ("down", "b"),
    ("left", "up"),
    ("right", "up"),
    ("left", "down"),
    ("right", "down"),
)

ACTION_NAMES = (
    "NOOP",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "A",
    "B",
    "LEFT+A",
    "RIGHT+A",
    "UP+A",
    "DOWN+A",
    "LEFT+B",
    "RIGHT+B",
    "UP+B",
    "DOWN+B",
    "LEFT+UP",
    "RIGHT+UP",
    "LEFT+DOWN",
    "RIGHT+DOWN",
)

assert len(ACTIONS) == len(ACTION_NAMES) == 19
