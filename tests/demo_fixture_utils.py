from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def write_demo_points(root: Path, length: int = 96) -> None:
    """Create tiny, deterministic, synthetic demonstration episodes for unit tests."""
    for outcome in ("winning", "losing"):
        folder = root / "level_1" / outcome / "match_fixture"
        folder.mkdir(parents=True, exist_ok=True)
        observations = np.zeros((length, 4, 84, 84), dtype=np.uint8)
        actions = np.zeros(length, dtype=np.int64)
        actions[:20] = 1   # shot activity; the very first windows are phase start
        actions[20:55] = 3 # directional movement for positioning windows
        actions[55:75] = 1 # stroke windows
        actions[75:] = 0   # late/neutral windows
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1.0 if outcome == "winning" else -1.0
        dones = np.zeros(length, dtype=np.bool_)
        dones[-1] = True
        np.savez_compressed(
            folder / "point_000.npz",
            observations=observations, actions=actions, rewards=rewards, dones=dones,
            level=np.asarray(1), outcome=np.asarray(outcome),
            match_id=np.asarray(f"fixture_{outcome}"), point_index=np.asarray(0),
        )
        (folder / "match.json").write_text(json.dumps({"match_outcome": outcome}), encoding="utf-8")
