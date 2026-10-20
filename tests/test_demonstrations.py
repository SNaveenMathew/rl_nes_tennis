import json
from pathlib import Path

import numpy as np
import torch

from tennis_rl.demonstrations import DemonstrationDataset, behavior_cloning_loss, collate_point_episodes, collate_windows, partial_supervision_losses
from tennis_rl.model import RecurrentActorCritic


def _write_point(root: Path, outcome: str, name: str = "point_0001.npz"):
    d = root / "level_1" / outcome / "match_x"
    d.mkdir(parents=True)
    obs = np.zeros((3, 4, 84, 84), dtype=np.uint8)
    actions = np.array([0, 1, 8], dtype=np.int64)
    rewards = np.array([0, 0, 1 if outcome == "winning" else -1], dtype=np.float32)
    dones = np.array([False, False, True])
    np.savez_compressed(d / name, observations=obs, actions=actions, rewards=rewards, dones=dones,
                        level=np.asarray(1), outcome=np.asarray(outcome), match_id=np.asarray("x"), point_index=np.asarray(1))
    (d / "match.json").write_text(json.dumps({"match_outcome": outcome}), encoding="utf-8")


def test_dataset_and_weighting(tmp_path):
    _write_point(tmp_path, "winning")
    _write_point(tmp_path, "losing")
    ds = DemonstrationDataset(tmp_path)
    assert len(ds) == 2
    assert ds.winning_points == 1
    assert ds.losing_points == 1


def test_collate_and_bc_loss(tmp_path):
    _write_point(tmp_path, "winning")
    ds = DemonstrationDataset(tmp_path)
    batch = ds.sample(1, np.random.default_rng(0))
    obs, actions, mask, weights = collate_point_episodes(batch, torch.device("cpu"))
    model = RecurrentActorCritic(4, 27, [32, 64, 64], 64, 64, input_size=(84, 84))
    loss = behavior_cloning_loss(model, obs, actions, mask, weights)
    assert loss.ndim == 0
    assert float(loss.detach()) >= 0.0


def test_partial_windows_and_outcome_supervision(tmp_path):
    _write_point(tmp_path, "winning")
    _write_point(tmp_path, "losing", "point_0002.npz")
    ds = DemonstrationDataset(tmp_path, window_length=2)
    windows = ds.sample_windows(2, np.random.default_rng(1))
    assert len(windows) == 2
    obs, actions, mask, outcome, rtg, weights = collate_windows(windows, torch.device("cpu"))
    model = RecurrentActorCritic(4, 27, [32, 64, 64], 64, 64, input_size=(84, 84))
    bc, aux = partial_supervision_losses(model, obs, actions, mask, outcome, rtg, weights)
    assert bc.ndim == 0 and aux.ndim == 0
    assert float(aux.detach()) >= 0.0
