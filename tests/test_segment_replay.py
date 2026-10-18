import pytest
import torch
import numpy as np
from tennis_rl.demonstrations import DemonstrationDataset
from tennis_rl.model import RecurrentActorCritic
from tennis_rl.segment_replay import DemoSegmentReplayer, SegmentReplayConfig


def test_segment_replay_extracts_and_updates(tmp_path):
    device = torch.device("cpu")
    from demo_fixture_utils import write_demo_points
    write_demo_points(tmp_path)
    dataset = DemonstrationDataset(tmp_path, window_length=32)
    assert len(dataset) > 0, "Synthetic demonstration fixture should not be empty"

    model = RecurrentActorCritic(
        4, 27, [32, 64, 64], 256, 256, 1, input_size=(84, 84)
    ).to(device)

    cfg = SegmentReplayConfig(
        seg_len=16,
        batch_size=4,
        steps_per_update=2,
    )
    replayer = DemoSegmentReplayer(dataset, model, device, cfg=cfg)
    assert len(replayer._segments) > 0

    metrics = replayer.update()
    assert "seg_bc_loss" in metrics
    assert "seg_outcome_loss" in metrics
    assert not np.isnan(metrics["seg_bc_loss"])
    assert not np.isnan(metrics["seg_outcome_loss"])

    rep = replayer.report()
    assert rep["seg_total_updates"] == 2
    assert rep["seg_n_segments"] == len(replayer._segments)
