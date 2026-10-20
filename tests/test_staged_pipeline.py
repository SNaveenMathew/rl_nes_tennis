from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tennis_rl.dataset import dataset_stats, extract_segment_catalog
from tennis_rl.hierarchy_schema import DEFAULT_LEAF_TASKS, phase_for_task
from tennis_rl.staged_train import configure_trainable, make_model, stage_losses
from tennis_rl.config import Config
from tennis_rl.task_data import TaskAnnotationDataset


def _write_point(root: Path, *, name: str, match_id: str, task: str, action: int = 1) -> Path:
    folder = root / "level_1" / "winning" / f"match_{match_id}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"point_{name}.npz"
    rng = np.random.default_rng(sum(ord(c) for c in f"{name}-{match_id}-{task}"))
    obs = rng.integers(0, 256, (12, 4, 84, 84), dtype=np.uint8)
    acts = np.full(12, action, dtype=np.int64)
    rewards = np.zeros(12, dtype=np.float32); rewards[-1] = 1.0
    pulses = np.full(12, -1, dtype=np.int16); pulses[2] = 2
    np.savez_compressed(path, observations=obs, actions=acts, rewards=rewards,
        dones=np.asarray([False] * 11 + [True]), pulse_frames=pulses,
        level=np.asarray(1), outcome=np.asarray("winning"), match_id=np.asarray(match_id),
        point_index=np.asarray(int(name)))
    path.with_suffix(".json").write_text(json.dumps({
        "match_id": match_id, "scenario_tag": task, "serve_mode": "serve-first",
    }), encoding="utf-8")
    return path


def test_dataset_stats_reads_scenario_tags_and_button_pulse_coverage(tmp_path: Path):
    path = _write_point(tmp_path, name="0001", match_id="stats-match", task="stroke_timing", action=1)
    result = dataset_stats(tmp_path)
    assert result["valid_points"] == 1
    assert result["scenario_tag_counts"] == {"stroke_timing": 1}
    assert result["button_pulse_label_coverage"]["shot_actions"] == 12
    assert result["button_pulse_label_coverage"]["known_pulse_labels"] == 1
    assert result["button_pulse_label_coverage"]["unknown_shot_pulse_labels"] == 11


def test_segment_catalog_counts_leaf_and_unlabeled_segments_without_inventing_labels(tmp_path: Path):
    path = _write_point(tmp_path, name="0001", match_id="catalog-match", task="stroke_timing", action=1)
    rel = path.relative_to(tmp_path).as_posix()
    (tmp_path / "task_annotations.jsonl").write_text(json.dumps({
        "path": rel, "start": 0, "end": 8, "leaf_task": "stroke_timing", "phase": "rally"
    }) + "\n", encoding="utf-8")
    result = extract_segment_catalog(tmp_path, segment_length=4, stride=4, holdout_fraction=0.2)
    assert result["segments"] == 3
    assert result["counts"]["task:stroke_timing"] == 2
    assert result["counts"]["label:leaf_labeled"] == 2
    assert result["counts"]["label:unlabeled"] == 1


def test_stage_losses_train_explicit_leaf_and_phase_but_not_unlabeled_duration(tmp_path: Path):
    path = _write_point(tmp_path, name="0001", match_id="stage-match", task="stroke_timing", action=4)
    rel = path.relative_to(tmp_path).as_posix()
    (tmp_path / "task_annotations.jsonl").write_text(json.dumps({
        "path": rel, "start": 0, "end": 8, "leaf_task": "stroke_timing", "phase": "rally"
    }) + "\n", encoding="utf-8")
    cfg = Config()
    # This fixture intentionally stores synthetic legacy-size 84x84 frames.
    cfg.mesen.observation_width = 84
    cfg.mesen.observation_height = 84
    cfg.network.channels = [8, 16, 16]
    cfg.network.embedding_dim = 32
    cfg.network.gru_hidden = 24
    cfg.hierarchical.hidden_dim = 24
    cfg.hierarchical.skills = list(DEFAULT_LEAF_TASKS)
    model = make_model(cfg)
    data = TaskAnnotationDataset(tmp_path, model.skills, model.durations,
        button_pulse_durations=model.button_pulse_durations, split="all")
    batch = data.sample_training_batch(4, np.random.default_rng(4))
    assert batch is not None
    assert batch["demo_obs"].shape[0] == 4
    assert batch["demo_manager_obs"].shape[0] == 4
    assert torch.all(batch["demo_manager_duration"] == -1), "Segment length must not be fabricated into an option-duration target"

    device = torch.device("cpu")
    configure_trainable(model, "leaf", "stroke_timing")
    loss, leaf_metrics = stage_losses(model, batch, "leaf", "stroke_timing", device)
    assert torch.isfinite(loss) and leaf_metrics["leaf_action_loss"] > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    configure_trainable(model, "gates", None)
    gate_loss, gate_metrics = stage_losses(model, batch, "gates", None, device)
    assert torch.isfinite(gate_loss)
    assert "phase_loss" in gate_metrics
    assert "gate_leaf_loss" in gate_metrics
    assert "duration_loss" not in gate_metrics
    assert phase_for_task("stroke_timing") == "rally"
