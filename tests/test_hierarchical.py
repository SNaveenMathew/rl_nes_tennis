from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tennis_rl.config import HierarchicalConfig
from tennis_rl.hierarchical import HierarchicalActorCritic, HierarchicalController, HierarchicalPPOTrainer
from tennis_rl.task_data import TaskAnnotationDataset


def small_model() -> HierarchicalActorCritic:
    return HierarchicalActorCritic(
        observation_channels=4,
        action_count=27,
        channels=[8, 16, 16],
        embedding_dim=32,
        hidden_dim=24,
        skills=["serve", "receive", "baseline", "shot", "wait"],
        durations=[1, 2, 4],
        input_size=(84, 84),
    )


def ppo_samples(batch_size: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    obs = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
    return {
        "manager_obs": obs.clone(),
        "manager_skill": torch.randint(0, 5, (batch_size,)),
        "manager_duration": torch.randint(0, 3, (batch_size,)),
        "manager_old_logp": torch.full((batch_size,), -2.0),
        "manager_advantages": torch.linspace(-1, 1, batch_size),
        "manager_returns": torch.linspace(-0.5, 0.5, batch_size),
        "manager_target_skill": torch.tensor([0, 1, -1, -1][:batch_size]),
        "expert_obs": obs.clone(),
        "expert_skill": torch.randint(0, 5, (batch_size,)),
        "expert_action": torch.randint(0, 27, (batch_size,)),
        "expert_old_logp": torch.full((batch_size,), -3.5),
        "expert_advantages": torch.linspace(1, -1, batch_size),
        "expert_returns": torch.linspace(0.5, -0.5, batch_size),
    }


def test_hierarchical_shapes_and_joint_action_space():
    model = small_model()
    obs = torch.zeros((3, 4, 84, 84), dtype=torch.uint8)
    out = model(obs)
    assert out["manager_skill_logits"].shape == (3, 5)
    assert out["manager_duration_logits"].shape == (3, 3)
    assert out["expert_logits"].shape == (3, 5, 27)
    assert out["expert_values"].shape == (3, 5)
    assert out["ball_xy"].shape == (3, 2)
    assert out["player_xy"].shape == (3, 2)
    assert out["event_logits"].shape == (3, 11)
    action = model.act(obs)
    assert action.actions.shape == (3,)
    assert torch.all((action.actions >= 0) & (action.actions < 27))
    assert torch.all((action.skills >= 0) & (action.skills < 5))
    assert torch.all((action.duration_indices >= 0) & (action.duration_indices < 3))


def test_controller_uses_learned_skill_and_duration_and_resets():
    model = small_model()
    controller = HierarchicalController(model, deterministic=False)
    obs = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    for _ in range(8):
        action = controller.act(obs)
        assert action.shape == (2,)
        assert torch.all((action >= 0) & (action < 27))
    assert controller.active_skills is not None
    assert controller.remaining is not None
    controller.reset(torch.tensor([True, False]))
    assert controller.active_skills[0].item() == -1
    assert controller.active_skills[1].item() >= 0


def test_hierarchical_ppo_update_is_finite_and_changes_parameters():
    model = small_model()
    cfg = HierarchicalConfig(batch_size=2, epochs=1, learning_rate=1e-3)
    trainer = HierarchicalPPOTrainer(model, cfg, torch.device("cpu"))
    batch = ppo_samples(4)
    batch.update({
        "demo_obs": torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8),
        "demo_skill": torch.tensor([0, 1, 3]),
        "demo_action": torch.tensor([1, 4, 8]),
        "demo_manager_obs": torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
        "demo_manager_skill": torch.tensor([0, 1]),
        "demo_manager_duration": torch.tensor([0, 1]),
        "state_obs": torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
        "state_ball_xy": torch.tensor([[0.2, 0.3], [float("nan"), float("nan")]]),
        "state_player_xy": torch.tensor([[0.6, 0.8], [0.4, 0.7]]),
        "state_events": torch.tensor([[0, 1, 0, -1], [-1, 0, -1, 1]], dtype=torch.float32),
    })
    before = next(model.parameters()).detach().clone()
    metrics = trainer.update(batch)
    after = next(model.parameters()).detach()
    assert metrics["optimizer_steps"] > 0
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["demo_bc_loss"] > 0
    assert metrics["perception_loss"] > 0
    assert not torch.equal(before, after)


def test_explicit_task_annotations_are_sampled_without_heuristic_labels(tmp_path: Path):
    point_path = tmp_path / "point.npz"
    observations = np.random.default_rng(1).integers(0, 256, (12, 4, 84, 84), dtype=np.uint8)
    actions = np.arange(12, dtype=np.int64) % 27
    np.savez_compressed(point_path, observations=observations, actions=actions)
    (tmp_path / "task_annotations.jsonl").write_text(
        json.dumps({"path": "point.npz", "start": 2, "end": 8, "skill": "serve"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "state_annotations.jsonl").write_text(
        json.dumps({"path": "point.npz", "step": 3, "ball_xy": [0.2, 0.3],
                    "player_xy": [0.4, 0.9], "events": {"ball_in_play": 1}}) + "\n",
        encoding="utf-8",
    )
    dataset = TaskAnnotationDataset(tmp_path, ["serve", "receive", "baseline", "shot", "wait"], [1, 2, 4])
    assert len(dataset) == 1
    assert len(dataset.state_annotations) == 1
    sampled = dataset.sample_training_batch(3, np.random.default_rng(2))
    assert sampled is not None
    # Legacy "serve" is phase-only. It must never fabricate an exact
    # serve-timing leaf action label.
    assert sampled["demo_obs"].shape == (0, 4, 84, 84)
    assert sampled["demo_skill"].numel() == 0
    assert sampled["demo_manager_obs"].shape[0] == 3
    assert torch.all(sampled["demo_manager_phase"] == 0)
    assert sampled["state_ball_xy"].shape == (3, 2)


def test_annotation_paths_cannot_escape_root(tmp_path: Path):
    outside = tmp_path.parent / "outside.npz"
    np.savez_compressed(
        outside,
        observations=np.zeros((3, 4, 84, 84), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int64),
    )
    (tmp_path / "task_annotations.jsonl").write_text(
        json.dumps({"path": "../outside.npz", "start": 0, "end": 2, "skill": "serve"}) + "\n",
        encoding="utf-8",
    )
    try:
        TaskAnnotationDataset(tmp_path, ["serve", "receive"], [1, 2])
        assert False, "outside paths must be rejected"
    except ValueError as exc:
        assert "escapes demonstration root" in str(exc)


class DummyWorker:
    height = width = 84

    def __init__(self, seed=0, done_every=0):
        self.rng = np.random.default_rng(seed)
        self.pending_token = None
        self.t = 0
        self.done_every = int(done_every)
        self.resets = 0

    def action(self, _action):
        if self.pending_token is not None:
            raise RuntimeError("action issued while observation pending")
        self.pending_token = True

    def reset(self):
        self.resets += 1
        self.t = 0
        self.action(0)

    def recv_observation(self):
        if self.pending_token is None:
            raise RuntimeError("observation requested without action")
        self.pending_token = None
        self.t += 1
        frame = self.rng.integers(0, 256, (84, 84), dtype=np.uint8)
        done = bool(self.done_every and self.t % self.done_every == 0)
        reward = float(self.rng.choice([-1.0, 0.0, 1.0])) if done else 0.0
        return frame, reward, done, {"player_games": 0, "opponent_games": 0}

    def close(self, force=False):
        self.pending_token = None


def test_hierarchical_rollout_and_real_optimizer_iteration_with_mock_worker():
    from tennis_rl.env import ParallelTennis
    from tennis_rl.config import Config
    from tennis_rl.hierarchical_train import collect_rollout

    cfg = Config()
    cfg.device = "cpu"
    cfg.mesen.observation_width = 84
    cfg.mesen.observation_height = 84
    cfg.environment.stack_size = 4
    cfg.environment.episode_max_steps = 32
    cfg.network.channels = [8, 16, 16]
    cfg.network.embedding_dim = 32
    cfg.hierarchical.hidden_dim = 24
    cfg.hierarchical.rollout_steps = 4
    cfg.hierarchical.batch_size = 2
    cfg.hierarchical.epochs = 1
    cfg.hierarchical.skills = ["serve", "receive", "baseline", "shot", "wait"]
    cfg.hierarchical.durations = [1, 2, 4]
    worker = DummyWorker(seed=13, done_every=3)
    env = ParallelTennis([worker], stack_size=4, max_steps=32)
    model = small_model()
    try:
        obs = env.reset()
        batch, obs, stats = collect_rollout(model, env, obs, cfg, torch.device("cpu"))
        assert batch["manager_obs"].shape[0] > 0
        assert batch["expert_obs"].shape[0] == 4
        assert batch["expert_action"].min() >= 0 and batch["expert_action"].max() < 27
        assert stats["manager_decisions"] > 0
        trainer = HierarchicalPPOTrainer(model, cfg.hierarchical, torch.device("cpu"))
        metrics = trainer.update(batch)
        assert metrics["optimizer_steps"] > 0
        assert all(np.isfinite(v) for v in metrics.values())
    finally:
        env.close()


def test_deterministic_controller_traverses_root_then_branch_not_global_leaf_argmax():
    """A singleton branch must not beat the root-selected phase just because it has fewer children."""
    from tennis_rl.hierarchy_schema import DEFAULT_LEAF_TASKS

    model = HierarchicalActorCritic(
        observation_channels=4, action_count=27, channels=[8, 16, 16],
        embedding_dim=32, hidden_dim=24, skills=list(DEFAULT_LEAF_TASKS),
        durations=[1, 2, 4, 8], input_size=(84, 84),
    )
    with torch.no_grad():
        model.phase_head.weight.zero_()
        # Root prefers serve, but an argmax over flattened joint leaf priors
        # would choose the singleton transition branch in this constructed case.
        model.phase_head.bias.copy_(torch.tensor([0.5, 0.0, 0.0, 0.0]))
        for gate in model.branch_gates.values():
            gate[-1].weight.zero_()
            gate[-1].bias.zero_()
        out = model(torch.zeros((1, 4, 84, 84), dtype=torch.uint8))
        assert int(out["manager_skill_logits"].argmax(-1)[0]) == model.skill_to_id["transition_wait_reposition"]
    controller = HierarchicalController(model, deterministic=True)
    controller.act(torch.zeros((1, 4, 84, 84), dtype=torch.uint8))
    assert controller.last_skills is not None
    assert int(controller.last_skills[0]) == model.skill_to_id["serve_timing"]
