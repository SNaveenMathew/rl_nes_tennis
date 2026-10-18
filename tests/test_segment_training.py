import pytest
import torch
import numpy as np
from tennis_rl.config import Config, ScenarioConfig, PPOConfig
from tennis_rl.demonstrations import DemonstrationDataset, PointEpisode
from tennis_rl.env import ParallelTennis, StepBatch
from tennis_rl.model import RecurrentActorCritic
from tennis_rl.ppo import PPOTrainer, Rollout
from tennis_rl.scenario_sampler import Assignment
from tennis_rl.segment_replay import (
    DemoSegmentReplayer,
    SegmentReplayConfig,
    _compute_point_rtg,
    _extract_segments,
    _classify_segment,
)


class MockWorker:
    height = 84
    width = 84

    def __init__(self, step_reward=0.0, step_done=False):
        self.step_reward = step_reward
        self.step_done = step_done
        self.t = 0
        self.pending_token = None

    def reset(self):
        self.t = 0
        self.pending_token = None

    def action(self, a):
        self.pending_token = 1

    def recv_observation(self):
        self.pending_token = None
        self.t += 1
        return np.zeros((84, 84), dtype=np.uint8), self.step_reward, self.step_done, {}

    def load_state_from(self, path, noop):
        self.t = 0
        self.pending_token = None

    def close(self, force=False):
        pass


def test_segment_start_serve_stall_penalty():
    """Verify that failing to serve within start_max_steps incurs serve stall penalty."""
    worker = MockWorker(step_reward=0.0, step_done=False)
    assignment = Assignment(kind="scenario", cell_id="serve|0-0", stratum="serve|deuce|opening", phase="start")
    env = ParallelTennis(
        [worker],
        stack_size=4,
        max_steps=200,
        point_episodes=True,
        scenario_provider=lambda i: assignment,
        segment_mode=True,
        start_max_steps=5,
        serve_stall_penalty=-1.0,
    )
    env.reset()

    # Step up to limit (5 steps)
    for _ in range(4):
        batch = env.step(np.array([0], dtype=np.int64))
        assert not batch.dones[0]
        assert batch.rewards[0] == 0.0

    # 5th step hits limit
    batch = env.step(np.array([0], dtype=np.int64))
    assert batch.dones[0]
    assert batch.rewards[0] == -1.0
    assert batch.infos[0]["service_fault"] is True
    assert batch.infos[0]["terminated"] is True
    assert batch.infos[0]["truncated"] is False
    assert batch.infos[0]["point_result"] == -1
    env.close()


def test_segment_mid_rally_survival_reward():
    """Verify that completing a mid rally segment without error grants rally progress reward."""
    worker = MockWorker(step_reward=0.0, step_done=False)
    assignment = Assignment(kind="scenario", cell_id="return|1-1", stratum="return|deuce|build_up", phase="mid")
    env = ParallelTennis(
        [worker],
        stack_size=4,
        max_steps=200,
        point_episodes=True,
        scenario_provider=lambda i: assignment,
        segment_mode=True,
        mid_max_steps=6,
        rally_progress_reward=0.25,
    )
    env.reset()

    for _ in range(5):
        batch = env.step(np.array([1], dtype=np.int64))
        assert not batch.dones[0]

    # 6th step finishes rally segment
    batch = env.step(np.array([1], dtype=np.int64))
    assert batch.dones[0]
    assert abs(batch.rewards[0] - 0.25) < 1e-5
    assert batch.infos[0]["rally_survival"] is True
    assert batch.infos[0]["terminated"] is False
    assert batch.infos[0]["truncated"] is True
    assert batch.truncateds[0] is True or batch.truncateds[0] == 1
    env.close()


def test_segment_natural_point_outcome():
    """Verify that a natural point decision (+1 / -1) ends the segment immediately."""
    worker = MockWorker(step_reward=1.0, step_done=False)
    assignment = Assignment(kind="scenario", cell_id="serve|3-1", stratum="serve|deuce|game_point_for", phase="late")
    env = ParallelTennis(
        [worker],
        stack_size=4,
        max_steps=200,
        point_episodes=True,
        scenario_provider=lambda i: assignment,
        segment_mode=True,
        late_max_steps=40,
    )
    env.reset()
    batch = env.step(np.array([2], dtype=np.int64))
    assert batch.dones[0]
    assert batch.rewards[0] == 1.0
    assert batch.infos[0]["terminated"] is True
    assert batch.infos[0]["truncated"] is False
    assert batch.infos[0]["point_result"] == 1
    env.close()


def test_segment_rtg_calculation_protects_rally_in_lost_points():
    """Verify that _compute_point_rtg gives continuation credit to non-terminal frames in lost points."""
    ep_win = PointEpisode(
        path="fake_win",
        observations=np.zeros((60, 4, 84, 84), dtype=np.uint8),
        actions=np.zeros(60, dtype=np.int64),
        rewards=np.zeros(60, dtype=np.float32),
        dones=np.zeros(60, dtype=np.bool_),
        outcome="winning",
        level=1,
        match_id="m1",
        point_index=1,
    )
    ep_win.rewards[-1] = 1.0
    rtg_win = _compute_point_rtg(ep_win, gamma=0.99)
    # Win should have strictly positive return throughout
    assert np.all(rtg_win > 0.0)
    assert rtg_win[-1] == 1.0
    assert rtg_win[0] > 0.4

    ep_loss = PointEpisode(
        path="fake_loss",
        observations=np.zeros((60, 4, 84, 84), dtype=np.uint8),
        actions=np.zeros(60, dtype=np.int64),
        rewards=np.zeros(60, dtype=np.float32),
        dones=np.zeros(60, dtype=np.bool_),
        outcome="losing",
        level=1,
        match_id="m2",
        point_index=2,
    )
    ep_loss.rewards[-1] = -1.0
    rtg_loss = _compute_point_rtg(ep_loss, gamma=0.99)
    # Terminal frame should be negative
    assert rtg_loss[-1] == -1.0
    # Early/mid rally frames before the blunder (first 35 steps) should be positive continuation credit (+0.15)
    assert rtg_loss[0] == 0.15
    assert rtg_loss[20] == 0.15


def test_stratified_segment_extraction_and_losses(tmp_path):
    """Verify stratified extraction across serve, stroke, late, and approach micro-segments."""
    from demo_fixture_utils import write_demo_points
    write_demo_points(tmp_path)
    device = torch.device("cpu")
    dataset = DemonstrationDataset(tmp_path, window_length=24)
    assert len(dataset) > 0

    model = RecurrentActorCritic(4, 27, [32, 64, 64], 256, 256, 1, input_size=(84, 84)).to(device)
    cfg = SegmentReplayConfig(seg_len=16, batch_size=8, serve_lead_steps=12, late_lead_steps=25)
    replayer = DemoSegmentReplayer(dataset, model, device, cfg=cfg)

    # Check segment types present in pool
    assert len(replayer._by_type["serve"]) > 0, "Should have serve segments"
    assert len(replayer._by_type["stroke"]) > 0, "Should have stroke segments"

    # Sample batch
    batch = replayer.sample_batch(batch_size=8)
    assert len(batch) == 8
    types_in_batch = {s.seg_type for s in batch}
    assert "serve" in types_in_batch or "stroke" in types_in_batch

    # Compute differentiable losses
    bc_loss, outcome_loss = replayer.compute_losses(model, batch)
    assert bc_loss.requires_grad
    assert outcome_loss.requires_grad
    assert not torch.isnan(bc_loss)
    assert not torch.isnan(outcome_loss)
    assert float(bc_loss.detach()) > 0.0


def test_ppo_update_with_segment_demonstration_regularization(tmp_path):
    """Verify that PPOTrainer updates policy and critic using rollouts and demo segment replayer."""
    from demo_fixture_utils import write_demo_points
    write_demo_points(tmp_path)
    device = torch.device("cpu")
    model = RecurrentActorCritic(4, 27, [32, 64, 64], 256, 256, 1, input_size=(84, 84)).to(device)
    dataset = DemonstrationDataset(tmp_path, window_length=16)
    seg_cfg = SegmentReplayConfig(seg_len=16, batch_size=4)
    replayer = DemoSegmentReplayer(dataset, model, device, cfg=seg_cfg)

    ppo_cfg = PPOConfig(
        rollout_steps=16,
        sequence_length=16,
        sequence_batch_size=2,
        epochs=1,
        learning_rate=1e-4,
        demo_bc_coef=0.5,
        demo_outcome_coef=0.25,
        demo_batches_per_update=1,
        amp=False,
    )
    trainer = PPOTrainer(
        model, ppo_cfg, device, run_dir="runs/test_run",
        demonstration_dataset=dataset, segment_replayer=replayer
    )

    # Build synthetic rollout
    T = 16
    N = 2
    rollout = Rollout(
        obs=torch.zeros((T, N, 4, 84, 84), dtype=torch.uint8),
        actions=torch.zeros((T, N), dtype=torch.long),
        rewards=torch.ones((T, N), dtype=torch.float32) * 0.1,
        dones=torch.zeros((T, N), dtype=torch.float32),
        values=torch.zeros((T, N), dtype=torch.float32),
        log_probs=torch.zeros((T, N), dtype=torch.float32),
        hiddens=torch.zeros((T, 1, N, 256), dtype=torch.float32),
        advantages=torch.ones((T, N), dtype=torch.float32) * 0.5,
        returns=torch.ones((T, N), dtype=torch.float32) * 0.5,
        truncateds=torch.zeros((T, N), dtype=torch.float32),
    )

    metrics = trainer.update(rollout)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "demo_bc_loss" in metrics
    assert "motion_loss" in metrics
    assert "demo_bc_coef" in metrics
    assert metrics["demo_used"] == 1.0
    assert not np.isnan(metrics["demo_bc_loss"])


def test_collect_segment_rollout_multi_worker():
    """Verify collect_segment_rollout vectorization across multiple workers with truncations and terminations."""
    from tennis_rl.train import collect_segment_rollout

    device = torch.device("cpu")
    model = RecurrentActorCritic(4, 27, [32, 64, 64], 256, 256, 1, input_size=(84, 84)).to(device)
    w1 = MockWorker(step_reward=0.0, step_done=False)
    w2 = MockWorker(step_reward=1.0, step_done=False)
    assignment = Assignment(kind="scenario", cell_id="serve|0-0", stratum="serve|deuce|opening", phase="start")
    env = ParallelTennis(
        [w1, w2],
        stack_size=4,
        max_steps=50,
        point_episodes=True,
        scenario_provider=lambda i: assignment,
        segment_mode=True,
        start_max_steps=5,
    )
    obs = env.reset()
    hidden = model.initial_hidden(2, device)

    cfg = Config()
    cfg.ppo.rollout_steps = 10
    cfg.ppo.gamma = 0.99
    cfg.ppo.gae_lambda = 0.95

    rollout, next_obs, next_hidden, stats = collect_segment_rollout(
        model, env, None, obs, hidden, cfg, sampler=None
    )

    assert rollout.obs.shape == (10, 2, 4, 84, 84)
    assert rollout.actions.shape == (10, 2)
    assert rollout.advantages.shape == (10, 2)
    assert rollout.returns.shape == (10, 2)
    assert not torch.isnan(rollout.advantages).any()
    assert not torch.isnan(rollout.returns).any()
    env.close()


def test_resilient_checkpoint_loading_and_bc_annealing(tmp_path):
    """Verify legacy checkpoint (missing motion_head) loads without error and BC anneals correctly."""
    device = torch.device("cpu")
    model = RecurrentActorCritic(4, 27, [32, 64, 64], 64, 64, 1, input_size=(84, 84)).to(device)
    state = model.state_dict()
    # Simulate a legacy checkpoint that lacks motion_head
    legacy_state = {k: v for k, v in state.items() if not k.startswith("motion_head")}
    ckpt_path = tmp_path / "legacy_ckpt.pt"
    torch.save({"model": legacy_state, "update_index": 10, "env_steps": 750000}, ckpt_path)

    ppo_cfg = PPOConfig(
        demo_bc_coef_start=0.5,
        demo_bc_coef_final=0.15,
        demo_bc_decay_steps=1500000,
        amp=False,
    )
    new_model = RecurrentActorCritic(4, 27, [32, 64, 64], 64, 64, 1, input_size=(84, 84)).to(device)
    trainer = PPOTrainer(new_model, ppo_cfg, device, run_dir=str(tmp_path))
    trainer.load(ckpt_path)
    assert trainer.update_index == 10
    assert trainer.env_steps == 750000

    # At 750k / 1.5M steps, progress is exactly 0.5, so cos(pi * 0.5) = 0
    # Expected bc_coef_now = 0.15 + 0.5 * (0.5 - 0.15) * (1 + 0) = 0.325
    T, N = 16, 2
    dummy_rollout = Rollout(
        obs=torch.zeros((T, N, 4, 84, 84), dtype=torch.uint8),
        actions=torch.zeros((T, N), dtype=torch.long),
        rewards=torch.zeros((T, N), dtype=torch.float32),
        dones=torch.zeros((T, N), dtype=torch.float32),
        values=torch.zeros((T, N), dtype=torch.float32),
        log_probs=torch.zeros((T, N), dtype=torch.float32),
        hiddens=torch.zeros((T, 1, N, 64), dtype=torch.float32),
        advantages=torch.zeros((T, N), dtype=torch.float32),
        returns=torch.zeros((T, N), dtype=torch.float32),
    )
    ppo_cfg.sequence_length = 16
    ppo_cfg.sequence_batch_size = 2
    ppo_cfg.epochs = 1
    metrics = trainer.update(dummy_rollout)
    assert abs(metrics["demo_bc_coef"] - 0.325) < 1e-4


