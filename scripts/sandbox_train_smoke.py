"""Run a short, emulator-free hierarchical trainer smoke test.

This validates the optimizer/rollout/checkpoint code path on synthetic mock
observations. It is not evidence of improved play against the NES Tennis CPU.
"""
from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tennis_rl.config import Config, HierarchicalConfig
from tennis_rl.env import ParallelTennis
from tennis_rl.hierarchical import HierarchicalActorCritic, HierarchicalPPOTrainer
from tennis_rl.hierarchical_train import collect_rollout


class MockWorker:
    height = width = 84

    def __init__(self, seed: int, done_every: int = 5):
        self.rng = np.random.default_rng(seed)
        self.done_every = done_every
        self.pending = False
        self.step_index = 0

    def action(self, _action: int):
        if self.pending:
            raise RuntimeError("mock worker received action before prior observation")
        self.pending = True

    def reset(self):
        self.step_index = 0
        self.action(0)

    def recv_observation(self):
        if not self.pending:
            raise RuntimeError("mock worker observation requested with no pending action")
        self.pending = False
        self.step_index += 1
        frame = self.rng.integers(0, 256, (84, 84), dtype=np.uint8)
        done = self.step_index % self.done_every == 0
        reward = float(self.rng.choice((-1.0, 1.0))) if done else 0.0
        return frame, reward, done, {"player_games": 0, "opponent_games": 0}

    def close(self, force: bool = False):
        self.pending = False


def tiny_model() -> HierarchicalActorCritic:
    return HierarchicalActorCritic(
        observation_channels=4, action_count=27, channels=[8, 16, 16],
        embedding_dim=32, hidden_dim=24,
        skills=["serve", "receive", "baseline", "shot", "wait"],
        durations=[1, 2, 4], input_size=(84, 84),
    )


def run_mock_ppo_updates(n_updates: int = 3) -> dict:
    cfg = Config()
    cfg.device = "cpu"
    cfg.environment.stack_size = 4
    cfg.environment.episode_max_steps = 64
    cfg.network.channels = [8, 16, 16]
    cfg.network.embedding_dim = 32
    cfg.hierarchical.hidden_dim = 24
    cfg.hierarchical.rollout_steps = 4
    cfg.hierarchical.batch_size = 4
    cfg.hierarchical.epochs = 1
    cfg.hierarchical.skills = ["serve", "receive", "baseline", "shot", "wait"]
    cfg.hierarchical.durations = [1, 2, 4]
    env = ParallelTennis([MockWorker(10), MockWorker(20)], 4, 64)
    model = tiny_model()
    trainer = HierarchicalPPOTrainer(model, cfg.hierarchical, torch.device("cpu"))
    metrics = []
    try:
        obs = env.reset()
        for i in range(n_updates):
            data, obs, stats = collect_rollout(model, env, obs, cfg, torch.device("cpu"))
            m = trainer.update(data)
            trainer.env_steps += env.n * cfg.hierarchical.rollout_steps
            metrics.append({"update": i + 1, "loss": m["loss"], "manager_policy_loss": m["manager_policy_loss"],
                            "expert_policy_loss": m["expert_policy_loss"], "manager_samples": m["manager_samples"],
                            "expert_samples": m["expert_samples"], "episodes": stats.get("episodes", 0)})
            if not all(np.isfinite(v) for v in m.values()):
                raise RuntimeError(f"non-finite mock training metrics: {m}")
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "hierarchical_latest.pt"
            trainer.save(str(save_path), {"purpose": "sandbox smoke test"})
            restored = tiny_model()
            restore_trainer = HierarchicalPPOTrainer(restored, cfg.hierarchical, torch.device("cpu"))
            restore_trainer.load(str(save_path))
            assert restore_trainer.update_index == trainer.update_index
        return {"updates": metrics, "checkpoint_round_trip": True}
    finally:
        env.close()


def run_supervised_sanity(n_updates: int = 10) -> dict:
    """Check that explicitly labelled synthetic demonstrations reduce BC loss."""
    model = tiny_model()
    cfg = HierarchicalConfig(
        skills=["serve", "receive", "baseline", "shot", "wait"],
        durations=[1, 2, 4], batch_size=16, epochs=1, learning_rate=3e-3,
        clip_range=0.2, value_coef=0.0, entropy_coef=0.0,
        manager_supervision_coef=1.0, demo_bc_coef=3.0, perception_coef=0.0,
    )
    trainer = HierarchicalPPOTrainer(model, cfg, torch.device("cpu"))
    # Identical mock states with explicit labels teach the wait expert to choose
    # the neutral controller action, and teach the manager the supplied task.
    state = torch.zeros((16, 4, 84, 84), dtype=torch.uint8)
    manager_state = torch.zeros((8, 4, 84, 84), dtype=torch.uint8)
    wait_id = model.skill_to_id["wait"]
    blank = torch.zeros((4, 4, 84, 84), dtype=torch.uint8)
    batch = {
        "manager_obs": blank.clone(),
        "manager_skill": torch.full((4,), wait_id, dtype=torch.long),
        "manager_duration": torch.zeros(4, dtype=torch.long),
        "manager_old_logp": torch.zeros(4),
        "manager_advantages": torch.zeros(4),
        "manager_returns": torch.zeros(4),
        "manager_target_skill": torch.full((4,), -1, dtype=torch.long),
        "expert_obs": blank.clone(),
        "expert_skill": torch.full((4,), wait_id, dtype=torch.long),
        "expert_action": torch.zeros(4, dtype=torch.long),
        "expert_old_logp": torch.zeros(4),
        "expert_advantages": torch.zeros(4),
        "expert_returns": torch.zeros(4),
        "demo_obs": state,
        "demo_skill": torch.full((16,), wait_id, dtype=torch.long),
        "demo_action": torch.zeros(16, dtype=torch.long),
        "demo_manager_obs": manager_state,
        "demo_manager_skill": torch.full((8,), wait_id, dtype=torch.long),
        "demo_manager_duration": torch.zeros(8, dtype=torch.long),
    }
    losses = []
    for _ in range(n_updates):
        m = trainer.update(batch)
        losses.append(float(m["demo_bc_loss"]))
    return {"demo_bc_loss_first": losses[0], "demo_bc_loss_last": losses[-1],
            "losses": losses, "improved": losses[-1] < losses[0]}


def main():
    torch.manual_seed(123)
    np.random.seed(123)
    report = {
        "validation_scope": "synthetic mock environment only; no Mesen executable or NES ROM was included",
        "mock_rl_smoke": run_mock_ppo_updates(3),
        "supervised_sanity": run_supervised_sanity(10),
    }
    print(json.dumps(report, indent=2))
    if not report["supervised_sanity"]["improved"]:
        raise SystemExit("supervised sanity check did not reduce demonstration loss")


if __name__ == "__main__":
    main()
