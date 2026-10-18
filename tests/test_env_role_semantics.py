"""Regression tests for scenario-role-specific rewards and horizons."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_rl.env import ParallelTennis
from tennis_rl.scenario_sampler import Assignment


class FlatWorker:
    height = width = 8

    def __init__(self):
        self.pending = False
        self.t = 0
        self.closed = False

    def reset(self):
        self.t = 0
        self.action(0)

    def action(self, _action):
        if self.pending:
            raise RuntimeError("unconsumed mock action")
        self.pending = True

    def recv_observation(self):
        if not self.pending:
            raise RuntimeError("no action pending")
        self.pending = False
        self.t += 1
        return np.full((8, 8), self.t, dtype=np.uint8), 0.0, False, {}

    def close(self, force=False):
        self.closed = True


def _timeout_assignment(role: str):
    return Assignment(kind="scenario", cell_id=f"{role}|0-0", phase="start")


def test_receiver_start_timeout_is_not_misclassified_as_service_fault():
    worker = FlatWorker()
    env = ParallelTennis(
        [worker], stack_size=4, max_steps=10, point_episodes=True,
        point_max_steps=5, scenario_provider=lambda _i: _timeout_assignment("return"),
        segment_mode=True, start_max_steps=2, serve_stall_penalty=-1.0,
    )
    try:
        env.reset()
        first = env.step(np.array([0]))
        assert not first.dones[0]
        second = env.step(np.array([0]))
        assert second.dones[0]
        assert second.rewards[0] == 0.0
        assert second.infos[0]["truncated"] is True
        assert second.infos[0]["terminated"] is False
        assert second.infos[0]["timeout"] is True
        assert "service_fault" not in second.infos[0]
    finally:
        env.close()


def test_server_start_timeout_is_a_service_fault():
    worker = FlatWorker()
    env = ParallelTennis(
        [worker], stack_size=4, max_steps=10, point_episodes=True,
        point_max_steps=5, scenario_provider=lambda _i: _timeout_assignment("serve"),
        segment_mode=True, start_max_steps=2, serve_stall_penalty=-1.0,
    )
    try:
        env.reset()
        env.step(np.array([0]))
        second = env.step(np.array([0]))
        assert second.dones[0]
        assert second.rewards[0] == -1.0
        assert second.infos[0]["terminated"] is True
        assert second.infos[0]["service_fault"] is True
    finally:
        env.close()
