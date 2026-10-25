from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .mesen_client import BridgeError, MesenWorker

log = logging.getLogger(__name__)


@dataclass
class StepBatch:
    observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: list[dict[str, Any]]
    truncateds: np.ndarray | None = None


class ParallelTennis:
    """Synchronous batched collection over independent headless Mesen workers."""

    def __init__(
        self,
        workers: list[MesenWorker],
        stack_size: int,
        max_steps: int,
        point_episodes: bool = False,
        point_max_steps: int = 900,
        scenario_provider: Callable[[int], Any] | None = None,
        segment_mode: bool = True,
        late_max_steps: int = 40,
        mid_max_steps: int = 50,
        start_max_steps: int = 25,
        serve_stall_penalty: float = -1.0,
        rally_progress_reward: float = 0.20,
        late_progress_reward: float = 0.10,
    ):
        """
        ``scenario_provider(i)`` returns a ``scenario_sampler.Assignment`` each time
        worker ``i`` needs a new episode.  "scenario"/"probe" assignments start from a
        savestate file and (with ``point_episodes``) run as point segments with
        appropriate phase-specific horizons and rewards.  "anchor" assignments are
        ordinary full games.
        """
        self.workers = workers
        self.n = len(workers)
        self.stack_size = stack_size
        self.height = workers[0].height
        self.width = workers[0].width
        self.max_steps = max_steps
        self.point_episodes = bool(point_episodes)
        self.point_max_steps = int(point_max_steps)
        self.scenario_provider = scenario_provider
        self.segment_mode = bool(segment_mode)
        self.late_max_steps = int(late_max_steps)
        self.mid_max_steps = int(mid_max_steps)
        self.start_max_steps = int(start_max_steps)
        self.serve_stall_penalty = float(serve_stall_penalty)
        self.rally_progress_reward = float(rally_progress_reward)
        self.late_progress_reward = float(late_progress_reward)
        self.assignments: list[Any] = [None] * self.n
        self.frames = [deque(maxlen=stack_size) for _ in workers]
        self.steps = np.zeros(self.n, dtype=np.int64)
        self.total_env_steps = 0
        self._pool = ThreadPoolExecutor(max_workers=self.n, thread_name_prefix="mesen-recv")

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return self.stack_size, self.height, self.width

    def _stack(self, i: int) -> np.ndarray:
        return np.stack(tuple(self.frames[i]), axis=0).astype(np.uint8, copy=False)

    def reset(self) -> np.ndarray:
        output = np.zeros((self.n, self.stack_size, self.height, self.width), dtype=np.uint8)
        if self.scenario_provider is not None:
            # Per-worker start (state loads are cheap, but anchors restart a process).
            for i in range(self.n):
                self.reset_one(i)
                output[i] = self._stack(i)
            return output
        for worker in self.workers:
            worker.reset()
        results = list(self._pool.map(lambda w: w.recv_observation(), self.workers))
        for i, (frame, _, _, _) in enumerate(results):
            self.frames[i].clear()
            for _ in range(self.stack_size):
                self.frames[i].append(frame)
            self.steps[i] = 0
            output[i] = self._stack(i)
        return output

    def current_observation(self, i: int) -> np.ndarray:
        """Frame stack for worker ``i`` as of *now* (i.e. after any reset)."""
        return self._stack(i)

    def _reset_to_state(self, i: int, assignment: Any) -> bool:
        worker = self.workers[i]
        try:
            worker.load_state_from(assignment.state.path, assignment.noop_frames)
            worker.action(0)
            frame, _, _, _ = worker.recv_observation()
        except BridgeError as exc:
            log.error("Worker %d could not start from %s (%s); falling back to a full restart",
                      i, getattr(assignment.state, "path", "?"), exc)
            worker.pending_token = None
            return False
        self.frames[i].clear()
        for _ in range(self.stack_size):
            self.frames[i].append(frame)
        self.steps[i] = 0
        return True

    def reset_one(self, i: int) -> None:
        assignment = self.scenario_provider(i) if self.scenario_provider else None
        if assignment is not None and assignment.kind in ("scenario", "probe") and assignment.state is not None:
            if self._reset_to_state(i, assignment):
                self.assignments[i] = assignment
                return
            assignment = type(assignment)(kind="anchor")
        self.assignments[i] = assignment
        worker = self.workers[i]
        worker.reset()
        frame, _, _, _ = worker.recv_observation()
        self.frames[i].clear()
        for _ in range(self.stack_size):
            self.frames[i].append(frame)
        self.steps[i] = 0

    def _hard_restart_one(self, i: int) -> None:
        """Force-restart a frozen worker (no pending token to consume first).
        Fills the frame buffer with the first observation so the episode starts
        cleanly. Called only from _recv_one error recovery, where the previous
        pending observation was never delivered.
        """
        worker = self.workers[i]
        # close(force=True) kills the process regardless of pending state.
        worker.close(force=True)
        worker.next_token = 1
        worker.pending_token = None
        worker._clear_mailbox()
        worker._start()
        # Request the first observation from the fresh ROM.
        token = worker._new_token()
        worker.pending_token = token
        worker._write_command(f"A {token} 0\n")
        try:
            frame, _, _, _ = worker.recv_observation()
        except Exception as exc:
            log.error("Worker %d failed to produce initial observation after hard restart: %s", i, exc)
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
        self.frames[i].clear()
        for _ in range(self.stack_size):
            self.frames[i].append(frame)
        self.steps[i] = 0
        self.assignments[i] = None  # now a plain full game from the match start

    def _recv_one(self, i: int) -> tuple[int, np.ndarray, float, bool, dict[str, Any]] | tuple[int, None, None, None, None]:
        """Call recv_observation for worker i, returning (i, frame, reward, done, info)
        or (i, None, ...) on BridgeError so the caller can restart the worker."""
        try:
            frame, reward, done, info = self.workers[i].recv_observation()
            return i, frame, reward, done, info
        except BridgeError as exc:
            log.error("Worker %d BridgeError (will restart): %s", i, exc)
            return i, None, None, None, None

    @staticmethod
    def _assignment_role(assignment: Any) -> str | None:
        """Return a role only when explicit scenario metadata identifies it.

        Scenario cell IDs have the documented ``serve|score`` / ``return|score``
        form. Missing or malformed metadata is unknown; it must not be treated as
        a serving episode just because its phase is ``start``.
        """
        if assignment is None:
            return None
        cell_id = str(getattr(assignment, "cell_id", "") or "")
        role = cell_id.split("|", 1)[0]
        return role if role in {"serve", "return"} else None

    def step(self, actions: np.ndarray, pulse_frames: np.ndarray | None = None) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"Expected actions shape {(self.n,)}, got {actions.shape}")
        pulse_values = None if pulse_frames is None else np.asarray(pulse_frames, dtype=np.int64)
        if pulse_values is not None and pulse_values.shape != (self.n,):
            raise ValueError(f"Expected pulse_frames shape {(self.n,)}, got {pulse_values.shape}")
        if pulse_values is not None and np.any(pulse_values < 1):
            raise ValueError("All pulse durations must be positive")

        for i, (worker, action) in enumerate(zip(self.workers, actions.tolist())):
            if pulse_values is not None and (int(action) % 3 != 0) and getattr(worker, "supports_button_pulses", False):
                worker.action(int(action), pulse_frames=int(pulse_values[i]))
            else:
                worker.action(int(action))

        observations: list[np.ndarray] = []
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        already_reset: set[int] = set()

        results = list(self._pool.map(self._recv_one, range(self.n)))
        for i, frame, reward, done, info in results:
            if frame is None:
                # Worker froze or crashed — hard-restart it and treat as episode end.
                log.warning("Restarting worker %d after BridgeError", i)
                self._hard_restart_one(i)
                already_reset.add(i)
                frame = np.zeros((self.height, self.width), dtype=np.uint8)
                reward = 0.0
                done = True
                info = {"bridge_restart": True}

            self.frames[i].append(frame)
            self.steps[i] += 1
            self.total_env_steps += 1

            assignment = self.assignments[i]
            point_mode = bool(
                self.point_episodes
                and (
                    assignment is None
                    or getattr(assignment, "kind", None) in ("scenario", "probe", "anchor")
                )
            )
            is_segment = bool(
                point_mode
                and self.segment_mode
                and assignment is not None
                and getattr(assignment, "kind", None) in ("scenario", "probe")
            )
            phase = getattr(assignment, "phase", None) if assignment is not None else None
            if is_segment:
                if phase == "late":
                    limit = self.late_max_steps
                elif phase == "mid":
                    limit = self.mid_max_steps
                elif phase == "start":
                    limit = self.start_max_steps
                else:
                    limit = self.point_max_steps
            else:
                limit = self.point_max_steps if point_mode else self.max_steps

            point_result = 0
            terminated = False
            truncated = False

            if point_mode and reward != 0.0:
                # Point ended naturally (score change in ROM).
                point_result = 1 if reward > 0 else -1
                reward = float(point_result)
                done = True
                terminated = True
                truncated = False
            elif point_mode and self.steps[i] >= limit:
                # Segment reached its horizon without a ROM score change.
                done = True
                if phase == "start" and self._assignment_role(assignment) == "serve":
                    # Apply the stall penalty only when this agent is the server.
                    # A receiving player can legitimately wait for the CPU to serve.
                    reward = self.serve_stall_penalty
                    point_result = -1
                    terminated = True
                    truncated = False
                    info["service_fault"] = True
                elif phase == "mid":
                    # Sustained the rally across the segment window without an unforced error.
                    # Reward rally survival and bootstrap remaining value via truncation.
                    reward = self.rally_progress_reward
                    point_result = 0
                    terminated = False
                    truncated = True
                    info["rally_survival"] = True
                elif phase == "late":
                    # Extended rally past the endgame window without losing the point.
                    reward = self.late_progress_reward
                    point_result = 0
                    terminated = False
                    truncated = True
                    info["late_survival"] = True
                else:
                    reward = 0.0
                    point_result = 0
                    terminated = False
                    truncated = True
                    info["timeout"] = True
            else:
                timeout = bool(self.steps[i] >= limit)
                done = bool(done or timeout)
                terminated = bool(done and not timeout)
                truncated = timeout

            info = dict(info)
            info["point_result"] = point_result
            info["assignment"] = assignment
            info["phase"] = phase
            info["terminated"] = terminated
            info["truncated"] = truncated
            info["timeout"] = bool(truncated or info.get("service_fault", False))
            info["episode_steps"] = int(self.steps[i])

            rewards[i] = reward
            dones[i] = done
            infos.append(info)
            observations.append(self._stack(i))

        truncateds = np.asarray([inf.get("truncated", False) for inf in infos], dtype=np.bool_)
        batch = StepBatch(
            observations=np.stack(observations, axis=0),
            rewards=rewards,
            dones=dones,
            infos=infos,
            truncateds=truncateds,
        )

        # Reset finished workers immediately so the next observation returned
        # from the vector stays valid. Workers that were already hard-restarted
        # (due to a BridgeError above) are skipped — they are already clean.
        for i, done in enumerate(dones.tolist()):
            if done and i not in already_reset:
                self.reset_one(i)

        return batch

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers:
            worker.close()
