from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any
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


class ParallelTennis:
    """Synchronous batched collection over independent headless Mesen workers."""

    def __init__(self, workers: list[MesenWorker], stack_size: int, max_steps: int):
        self.workers = workers
        self.n = len(workers)
        self.stack_size = stack_size
        self.height = workers[0].height
        self.width = workers[0].width
        self.max_steps = max_steps
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
        for worker in self.workers:
            worker.reset()
        results = list(self._pool.map(lambda w: w.recv_observation(), self.workers))
        output = np.zeros((self.n, self.stack_size, self.height, self.width), dtype=np.uint8)
        for i, (frame, _, _, _) in enumerate(results):
            self.frames[i].clear()
            for _ in range(self.stack_size):
                self.frames[i].append(frame)
            self.steps[i] = 0
            output[i] = self._stack(i)
        return output

    def reset_one(self, i: int) -> None:
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

    def _recv_one(self, i: int) -> tuple[int, np.ndarray, float, bool, dict[str, Any]] | tuple[int, None, None, None, None]:
        """Call recv_observation for worker i, returning (i, frame, reward, done, info)
        or (i, None, ...) on BridgeError so the caller can restart the worker."""
        try:
            frame, reward, done, info = self.workers[i].recv_observation()
            return i, frame, reward, done, info
        except BridgeError as exc:
            log.error("Worker %d BridgeError (will restart): %s", i, exc)
            return i, None, None, None, None

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"Expected actions shape {(self.n,)}, got {actions.shape}")

        for worker, action in zip(self.workers, actions.tolist()):
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

            timeout = self.steps[i] >= self.max_steps
            done = bool(done or timeout)
            info = dict(info)
            info["timeout"] = timeout
            info["episode_steps"] = int(self.steps[i])

            rewards[i] = reward
            dones[i] = done
            infos.append(info)
            observations.append(self._stack(i))

        batch = StepBatch(
            observations=np.stack(observations, axis=0),
            rewards=rewards,
            dones=dones,
            infos=infos,
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
