from __future__ import annotations

from pathlib import Path

import numpy as np

from tennis_rl.mesen_client import MesenWorker


def test_consume_observation_splits_grayscale_and_native_rgb(tmp_path: Path):
    worker = MesenWorker.__new__(MesenWorker)
    worker.pending_token = "0000000000000001"
    worker.ipc_dir = tmp_path
    worker.socket_timeout = 1.0
    worker.width = 4
    worker.height = 3
    worker.capture_native_frames = True
    worker.process = None
    worker.pending_token = "0000000000000001"
    header = b"O 0000000000000001 0.0 0 1 2 3 4 0 0 0 0\n"
    gray = np.arange(12, dtype=np.uint8).tobytes()
    native = np.zeros((240, 256, 3), dtype=np.uint8)
    native[7, 11] = [17, 128, 255]
    (tmp_path / "observation_0000000000000001.bin").write_bytes(header + gray + native.tobytes())
    (tmp_path / "observation_ready_0000000000000001.ready").write_text("READY\n", encoding="ascii")

    frame, reward, done, info = worker._consume_observation()
    assert frame.shape == (3, 4) and frame.dtype == np.uint8
    assert frame.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert reward == 0.0 and not done
    assert info["player_score"] == 1 and info["opponent_games"] == 4
    assert info["native_frame"].shape == (240, 256, 3)
    assert info["native_frame"][7, 11].tolist() == [17, 128, 255]
    assert worker.pending_token is None
