import json
from pathlib import Path

import numpy as np

from app.annotation_app import enumerate_points, load_point_bundle


def test_enumerate_point_and_load_full_resolution_frames(tmp_path):
    root = tmp_path / "demonstrations"
    folder = root / "level_1" / "winning" / "match_demo"
    folder.mkdir(parents=True)
    point = folder / "point_0007.npz"
    frames = np.zeros((3, 240, 256, 3), dtype=np.uint8)
    frames[1, 100, 120] = [255, 0, 0]
    observations = np.zeros((3, 4, 240, 256), dtype=np.uint8)
    np.savez_compressed(point, actions=np.asarray([0, 1, 2]), pulse_frames=np.asarray([-1, 1, 2]),
                        observations=observations, native_frames=frames, level=np.asarray(1),
                        point_index=np.asarray(7), match_id=np.asarray("demo"), outcome=np.asarray("winning"))
    point.with_suffix(".json").write_text(json.dumps({"frame_skip": 1, "scenario_tag": "serve-first"}), encoding="utf-8")

    rows = enumerate_points(root)
    assert len(rows) == 1
    assert rows[0]["rel"] == "level_1/winning/match_demo/point_0007.npz"
    assert rows[0]["frame_skip"] == 1
    assert rows[0]["scenario_tag"] == "serve-first"
    bundle = load_point_bundle(point, point.stat().st_mtime_ns, point.stat().st_size)
    assert bundle["has_native"] is True
    assert bundle["frames"].shape == (3, 240, 256, 3)
    assert bundle["frames"][1, 100, 120].tolist() == [255, 0, 0]
    assert bundle["actions"].tolist() == [0, 1, 2]
    assert bundle["pulses"].tolist() == [-1, 1, 2]


def test_video_encoder_creates_browser_playable_mp4(tmp_path):
    import importlib.util
    import pytest
    if importlib.util.find_spec("imageio_ffmpeg") is None:
        pytest.skip("install app/requirements.txt to test MP4 preview generation")
    from app.annotation_app import ensure_video
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 255
    point = {"path": tmp_path / "point_0000.npz", "mtime_ns": 1, "size": 2}
    video = ensure_video(point, frames, frame_skip=1)
    assert video.suffix == ".mp4"
    assert video.is_file() and video.stat().st_size > 0
