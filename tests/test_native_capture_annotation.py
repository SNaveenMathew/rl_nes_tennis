from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tennis_rl.annotate_demonstrations import make_contact_sheets, make_review_gif
from tennis_rl.dataset import dataset_stats
from tennis_rl.demonstrations import _safe_load
from tennis_rl.record import save_point


def _write_point(root: Path, *, native: bool = True) -> Path:
    out = root / "level_1" / "winning" / "match_native" / "point_0001.npz"
    out.parent.mkdir(parents=True)
    observations = np.zeros((3, 4, 120, 128), dtype=np.uint8)
    observations[:, :, 30:40, 60:70] = 200
    frames = None
    if native:
        frames = np.zeros((3, 240, 256, 3), dtype=np.uint8)
        frames[:, 40:60, 100:120, 0] = 255
    metadata = {
        "match_id": "native", "scenario_tag": "test", "frame_skip": 2,
        "capture_native_frames": native, "native_frame_shape": [240, 256, 3] if native else None,
    }
    save_point(out, observations, [0, 1, 2], [0, 0, 1], [False, False, True],
               level=1, outcome="winning", match_id="native", point_index=1,
               metadata=metadata, native_frames=frames)
    return out


def test_native_frames_round_trip_and_dataset_audit(tmp_path: Path):
    path = _write_point(tmp_path)
    ep = _safe_load(path)
    assert ep is not None and ep.observations.shape == (3, 4, 120, 128)
    with np.load(path, allow_pickle=False) as data:
        assert data["native_frames"].shape == (3, 240, 256, 3)
        assert int(data["native_frames"][0, 45, 105, 0]) == 255
    stats = dataset_stats(tmp_path)
    assert stats["stored_observation_resolution_counts_by_point"] == {"128x120": 1}
    assert stats["recorded_frame_skip_counts_by_point"] == {"2": 1}
    assert stats["native_capture_coverage"]["points_with_complete_native_rgb"] == 1
    assert stats["native_capture_coverage"]["native_rgb_steps"] == 3
    assert stats["controller_action_counts"] == {"NO_MOVE+A": 1, "NO_MOVE+B": 1, "NO_MOVE+NONE": 1}


def test_contact_sheet_prefers_native_rgb_and_review_gif_works(tmp_path: Path):
    path = _write_point(tmp_path)
    sheets = make_contact_sheets(path, tmp_path / "sheets" / "point.png", sample_every=1,
                                 columns=1, scale=1, max_cells=2)
    assert len(sheets) == 2
    assert all(sheet.exists() for sheet in sheets)
    # A native frame is RGB and is not resized to the policy's 128x120 dimensions.
    from PIL import Image
    with Image.open(sheets[0]) as sheet:
        assert sheet.width >= 256
        assert sheet.height >= 240
    gif = make_review_gif(path, tmp_path / "sheets" / "point.gif", gif_scale=1)
    assert gif.exists() and gif.stat().st_size > 0
    with Image.open(gif) as animation:
        assert animation.size == (256, 240)


def test_legacy_point_audit_and_sheet_are_marked_legacy(tmp_path: Path):
    path = _write_point(tmp_path, native=False)
    stats = dataset_stats(tmp_path)
    assert stats["native_capture_coverage"]["points_with_complete_native_rgb"] == 0
    sheet = make_contact_sheets(path, tmp_path / "legacy.png", sample_every=1, columns=1, scale=1)[0]
    from PIL import Image
    with Image.open(sheet) as image:
        assert image.width >= 128
        # The warning is drawn into the contact sheet for legacy upscaled data.
        assert image.height >= 120
