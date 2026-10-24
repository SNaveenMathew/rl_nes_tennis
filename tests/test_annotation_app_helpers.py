import json
from pathlib import Path

import numpy as np

from app.annotation_app import (
    enumerate_points, load_point_bundle, save_task_draft, task_rows_signature,
)
from app.annotation_store import AnnotationStore


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


def test_ffmpeg_resolver_falls_back_to_system_binary(monkeypatch):
    import builtins
    from app import annotation_app

    original_import = builtins.__import__
    def import_without_imageio_ffmpeg(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ModuleNotFoundError("No module named 'imageio_ffmpeg'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_imageio_ffmpeg)
    monkeypatch.setattr(annotation_app.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert annotation_app.resolve_ffmpeg_executable() == "/usr/bin/ffmpeg"


def test_ffmpeg_resolver_reports_install_command_when_no_backend(monkeypatch):
    import builtins
    import pytest
    from app import annotation_app

    original_import = builtins.__import__
    def import_without_imageio_ffmpeg(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ModuleNotFoundError("No module named 'imageio_ffmpeg'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_imageio_ffmpeg)
    monkeypatch.setattr(annotation_app.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="python -m pip install imageio-ffmpeg"):
        annotation_app.resolve_ffmpeg_executable()



def test_added_task_segment_is_saved_and_summary_updates_immediately(tmp_path):
    demo_root = tmp_path / "demonstrations"
    point_path = "level_1/winning/match_demo/point_0001.npz"
    store = AnnotationStore(tmp_path / "app" / "data" / "annotations.sqlite3")
    draft = [{
        "id": None, "start": 12, "end": 28, "skill": "hold_cover",
        "manager_duration": None, "confidence": 1.0, "annotator": "alice", "notes": "",
    }]

    count, refreshed = save_task_draft(store, demo_root, point_path, draft, 40, "alice")

    assert count == 1
    assert len(refreshed) == 1
    assert refreshed[0]["id"] is not None
    summary = store.point_summary(demo_root)
    assert summary[point_path]["segments"] == 1
    assert summary[point_path]["skills"] == ["hold_cover"]
    # An added row (without an id in the table) compares equal to the row now
    # persisted to SQLite, preventing autosave from repeatedly reinserting it.
    assert task_rows_signature(draft, "alice") == task_rows_signature(refreshed)


def test_task_row_signature_waits_until_new_row_is_complete():
    incomplete = [
        {"id": None, "start": 14, "end": None, "skill": "", "confidence": 1.0},
    ]
    assert task_rows_signature(incomplete, "alice") is None
    assert task_rows_signature([{"start": None, "end": None, "skill": ""}]) == ()


def test_task_row_signature_detects_label_change():
    original = [{"id": 7, "start": 2, "end": 10, "skill": "intercept_track",
                 "manager_duration": None, "confidence": 1.0, "annotator": "alice", "notes": ""}]
    edited = [dict(original[0], skill="hold_cover")]
    assert task_rows_signature(original) != task_rows_signature(edited)
