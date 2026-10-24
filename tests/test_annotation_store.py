import json
from pathlib import Path

import pytest

from app.annotation_store import AnnotationStore


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_bootstrap_existing_task_labels_and_export_training_schema(tmp_path):
    root = tmp_path / "demonstrations"
    root.mkdir()
    task_jsonl = root / "task_annotations.jsonl"
    write_jsonl(task_jsonl, [
        {"path": "level_1/winning/match_x/point_0000.npz", "start": 2, "end": 10,
         "skill": "intercept_track", "phase": "rally", "leaf_task": "intercept_track", "manager_duration": 2}
    ])
    db = AnnotationStore(tmp_path / "app" / "data" / "annotations.sqlite3")
    counts = db.bootstrap_from_jsonl(root, task_jsonl)
    assert counts["task_imported"] == 1
    loaded = db.annotations_for_point(root, "level_1/winning/match_x/point_0000.npz")
    assert len(loaded) == 1
    assert loaded[0]["start"] == 2 and loaded[0]["end"] == 10
    assert db.review_for_point(root, "level_1/winning/match_x/point_0000.npz")["status"] == "In progress"

    out = root / "out.jsonl"
    assert db.export_task_jsonl(root, out) == 1
    exported = json.loads(out.read_text(encoding="utf-8").strip())
    assert exported == {
        "path": "level_1/winning/match_x/point_0000.npz", "start": 2, "end": 10,
        "skill": "intercept_track", "phase": "rally", "leaf_task": "intercept_track", "manager_duration": 2,
    }


def test_save_edit_delete_and_audit(tmp_path):
    root = tmp_path / "demos"
    db = AnnotationStore(tmp_path / "db.sqlite3")
    path = "level_1/winning/match_a/point_0001.npz"
    db.save_point_annotations(root, path, [
        {"start": 1, "end": 5, "skill": "serve_timing", "confidence": 0.9, "annotator": "alice"},
        {"start": 8, "end": 12, "skill": "intercept_track", "confidence": 1.0, "annotator": "alice"},
    ], 20, default_annotator="alice")
    rows = db.annotations_for_point(root, path)
    assert len(rows) == 2
    edited = [dict(rows[0], end=6, notes="clear toss window")]
    assert db.save_point_annotations(root, path, edited, 20, default_annotator="alice") == 1
    rows = db.annotations_for_point(root, path)
    assert len(rows) == 1 and rows[0]["end"] == 6 and rows[0]["notes"] == "clear toss window"
    audit_actions = {r["action"] for r in db.audit_for_point(root, path)}
    assert {"create", "update", "delete"}.issubset(audit_actions)


def test_overlaps_and_bad_ranges_are_rejected_atomically(tmp_path):
    root = tmp_path / "demos"
    db = AnnotationStore(tmp_path / "db.sqlite3")
    path = "level_1/winning/match_a/point_0001.npz"
    good = [{"start": 1, "end": 5, "skill": "serve_timing"}]
    db.save_point_annotations(root, path, good, 10)
    with pytest.raises(ValueError, match="Overlapping"):
        db.save_point_annotations(root, path, [
            {"start": 1, "end": 6, "skill": "serve_timing"},
            {"start": 5, "end": 8, "skill": "intercept_track"},
        ], 10)
    with pytest.raises(ValueError, match="manager_duration"):
        db.save_point_annotations(root, path, [{"start": 1, "end": 5, "skill": "serve_timing", "manager_duration": 3}], 10)
    with pytest.raises(ValueError, match="0 <= start < end"):
        db.save_point_annotations(root, path, [{"start": -1, "end": 5, "skill": "serve_timing"}], 10)
    assert db.annotations_for_point(root, path)[0]["end"] == 5


def test_state_rows_round_trip_edit_and_export(tmp_path):
    root = tmp_path / "demos"
    db = AnnotationStore(tmp_path / "db.sqlite3")
    path = "level_1/losing/match_a/point_0002.npz"
    rows = [{"step": 7, "ball_xy": "[0.3, 0.4]", "player_xy": "[0.6, 0.8]",
             "time_to_contact": 2.0, "events_json": '{"agent_contact_window": 1}'}]
    assert db.save_state_for_point(root, path, rows, 20, annotator="reviewer") == 1
    loaded = db.state_for_point(root, path)
    assert loaded[0]["step"] == 7
    assert loaded[0]["ball_xy"] == [0.3, 0.4]
    assert loaded[0]["events"] == {"agent_contact_window": 1}
    target = root / "state_annotations.jsonl"
    assert db.export_state_jsonl(root, target) == 1
    exported = json.loads(target.read_text(encoding="utf-8").strip())
    assert exported["path"] == path and exported["step"] == 7
    assert exported["events"]["agent_contact_window"] == 1


def test_merge_is_idempotent_and_scoped_by_dataset_root(tmp_path):
    root = tmp_path / "demos"
    root.mkdir()
    src = root / "task_annotations.jsonl"
    row = {"path": "level_1/winning/match_a/point_0001.npz", "start": 0, "end": 4, "skill": "hold_cover"}
    write_jsonl(src, [row])
    db = AnnotationStore(tmp_path / "db.sqlite3")
    db.bootstrap_from_jsonl(root, src)
    assert db.merge_task_jsonl(root, src) == 0
    other_root = tmp_path / "other_demos"
    db.bootstrap_from_jsonl(other_root, src)
    assert len(db.annotations_for_point(root, row["path"])) == 1
    assert len(db.annotations_for_point(other_root, row["path"])) == 1


def test_bootstrap_imports_state_labels_once_and_audits_import(tmp_path):
    root = tmp_path / "demos"
    root.mkdir()
    task = root / "task_annotations.jsonl"
    state = root / "state_annotations.jsonl"
    write_jsonl(task, [])
    write_jsonl(state, [{"path": "level_1/winning/match_x/point_0000.npz", "step": 3,
                        "ball_xy": [0.1, 0.2], "events": {"ball_in_play": 1}}])
    store = AnnotationStore(tmp_path / "db.sqlite3")
    counts = store.bootstrap_from_jsonl(root, task, state)
    assert counts == {"task_imported": 0, "state_imported": 1}
    assert store.bootstrap_from_jsonl(root, task, state) == {"task_imported": 0, "state_imported": 0}
    audit = store.audit_for_point(root, "level_1/winning/match_x/point_0000.npz")
    assert any(event["action"] == "import" and event["annotation_type"] == "state" for event in audit)


def test_invalid_state_event_name_is_rejected(tmp_path):
    store = AnnotationStore(tmp_path / "db.sqlite3")
    with pytest.raises(ValueError, match="unknown event field"):
        store.save_state_for_point(tmp_path / "demos", "level_1/winning/match_x/point_0.npz",
                                  [{"step": 0, "events_json": '{"made_up_event": 1}'}], 10)
