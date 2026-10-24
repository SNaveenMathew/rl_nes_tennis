"""SQLite-backed annotation persistence for the Streamlit review app.

The database is canonical for the app. Task labels can be exported to the
project's task_annotations.jsonl format for the existing training pipeline.
Raw demonstrations are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from tennis_rl.hierarchy_schema import DEFAULT_LEAF_TASKS, LEGACY_PHASE_LABELS, TASK_TO_PHASE, phase_for_task
except ImportError:  # Allows standalone tests/tools when the repository root isn't on sys.path.
    DEFAULT_LEAF_TASKS = (
        "serve_timing", "serve_placement_power", "serve_recovery",
        "receive_position", "return_stroke_timing", "intercept_track",
        "hold_cover", "recover_reposition", "approach_net", "retreat_lob",
        "stroke_selector", "stroke_timing", "transition_wait_reposition",
    )
    LEGACY_PHASE_LABELS = {"serve": "serve", "receive": "receive", "return": "receive",
                           "baseline": "rally", "recover": "rally", "shot": "rally",
                           "wait": "transition", "transition": "transition"}
    TASK_TO_PHASE = {name: phase for phase, names in {
        "serve": ("serve_timing", "serve_placement_power", "serve_recovery"),
        "receive": ("receive_position", "return_stroke_timing"),
        "rally": ("intercept_track", "hold_cover", "recover_reposition", "approach_net",
                  "retreat_lob", "stroke_selector", "stroke_timing"),
        "transition": ("transition_wait_reposition",),
    }.items() for name in names}
    def phase_for_task(task: str) -> str | None:
        return TASK_TO_PHASE.get(task, LEGACY_PHASE_LABELS.get(task))

LABEL_OPTIONS = tuple(dict.fromkeys((*DEFAULT_LEAF_TASKS, *LEGACY_PHASE_LABELS.keys())))
DURATIONS = (1, 2, 4, 8)
REVIEW_STATUSES = ("Not started", "In progress", "Needs review", "Reviewed", "Rejected")
STATE_EVENT_NAMES = (
    "serve_ready", "agent_serve", "cpu_serve", "ball_in_play", "ball_toward_agent",
    "ball_away_from_agent", "opponent_contact_window", "agent_contact_window",
    "high_floater_lob", "point_resolved", "uncertain",
)
STATE_VALUE_FIELDS = ("ball_xy", "player_xy", "ball_velocity_xy", "time_to_contact")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def row_fingerprint(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {source} at line {line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object in {source} at line {line_no}")
        rows.append(item)
    return rows


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class AnnotationStore:
    """Small SQLite repository with import, audit, editing, and training export."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def _initialize(self) -> None:
        with self.connect() as con:
            con.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_root TEXT NOT NULL,
                    point_path TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    skill TEXT NOT NULL,
                    phase TEXT,
                    leaf_task TEXT,
                    manager_duration INTEGER,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    notes TEXT NOT NULL DEFAULT '',
                    annotator TEXT NOT NULL DEFAULT '',
                    source_key TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dataset_root, source_key)
                );
                CREATE INDEX IF NOT EXISTS idx_task_point
                    ON task_annotations(dataset_root, point_path, start, end);
                CREATE TABLE IF NOT EXISTS point_reviews (
                    dataset_root TEXT NOT NULL,
                    point_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Not started',
                    annotator TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(dataset_root, point_path)
                );
                CREATE TABLE IF NOT EXISTS state_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_root TEXT NOT NULL,
                    point_path TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    source_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dataset_root, source_key)
                );
                CREATE INDEX IF NOT EXISTS idx_state_point
                    ON state_annotations(dataset_root, point_path, step);
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_root TEXT NOT NULL,
                    point_path TEXT NOT NULL,
                    annotation_type TEXT NOT NULL,
                    annotation_id INTEGER,
                    action TEXT NOT NULL,
                    annotator TEXT NOT NULL DEFAULT '',
                    old_json TEXT,
                    new_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_point
                    ON audit_events(dataset_root, point_path, created_at);
            """)

    @staticmethod
    def _meta_key(kind: str, dataset_root: str) -> str:
        return f"{kind}_bootstrap:{dataset_root}"

    def _has_meta(self, key: str) -> bool:
        with self.connect() as con:
            return con.execute("SELECT 1 FROM app_meta WHERE key=?", (key,)).fetchone() is not None

    def _set_meta(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO app_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def bootstrap_from_jsonl(self, dataset_root: str | Path, task_jsonl: str | Path, state_jsonl: str | Path | None = None) -> dict[str, int]:
        """One-time import of current project JSONL labels; later edits stay in SQLite."""
        scope = str(Path(dataset_root).resolve())
        counts = {"task_imported": 0, "state_imported": 0}
        task_key = self._meta_key("task", scope)
        if not self._has_meta(task_key):
            counts["task_imported"] = self._import_task_rows(scope, Path(task_jsonl), replace=False)
            self._set_meta(task_key, "1")
        if state_jsonl is not None:
            state_key = self._meta_key("state", scope)
            if not self._has_meta(state_key):
                counts["state_imported"] = self._import_state_rows(scope, Path(state_jsonl), replace=False)
                self._set_meta(state_key, "1")
        return counts

    def merge_task_jsonl(self, dataset_root: str | Path, task_jsonl: str | Path) -> int:
        """Merge task labels from an external JSONL file without overwriting DB edits."""
        return self._import_task_rows(str(Path(dataset_root).resolve()), Path(task_jsonl), replace=False)

    def _import_task_rows(self, scope: str, source: Path, replace: bool) -> int:
        rows = read_jsonl(source)
        now = utc_now()
        imported = 0
        with self.connect() as con:
            if replace:
                con.execute("DELETE FROM task_annotations WHERE dataset_root=?", (scope,))
            existing_signatures = {
                (r["point_path"], int(r["start"]), int(r["end"]), r["skill"])
                for r in con.execute("SELECT point_path,start,end,skill FROM task_annotations WHERE dataset_root=?", (scope,))
            }
            for row in rows:
                point = str(row.get("path", "")).replace("\\", "/").strip()
                skill = str(row.get("skill", row.get("leaf_task", ""))).strip()
                if not point or not skill or "start" not in row or "end" not in row:
                    continue
                try:
                    start, end = int(row["start"]), int(row["end"])
                except (TypeError, ValueError, OverflowError):
                    continue
                signature = (point, start, end, skill)
                source_key = row_fingerprint(row)
                if signature in existing_signatures:
                    continue
                if con.execute("SELECT 1 FROM task_annotations WHERE dataset_root=? AND source_key=?", (scope, source_key)).fetchone():
                    continue
                phase = row.get("phase") or phase_for_task(skill)
                leaf_task = row.get("leaf_task") or (skill if skill in TASK_TO_PHASE else None)
                duration = row.get("manager_duration")
                try:
                    duration = int(duration) if duration not in (None, "") else None
                except (TypeError, ValueError):
                    duration = None
                extras = {k: v for k, v in row.items() if k not in {"path", "start", "end", "skill", "phase", "leaf_task", "manager_duration"}}
                con.execute("""INSERT OR IGNORE INTO task_annotations
                    (dataset_root,point_path,start,end,skill,phase,leaf_task,manager_duration,
                     source_key,extra_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (scope, point, start, end, skill, phase, leaf_task, duration,
                     source_key, canonical_json(extras), now, now))
                if con.execute("SELECT changes()").fetchone()[0]:
                    imported += 1
                    existing_signatures.add(signature)
            if imported:
                con.execute("INSERT INTO audit_events(dataset_root,point_path,annotation_type,action,old_json,new_json,created_at) VALUES(?,?, ?, ?, NULL, ?, ?)" ,
                            (scope, "*", "task", "import", canonical_json({"imported": imported, "source": str(source)}), now))
                for point_path in {r["point_path"] for r in con.execute("SELECT point_path FROM task_annotations WHERE dataset_root=?", (scope,))}:
                    con.execute("INSERT OR IGNORE INTO point_reviews(dataset_root,point_path,status,annotator,notes,updated_at) VALUES(?,?,'In progress','','',?)",
                                (scope, point_path, now))
        return imported

    def merge_state_jsonl(self, dataset_root: str | Path, state_jsonl: str | Path) -> int:
        """Merge unseen per-frame state labels from JSONL without overwriting DB edits."""
        return self._import_state_rows(str(Path(dataset_root).resolve()), Path(state_jsonl), replace=False)

    def _import_state_rows(self, scope: str, source: Path, replace: bool) -> int:
        rows = read_jsonl(source)
        now = utc_now()
        imported = 0
        with self.connect() as con:
            if replace:
                con.execute("DELETE FROM state_annotations WHERE dataset_root=?", (scope,))
            existing = {(r["point_path"], int(r["step"])) for r in con.execute(
                "SELECT point_path,step FROM state_annotations WHERE dataset_root=?", (scope,))}
            for row in rows:
                point = str(row.get("path", "")).replace("\\", "/").strip()
                if not point or "step" not in row:
                    continue
                try:
                    step = int(row["step"])
                except (TypeError, ValueError):
                    continue
                payload = {k: v for k, v in row.items() if k not in {"path", "step"}}
                signature = (point, step)
                source_key = row_fingerprint(row)
                if signature in existing:
                    continue
                con.execute("INSERT OR IGNORE INTO state_annotations(dataset_root,point_path,step,payload_json,source_key,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                            (scope, point, step, canonical_json(payload), source_key, now, now))
                if con.execute("SELECT changes()").fetchone()[0]:
                    imported += 1
                    existing.add(signature)
            if imported:
                con.execute("INSERT INTO audit_events(dataset_root,point_path,annotation_type,action,old_json,new_json,created_at) VALUES(?,?,?, ?, NULL, ?, ?)",
                            (scope, "*", "state", "import", canonical_json({"imported": imported, "source": str(source)}), now))
        return imported

    def annotations_for_point(self, dataset_root: str | Path, point_path: str) -> list[dict[str, Any]]:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            rows = con.execute("SELECT * FROM task_annotations WHERE dataset_root=? AND point_path=? ORDER BY start,end,id", (scope, point_path)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["notes"] = item.get("notes", "") or ""
            item["annotator"] = item.get("annotator", "") or ""
            result.append(item)
        return result

    def save_point_annotations(
        self, dataset_root: str | Path, point_path: str, rows: list[dict[str, Any]],
        point_length: int, *, default_annotator: str = "",
    ) -> int:
        """Validate and replace one point's interval set atomically, auditing every change."""
        scope, now = str(Path(dataset_root).resolve()), utc_now()
        normalized: list[dict[str, Any]] = []
        for index, source in enumerate(rows):
            start_raw, end_raw = source.get("start"), source.get("end")
            skill_raw = source.get("skill")
            if _blank(start_raw) and _blank(end_raw) and _blank(skill_raw):
                continue  # Ignore the blank new row supplied by the data editor.
            start = _strict_int(start_raw, f"row {index + 1} start")
            end = _strict_int(end_raw, f"row {index + 1} end")
            skill = str(skill_raw or "").strip()
            if skill not in LABEL_OPTIONS:
                raise ValueError(f"Row {index + 1}: unsupported task label {skill!r}")
            if not 0 <= start < end <= int(point_length):
                raise ValueError(f"Row {index + 1}: range must satisfy 0 <= start < end <= {point_length}")
            duration_raw = source.get("manager_duration")
            duration = None if _blank(duration_raw) else _strict_int(duration_raw, f"row {index + 1} manager_duration")
            if duration is not None and duration not in DURATIONS:
                raise ValueError(f"Row {index + 1}: manager_duration must be one of {DURATIONS} or blank")
            try:
                confidence = 1.0 if _blank(source.get("confidence")) else float(source.get("confidence"))
            except (TypeError, ValueError):
                raise ValueError(f"Row {index + 1}: confidence must be numeric from 0 to 1")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Row {index + 1}: confidence must be between 0 and 1")
            item_id = None if _blank(source.get("id")) else _strict_int(source.get("id"), f"row {index + 1} id")
            normalized.append({
                "id": item_id, "start": start, "end": end, "skill": skill,
                "phase": phase_for_task(skill), "leaf_task": skill if skill in TASK_TO_PHASE else None,
                "manager_duration": duration, "confidence": confidence,
                "notes": str(source.get("notes") or ""),
                "annotator": str(source.get("annotator") or default_annotator or ""),
            })
        normalized.sort(key=lambda row: (row["start"], row["end"]))
        for previous, current in zip(normalized, normalized[1:]):
            if current["start"] < previous["end"]:
                raise ValueError(f"Overlapping ranges are not allowed: [{previous['start']},{previous['end']}) and [{current['start']},{current['end']})")

        with self.connect() as con:
            old_rows = [dict(r) for r in con.execute("SELECT * FROM task_annotations WHERE dataset_root=? AND point_path=? ORDER BY id", (scope, point_path))]
            old_by_id = {int(row["id"]): row for row in old_rows}
            retained: set[int] = set()
            for row in normalized:
                item_id = row["id"]
                existing = old_by_id.get(item_id) if item_id is not None else None
                if existing:
                    retained.add(item_id)
                    new_db = {k: row[k] for k in ("start", "end", "skill", "phase", "leaf_task", "manager_duration", "confidence", "notes", "annotator")}
                    old_db = {k: existing.get(k) for k in new_db}
                    if old_db != new_db:
                        con.execute("UPDATE task_annotations SET start=?,end=?,skill=?,phase=?,leaf_task=?,manager_duration=?,confidence=?,notes=?,annotator=?,updated_at=? WHERE id=? AND dataset_root=?",
                                    (row["start"], row["end"], row["skill"], row["phase"], row["leaf_task"], row["manager_duration"], row["confidence"], row["notes"], row["annotator"], now, item_id, scope))
                        self._audit(con, scope, point_path, "task", item_id, "update", row["annotator"], old_db, new_db, now)
                else:
                    cur = con.execute("""INSERT INTO task_annotations
                        (dataset_root,point_path,start,end,skill,phase,leaf_task,manager_duration,confidence,notes,annotator,source_key,extra_json,created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL,'{}',?,?)""",
                        (scope, point_path, row["start"], row["end"], row["skill"], row["phase"], row["leaf_task"], row["manager_duration"], row["confidence"], row["notes"], row["annotator"], now, now))
                    new_id = int(cur.lastrowid)
                    retained.add(new_id)
                    self._audit(con, scope, point_path, "task", new_id, "create", row["annotator"], None,
                                {k: v for k, v in row.items() if k != "id"}, now)
            for item_id, old in old_by_id.items():
                if item_id not in retained:
                    self._audit(con, scope, point_path, "task", item_id, "delete", old.get("annotator", ""), old, None, now)
                    con.execute("DELETE FROM task_annotations WHERE id=? AND dataset_root=?", (item_id, scope))
            # Track point-level review progress without overriding an explicit Reviewed/Rejected state.
            review = con.execute("SELECT status FROM point_reviews WHERE dataset_root=? AND point_path=?", (scope, point_path)).fetchone()
            current_status = review["status"] if review else "Not started"
            if normalized and current_status == "Not started":
                con.execute("INSERT INTO point_reviews(dataset_root,point_path,status,annotator,notes,updated_at) VALUES(?,?, 'In progress', ?, '', ?) ON CONFLICT(dataset_root,point_path) DO UPDATE SET status='In progress',updated_at=excluded.updated_at",
                            (scope, point_path, default_annotator, now))
            elif review is None:
                con.execute("INSERT INTO point_reviews(dataset_root,point_path,status,annotator,notes,updated_at) VALUES(?,?, 'Not started', ?, '', ?)",
                            (scope, point_path, default_annotator, now))
        return len(normalized)

    def review_for_point(self, dataset_root: str | Path, point_path: str) -> dict[str, Any]:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            row = con.execute("SELECT * FROM point_reviews WHERE dataset_root=? AND point_path=?", (scope, point_path)).fetchone()
        return dict(row) if row else {"status": "Not started", "annotator": "", "notes": ""}

    def set_review(self, dataset_root: str | Path, point_path: str, status: str, annotator: str, notes: str) -> None:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status {status!r}")
        scope, now = str(Path(dataset_root).resolve()), utc_now()
        with self.connect() as con:
            old_row = con.execute("SELECT * FROM point_reviews WHERE dataset_root=? AND point_path=?", (scope, point_path)).fetchone()
            old = dict(old_row) if old_row else None
            con.execute("INSERT INTO point_reviews(dataset_root,point_path,status,annotator,notes,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(dataset_root,point_path) DO UPDATE SET status=excluded.status,annotator=excluded.annotator,notes=excluded.notes,updated_at=excluded.updated_at",
                        (scope, point_path, status, annotator, notes, now))
            new = {"status": status, "annotator": annotator, "notes": notes}
            comparable_old = {k: old.get(k) for k in ("status", "annotator", "notes")} if old else None
            if comparable_old != new:
                self._audit(con, scope, point_path, "review", None, "update", annotator, comparable_old, new, now)

    def state_for_point(self, dataset_root: str | Path, point_path: str) -> list[dict[str, Any]]:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            rows = con.execute("SELECT * FROM state_annotations WHERE dataset_root=? AND point_path=? ORDER BY step,id", (scope, point_path)).fetchall()
        out = []
        for raw in rows:
            item = dict(raw)
            item.update(json.loads(item.pop("payload_json")))
            out.append(item)
        return out

    def save_state_for_point(self, dataset_root: str | Path, point_path: str, rows: list[dict[str, Any]], point_length: int, annotator: str = "") -> int:
        scope, now = str(Path(dataset_root).resolve()), utc_now()
        normalized: list[dict[str, Any]] = []
        seen: set[int] = set()
        for i, row in enumerate(rows):
            if _blank(row.get("step")) and not any(not _blank(row.get(k)) for k in (*STATE_VALUE_FIELDS, *STATE_EVENT_NAMES, "events_json")):
                continue
            step = _strict_int(row.get("step"), f"state row {i + 1} step")
            if not 0 <= step < int(point_length):
                raise ValueError(f"State row {i + 1}: step must be between 0 and {point_length - 1}")
            if step in seen:
                raise ValueError(f"Only one state row per point/step is allowed (duplicate step {step})")
            seen.add(step)
            payload: dict[str, Any] = {}
            for field in STATE_VALUE_FIELDS:
                value = row.get(field)
                if _blank(value):
                    continue
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"State row {i + 1}: {field} must be valid JSON") from exc
                if field.endswith("_xy"):
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        raise ValueError(f"State row {i + 1}: {field} must be a pair [x,y]")
                    nums = [float(x) for x in value]
                    if not all(math.isfinite(x) for x in nums):
                        raise ValueError(f"State row {i + 1}: {field} values must be finite")
                    if field in ("ball_xy", "player_xy") and not all(0 <= x <= 1 for x in nums):
                        raise ValueError(f"State row {i + 1}: {field} coordinates must be normalized to [0,1]")
                    payload[field] = nums
                else:
                    number = float(value)
                    if not math.isfinite(number) or (field == "time_to_contact" and number < 0):
                        raise ValueError(f"State row {i + 1}: invalid {field}")
                    payload[field] = number
            events: dict[str, int] = {}
            for event in STATE_EVENT_NAMES:
                value = row.get(event)
                if _blank(value) or str(value).lower() in {"unknown", "none", "nan"}:
                    continue
                if isinstance(value, str) and value.lower() in {"yes", "true"}:
                    value = 1
                elif isinstance(value, str) and value.lower() in {"no", "false"}:
                    value = 0
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise ValueError(f"State row {i + 1}: {event} must be yes/no/unknown")
                if value not in (0, 1):
                    raise ValueError(f"State row {i + 1}: {event} must be yes/no/unknown")
                events[event] = value
            events_raw = row.get("events_json")
            if not _blank(events_raw):
                parsed_events = json.loads(events_raw) if isinstance(events_raw, str) else events_raw
                if not isinstance(parsed_events, dict):
                    raise ValueError(f"State row {i + 1}: events_json must be a JSON object")
                for event_name, event_value in parsed_events.items():
                    if event_name not in STATE_EVENT_NAMES:
                        raise ValueError(f"State row {i + 1}: unknown event field {event_name!r}")
                    if event_value not in (0, 1, False, True):
                        raise ValueError(f"State row {i + 1}: event {event_name} must be 0, 1, or omitted for unknown")
                    events[event_name] = int(event_value)
            if events:
                payload["events"] = events
            item_id = None if _blank(row.get("id")) else _strict_int(row.get("id"), f"state row {i + 1} id")
            normalized.append({"id": item_id, "step": step, "payload": payload})
        normalized.sort(key=lambda r: r["step"])
        with self.connect() as con:
            old_rows = [dict(r) for r in con.execute("SELECT * FROM state_annotations WHERE dataset_root=? AND point_path=? ORDER BY id", (scope, point_path))]
            old_by_id = {int(r["id"]): r for r in old_rows}
            retained: set[int] = set()
            for row in normalized:
                old = old_by_id.get(row["id"]) if row["id"] is not None else None
                payload_json = canonical_json(row["payload"])
                if old:
                    item_id = int(old["id"])
                    retained.add(item_id)
                    if int(old["step"]) != row["step"] or old["payload_json"] != payload_json:
                        con.execute("UPDATE state_annotations SET step=?,payload_json=?,updated_at=? WHERE id=? AND dataset_root=?",
                                    (row["step"], payload_json, now, item_id, scope))
                        self._audit(con, scope, point_path, "state", item_id, "update", annotator,
                                    {"step": old["step"], "payload": json.loads(old["payload_json"])},
                                    {"step": row["step"], "payload": row["payload"]}, now)
                else:
                    cur = con.execute("INSERT INTO state_annotations(dataset_root,point_path,step,payload_json,source_key,created_at,updated_at) VALUES(?,?,?,?,NULL,?,?)",
                                      (scope, point_path, row["step"], payload_json, now, now))
                    item_id = int(cur.lastrowid)
                    retained.add(item_id)
                    self._audit(con, scope, point_path, "state", item_id, "create", annotator, None,
                                {"step": row["step"], "payload": row["payload"]}, now)
            for item_id, old in old_by_id.items():
                if item_id not in retained:
                    self._audit(con, scope, point_path, "state", item_id, "delete", annotator,
                                {"step": old["step"], "payload": json.loads(old["payload_json"])}, None, now)
                    con.execute("DELETE FROM state_annotations WHERE id=? AND dataset_root=?", (item_id, scope))
        return len(normalized)

    def export_task_jsonl(self, dataset_root: str | Path, destination: str | Path) -> int:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            rows = con.execute("SELECT point_path,start,end,skill,phase,leaf_task,manager_duration FROM task_annotations WHERE dataset_root=? ORDER BY point_path,start,end,id", (scope,)).fetchall()
        output: list[dict[str, Any]] = []
        for r in rows:
            row: dict[str, Any] = {"path": r["point_path"], "start": int(r["start"]), "end": int(r["end"]),
                                   "skill": r["skill"], "phase": r["phase"]}
            if r["leaf_task"]:
                row["leaf_task"] = r["leaf_task"]
            if r["manager_duration"] is not None:
                row["manager_duration"] = int(r["manager_duration"])
            output.append(row)
        atomic_write_jsonl(destination, output)
        return len(output)

    def export_state_jsonl(self, dataset_root: str | Path, destination: str | Path) -> int:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            rows = con.execute("SELECT point_path,step,payload_json FROM state_annotations WHERE dataset_root=? ORDER BY point_path,step,id", (scope,)).fetchall()
        output = [{"path": r["point_path"], "step": int(r["step"]), **json.loads(r["payload_json"])} for r in rows]
        atomic_write_jsonl(destination, output)
        return len(output)

    def audit_for_point(self, dataset_root: str | Path, point_path: str, limit: int = 30) -> list[dict[str, Any]]:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            rows = con.execute("SELECT * FROM audit_events WHERE dataset_root=? AND point_path IN (?, '*') ORDER BY id DESC LIMIT ?", (scope, point_path, int(limit))).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for k in ("old_json", "new_json"):
                if item.get(k):
                    try:
                        item[k] = json.loads(item[k])
                    except json.JSONDecodeError:
                        pass
            result.append(item)
        return result

    def point_summary(self, dataset_root: str | Path) -> dict[str, dict[str, Any]]:
        scope = str(Path(dataset_root).resolve())
        with self.connect() as con:
            segment_rows = con.execute(
                "SELECT point_path,skill,COUNT(*) AS count FROM task_annotations WHERE dataset_root=? GROUP BY point_path,skill", (scope,)
            ).fetchall()
            statuses = {r["point_path"]: {"status": r["status"], "annotator": r["annotator"], "notes": r["notes"]}
                        for r in con.execute("SELECT * FROM point_reviews WHERE dataset_root=?", (scope,))}
        task_counts: dict[str, int] = {}
        skills_by_point: dict[str, set[str]] = {}
        for row in segment_rows:
            task_counts[row["point_path"]] = task_counts.get(row["point_path"], 0) + int(row["count"])
            skills_by_point.setdefault(row["point_path"], set()).add(row["skill"])
        return {k: {"segments": task_counts.get(k, 0), "skills": sorted(skills_by_point.get(k, set())),
                    **statuses.get(k, {"status": "Not started", "annotator": "", "notes": ""})}
                for k in set(task_counts) | set(statuses)}

    @staticmethod
    def _audit(con: sqlite3.Connection, scope: str, point_path: str, annotation_type: str,
               annotation_id: int | None, action: str, annotator: str,
               old: Any, new: Any, timestamp: str) -> None:
        con.execute("INSERT INTO audit_events(dataset_root,point_path,annotation_type,annotation_id,action,annotator,old_json,new_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (scope, point_path, annotation_type, annotation_id, action, annotator,
                     canonical_json(old) if old is not None else None,
                     canonical_json(new) if new is not None else None, timestamp))


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _strict_int(value: Any, name: str) -> int:
    if _blank(value):
        raise ValueError(f"{name} is required")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)
