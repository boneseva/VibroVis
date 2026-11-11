"""
Persistent Raven annotation database (SQLite, separate from the manual-label store).

Location: $VIBROVIS_ANNOTATIONS_DIR/raven_annotations.sqlite3, defaulting to
<repo>/cache/annotations/. The directory is git- and docker-ignored and must be on
persistent storage in deployments.

Original Raven data (batches, selections, annotation values) is never modified after
import. Batch status changes (superseded/reverted) are recorded in batch_events.
Clip links are derived data tied to a model and partition signature.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import INTERPRETATION_RULESET
from .interpret import interpret_label
from .matcher import MatchResult
from .parser import ParsedSelection, ParsedTable

SCHEMA_VERSION = 3
DB_FILENAME = "raven_annotations.sqlite3"

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS annotators (
    annotator_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS import_batches (
    batch_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    annotator_id INTEGER NOT NULL REFERENCES annotators(annotator_id),
    source_filename TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    encoding TEXT,
    raw_content BLOB NOT NULL,
    headers_json TEXT NOT NULL,
    column_map_json TEXT NOT NULL,
    annotation_columns_json TEXT NOT NULL,
    primary_label_column TEXT,
    options_json TEXT NOT NULL,
    model_name TEXT NOT NULL,
    partition_signature TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'reverted')),
    superseded_by TEXT REFERENCES import_batches(batch_id),
    status_changed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_batches_active_file
    ON import_batches(project, annotator_id, file_sha256) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS ix_batches_filename ON import_batches(project, annotator_id, source_filename);

CREATE TABLE IF NOT EXISTS batch_events (
    event_id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES import_batches(batch_id),
    event TEXT NOT NULL,
    at TEXT NOT NULL,
    actor TEXT,
    detail_json TEXT
);
CREATE TABLE IF NOT EXISTS selections (
    selection_pk INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES import_batches(batch_id),
    ordinal INTEGER NOT NULL,
    raven_selection TEXT NOT NULL,
    source_lines_json TEXT NOT NULL,
    views_json TEXT NOT NULL,
    raw_rows_json TEXT NOT NULL,
    channel_raw TEXT,
    begin_time_raw TEXT,
    end_time_raw TEXT,
    file_offset_raw TEXT,
    begin_file TEXT,
    end_file TEXT,
    begin_path TEXT,
    end_path TEXT,
    low_freq_raw TEXT,
    high_freq_raw TEXT,
    begin_time_s REAL,
    end_time_s REAL,
    file_offset_s REAL,
    low_freq_hz REAL,
    high_freq_hz REAL,
    location TEXT,
    location_source TEXT,
    source_stem TEXT,
    channel INTEGER,
    recording_id TEXT,
    file_t0_s REAL,
    file_t1_s REAL,
    time_mode_used TEXT,
    status TEXT NOT NULL,
    status_message TEXT,
    flags_json TEXT NOT NULL,
    UNIQUE (batch_id, ordinal)
);
CREATE INDEX IF NOT EXISTS ix_selections_recording ON selections(recording_id);

CREATE TABLE IF NOT EXISTS selection_annotations (
    selection_pk INTEGER NOT NULL REFERENCES selections(selection_pk),
    column_name TEXT NOT NULL,
    column_order INTEGER NOT NULL,
    value TEXT NOT NULL,
    is_primary INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (selection_pk, column_name)
);
CREATE TABLE IF NOT EXISTS selection_clip_links (
    selection_pk INTEGER NOT NULL REFERENCES selections(selection_pk),
    model_name TEXT NOT NULL,
    partition_signature TEXT NOT NULL,
    semantic_clip_key TEXT NOT NULL,
    recording_id TEXT NOT NULL,
    clip_time_s REAL NOT NULL,
    clip_duration_s REAL NOT NULL,
    overlap_s REAL NOT NULL,
    frac_of_clip REAL NOT NULL,
    frac_of_selection REAL NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (selection_pk, model_name, semantic_clip_key)
);
CREATE INDEX IF NOT EXISTS ix_links_clip ON selection_clip_links(semantic_clip_key);
"""

# {from_version: statements reaching from_version + 1}. New databases are created at v1 and migrated.
MIGRATIONS: Dict[int, str] = {
    1: """
ALTER TABLE import_batches ADD COLUMN table_meta_json TEXT;
ALTER TABLE selections ADD COLUMN file_t1_original_s REAL;
CREATE TABLE IF NOT EXISTS annotation_interpretations (
    selection_pk INTEGER NOT NULL REFERENCES selections(selection_pk),
    column_name TEXT NOT NULL,
    ruleset TEXT NOT NULL,
    species TEXT,
    uncertain INTEGER NOT NULL,
    rule TEXT NOT NULL,
    PRIMARY KEY (selection_pk, column_name, ruleset)
);
CREATE INDEX IF NOT EXISTS ix_interp_species ON annotation_interpretations(species)
""",
    # Re-import identity includes the location when it comes from the table name: two byte-identical
    # (e.g. header-only) tables for different locations are different imports.
    2: """
ALTER TABLE import_batches ADD COLUMN identity_scope TEXT NOT NULL DEFAULT '';
UPDATE import_batches SET identity_scope = 'location:' || json_extract(table_meta_json, '$.location')
    WHERE json_extract(options_json, '$.location_mode') = 'table_name'
      AND json_extract(table_meta_json, '$.location') IS NOT NULL;
DROP INDEX IF EXISTS ux_batches_active_file;
CREATE UNIQUE INDEX IF NOT EXISTS ux_batches_active_identity
    ON import_batches(project, annotator_id, file_sha256, identity_scope) WHERE status = 'active'
""",
}


def default_db_path() -> str:
    base = os.environ.get("VIBROVIS_ANNOTATIONS_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "annotations")
    return os.path.join(base, DB_FILENAME)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _execute_statements(conn: sqlite3.Connection, script: str) -> None:
    # executescript() would COMMIT the surrounding transaction; run statements one by one instead.
    for stmt in script.split(";"):
        if stmt.strip():
            conn.execute(stmt)


def selection_flags(sel: ParsedSelection, res: MatchResult) -> List[str]:
    return sel.flags + list(res.flags) + sorted(f"number_error:{k}" for k in sel.number_errors)


class SchemaVersionError(RuntimeError):
    pass


@dataclass
class BatchPayload:
    project: str
    annotator: str
    raw_content: bytes
    table: ParsedTable
    results: List[MatchResult]
    options: dict
    model_name: str
    partition_signature: str
    stats: dict
    supersede_batch_ids: List[str] = field(default_factory=list)
    identity_scope: str = ""
    note: Optional[str] = None


class AnnotationStore:
    _init_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_db_path()
        self._initialized = False

    @contextmanager
    def connect(self):
        self._ensure_schema()
        conn = self._raw_connect()
        try:
            yield conn
        finally:
            conn.close()

    def _raw_connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _ensure_schema(self):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._raw_connect()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("BEGIN IMMEDIATE")
                has_meta = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'").fetchone()
                version = 0
                if has_meta:
                    row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
                    version = int(row[0]) if row else 0
                if version > SCHEMA_VERSION:
                    conn.execute("ROLLBACK")
                    raise SchemaVersionError(
                        f"{self.db_path} has schema v{version}; this code supports up to v{SCHEMA_VERSION}.")
                if version == 0:
                    _execute_statements(conn, SCHEMA_V1)
                    version = 1
                    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('created_at', ?)", (_now(),))
                while version < SCHEMA_VERSION:
                    _execute_statements(conn, MIGRATIONS[version])
                    version += 1
                conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                             (str(version),))
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
            self._initialized = True

    # ---- reads -------------------------------------------------------------

    def schema_version(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0])

    def list_annotators(self) -> List[str]:
        with self.connect() as conn:
            return [r["name"] for r in conn.execute("SELECT name FROM annotators ORDER BY name COLLATE NOCASE")]

    def list_batches(self, project: Optional[str] = None) -> List[dict]:
        sql = ("SELECT b.batch_id, b.project, a.name AS annotator, b.source_filename, b.file_sha256, b.encoding, "
               "b.model_name, b.created_at, b.created_by, b.status, b.superseded_by, b.status_changed_at, b.stats_json "
               "FROM import_batches b JOIN annotators a USING (annotator_id)")
        args = ()
        if project:
            sql += " WHERE b.project = ?"
            args = (project,)
        sql += " ORDER BY b.created_at DESC, b.batch_id"
        with self.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, args)]
        for r in rows:
            r["stats"] = json.loads(r.pop("stats_json"))
        return rows

    def find_active_duplicate(self, project: str, annotator: str, sha256: str, identity_scope: str = "") -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT b.batch_id FROM import_batches b JOIN annotators a USING (annotator_id) "
                "WHERE b.project=? AND a.name=? AND b.file_sha256=? AND b.identity_scope=? AND b.status='active'",
                (project, annotator.strip(), sha256, identity_scope)).fetchone()
        return row["batch_id"] if row else None

    def find_supersede_candidates(self, project: str, annotator: str, source_filename: str, sha256: str) -> List[dict]:
        """Active batches by the same annotator with the same file name but different content."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT b.batch_id, b.source_filename, b.created_at FROM import_batches b "
                "JOIN annotators a USING (annotator_id) "
                "WHERE b.project=? AND a.name=? AND b.source_filename=? AND b.file_sha256<>? AND b.status='active' "
                "ORDER BY b.created_at", (project, annotator.strip(), source_filename, sha256)).fetchall()
        return [dict(r) for r in rows]

    def batch_events(self, batch_id: str) -> List[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT event, at, actor, detail_json FROM batch_events WHERE batch_id=? ORDER BY event_id",
                (batch_id,))]

    def selections(self, batch_id: str) -> List[dict]:
        with self.connect() as conn:
            sels = [dict(r) for r in conn.execute(
                "SELECT * FROM selections WHERE batch_id=? ORDER BY ordinal", (batch_id,))]
            for s in sels:
                s["annotations"] = {r["column_name"]: r["value"] for r in conn.execute(
                    "SELECT column_name, value FROM selection_annotations WHERE selection_pk=? ORDER BY column_order",
                    (s["selection_pk"],))}
                s["links"] = [dict(r) for r in conn.execute(
                    "SELECT * FROM selection_clip_links WHERE selection_pk=? ORDER BY clip_time_s",
                    (s["selection_pk"],))]
                s["interpretations"] = {r["column_name"]: dict(r) for r in conn.execute(
                    "SELECT column_name, ruleset, species, uncertain, rule FROM annotation_interpretations "
                    "WHERE selection_pk=?", (s["selection_pk"],))}
        return sels

    def active_clip_annotations(self, project: str, model_name: Optional[str] = None) -> List[dict]:
        """Flat view of active annotation values per linked clip, per annotator (for display/export)."""
        sql = ("SELECT a.name AS annotator, b.batch_id, s.selection_pk, s.raven_selection, s.recording_id, "
               "s.file_t0_s, s.file_t1_s, l.semantic_clip_key, l.model_name, l.overlap_s, l.frac_of_clip, "
               "l.frac_of_selection, l.relation, sa.column_name, sa.value, sa.is_primary, "
               "ai.species AS interpreted_species, ai.uncertain AS interpreted_uncertain, ai.rule AS interpretation_rule "
               "FROM selection_clip_links l JOIN selections s USING (selection_pk) "
               "JOIN import_batches b ON b.batch_id = s.batch_id JOIN annotators a USING (annotator_id) "
               "JOIN selection_annotations sa ON sa.selection_pk = s.selection_pk "
               "LEFT JOIN annotation_interpretations ai ON ai.selection_pk = sa.selection_pk "
               "AND ai.column_name = sa.column_name AND ai.ruleset = ? "
               "WHERE b.status='active' AND b.project=?")
        args = [INTERPRETATION_RULESET, project]
        if model_name:
            sql += " AND l.model_name=?"
            args.append(model_name)
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, args)]

    # ---- writes ------------------------------------------------------------

    def commit_batches(self, payloads: List[BatchPayload], actor: Optional[str] = None) -> List[dict]:
        """Insert all payloads in ONE transaction. Exact re-imports are skipped (idempotent)."""
        outcomes = []
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for p in payloads:
                    outcomes.append(self._insert_batch(conn, p, actor))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return outcomes

    def _annotator_id(self, conn, name: str) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("Annotator name is required.")
        row = conn.execute("SELECT annotator_id FROM annotators WHERE name=?", (name,)).fetchone()
        if row:
            return row[0]
        return conn.execute("INSERT INTO annotators(name, created_at) VALUES (?, ?)", (name, _now())).lastrowid

    def _insert_batch(self, conn, p: BatchPayload, actor: Optional[str]) -> dict:
        t = p.table
        if t.fatal_error:
            raise ValueError(f"{t.filename}: cannot import a file that failed to parse ({t.fatal_error}).")
        if len(p.results) != len(t.selections):
            raise ValueError(f"{t.filename}: match results do not align with selections.")
        annotator_id = self._annotator_id(conn, p.annotator)
        dup = conn.execute(
            "SELECT batch_id FROM import_batches WHERE project=? AND annotator_id=? AND file_sha256=? "
            "AND identity_scope=? AND status='active'",
            (p.project, annotator_id, t.sha256, p.identity_scope)).fetchone()
        if dup:
            return {"filename": t.filename, "outcome": "duplicate", "batch_id": dup[0]}

        for old_id in p.supersede_batch_ids:
            old = conn.execute("SELECT project, annotator_id, status FROM import_batches WHERE batch_id=?",
                               (old_id,)).fetchone()
            if not old or old["status"] != "active" or old["project"] != p.project or old["annotator_id"] != annotator_id:
                raise ValueError(f"Batch {old_id} cannot be superseded (missing, not active, or different annotator).")

        batch_id = uuid.uuid4().hex
        now = _now()
        conn.execute(
            "INSERT INTO import_batches(batch_id, project, annotator_id, source_filename, file_sha256, file_size, "
            "encoding, raw_content, headers_json, column_map_json, annotation_columns_json, primary_label_column, "
            "options_json, model_name, partition_signature, stats_json, warnings_json, note, created_at, created_by, "
            "table_meta_json, identity_scope, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'active')",
            (batch_id, p.project, annotator_id, t.filename, t.sha256, t.size, t.encoding, sqlite3.Binary(p.raw_content),
             _dumps(t.headers), _dumps(t.column_map), _dumps(t.annotation_columns), t.primary_label_column,
             _dumps(p.options), p.model_name, p.partition_signature, _dumps(p.stats), _dumps(t.warnings), p.note,
             now, actor, _dumps(t.table_meta), p.identity_scope))
        self._event(conn, batch_id, "imported", actor, {"selections": len(t.selections), "stats": p.stats})

        for ordinal, (sel, res) in enumerate(zip(t.selections, p.results)):
            f, n = sel.fields, sel.numbers
            pk = conn.execute(
                "INSERT INTO selections(batch_id, ordinal, raven_selection, source_lines_json, views_json, raw_rows_json, "
                "channel_raw, begin_time_raw, end_time_raw, file_offset_raw, begin_file, end_file, begin_path, end_path, "
                "low_freq_raw, high_freq_raw, begin_time_s, end_time_s, file_offset_s, low_freq_hz, high_freq_hz, "
                "location, location_source, source_stem, channel, recording_id, file_t0_s, file_t1_s, time_mode_used, "
                "status, status_message, flags_json, file_t1_original_s) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (batch_id, ordinal, sel.key, _dumps(sel.row_numbers), _dumps(sel.views), _dumps(sel.raw_rows),
                 f.get("channel"), f.get("begin_time"), f.get("end_time"), f.get("file_offset"), f.get("begin_file"),
                 f.get("end_file"), f.get("begin_path"), f.get("end_path"), f.get("low_freq"), f.get("high_freq"),
                 n.get("begin_time"), n.get("end_time"), n.get("file_offset"), n.get("low_freq"), n.get("high_freq"),
                 res.location, res.location_source, res.source_stem, res.channel, res.recording_id,
                 res.file_t0_s, res.file_t1_s, res.time_mode_used, res.status, res.message,
                 _dumps(selection_flags(sel, res)), res.file_t1_original_s)).lastrowid
            for order, col in enumerate(t.annotation_columns):
                conn.execute(
                    "INSERT INTO selection_annotations(selection_pk, column_name, column_order, value, is_primary) "
                    "VALUES (?,?,?,?,?)",
                    (pk, col, order, sel.annotations.get(col, ""), int(col == t.primary_label_column)))
            self._insert_interpretations(conn, pk, sel, t, p.options)
            self._insert_links(conn, pk, res, p.model_name, p.partition_signature)

        for old_id in p.supersede_batch_ids:
            conn.execute("UPDATE import_batches SET status='superseded', superseded_by=?, status_changed_at=? "
                         "WHERE batch_id=?", (batch_id, now, old_id))
            self._event(conn, old_id, "superseded", actor, {"by": batch_id})
        return {"filename": t.filename, "outcome": "imported", "batch_id": batch_id,
                "superseded": list(p.supersede_batch_ids)}

    def revert_batch(self, batch_id: str, actor: Optional[str] = None, reason: str = "") -> None:
        """Mark an active batch as reverted. Data is retained for traceability."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT status FROM import_batches WHERE batch_id=?", (batch_id,)).fetchone()
                if not row:
                    raise ValueError(f"Unknown batch {batch_id}.")
                if row["status"] != "active":
                    raise ValueError(f"Batch {batch_id} is {row['status']}, only active batches can be reverted.")
                conn.execute("UPDATE import_batches SET status='reverted', status_changed_at=? WHERE batch_id=?",
                             (_now(), batch_id))
                self._event(conn, batch_id, "reverted", actor, {"reason": reason})
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _insert_links(conn, pk: int, res: MatchResult, model_name: str, signature: str) -> None:
        for link in res.links:
            conn.execute(
                "INSERT INTO selection_clip_links(selection_pk, model_name, partition_signature, semantic_clip_key, "
                "recording_id, clip_time_s, clip_duration_s, overlap_s, frac_of_clip, frac_of_selection, relation) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (pk, model_name, signature, link.semantic_clip_key, link.recording_id, link.clip_time_s,
                 link.clip_duration_s, link.overlap_s, link.frac_of_clip, link.frac_of_selection, link.relation))

    @staticmethod
    def _insert_interpretations(conn, pk: int, sel: ParsedSelection, t: ParsedTable, options: dict) -> None:
        columns = [t.primary_label_column] if t.primary_label_column else t.annotation_columns
        vocab = options.get("species_vocabulary") or []
        synonyms = options.get("species_synonyms") or {}
        for col in columns:
            it = interpret_label(sel.annotations.get(col, ""), vocab, synonyms)
            conn.execute(
                "INSERT OR REPLACE INTO annotation_interpretations(selection_pk, column_name, ruleset, species, "
                "uncertain, rule) VALUES (?,?,?,?,?,?)",
                (pk, col, INTERPRETATION_RULESET, it.species, int(it.uncertain), it.rule))

    def batch_for_relink(self, batch_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT b.batch_id, b.project, b.status, b.source_filename, b.raw_content, b.options_json, "
                "b.model_name, a.name AS annotator FROM import_batches b JOIN annotators a USING (annotator_id) "
                "WHERE b.batch_id=?", (batch_id,)).fetchone()
            if not row:
                raise ValueError(f"Unknown batch {batch_id}.")
            out = dict(row)
            out["selection_keys"] = [r["raven_selection"] for r in conn.execute(
                "SELECT raven_selection FROM selections WHERE batch_id=? ORDER BY ordinal", (batch_id,))]
            out["status_counts"] = {r[0]: r[1] for r in conn.execute(
                "SELECT status, COUNT(*) FROM selections WHERE batch_id=? GROUP BY status", (batch_id,))}
        return out

    def replace_resolution(self, batch_id: str, table: ParsedTable, results: List[MatchResult], model_name: str,
                           signature: str, actor: Optional[str], detail: dict) -> None:
        """Re-link: overwrite derived resolution columns and links of one batch; original Raven data untouched."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                status = conn.execute("SELECT status FROM import_batches WHERE batch_id=?", (batch_id,)).fetchone()
                if not status or status[0] != "active":
                    raise ValueError(f"Batch {batch_id} is not active.")
                pks = [r[0] for r in conn.execute(
                    "SELECT selection_pk FROM selections WHERE batch_id=? ORDER BY ordinal", (batch_id,))]
                if len(pks) != len(results):
                    raise ValueError(f"Batch {batch_id}: {len(pks)} stored selections vs {len(results)} re-matched.")
                for pk, sel, res in zip(pks, table.selections, results):
                    conn.execute(
                        "UPDATE selections SET location=?, location_source=?, source_stem=?, channel=?, recording_id=?, "
                        "file_t0_s=?, file_t1_s=?, file_t1_original_s=?, time_mode_used=?, status=?, status_message=?, "
                        "flags_json=? WHERE selection_pk=?",
                        (res.location, res.location_source, res.source_stem, res.channel, res.recording_id,
                         res.file_t0_s, res.file_t1_s, res.file_t1_original_s, res.time_mode_used, res.status,
                         res.message, _dumps(selection_flags(sel, res)), pk))
                    conn.execute("DELETE FROM selection_clip_links WHERE selection_pk=? AND model_name=?",
                                 (pk, model_name))
                    self._insert_links(conn, pk, res, model_name, signature)
                self._event(conn, batch_id, "relinked", actor, detail)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _event(conn, batch_id, event, actor, detail):
        conn.execute("INSERT INTO batch_events(batch_id, event, at, actor, detail_json) VALUES (?,?,?,?,?)",
                     (batch_id, event, _now(), actor, _dumps(detail)))

    def backup_to(self, dest_path: str) -> str:
        """Consistent online copy using SQLite's backup API (safe while the app is running)."""
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with self.connect() as src:
            dest = sqlite3.connect(dest_path)
            try:
                src.backup(dest)
                # Single self-contained file (no -wal/-shm side files) for archiving.
                dest.execute("PRAGMA journal_mode = DELETE")
            finally:
                dest.close()
        return dest_path
