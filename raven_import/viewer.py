"""
Read-only access to imported Raven annotations for colouring VibroVis points (see aggregate.py).

Nothing here writes: the database is opened with SQLite's read-only URI flags (never through
AnnotationStore, whose schema check writes), and no manual-label state is touched.

Point matching: a stored clip link names a recording (``<project>::<audio path>::ch<N>``) and a clip start.
Plotted rows carry ``mp3_file``, ``channel`` and ``clip_time``. A link maps to
  - "exact":      the plotted single clip with the same recording and start;
  - "merged":     the one merged point (merge mode) whose span contains the linked clip;
  - "ambiguous":  several merged points contain it (different clusters) -> not highlighted;
  - "not_in_view": no plotted point (other location, filters, sampling).
A selection's original interval is never altered; points are only those of its stored clip links.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .service import outcome_class

EPS = 1e-6

SELECTION_COLUMNS = [
    "selection_pk", "batch_id", "annotator", "table_file", "quick_pass", "raven_selection", "status", "outcome",
    "reason", "location", "location_source", "source_stem", "channel", "recording_id", "t0", "t1", "t1_original",
    "begin_file", "flags", "label", "label_column", "species", "uncertain", "interpretation_rule", "n_clips",
    "imported_at",
]


def connect_readonly(db_path: str) -> Optional[sqlite3.Connection]:
    """Open without writing: immutable when no WAL content is pending, plain read-only otherwise."""
    if not db_path or not os.path.exists(db_path):
        return None
    wal = db_path + "-wal"
    uri_path = db_path.replace("\\", "/")
    if os.path.exists(wal) and os.path.getsize(wal) > 0:
        uri = f"file:{uri_path}?mode=ro"
    else:
        uri = f"file:{uri_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_recording_id(recording_id: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """'<project>::<audio path>::ch<N>' -> (audio path with '/', channel)."""
    parts = str(recording_id or "").split("::")
    if len(parts) < 3 or not parts[2].lower().startswith("ch"):
        return None, None
    try:
        return parts[1].replace("\\", "/"), int(parts[2][2:])
    except ValueError:
        return None, None


def load_selections(db_path: str, project: str = "Citimon") -> pd.DataFrame:
    """One row per selection of every active import, with its primary label and interpretation."""
    conn = connect_readonly(db_path)
    if conn is None:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        has_v2 = "annotation_interpretations" in tables
        rows = conn.execute(
            "SELECT s.selection_pk, s.batch_id, a.name AS annotator, b.source_filename AS table_file, "
            + ("b.table_meta_json, " if has_v2 else "NULL AS table_meta_json, ")
            + "s.raven_selection, s.status, s.status_message AS reason, s.location, s.location_source, "
            "s.source_stem, s.channel, s.recording_id, s.file_t0_s AS t0, s.file_t1_s AS t1, "
            + ("s.file_t1_original_s AS t1_original, " if has_v2 else "NULL AS t1_original, ")
            + "s.begin_file, s.flags_json, b.created_at AS imported_at, "
            "(SELECT COUNT(*) FROM selection_clip_links l WHERE l.selection_pk = s.selection_pk) AS n_clips "
            "FROM selections s JOIN import_batches b ON b.batch_id = s.batch_id "
            "JOIN annotators a ON a.annotator_id = b.annotator_id "
            "WHERE b.status = 'active' AND b.project = ? ORDER BY b.source_filename, s.ordinal", (project,)
        ).fetchall()
        labels: Dict[int, Tuple[str, str]] = {}
        for r in conn.execute("SELECT selection_pk, column_name, value, is_primary, column_order "
                              "FROM selection_annotations ORDER BY selection_pk, is_primary DESC, column_order"):
            labels.setdefault(r["selection_pk"], (r["column_name"], r["value"]))
        interp: Dict[Tuple[int, str], sqlite3.Row] = {}
        if has_v2:
            for r in conn.execute("SELECT selection_pk, column_name, species, uncertain, rule "
                                  "FROM annotation_interpretations"):
                interp[(r["selection_pk"], r["column_name"])] = r
    finally:
        conn.close()

    out = []
    for r in rows:
        meta = json.loads(r["table_meta_json"]) if r["table_meta_json"] else {}
        col, value = labels.get(r["selection_pk"], (None, ""))
        it = interp.get((r["selection_pk"], col)) if col else None
        out.append({
            "selection_pk": r["selection_pk"], "batch_id": r["batch_id"], "annotator": r["annotator"],
            "table_file": r["table_file"], "quick_pass": bool((meta or {}).get("quick_pass")),
            "raven_selection": r["raven_selection"], "status": r["status"], "outcome": outcome_class(r["status"]),
            "reason": r["reason"] or "", "location": r["location"] or (meta or {}).get("location"),
            "location_source": r["location_source"], "source_stem": r["source_stem"], "channel": r["channel"],
            "recording_id": r["recording_id"], "t0": r["t0"], "t1": r["t1"], "t1_original": r["t1_original"],
            "begin_file": r["begin_file"], "flags": json.loads(r["flags_json"] or "[]"),
            "label": value or "", "label_column": col,
            "species": it["species"] if it is not None else None,
            "uncertain": bool(it["uncertain"]) if it is not None else ("?" in (value or "")),
            "interpretation_rule": it["rule"] if it is not None else None,
            "n_clips": r["n_clips"], "imported_at": r["imported_at"],
        })
    return pd.DataFrame(out, columns=SELECTION_COLUMNS)


def load_links(db_path: str, selection_pks: Optional[Iterable[int]] = None,
               model_name: Optional[str] = None) -> pd.DataFrame:
    cols = ["selection_pk", "model_name", "recording_id", "audio_path", "channel", "clip_time_s",
            "clip_duration_s", "overlap_s", "frac_of_clip", "frac_of_selection", "relation"]
    conn = connect_readonly(db_path)
    if conn is None:
        return pd.DataFrame(columns=cols)
    try:
        sql = ("SELECT selection_pk, model_name, recording_id, clip_time_s, clip_duration_s, overlap_s, "
               "frac_of_clip, frac_of_selection, relation FROM selection_clip_links")
        args: list = []
        if model_name:
            sql += " WHERE model_name = ?"
            args.append(model_name)
        df = pd.DataFrame([dict(r) for r in conn.execute(sql, args)])
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=cols)
    if selection_pks is not None:
        df = df[df["selection_pk"].isin(set(selection_pks))]
    parsed = df["recording_id"].map(parse_recording_id)
    df["audio_path"] = parsed.map(lambda p: p[0])
    df["channel"] = parsed.map(lambda p: p[1])
    return df[cols].reset_index(drop=True)


def _plotted_keys(dff: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "plot_id": dff["plot_id"] if "plot_id" in dff.columns else dff.index,
        "audio_path": dff["mp3_file"].astype(str).str.replace("\\", "/", regex=False),
        "channel": pd.to_numeric(dff["channel"].astype(str), errors="coerce"),
        "clip_time": pd.to_numeric(dff["clip_time"], errors="coerce").astype(float),
        "clip_duration": pd.to_numeric(dff.get("clip_duration", 5.0), errors="coerce").astype(float),
        "clip_count": pd.to_numeric(dff.get("clip_count", 1), errors="coerce").fillna(1).astype(int),
    })
    return out.reset_index(drop=True)


def map_links_to_points(links: pd.DataFrame, dff: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add plot_id + match ('exact' | 'merged' | 'ambiguous' | 'not_in_view') to every link."""
    result = links.copy()
    result["plot_id"] = None
    result["match"] = "not_in_view"
    if result.empty or dff is None or dff.empty:
        return result
    pts = _plotted_keys(dff)
    groups = {k: g for k, g in pts.groupby(["audio_path", "channel"])}
    for i, link in result.iterrows():
        g = groups.get((link["audio_path"], link["channel"]))
        if g is None:
            continue
        start, end = float(link["clip_time_s"]), float(link["clip_time_s"]) + float(link["clip_duration_s"])
        exact = g[(g["clip_count"] == 1) & (np.abs(g["clip_time"] - start) < EPS)]
        if len(exact) == 1:
            result.at[i, "plot_id"], result.at[i, "match"] = exact["plot_id"].iloc[0], "exact"
            continue
        containing = g[(g["clip_count"] > 1) & (g["clip_time"] <= start + EPS)
                       & (g["clip_time"] + g["clip_duration"] >= end - EPS)]
        if len(containing) == 1:
            result.at[i, "plot_id"], result.at[i, "match"] = containing["plot_id"].iloc[0], "merged"
        elif len(containing) > 1 or len(exact) > 1:
            result.at[i, "match"] = "ambiguous"
    return result


def clean_row(row: dict) -> dict:
    """Selection row as a dict with pandas NaN replaced by None."""
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}


def selection_audio_path(row: dict, recordings_root: str) -> Optional[str]:
    """Audio path (relative to the recordings root) for playing a selection's original interval.

    Linked selections use their stored recording. Others use the stored resolved location and file stem when that
    audio file exists on disk (e.g. a recording not yet in the partition); otherwise None.
    """
    row = clean_row(row)
    audio, _ = parse_recording_id(row.get("recording_id"))
    if audio:
        return audio
    loc, stem, ch = row.get("location"), row.get("source_stem"), row.get("channel")
    if not (loc and stem and ch is not None and row.get("t0") is not None):
        return None
    for project_dir in ("Citimon",):
        for name in (f"{stem}_ch{int(ch)}.mp3", f"{stem.upper()}_ch{int(ch)}.mp3"):
            rel = f"{project_dir}/{loc}/{name}"
            if os.path.exists(os.path.join(recordings_root, *rel.split("/"))):
                return rel
    return None
