"""
Parse Raven Pro selection tables (tab-delimited .txt/.tsv).

Every input row is preserved verbatim. Rows are grouped by the ``Selection``
column because Raven writes one row per view (e.g. "Waveform 1" and
"Spectrogram 1") for the same selection. Problems never drop a row: they mark
the selection with a blocking status that the matcher and store carry through.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import ROLES, MEASUREMENT_UNIT_RE, ImportOptions, RavenFormatConfig


class RavenDecodeError(ValueError):
    pass


@dataclass
class ParsedSelection:
    key: str
    row_numbers: List[int]
    views: List[str]
    raw_rows: List[Dict[str, str]]
    fields: Dict[str, Optional[str]]          # role -> raw string from the canonical row
    annotations: Dict[str, str]               # annotation column -> raw value (verbatim)
    numbers: Dict[str, Optional[float]] = field(default_factory=dict)
    number_errors: Dict[str, str] = field(default_factory=dict)
    status: Optional[str] = None              # blocking status found while parsing
    message: str = ""
    flags: List[str] = field(default_factory=list)

    @property
    def duplicate_view_rows(self) -> int:
        return max(0, len(self.raw_rows) - 1)


@dataclass
class ParsedTable:
    filename: str
    sha256: str
    size: int
    encoding: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    column_map: Dict[str, Optional[str]] = field(default_factory=dict)
    annotation_columns: List[str] = field(default_factory=list)
    primary_label_column: Optional[str] = None
    selections: List[ParsedSelection] = field(default_factory=list)
    data_row_count: int = 0
    warnings: List[str] = field(default_factory=list)
    fatal_error: Optional[str] = None
    table_name_stem: Optional[str] = None
    # From format.table_meta_re: {"location", "annotator", "quick_pass"}; None when the name doesn't match.
    table_meta: Optional[Dict[str, object]] = None

    @property
    def duplicate_view_rows(self) -> int:
        return sum(s.duplicate_view_rows for s in self.selections)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_table_meta(filename: str, fmt: RavenFormatConfig) -> Optional[Dict[str, object]]:
    m = re.match(fmt.table_meta_re, os.path.basename(filename or ""))
    if not m:
        return None
    groups = m.groupdict()
    return {
        "location": groups.get("location"),
        "annotator": groups.get("annotator"),
        "quick_pass": bool(groups.get("quick_pass")),
    }


def decode_bytes(data: bytes, encoding: str, fmt: RavenFormatConfig) -> Tuple[str, str, Optional[str]]:
    """Return (text, encoding_used, warning). Raises RavenDecodeError if decoding is impossible."""
    if encoding == "auto":
        try:
            return data.decode("utf-8-sig"), "utf-8", None
        except UnicodeDecodeError:
            try:
                text = data.decode(fmt.fallback_encoding)
            except UnicodeDecodeError as exc:
                raise RavenDecodeError(f"Not valid UTF-8 or {fmt.fallback_encoding}: {exc}") from exc
            return text, fmt.fallback_encoding, (
                f"File is not valid UTF-8; decoded as {fmt.fallback_encoding}. "
                "Check that characters such as š/č/ž display correctly."
            )
    codec = "utf-8-sig" if encoding.lower() in ("utf-8", "utf8") else encoding
    try:
        return data.decode(codec), encoding, None
    except UnicodeDecodeError as exc:
        raise RavenDecodeError(f"Could not decode as {encoding}: {exc}") from exc


def parse_number(raw: Optional[str], decimal_separator: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, error). Empty input returns (None, None)."""
    if raw is None:
        return None, None
    text = raw.strip()
    if not text:
        return None, None
    if decimal_separator == ",":
        if "." in text:
            return None, f"'{text}' contains '.', but decimal separator is ','"
        text = text.replace(",", ".")
    elif "," in text:
        return None, f"'{text}' contains ',' (decimal comma?); decimal separator is '.'"
    try:
        value = float(text)
    except ValueError:
        return None, f"'{text}' is not a number"
    if not math.isfinite(value):
        return None, f"'{text}' is not finite"
    return value, None


def _resolve_column_map(headers: List[str], options: ImportOptions, warnings: List[str]) -> Dict[str, Optional[str]]:
    lowered = {h.lower(): h for h in headers}
    resolved: Dict[str, Optional[str]] = {}
    for role in ROLES:
        mapped = options.column_mapping.get(role)
        if mapped is None:
            # Auto-detect from aliases (case-insensitive).
            resolved[role] = next(
                (lowered[a.lower()] for a in options.format.role_aliases.get(role, []) if a.lower() in lowered),
                None,
            )
        elif mapped == "":
            resolved[role] = None
        elif mapped in headers:
            resolved[role] = mapped
        else:
            resolved[role] = None
            warnings.append(f"Column '{mapped}' mapped to {role} is not in this file; {role} treated as missing.")
    return resolved


def suggest_annotation_columns(headers: List[str], column_map: Dict[str, Optional[str]], fmt: RavenFormatConfig) -> List[str]:
    used = {c for c in column_map.values() if c}
    measurement = set(fmt.measurement_columns)
    return [h for h in headers if h not in used and h not in measurement and not MEASUREMENT_UNIT_RE.search(h)]


def read_headers(data: bytes, options: ImportOptions) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Lightweight header read for the wizard. Returns (headers, encoding_used, error)."""
    try:
        text, used, _ = decode_bytes(data, options.encoding, options.format)
    except RavenDecodeError as exc:
        return [], None, str(exc)
    for line in text.splitlines():
        if line.strip():
            return _split_header(line, options.format.delimiter, []), used, None
    return [], used, "File is empty."


def _split_header(line: str, delimiter: str, warnings: List[str]) -> List[str]:
    headers = []
    seen: Dict[str, int] = {}
    raw_headers = [h.strip().lstrip("﻿") for h in line.split(delimiter)]
    # Raven may end lines with a trailing delimiter; drop trailing empty header cells.
    while raw_headers and not raw_headers[-1]:
        raw_headers.pop()
    for h in raw_headers:
        if h in seen:
            seen[h] += 1
            new_h = f"{h} [{seen[h]}]"
            warnings.append(f"Duplicate column '{h}' renamed to '{new_h}'.")
            h = new_h
        else:
            seen[h] = 1
        headers.append(h)
    return headers


def parse_raven_table(data: bytes, filename: str, options: ImportOptions) -> ParsedTable:
    fmt = options.format
    table = ParsedTable(filename=os.path.basename(filename or ""), sha256=sha256_bytes(data), size=len(data))

    m = re.match(fmt.table_name_stem_re, table.filename, flags=re.IGNORECASE)
    table.table_name_stem = m.group("stem") if m else None
    table.table_meta = parse_table_meta(table.filename, fmt)

    try:
        text, used, warning = decode_bytes(data, options.encoding, fmt)
    except RavenDecodeError as exc:
        table.fatal_error = str(exc)
        return table
    table.encoding = used
    if warning:
        table.warnings.append(warning)

    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if header_idx is None:
        table.fatal_error = "File is empty."
        return table

    headers = _split_header(lines[header_idx], fmt.delimiter, table.warnings)
    table.headers = headers
    table.column_map = _resolve_column_map(headers, options, table.warnings)
    if not table.column_map.get("begin_time") or not table.column_map.get("end_time"):
        table.fatal_error = (
            "Begin/End time columns not found (expected e.g. 'Begin Time (s)' and 'End Time (s)'). "
            "Map them explicitly if the headers differ."
        )
        return table

    if options.annotation_columns is None:
        table.annotation_columns = suggest_annotation_columns(headers, table.column_map, fmt)
    else:
        table.annotation_columns = [c for c in options.annotation_columns if c in headers]
        missing = [c for c in options.annotation_columns if c not in headers]
        if missing:
            table.warnings.append(f"Annotation column(s) not in this file: {', '.join(missing)}.")
    if options.primary_label_column and options.primary_label_column in table.annotation_columns:
        table.primary_label_column = options.primary_label_column
    elif options.primary_label_column:
        table.warnings.append(f"Primary label column '{options.primary_label_column}' not in this file.")
    elif options.primary_label_auto:
        table.primary_label_column = next(
            (c for c in options.primary_label_candidates if c in table.annotation_columns), None)

    groups: Dict[str, List[Tuple[int, Dict[str, str], bool]]] = {}
    order: List[str] = []
    blank_inside = 0
    sel_col = table.column_map.get("selection")
    view_col = table.column_map.get("view")

    for idx in range(header_idx + 1, len(lines)):
        line = lines[idx]
        line_no = idx + 1
        if not line.strip():
            blank_inside += 1
            continue
        cells = line.split(fmt.delimiter)
        malformed = False
        if len(cells) > len(headers):
            extra = cells[len(headers):]
            if any(c.strip() for c in extra):
                malformed = True
            # Extra cells survive in __raw_line__ below when they carry content.
            cells = cells[:len(headers)]
        if len(cells) < len(headers):
            malformed = True
        raw = {h: (cells[i] if i < len(cells) else "") for i, h in enumerate(headers)}
        if malformed:
            raw["__raw_line__"] = line
        raw["__line__"] = str(line_no)
        table.data_row_count += 1

        key = raw.get(sel_col, "").strip() if sel_col else ""
        key = key or f"row:{line_no}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((line_no, raw, malformed))

    trailing_blank = 0
    for line in reversed(lines):
        if line.strip():
            break
        trailing_blank += 1
    if blank_inside - trailing_blank > 0:
        table.warnings.append(f"{blank_inside - trailing_blank} blank line(s) inside the table were ignored.")

    for key in order:
        table.selections.append(_build_selection(key, groups[key], table, view_col, options))
    return table


def _build_selection(key, rows, table: ParsedTable, view_col, options: ImportOptions) -> ParsedSelection:
    fmt = options.format
    views = [(r[1].get(view_col, "") if view_col else "") for r in rows]
    prefix = fmt.preferred_view_prefix.lower()
    canon_i = next((i for i, v in enumerate(views) if v.strip().lower().startswith(prefix)), 0)
    canon = rows[canon_i][1]

    fields = {role: (canon.get(col) if col else None) for role, col in table.column_map.items()}
    annotations = {c: canon.get(c, "") for c in table.annotation_columns}
    sel = ParsedSelection(
        key=key,
        row_numbers=[r[0] for r in rows],
        views=views,
        raw_rows=[r[1] for r in rows],
        fields=fields,
        annotations=annotations,
    )

    for role in ("begin_time", "end_time", "file_offset", "channel", "low_freq", "high_freq"):
        value, err = parse_number(fields.get(role), fmt.decimal_separator)
        sel.numbers[role] = value
        if err:
            sel.number_errors[role] = err

    if len(rows) > 1:
        sel.flags.append("duplicate_views")
        if len(set(v.strip() for v in views)) < len(views):
            sel.flags.append("repeated_view_name")

    if any(r[2] for r in rows):
        bad = [str(r[0]) for r in rows if r[2]]
        sel.status, sel.message = "malformed_row", f"Line(s) {', '.join(bad)} have a different number of fields than the header."
        return sel

    conflict = _view_conflict(rows, canon_i, table, options)
    if conflict:
        sel.status, sel.message = "view_conflict", conflict
        return sel

    label_cols = [table.primary_label_column] if table.primary_label_column else table.annotation_columns
    if not label_cols or all(not str(annotations.get(c, "")).strip() for c in label_cols):
        sel.flags.append("missing_label")
    return sel


def _view_conflict(rows, canon_i, table: ParsedTable, options: ImportOptions) -> Optional[str]:
    if len(rows) < 2:
        return None
    fmt = options.format
    canon = rows[canon_i][1]
    numeric_roles = ("begin_time", "end_time", "file_offset")
    text_roles = ("channel", "begin_file", "end_file", "begin_path", "end_path")
    for line_no, raw, _ in rows:
        for role in numeric_roles:
            col = table.column_map.get(role)
            if not col:
                continue
            a, _ = parse_number(canon.get(col), fmt.decimal_separator)
            b, _ = parse_number(raw.get(col), fmt.decimal_separator)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > fmt.time_tolerance_s):
                return f"'{col}' differs between views (line {line_no}: {raw.get(col)!r} vs {canon.get(col)!r})."
        for role in text_roles:
            col = table.column_map.get(role)
            if col and raw.get(col, "").strip() != canon.get(col, "").strip():
                return f"'{col}' differs between views (line {line_no})."
        for col in table.annotation_columns:
            if raw.get(col, "") != canon.get(col, ""):
                return f"Annotation '{col}' differs between views (line {line_no})."
    return None
