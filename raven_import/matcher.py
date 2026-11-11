"""
Resolve parsed Raven selections to Citimon recordings and compute clip overlaps.

Rules:
- A recording is identified by location + file stem + channel. A file name alone
  never identifies a recording.
- Times are converted to seconds within one recording. If the time origin cannot be
  established (sequence-relative times without File Offset), the row is blocked.
- Overlaps are computed against clips de-duplicated by ``semantic_clip_key``.
- Ambiguity blocks the row with a reason; nothing is guessed. The only corrections
  applied (unpadded file dates, explicit location aliases, truncation at the
  recording end) are opt-in, verified, and recorded as flags on the row.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from .clip_index import ClipIndex, file_stem
from .config import (
    BLOCKING_STATUSES, STATUS_MATCHED_MULTIPLE, STATUS_MATCHED_SINGLE, STATUS_NO_CLIP_OVERLAP, ImportOptions,
)
from .parser import ParsedSelection, ParsedTable

ROUND = 6
OVERLAP_EPS = 1e-9
DATE_STEM_RE = re.compile(r"^(?P<date>\d{6,8})_(?P<time>\d{6})$")
CLOCK_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\s*$")


@dataclass
class ClipLink:
    semantic_clip_key: str
    recording_id: str
    clip_time_s: float
    clip_duration_s: float
    overlap_s: float
    frac_of_clip: float
    frac_of_selection: float
    relation: str  # selection_within_clip | clip_within_selection | partial


@dataclass
class MatchResult:
    status: str
    message: str = ""
    location: Optional[str] = None
    location_source: Optional[str] = None  # path | fixed | table_name (+ "+alias")
    source_stem: Optional[str] = None
    channel: Optional[int] = None
    recording_id: Optional[str] = None
    file_t0_s: Optional[float] = None
    file_t1_s: Optional[float] = None
    file_t1_original_s: Optional[float] = None  # set only when the interval was truncated
    time_mode_used: Optional[str] = None
    links: List[ClipLink] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.status in BLOCKING_STATUSES


def _path_parts(path: Optional[str]) -> List[str]:
    return [p for p in str(path or "").replace("\\", "/").split("/") if p.strip()]


def path_folder(path: Optional[str]) -> Optional[str]:
    parts = _path_parts(path)
    return parts[-2].strip() if len(parts) >= 2 else None


def raw_source_stem(sel: ParsedSelection, table: ParsedTable, options: ImportOptions) -> Optional[str]:
    begin_file = (sel.fields.get("begin_file") or "").strip()
    begin_path = (sel.fields.get("begin_path") or "").strip()
    if begin_file:
        return file_stem(begin_file)
    if begin_path:
        return file_stem(begin_path)
    if options.file_from_table_name and table.table_name_stem:
        return file_stem(table.table_name_stem)
    return None


# Kept for callers that only need the unrepaired stem.
source_stem = raw_source_stem


def _begin_date(sel: ParsedSelection, options: ImportOptions) -> Optional[datetime.date]:
    raw_date = (sel.fields.get("begin_date") or "").strip()
    for fmt in options.format.begin_date_formats if raw_date else []:
        try:
            return datetime.datetime.strptime(raw_date, fmt).date()
        except ValueError:
            continue
    return None


def _clock_seconds(sel: ParsedSelection) -> Optional[float]:
    clock = CLOCK_RE.match(sel.fields.get("begin_clock_time") or "")
    if not clock:
        return None
    return int(clock.group(1)) * 3600 + int(clock.group(2)) * 60 + float(clock.group(3))


def expected_file_dates(sel: ParsedSelection, options: ImportOptions) -> List[datetime.date]:
    """Dates the recording may start on: Begin Date, and the date of (Begin Date + Clock - File Offset).

    The second covers a selection that begins after midnight in a file started before midnight.
    Raven's clock time can differ from file-name time + offset by up to ~1 s (seen in real exports),
    so the clock is only used to widen the accepted dates, never to narrow them.
    """
    day = _begin_date(sel, options)
    if day is None:
        return []
    dates = [day]
    secs, offset = _clock_seconds(sel), sel.numbers.get("file_offset")
    if secs is not None and offset is not None:
        derived = (datetime.datetime.combine(day, datetime.time()) + datetime.timedelta(seconds=secs - offset)).date()
        if derived != day:
            dates.append(derived)
    return dates


def clock_offset_discrepancy(sel: ParsedSelection, stem: str) -> Optional[float]:
    """Seconds between Begin Clock Time and (file-name start time + File Offset); None if not computable."""
    m = DATE_STEM_RE.match(stem or "")
    secs, offset = _clock_seconds(sel), sel.numbers.get("file_offset")
    if not m or secs is None or offset is None:
        return None
    t = m.group("time")
    start = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:])
    delta = (secs - start) % 86400 - offset
    return delta - 86400 if delta > 43200 else delta


def resolve_source_stem(sel: ParsedSelection, table: ParsedTable,
                        options: ImportOptions) -> Tuple[Optional[str], List[str], Optional[Tuple[str, str]]]:
    """Return (stem, flags, blocking (status, detail) or None). Repairs unpadded dates only when verified."""
    stem = raw_source_stem(sel, table, options)
    if not stem:
        return None, [], None
    m = DATE_STEM_RE.match(stem)
    if not m:
        return stem, [], None
    date_part = m.group("date")
    dates = expected_file_dates(sel, options)
    accepted = " or ".join(d.isoformat() for d in dates)
    if len(date_part) == 8:
        if dates and date_part not in {d.strftime("%Y%m%d") for d in dates}:
            return stem, [], ("source_file_date_mismatch", f"File {stem!r}, Begin Date implies {accepted}.")
        return stem, [], None
    if not options.repair_unpadded_file_dates:
        return stem, [], None
    if not dates:
        return stem, [], ("source_file_date_unverified", f"File {stem!r}.")
    for day in dates:
        y, mo, d = day.year, day.month, day.day
        if date_part in {f"{y}{mo}{d}", f"{y}{mo:02d}{d}", f"{y}{mo}{d:02d}"}:
            return f"{day.strftime('%Y%m%d')}_{m.group('time')}", ["begin_file_date_repaired"], None
    return stem, [], ("source_file_date_mismatch", f"File {stem!r} is not an unpadded form of {accepted}.")


def table_source_stems(table: ParsedTable, options: ImportOptions) -> Set[str]:
    stems = set()
    for sel in table.selections:
        stem, _, _ = resolve_source_stem(sel, table, options)
        if stem:
            stems.add(stem)
        end_file = (sel.fields.get("end_file") or "").strip()
        if end_file and file_stem(end_file) != raw_source_stem(sel, table, options):
            stems.add(file_stem(end_file))
    return stems


def _alias(location: Optional[str], options: ImportOptions) -> Tuple[Optional[str], bool]:
    aliases = options.location_aliases or {}
    if location in aliases:
        return aliases[location], True
    return location, False


def locations_needed(table: ParsedTable, options: ImportOptions, known_locations: Iterable[str]) -> Set[str]:
    known = set(known_locations)
    candidates = [options.fixed_location, (options.location_overrides or {}).get(table.filename)]
    if table.table_meta:
        candidates.append(table.table_meta.get("location"))
    candidates += [path_folder(sel.fields.get("begin_path")) for sel in table.selections]
    return {loc for loc in (_alias(c, options)[0] for c in candidates if c) if loc in known}


def compute_overlaps(t0: float, t1: float, recording_id: str, index: ClipIndex) -> List[ClipLink]:
    clips = index.clips(recording_id)
    if clips is None or not len(clips.starts):
        return []
    starts, ends = clips.starts, clips.starts + clips.durations
    overlap = np.minimum(ends, t1) - np.maximum(starts, t0)
    hits = np.nonzero(overlap > OVERLAP_EPS)[0]
    sel_dur = t1 - t0
    links = []
    for i in hits:
        c0, c1, ov = float(starts[i]), float(ends[i]), float(overlap[i])
        if t0 >= c0 - OVERLAP_EPS and t1 <= c1 + OVERLAP_EPS:
            relation = "selection_within_clip"
        elif c0 >= t0 - OVERLAP_EPS and c1 <= t1 + OVERLAP_EPS:
            relation = "clip_within_selection"
        else:
            relation = "partial"
        links.append(ClipLink(
            semantic_clip_key=clips.keys[i],
            recording_id=recording_id,
            clip_time_s=round(c0, ROUND),
            clip_duration_s=round(float(clips.durations[i]), ROUND),
            overlap_s=round(ov, ROUND),
            frac_of_clip=round(ov / float(clips.durations[i]), ROUND) if clips.durations[i] > 0 else 0.0,
            frac_of_selection=round(ov / sel_dur, ROUND),
            relation=relation,
        ))
    return links


def _blocked(status: str, detail: str = "", **kw) -> MatchResult:
    message = BLOCKING_STATUSES[status] + (f" {detail}" if detail else "")
    return MatchResult(status=status, message=message, **kw)


def _resolve_location(table: ParsedTable, options: ImportOptions, begin_path: str, stem: str,
                      known: Set[str]) -> Tuple[Optional[str], Optional[str], List[str], Optional[Tuple[str, str]]]:
    """Return (location, source, flags, blocking (status, detail) or None)."""
    folder_raw = path_folder(begin_path) if begin_path else None
    folder, folder_aliased = _alias(folder_raw, options)
    folder_known = folder in known if folder else False
    fixed, fixed_aliased = _alias(options.fixed_location, options)
    meta_raw = (table.table_meta or {}).get("location")

    # Explicit per-table override (wizard fix for tables whose location cannot be detected).
    override = (options.location_overrides or {}).get(table.filename)
    if override:
        loc, aliased = _alias(override, options)
        if loc not in known:
            return None, None, [], ("unknown_location", f"Override {override!r} for {table.filename!r}.")
        if folder_known and folder != loc:
            return None, None, [], ("location_conflict", f"Path folder {folder!r} vs override {loc!r}.")
        meta_loc = _alias(meta_raw, options)[0] if meta_raw else None
        if meta_loc in known and meta_loc != loc:
            return None, None, [], ("location_conflict", f"Table name {meta_loc!r} vs override {loc!r}.")
        flags = ["location_override"] + ([f"location_alias:{override}->{loc}"] if aliased else [])
        return loc, "override" + ("+alias" if aliased else ""), flags, None

    if options.location_mode == "auto":
        # Table-name location when it is a known location (after aliases); otherwise the Begin Path
        # rules below, then the fallback. Same conflict checks as the explicit modes.
        if meta_raw:
            loc, aliased = _alias(meta_raw, options)
            if loc in known:
                if folder_known and folder != loc:
                    return None, None, [], ("location_conflict", f"Path folder {folder!r} vs table name {loc!r}.")
                flags = [f"location_alias:{meta_raw}->{loc}"] if aliased else []
                return loc, "table_name" + ("+alias" if aliased else ""), flags, None
            if not begin_path:
                return None, None, [], ("unknown_location", f"{meta_raw!r} (from table name).")
        elif not begin_path and not options.fixed_location:
            return None, None, [], ("missing_location",
                                    f"File {stem!r}: no location in the table name or Begin Path, and no fallback.")

    if options.location_mode == "table_name":
        meta_loc = (table.table_meta or {}).get("location")
        if not meta_loc:
            return None, None, [], ("missing_location",
                                    f"Table name {table.filename!r} does not match the configured pattern.")
        loc, aliased = _alias(meta_loc, options)
        if loc not in known:
            return None, None, [], ("unknown_location", f"{meta_loc!r} (from table name).")
        if folder_known and folder != loc:
            return None, None, [], ("location_conflict", f"Path folder {folder!r} vs table name {loc!r}.")
        flags = [f"location_alias:{meta_loc}->{loc}"] if aliased else []
        return loc, "table_name" + ("+alias" if aliased else ""), flags, None

    if options.location_mode == "fixed":
        if folder_known and folder != fixed:
            return None, None, [], ("location_conflict", f"Path folder {folder!r} vs fixed {fixed!r}.")
        if fixed not in known:
            return None, None, [], ("unknown_location", f"Fixed location {options.fixed_location!r}.")
        flags = [f"location_alias:{options.fixed_location}->{fixed}"] if fixed_aliased else []
        return fixed, "fixed" + ("+alias" if fixed_aliased else ""), flags, None

    if begin_path:
        if not folder:
            return None, None, [], ("missing_location", f"Begin Path {begin_path!r} has no folder.")
        if not folder_known:
            return None, None, [], ("unknown_location_folder", f"Folder {folder_raw!r}.")
        if options.fixed_location and fixed != folder:
            return None, None, [], ("location_conflict", f"Path folder {folder!r} vs fallback {fixed!r}.")
        flags = [f"location_alias:{folder_raw}->{folder}"] if folder_aliased else []
        return folder, "path" + ("+alias" if folder_aliased else ""), flags, None
    if options.fixed_location:
        if fixed not in known:
            return None, None, [], ("unknown_location", f"Fallback location {options.fixed_location!r}.")
        flags = [f"location_alias:{options.fixed_location}->{fixed}"] if fixed_aliased else []
        return fixed, "fixed" + ("+alias" if fixed_aliased else ""), flags, None
    return None, None, [], ("missing_location", f"File {stem!r} has no Begin Path and no fallback location is set.")


def match_selection(sel: ParsedSelection, table: ParsedTable, options: ImportOptions, index: ClipIndex,
                    known_locations: Set[str], single_file_table: bool) -> MatchResult:
    tol = options.format.time_tolerance_s
    duration = options.recording_duration_s
    if sel.status:
        return MatchResult(status=sel.status, message=sel.message)

    # 1. Interval values.
    begin, end = sel.numbers.get("begin_time"), sel.numbers.get("end_time")
    for role in ("begin_time", "end_time"):
        if role in sel.number_errors:
            return _blocked("invalid_interval", sel.number_errors[role])
    if begin is None or end is None:
        return _blocked("invalid_interval", "Begin or End time is empty.")
    if begin < 0 or end <= begin:
        return _blocked("invalid_interval", f"Begin={begin}, End={end}.")

    # 2. Source file (with verified repair of unpadded dates).
    begin_file = (sel.fields.get("begin_file") or "").strip()
    begin_path = (sel.fields.get("begin_path") or "").strip()
    if begin_file and begin_path and file_stem(begin_file) != file_stem(begin_path):
        return _blocked("source_file_conflict", f"Begin File={begin_file!r}, Begin Path={begin_path!r}.")
    stem, flags, problem = resolve_source_stem(sel, table, options)
    if not stem:
        return _blocked("missing_source_file")
    if problem:
        return _blocked(problem[0], problem[1], source_stem=stem)
    drift = clock_offset_discrepancy(sel, stem)
    if drift is not None and abs(drift) > options.format.clock_offset_tolerance_s:
        flags = flags + [f"clock_offset_discrepancy:{drift:+.3f}s"]

    # 3. Location.
    location, loc_source, loc_flags, problem = _resolve_location(table, options, begin_path, stem, known_locations)
    flags = flags + loc_flags
    if problem:
        return _blocked(problem[0], problem[1], source_stem=stem, flags=flags)

    # 4. Channel and recording.
    channel = sel.numbers.get("channel")
    if "channel" in sel.number_errors:
        return _blocked("missing_channel", sel.number_errors["channel"], flags=flags)
    if channel is None:
        if options.default_channel is None:
            return _blocked("missing_channel", flags=flags)
        channel = options.default_channel
    channel = int(channel)
    ctx = dict(location=location, location_source=loc_source, source_stem=stem, channel=channel, flags=flags)

    by_channel = index.channels_for(location, stem)
    if not by_channel:
        return _blocked("recording_not_found", f"{location}/{stem}.", **ctx)
    rec_ids = by_channel.get(channel)
    if not rec_ids:
        return _blocked("channel_not_found", f"Channel {channel}; available {sorted(by_channel)}.", **ctx)
    if len(rec_ids) > 1:
        return _blocked("ambiguous_recording", f"Candidates: {sorted(rec_ids)}.", **ctx)
    recording_id = next(iter(rec_ids))
    ctx["recording_id"] = recording_id
    actual = index.recording_duration(recording_id) if options.recording_duration_from_audio else None
    if actual is not None and actual > 0:
        if abs(actual - duration) > tol:
            ctx["flags"] = flags = flags + [f"recording_duration:{actual:g}s"]
        duration = actual

    # 5. Time origin.
    offset = sel.numbers.get("file_offset")
    if "file_offset" in sel.number_errors:
        return _blocked("missing_file_offset", sel.number_errors["file_offset"], **ctx)
    mode = options.time_mode
    if mode == "auto":
        if offset is not None:
            mode = "file_offset"
        elif single_file_table:
            mode = "single_file"
        else:
            return _blocked("ambiguous_time_origin", **ctx)
    if mode == "file_offset":
        if offset is None:
            return _blocked("missing_file_offset", **ctx)
        t0, t1 = offset, offset + (end - begin)
    else:
        if not single_file_table:
            return _blocked("ambiguous_time_origin", "Single-file mode chosen, but the table names several files.", **ctx)
        if offset is not None and abs(offset - begin) > tol:
            return _blocked("time_origin_conflict", f"Offset={offset}, Begin={begin}.", **ctx)
        t0, t1 = begin, end
    ctx.update(file_t0_s=round(t0, ROUND), file_t1_s=round(t1, ROUND), time_mode_used=mode)

    # 6. Bounds (optionally truncating at the recording end; the original end is kept).
    if t0 < 0:
        return _blocked("invalid_interval", f"File-relative start {t0} < 0.", **ctx)
    end_file = (sel.fields.get("end_file") or "").strip()
    spans = bool(end_file) and file_stem(end_file) != raw_source_stem(sel, table, options)
    over = t1 > duration + tol
    if spans and not over:
        return _blocked("spans_file_boundary",
                        f"End File {file_stem(end_file)!r} differs but the interval ends within the recording.", **ctx)
    if spans or over:
        if options.truncate_at_recording_end and t0 < duration - tol:
            ctx.update(file_t1_s=round(duration, ROUND), file_t1_original_s=round(t1, ROUND),
                       flags=flags + ["truncated_at_recording_end"])
            t1 = duration
        elif spans:
            return _blocked("spans_file_boundary", f"Begin {stem!r}, End {file_stem(end_file)!r}.", **ctx)
        else:
            return _blocked(
                "exceeds_recording_duration",
                f"Ends at {t1:.3f}s, recording is {duration:g}s "
                "(likely spans into the next file or uses the wrong time mode).", **ctx)

    # 7. Overlaps.
    links = compute_overlaps(t0, t1, recording_id, index)
    if not links:
        status = STATUS_NO_CLIP_OVERLAP
        message = "Valid interval, but no clip covers it (gap between available clips)."
    elif len(links) == 1:
        status, message = STATUS_MATCHED_SINGLE, ""
    else:
        status, message = STATUS_MATCHED_MULTIPLE, f"Overlaps {len(links)} clips."
    if "truncated_at_recording_end" in ctx["flags"]:
        message = (message + f" Truncated at {duration:g}s (original end {ctx['file_t1_original_s']:g}s).").strip()
    return MatchResult(status=status, message=message, links=links, **ctx)


def match_table(table: ParsedTable, options: ImportOptions, index: ClipIndex,
                known_locations: Iterable[str]) -> List[MatchResult]:
    known = set(known_locations)
    single_file_table = len(table_source_stems(table, options)) <= 1
    return [match_selection(s, table, options, index, known, single_file_table) for s in table.selections]
