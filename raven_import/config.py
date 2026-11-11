"""
Configurable assumptions about Raven Pro selection tables and import options.

Nothing here has been verified against the biologists' real exports yet; every
assumption that could differ between Raven versions or user settings is a field
so it can be changed without code edits.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Column roles the importer understands. Annotation columns are handled separately.
ROLES = (
    "selection",
    "view",
    "channel",
    "begin_time",
    "end_time",
    "begin_file",
    "end_file",
    "begin_path",
    "end_path",
    "file_offset",
    "low_freq",
    "high_freq",
    "begin_date",
    "begin_clock_time",
)

# Raven default header names per role (first match wins during auto-detection).
DEFAULT_ROLE_ALIASES: Dict[str, List[str]] = {
    "selection": ["Selection"],
    "view": ["View"],
    "channel": ["Channel"],
    "begin_time": ["Begin Time (s)"],
    "end_time": ["End Time (s)"],
    "begin_file": ["Begin File"],
    "end_file": ["End File"],
    "begin_path": ["Begin Path"],
    "end_path": ["End Path"],
    "file_offset": ["File Offset (s)"],
    "low_freq": ["Low Freq (Hz)"],
    "high_freq": ["High Freq (Hz)"],
    "begin_date": ["Begin Date"],
    "begin_clock_time": ["Begin Clock Time"],
}

# Species vocabulary for interpreting the primary label (supplied by the Citimon biologists, 2026-10).
# Interpretation is stored next to, never instead of, the original value.
DEFAULT_SPECIES_VOCABULARY = [
    "Bufo bufo", "Rana dalmatina", "Rana temporaria", "Hyla arborea", "Bombina variegata", "Pelophylax sp.",
]
# Case-insensitive label -> canonical species (after stripping '?' and whitespace).
DEFAULT_SPECIES_SYNONYMS = {"Pelophylax": "Pelophylax sp."}
INTERPRETATION_RULESET = "species-v1"
# Column names preferred as the primary label when ImportOptions.primary_label_auto is on.
DEFAULT_PRIMARY_LABEL_CANDIDATES = ["Annotation"]

# Raven measurement columns that are never offered as annotation columns by default.
DEFAULT_MEASUREMENT_COLUMNS = {
    "Begin Date", "Begin Clock Time", "Begin Hour", "Begin Date Time", "End Date",
    "End Clock Time", "Begin Sample", "End Sample", "Sound Files", "Table",
}
# Headers ending with a unit in parentheses, e.g. "Delta Time (s)", "Peak Freq (Hz)".
MEASUREMENT_UNIT_RE = re.compile(r"\((s|ms|Hz|kHz|dB|dB FS|dB FS/Hz|%|u|samples|bits)\)\s*$", re.IGNORECASE)

# Selection statuses. "Blocking" statuses never produce clip links.
STATUS_MATCHED_SINGLE = "matched_single"
STATUS_MATCHED_MULTIPLE = "matched_multiple"
STATUS_NO_CLIP_OVERLAP = "no_clip_overlap"

BLOCKING_STATUSES = {
    "malformed_row": "Row has a different number of fields than the header.",
    "view_conflict": "Views of the same selection disagree on time, channel or file.",
    "invalid_interval": "Begin/End time missing, non-numeric, negative or End <= Begin.",
    "missing_source_file": "No Begin File / Begin Path, so the source recording is unknown.",
    "source_file_conflict": "Begin File and Begin Path name different files.",
    "missing_location": "Location cannot be determined (no usable Begin Path folder and no fixed location).",
    "unknown_location_folder": "Begin Path folder is not a known Citimon location.",
    "location_conflict": "Begin Path folder and the chosen fixed location disagree.",
    "unknown_location": "Location is not a known Citimon location (and has no alias).",
    "source_file_date_mismatch": "The date in the file name disagrees with Begin Date.",
    "source_file_date_unverified": "File name has an unpadded date that cannot be verified (no usable Begin Date).",
    "missing_channel": "No Channel column and no default channel configured.",
    "recording_not_found": "No Citimon recording with this file name at this location.",
    "channel_not_found": "Recording exists but not for this channel.",
    "ambiguous_recording": "More than one recording matches this location/file/channel.",
    "missing_file_offset": "File Offset (s) is required by the chosen time mode but missing.",
    "ambiguous_time_origin": "Table covers several files but has no File Offset (s); times are sequence-relative.",
    "time_origin_conflict": "Single-file mode, but File Offset (s) differs from Begin Time (s).",
    "spans_file_boundary": "Selection starts and ends in different files.",
    "exceeds_recording_duration": "Selection ends after the end of the recording.",
}

ALL_STATUSES = [STATUS_MATCHED_SINGLE, STATUS_MATCHED_MULTIPLE, STATUS_NO_CLIP_OVERLAP] + list(BLOCKING_STATUSES)

TIME_MODES = ("auto", "file_offset", "single_file")
LOCATION_MODES = ("path", "fixed", "table_name", "auto")
ENCODINGS = ("auto", "utf-8", "cp1250", "cp1252")


@dataclass
class RavenFormatConfig:
    """Assumptions about the file format itself."""
    delimiter: str = "\t"
    decimal_separator: str = "."
    # Encoding tried after UTF-8 fails in 'auto' mode. Slovenian Windows default is cp1250.
    fallback_encoding: str = "cp1250"
    role_aliases: Dict[str, List[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ROLE_ALIASES.items()})
    measurement_columns: List[str] = field(default_factory=lambda: sorted(DEFAULT_MEASUREMENT_COLUMNS))
    # View kept as canonical when a selection appears once per view (prefix match, case-insensitive).
    preferred_view_prefix: str = "Spectrogram"
    # Raven default table file name: "<sound file>.Table.<n>.selections.txt"
    table_name_stem_re: str = r"^(?P<stem>.+?)\.Table\.\d+\.selections\.(txt|tsv)$"
    # Citimon delivery naming "<location>-<annotator>[-hitro].txt" ("hitro" = quick pass).
    # Named groups used: location, annotator, quick_pass (optional).
    table_meta_re: str = r"^(?P<location>.+?)-(?P<annotator>[A-Za-z]{1,10})(?P<quick_pass>-hitro)?\.(?:txt|tsv)$"
    # Formats tried for the Begin Date column (strptime accepts unpadded month/day).
    begin_date_formats: List[str] = field(default_factory=lambda: ["%Y/%m/%d", "%Y-%m-%d", "%d.%m.%Y"])
    # Flag (not block) rows whose Begin Clock Time differs from file-name time + File Offset by more than this.
    clock_offset_tolerance_s: float = 0.1
    # Tolerance when comparing times across views / against recording duration (seconds).
    time_tolerance_s: float = 1e-3


@dataclass
class ImportOptions:
    """Per-import choices made in the wizard."""
    project: str = "Citimon"
    model_name: str = "perchv2"
    encoding: str = "auto"
    time_mode: str = "auto"
    location_mode: str = "path"
    # 'path' mode: used only when Begin Path is missing. 'fixed' mode: required.
    fixed_location: Optional[str] = None
    # role -> header name; None/absent role means auto-detect.
    column_mapping: Dict[str, Optional[str]] = field(default_factory=dict)
    annotation_columns: Optional[List[str]] = None  # None -> auto-detect
    primary_label_column: Optional[str] = None
    # When no primary label column is given, use the first of primary_label_candidates present in the table.
    primary_label_auto: bool = False
    primary_label_candidates: List[str] = field(default_factory=lambda: list(DEFAULT_PRIMARY_LABEL_CANDIDATES))
    default_channel: Optional[int] = 1
    # Use "<stem>.Table.N.selections.txt" to infer the source file when no file column exists.
    file_from_table_name: bool = False
    # Take the annotator ID from the table file name (table_meta_re) instead of one name for all files.
    annotator_from_table_name: bool = False
    # Exact location name -> known location, e.g. {"2026_03-r26": "2026_03-R26"}. Never case-folded.
    location_aliases: Dict[str, str] = field(default_factory=dict)
    # Explicit per-table fixes, keyed by uploaded table file name (recorded in the batch options).
    annotator_overrides: Dict[str, str] = field(default_factory=dict)
    location_overrides: Dict[str, str] = field(default_factory=dict)
    # Fix "2026223_122000.WAV" -> "20260223_122000.WAV" only when Begin Date confirms the date.
    repair_unpadded_file_dates: bool = True
    # Clip selections that run past the recording end at the end (original end kept, row flagged).
    truncate_at_recording_end: bool = False
    recording_duration_s: float = 60.0
    # Prefer the real length from the audio header when the clip source can read it (e.g. 55 s at 2026_04-C31).
    recording_duration_from_audio: bool = True
    species_vocabulary: List[str] = field(default_factory=lambda: list(DEFAULT_SPECIES_VOCABULARY))
    species_synonyms: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SPECIES_SYNONYMS))
    format: RavenFormatConfig = field(default_factory=RavenFormatConfig)

    def validate(self) -> List[str]:
        errors = []
        if self.time_mode not in TIME_MODES:
            errors.append(f"Unknown time mode '{self.time_mode}'.")
        if self.location_mode not in LOCATION_MODES:
            errors.append(f"Unknown location mode '{self.location_mode}'.")
        try:
            re.compile(self.format.table_meta_re)
        except re.error as exc:
            errors.append(f"Invalid table name pattern: {exc}.")
        for src, dst in (self.location_aliases or {}).items():
            if not src or not dst:
                errors.append(f"Invalid location alias {src!r} -> {dst!r}.")
        if self.location_mode == "fixed" and not self.fixed_location:
            errors.append("Fixed location mode requires a location.")
        if self.encoding not in ENCODINGS:
            errors.append(f"Unknown encoding '{self.encoding}'.")
        if not self.recording_duration_s or self.recording_duration_s <= 0:
            errors.append("Recording duration must be positive.")
        if self.format.decimal_separator not in (".", ","):
            errors.append("Decimal separator must be '.' or ','.")
        return errors

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ImportOptions":
        data = dict(data or {})
        fmt_data = data.pop("format", None) or {}
        fmt_known = {f.name for f in dataclasses.fields(RavenFormatConfig)}
        fmt = RavenFormatConfig(**{k: v for k, v in fmt_data.items() if k in fmt_known})
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(format=fmt, **{k: v for k, v in data.items() if k in known})

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("ascii")).hexdigest()
