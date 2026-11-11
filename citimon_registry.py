"""
citimon_registry.py - Citimon hierarchy registry (config/citimon_registry.json).

Hierarchy: Citimon -> recorder (the ``location`` column, e.g. "C08")
                   -> microlocation (a physical position, e.g. "C08-P1").

Original source-folder names (e.g. "2026_03-C08") are internal metadata. They are stored in the
``source_folder`` column, keep recording IDs and audio paths unchanged, and are what Raven
annotations refer to. Historical and variant spellings resolve through ``name_index``.

Folders without coordinates get the microlocation ``Unassigned``: they stay under their recorder
and have no map marker. Folders listed in ``excluded_source_folders`` are not ingested. They are
matched by their literal on-disk name before alias resolution, so an excluded former name cannot
be read in as its successor.

The registry path can be overridden with the VIBROVIS_CITIMON_REGISTRY environment variable.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

PROJECT = "Citimon"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "citimon_registry.json")
ENV_VAR = "VIBROVIS_CITIMON_REGISTRY"
UNASSIGNED = "Unassigned"


def registry_path() -> str:
    return os.environ.get(ENV_VAR) or DEFAULT_PATH


def _norm(name: Any) -> str:
    """Spelling-insensitive key: case-folded, '-' and '_' treated alike."""
    return str(name or "").strip().casefold().replace("-", "_")


@dataclass(frozen=True)
class FolderInfo:
    source_folder: str          # canonical source-folder name
    recorder: str               # location
    microlocation: str          # microlocation ID or UNASSIGNED
    latitude: Optional[float]
    longitude: Optional[float]


class CitimonRegistry:
    def __init__(self, data: Dict[str, Any], path: Optional[str] = None):
        if data.get("schema") != "vibrovis.citimon.registry":
            raise ValueError(f"Not a Citimon registry: {path or '<dict>'}")
        self.path = path
        self.data = data
        self.unassigned = data.get("unassigned_microlocation") or UNASSIGNED
        self.recorders: Dict[str, Dict[str, Any]] = data.get("recorders", {})
        self.microlocations: Dict[str, Dict[str, Any]] = data.get("microlocations", {})
        self.source_folders: Dict[str, Dict[str, Any]] = data.get("source_folders", {})
        self.excluded: Dict[str, str] = data.get("excluded_source_folders", {})
        self._excluded_norm = {_norm(k) for k in self.excluded}
        self._exact = dict(data.get("name_index", {}))
        for sf in self.source_folders:
            self._exact.setdefault(sf, sf)
        # Spelling-insensitive fallback; a key shared by two different targets is ambiguous and dropped.
        loose: Dict[str, Optional[str]] = {}
        for name, target in self._exact.items():
            k = _norm(name)
            loose[k] = target if loose.get(k, target) == target else None
        self._loose = {k: v for k, v in loose.items() if v}

    # ------------------------------------------------------------------ lookups
    def is_excluded(self, folder_name: str) -> bool:
        """True for a folder that must not be ingested (literal on-disk name, spelling-insensitive)."""
        return _norm(folder_name) in self._excluded_norm

    def canonical_folder(self, name: str) -> Optional[str]:
        if name is None:
            return None
        s = str(name).strip()
        return self._exact.get(s) or self._loose.get(_norm(s))

    def folder_info(self, name: str) -> Optional[FolderInfo]:
        sf = self.canonical_folder(name)
        if not sf:
            return None
        e = self.source_folders[sf]
        mid = e.get("microlocation")
        m = self.microlocations.get(mid) if mid else None
        return FolderInfo(sf, e["recorder"], mid or self.unassigned,
                          m["latitude"] if m else None, m["longitude"] if m else None)

    def recorder_microlocations(self, recorder: str) -> List[str]:
        return list(self.recorders.get(recorder, {}).get("microlocations", []))

    def microlocation_coordinates(self, mid: str) -> Optional[Tuple[float, float]]:
        m = self.microlocations.get(mid)
        return (m["latitude"], m["longitude"]) if m else None

    def resolve_display(self, location: Optional[str]) -> Tuple[Optional[str], List[str]]:
        """(recorder, its microlocations) for a recorder ID or any (historical) folder name."""
        if not location:
            return None, []
        loc = str(location).strip()
        if loc in self.recorders:
            return loc, self.recorder_microlocations(loc)
        info = self.folder_info(loc)
        if info:
            return info.recorder, ([info.microlocation] if info.microlocation in self.microlocations else [])
        return None, []

    # ------------------------------------------------------------------ validation
    def validate(self) -> List[str]:
        """Structural checks; returns a list of error strings (empty when consistent)."""
        errs: List[str] = []
        seen_pos: Dict[Tuple[str, float, float], str] = {}
        for mid, m in self.microlocations.items():
            if m.get("recorder") not in self.recorders:
                errs.append(f"{mid}: unknown recorder {m.get('recorder')!r}")
            lat, lon = m.get("latitude"), m.get("longitude")
            if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and -90 <= lat <= 90 and -180 <= lon <= 180):
                errs.append(f"{mid}: invalid coordinates")
            key = (m.get("recorder"), lat, lon)
            if key in seen_pos:
                errs.append(f"{mid} duplicates {seen_pos[key]}")
            seen_pos[key] = mid
            if not str(mid).startswith(f"{m.get('recorder')}-P"):
                errs.append(f"{mid}: ID does not match recorder")
        for sf, e in self.source_folders.items():
            if e.get("recorder") not in self.recorders:
                errs.append(f"{sf}: unknown recorder")
            mid = e.get("microlocation")
            if mid and mid not in self.microlocations:
                errs.append(f"{sf}: unknown microlocation {mid}")
            if mid and self.microlocations.get(mid, {}).get("recorder") != e.get("recorder"):
                errs.append(f"{sf}: microlocation {mid} belongs to another recorder")
        for name, target in self._exact.items():
            if target not in self.source_folders:
                errs.append(f"name {name!r} -> unknown folder {target!r}")
        for name in self.excluded:
            if not self.canonical_folder(name):
                errs.append(f"excluded folder {name!r} is not in the catalogue")
        for rec, r in self.recorders.items():
            for mid in r.get("microlocations", []):
                if mid not in self.microlocations:
                    errs.append(f"{rec}: unknown microlocation {mid}")
        for rec, led in self.data.get("microlocation_id_ledger", {}).items():
            nxt = int(str(led.get("next_free", "-P0")).split("-P")[-1])
            used = [int(m.split("-P")[-1]) for m in self.microlocations if self.microlocations[m].get("recorder") == rec]
            if used and nxt <= max(used):
                errs.append(f"{rec}: ledger next_free {led.get('next_free')} would reuse an assigned ID")
        return errs


_lock = threading.Lock()
_cached: Dict[Tuple[str, int, int], Optional[CitimonRegistry]] = {}


def load_registry(path: Optional[str] = None) -> Optional[CitimonRegistry]:
    """Load (and cache by file size/mtime) the registry; None when the file does not exist."""
    p = path or registry_path()
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (os.path.abspath(p), st.st_size, st.st_mtime_ns)
    with _lock:
        if key not in _cached:
            with open(p, "r", encoding="utf-8") as f:
                _cached[key] = CitimonRegistry(json.load(f), p)
        return _cached[key]


def registry_signature(path: Optional[str] = None) -> str:
    """Short signature of the registry file for cache invalidation ('' when absent)."""
    try:
        st = os.stat(path or registry_path())
    except OSError:
        return ""
    return f"reg{st.st_size}_{st.st_mtime_ns % 1000000}"


# ---------------------------------------------------------------------- sunrise / sunset
def _central_europe_utc_offset(day: dt.date) -> int:
    """Slovenian civil time: CET (+1), CEST (+2) from the last Sunday of March to the last Sunday of October.
    Sunrise/sunset never fall in the 01:00 UTC switch hour, so the date-level rule is exact here."""
    def last_sunday(year: int, month: int) -> dt.date:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        return d - dt.timedelta(days=(d.weekday() + 1) % 7)
    return 2 if last_sunday(day.year, 3) <= day < last_sunday(day.year, 10) else 1


SUN_ZENITH_DEG = 91.0


def sun_times(day: dt.date, latitude: float, longitude: float) -> Tuple[Optional[str], Optional[str]]:
    """Local ("HH:MM", truncated) sunrise and sunset by the NOAA solar equations.

    Used only where the master overview TSV has no value. The 91.0 deg zenith and truncation were fitted
    to the master's own Citimon values (2026-11-10): within 1 minute for 99% of 1,206 folder-days.
    Returns (None, None) for polar day/night (not reachable in Slovenia)."""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    lat = math.radians(latitude)
    cos_ha = math.cos(math.radians(SUN_ZENITH_DEG)) / (math.cos(lat) * math.cos(decl)) - math.tan(lat) * math.tan(decl)
    if not -1.0 <= cos_ha <= 1.0:
        return None, None
    ha = math.degrees(math.acos(cos_ha))
    offset_min = _central_europe_utc_offset(day) * 60

    def fmt(minutes_utc: float) -> str:
        m = int(math.floor(minutes_utc + offset_min)) % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}"

    return fmt(720 - 4 * (longitude + ha) - eqtime), fmt(720 - 4 * (longitude - ha) - eqtime)
