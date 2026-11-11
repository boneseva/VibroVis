"""
multimon_locations.py - Map positions for the Multimon project.

Reads config/multimon_locations.json (override with VIBROVIS_MULTIMON_LOCATIONS). Keys of ``locations`` are the
app's Multimon location values; each has zero or more positions (zero = "No coordinates"). Recording folders that
are not in the cache yet are matched by folder prefix via ``folder_prefix_sites`` (e.g. 'Krakovski_gozd-2025').

Point keys are display strings: the location name, or "<location> · <label>" when a location has several positions.
"""
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "multimon_locations.json")
ENV_VAR = "VIBROVIS_MULTIMON_LOCATIONS"
SEPARATOR = " · "

_lock = threading.Lock()
_cache: Dict[Tuple[str, int, float], "MultimonLocations"] = {}


def _norm(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").casefold().split())


@dataclass
class MultimonLocations:
    points: Dict[str, Tuple[float, float]] = field(default_factory=dict)        # key -> (lat, lon)
    owners: Dict[str, str] = field(default_factory=dict)                        # key -> location value
    location_points: Dict[str, List[str]] = field(default_factory=dict)         # location -> keys
    folder_sites: Dict[str, str] = field(default_factory=dict)                  # folder prefix -> key

    def resolve(self, location: Optional[str]) -> List[str]:
        """Point keys for a location value (exact, then spelling-insensitive, then recording-folder prefix)."""
        if not location:
            return []
        loc = str(location).strip()
        if loc in self.location_points:
            return list(self.location_points[loc])
        n = _norm(loc)
        for name, keys in self.location_points.items():
            if _norm(name) == n:
                return list(keys)
        for prefix, key in self.folder_sites.items():
            if loc.startswith(prefix) or n.startswith(_norm(prefix)):
                return [key]
        return []


def _parse(raw: dict) -> MultimonLocations:
    out = MultimonLocations()
    for loc, positions in (raw.get("locations") or {}).items():
        keys = []
        for i, p in enumerate(positions or []):
            label = p.get("label") or (f"P{i + 1}" if len(positions) > 1 else None)
            key = loc if len(positions) == 1 else f"{loc}{SEPARATOR}{label}"
            out.points[key] = (float(p["latitude"]), float(p["longitude"]))
            out.owners[key] = loc
            keys.append(key)
        out.location_points[loc] = keys
    for prefix, p in (raw.get("folder_prefix_sites") or {}).items():
        key = prefix.replace("_", " ")
        if key in out.points:
            continue
        out.points[key] = (float(p["latitude"]), float(p["longitude"]))
        out.owners[key] = prefix
        out.folder_sites[prefix] = key
    return out


def load(path: Optional[str] = None) -> Optional[MultimonLocations]:
    """Parsed config (cached by file size and mtime), or None when the file is absent or unreadable."""
    path = path or os.environ.get(ENV_VAR) or DEFAULT_PATH
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime)
    with _lock:
        if key not in _cache:
            try:
                with open(path, encoding="utf-8") as f:
                    _cache.clear()
                    _cache[key] = _parse(json.load(f))
            except (OSError, ValueError, KeyError, TypeError):
                return None
        return _cache[key]
