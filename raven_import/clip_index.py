"""
Recording and clip lookup for Raven matching, built from Citimon partition data.

Recordings are identified by (location, file stem, channel) -> ``recording_id``.
The file stem alone is never enough: the same stem exists at many Citimon
locations because all recorders share one schedule.

Partition rows are duplicated once per clustering setting (``cluster_num``);
clips are de-duplicated by ``semantic_clip_key`` before any overlap is computed.

"Location" here is always the original source-folder name (e.g. "2026_03-C08"), which is what Raven
table names, Begin Paths, stored selections and ``identity_scope`` use. Caches built with the Citimon
registry keep it in ``source_folder`` (``location`` holds the recorder there); older caches have it in
``location``. The column is chosen per partition, so stored annotations never need rewriting.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

CLIP_COLUMNS = ["location", "wav_file", "channel", "recording_id", "semantic_clip_key",
                "clip_time", "clip_duration", "model_name"]
SOURCE_FOLDER_COLUMN = "source_folder"


def folder_column(columns: Iterable[str]) -> str:
    """Column holding the source-folder name: 'source_folder' when present, else 'location'."""
    return SOURCE_FOLDER_COLUMN if SOURCE_FOLDER_COLUMN in set(columns) else "location"


def with_folder_as_location(df: pd.DataFrame) -> pd.DataFrame:
    """Frame whose 'location' column is the source-folder name (no-op for older caches)."""
    if folder_column(df.columns) == SOURCE_FOLDER_COLUMN:
        df = df.drop(columns=["location"], errors="ignore").rename(columns={SOURCE_FOLDER_COLUMN: "location"})
    return df


def file_stem(name: Optional[str]) -> str:
    """Case-folded file name without directories or extension (handles / and \\)."""
    base = str(name or "").replace("\\", "/").split("/")[-1].strip()
    return os.path.splitext(base)[0].casefold()


@dataclass
class RecordingClips:
    recording_id: str
    starts: np.ndarray
    durations: np.ndarray
    keys: List[str]


class ClipIndex:
    def __init__(self, df: pd.DataFrame, model_name: str, partition_signature: str,
                 duration_provider: Optional[Callable[[str], Optional[float]]] = None):
        self.model_name = model_name
        self.partition_signature = partition_signature
        self.warnings: List[str] = []
        self._duration_provider = duration_provider
        self._durations: Dict[str, Optional[float]] = {}

        if "model_name" in df.columns:
            df = df[df["model_name"].astype(str) == str(model_name)]
        name_col = "wav_file" if "wav_file" in df.columns else "file_name"
        self.rows_before_dedup = len(df)

        distinct = df.drop_duplicates(["semantic_clip_key", "recording_id", "clip_time", "clip_duration"])
        inconsistent = distinct["semantic_clip_key"][distinct["semantic_clip_key"].duplicated()].unique()
        if len(inconsistent):
            self.warnings.append(
                f"{len(inconsistent)} clip key(s) have inconsistent timing across rows; first occurrence used "
                f"(e.g. {inconsistent[0]})."
            )
        clips = df.drop_duplicates("semantic_clip_key")
        self.duplicate_rows_removed = self.rows_before_dedup - len(clips)

        # (location, stem) -> channel -> set(recording_id)
        self._recordings: Dict[Tuple[str, str], Dict[int, Set[str]]] = {}
        rec_frame = clips[["location", name_col, "channel", "recording_id"]].drop_duplicates()
        for loc, name, ch, rid in rec_frame.itertuples(index=False):
            by_ch = self._recordings.setdefault((str(loc), file_stem(name)), {})
            by_ch.setdefault(int(ch), set()).add(str(rid))

        self._clips: Dict[str, RecordingClips] = {}
        clips = clips.sort_values(["recording_id", "clip_time"])
        for rid, grp in clips.groupby("recording_id", sort=False):
            self._clips[str(rid)] = RecordingClips(
                recording_id=str(rid),
                starts=grp["clip_time"].astype(float).to_numpy(),
                durations=grp["clip_duration"].astype(float).to_numpy(),
                keys=grp["semantic_clip_key"].astype(str).tolist(),
            )

    def channels_for(self, location: str, stem: str) -> Dict[int, Set[str]]:
        return self._recordings.get((location, stem.casefold()), {})

    def recording_duration(self, recording_id: str) -> Optional[float]:
        """Actual recording length in seconds if a provider can tell, else None (caller uses its default)."""
        if self._duration_provider is None:
            return None
        if recording_id not in self._durations:
            self._durations[recording_id] = self._duration_provider(recording_id)
        return self._durations[recording_id]

    def clips(self, recording_id: str) -> Optional[RecordingClips]:
        return self._clips.get(recording_id)


class DataFrameClipSource:
    """Clip source over an in-memory frame (tests, dry runs)."""

    def __init__(self, df: pd.DataFrame, signature: str = "in-memory", project: str = "Citimon",
                 durations: Optional[Dict[str, float]] = None):
        self.df = with_folder_as_location(df)
        self.signature = signature
        self.project = project
        self.durations = durations  # recording_id -> seconds; None means "unknown, use configured length"

    def models(self) -> List[str]:
        return sorted(self.df["model_name"].astype(str).unique()) if "model_name" in self.df.columns else []

    def known_locations(self, model_name: str) -> List[str]:
        return sorted(self.df["location"].astype(str).unique())

    def partition_signature(self, model_name: str) -> str:
        return self.signature

    def load_index(self, model_name: str, locations: Iterable[str]) -> ClipIndex:
        locs = set(locations)
        provider = self.durations.get if self.durations is not None else None
        return ClipIndex(self.df[self.df["location"].isin(locs)], model_name, self.signature, provider)


class PartitionClipSource:
    """Clip source reading the Citimon parquet partition listed in the cache manifest."""

    _lock = threading.Lock()
    _location_cache: Dict[Tuple[str, int, int], List[str]] = {}

    def __init__(self, project: str = "Citimon", base_dir: str = "data"):
        self.project = project
        self.base_dir = base_dir

    def _manifest(self) -> dict:
        path = os.path.join(self.base_dir, "cache", "manifest.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def models(self) -> List[str]:
        models = (self._manifest().get("models_by_location") or {}).get(self.project)
        if models:
            return list(models)
        return ["perchv2"] if self._partition_path("perchv2") else []

    def _partition_entry(self, model_name: str) -> Optional[dict]:
        for entry in self._manifest().get("partitions", []):
            if entry.get("location") == self.project and entry.get("model") == model_name:
                return entry
        return None

    def _partition_path(self, model_name: str) -> Optional[str]:
        entry = self._partition_entry(model_name)
        if entry and entry.get("partition_rel_path"):
            path = os.path.join(self.base_dir, entry["partition_rel_path"])
            if os.path.exists(path):
                return path
        path = os.path.join(self.base_dir, "cache", "partitions", self.project, f"{model_name}.parquet")
        return path if os.path.exists(path) else None

    def partition_signature(self, model_name: str) -> str:
        entry = self._partition_entry(model_name)
        if entry and entry.get("signature"):
            return str(entry["signature"])
        path = self._partition_path(model_name)
        if not path:
            return "missing"
        st = os.stat(path)
        return f"{st.st_size}_{st.st_mtime_ns}"

    def known_locations(self, model_name: str) -> List[str]:
        import pyarrow.parquet as pq
        path = self._partition_path(model_name)
        if not path:
            return []
        st = os.stat(path)
        cache_key = (path, st.st_size, st.st_mtime_ns)
        with self._lock:
            if cache_key not in self._location_cache:
                col_name = folder_column(pq.read_schema(path).names)
                col = pq.read_table(path, columns=[col_name]).column(col_name)
                self._location_cache[cache_key] = sorted(str(v) for v in col.unique().to_pylist() if v is not None)
            return list(self._location_cache[cache_key])

    def load_index(self, model_name: str, locations: Iterable[str]) -> ClipIndex:
        import pyarrow.parquet as pq
        path = self._partition_path(model_name)
        if not path:
            raise FileNotFoundError(f"No partition for {self.project}/{model_name} under {self.base_dir}")
        locs = sorted(set(locations))
        if not locs:
            return ClipIndex(pd.DataFrame(columns=CLIP_COLUMNS), model_name, self.partition_signature(model_name),
                             self.audio_duration)
        col_name = folder_column(pq.read_schema(path).names)
        columns = CLIP_COLUMNS if col_name == "location" else [c for c in CLIP_COLUMNS if c != "location"] + [col_name]
        table = pq.read_table(path, columns=columns, filters=[(col_name, "in", locs)])
        return ClipIndex(with_folder_as_location(table.to_pandas()), model_name, self.partition_signature(model_name),
                         self.audio_duration)

    def audio_duration(self, recording_id: str) -> Optional[float]:
        """Read the length from the audio header: '<project>::<rel_audio_path>::ch<N>' -> recordings/<rel path>."""
        parts = str(recording_id).split("::")
        if len(parts) < 3:
            return None
        path = os.path.join(self.base_dir, "recordings", *parts[1].split("/"))
        try:
            import soundfile as sf
            return float(sf.info(path).duration)
        except Exception:
            return None
