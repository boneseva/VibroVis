#!/usr/bin/env python3
"""
build_cache.py — Phase 1.15 Partitioned Cache Ingestion Prototype

Transforms real source audio and positions TSVs under:
    data/<project>/recordings/<Location>/...
into partitioned Parquet caches:
    data/<project>/cache/partitions/<location>/<model>.parquet
and deterministic cache manifest:
    data/<project>/cache/manifest.json

Non-destructive prototype living alongside the existing system.
"""

import argparse
import csv
import datetime
import filecmp
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


MANIFEST_SCHEMA_VERSION = "1.0.0"
BUILDER_VERSION = "Phase-1.18"

CANONICAL_PARQUET_SCHEMA = pa.schema([
    ("file_name", pa.string()),
    ("channel", pa.int64()),
    ("clip_time", pa.float64()),
    ("clip_duration", pa.float64()),
    ("model_name", pa.string()),
    ("cluster_id", pa.int64()),
    ("wav_file", pa.string()),
    ("cluster_num", pa.int64()),
    ("location", pa.string()),
    ("microlocation", pa.string()),
    ("channel_name", pa.string()),
    ("recorder_type", pa.string()),
    ("sunrise", pa.string()),
    ("sunset", pa.string()),
    ("day_dt", pa.timestamp("ns")),
    ("start_dt", pa.timestamp("ns")),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("time_of_day", pa.timestamp("ns")),
    ("start_hour_float", pa.float64()),
    ("day_str", pa.string()),
    ("mp3_file", pa.string()),
    ("abs_file_name", pa.string()),
    ("row_idx", pa.int64()),
    ("source_file_name", pa.string()),
    ("source_tsv_path", pa.string()),
    ("rel_audio_path", pa.string()),
    ("recording_id", pa.string()),
    ("raw_clip_id", pa.string()),
    ("semantic_clip_key", pa.string()),
])


def normalize_posix(path: str) -> str:
    """Normalize filesystem path to POSIX forward-slash format."""
    return os.path.normpath(path).replace("\\", "/")


def normalize_path_ext(path: str) -> str:
    """Normalize filesystem path to POSIX format with lowercased file extension."""
    norm = normalize_posix(path)
    base, ext = os.path.splitext(norm)
    return f"{base}{ext.lower()}"


def get_process_memory_mb() -> Tuple[Optional[float], Optional[float]]:
    """Returns (current_working_set_mb, peak_working_set_mb) on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            class _PMC(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if fn(h, ctypes.byref(pmc), pmc.cb):
                return (pmc.WorkingSetSize / (1024 * 1024), pmc.PeakWorkingSetSize / (1024 * 1024))
        except Exception:
            pass
    return (None, None)


def parse_filename_timestamp(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract date ('YYYY-MM-DD') and time ('HH:MM:SS') from filename.
    Supports:
      Pattern A: YYYYMMDD_HHMMSS (e.g. 20240215_110000)
      Pattern B: YYMMDD.HHMMSS (e.g. 240429.094803)
    Returns (day_str, time_str) or (None, None).
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # Pattern A: 8 digits _ 6 digits
    mA = re.search(r"(\d{8})_(\d{6})", base)
    if mA:
        d_raw, t_raw = mA.group(1), mA.group(2)
        try:
            dt = datetime.datetime.strptime(f"{d_raw}_{t_raw}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except ValueError:
            pass

    # Pattern B: 6 digits . 6 digits
    mB = re.search(r"\b(\d{6})\.(\d{6})\b", base)
    if mB:
        d_raw, t_raw = mB.group(1), mB.group(2)
        try:
            dt = datetime.datetime.strptime(f"20{d_raw}_{t_raw}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except ValueError:
            pass

    return None, None


def extract_channel_from_name(name: str) -> Optional[int]:
    """Extract channel number from filename if present (e.g. _ch1, _ch2)."""
    m = re.search(r"_ch(\d+)", name, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def extract_cluster_num_from_tsv(tsv_name: str) -> int:
    """Extract cluster count k from TSV filename suffix."""
    base = tsv_name.replace(".tsv", "").replace(".umap", "")
    parts = base.split("_")
    for p in reversed(parts):
        if p.isnumeric():
            return int(p)
    return 10


def extract_backup_timestamp(backup_path: str) -> Optional[datetime.datetime]:
    """
    Extract embedded UTC creation timestamp from backup filename.
    Format: *.bak.YYYYMMDD_HHMMSS_ffffff or *.bak.YYYYMMDD_HHMMSS
    Returns datetime with UTC tzinfo, or None if unparseable/unversioned.
    """
    m = re.search(r"\.bak\.(\d{8}_\d{6}(?:_\d+)?)$", backup_path)
    if m:
        ts_str = m.group(1)
        for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
            try:
                return datetime.datetime.strptime(ts_str, fmt).replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                pass
    return None


def backup_sort_key(backup_path: str) -> Tuple[int, datetime.datetime]:
    """
    Sort key for backup files based strictly on embedded UTC timestamp.
    Does NOT use filesystem st_mtime, which is preserved by shutil.copy2().
    """
    dt = extract_backup_timestamp(backup_path)
    if dt is not None:
        return (1, dt)
    return (0, datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))


def find_backups(file_path: str) -> List[str]:
    """
    Find all backup files associated with file_path, sorted chronologically by embedded
    UTC timestamp (newest first). Never uses filesystem mtime.
    """
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    if not os.path.isdir(dir_name):
        return []
    baks = []
    try:
        for f in os.listdir(dir_name):
            if f == base_name + ".bak" or f.startswith(base_name + ".bak."):
                baks.append(os.path.join(dir_name, f))
    except OSError:
        return []
    baks.sort(key=backup_sort_key, reverse=True)
    return baks


def get_backup_journal_path(project_dir: str) -> str:
    """Return path to backup_journal.json."""
    return os.path.join(project_dir, "cache", "backup_journal.json")


def load_backup_journal(project_dir: str) -> List[Dict[str, Any]]:
    """Load backup journal entries."""
    j_path = get_backup_journal_path(project_dir)
    if not os.path.isfile(j_path):
        return []
    try:
        with open(j_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def record_backup_journal_entry(
    project_dir: str,
    commit_id: str,
    location: str,
    model: str,
    partition_backup: Optional[str],
    manifest_backup: Optional[str],
    partition_rows: Optional[int] = None,
) -> None:
    """Append a coordinated commit record to backup_journal.json."""
    j_path = get_backup_journal_path(project_dir)
    os.makedirs(os.path.dirname(j_path), exist_ok=True)
    journal = load_backup_journal(project_dir)
    entry = {
        "commit_id": commit_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "location": location,
        "model": model,
        "partition_backup": partition_backup,
        "manifest_backup": manifest_backup,
        "partition_rows": partition_rows,
    }
    journal.append(entry)
    tmp_j = j_path + ".tmp"
    with open(tmp_j, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2)
    os.replace(tmp_j, j_path)


def create_safe_backup(
    src_path: str,
    commit_id: Optional[str] = None,
    is_parquet: bool = False,
    is_json: bool = False,
) -> str:
    """
    Creates a verified backup of src_path before replacement.
    Safety guarantees:
    1. Embeds UTC timestamp in backup filename: src_path + '.bak.<timestamp>'.
    2. Never overwrites an existing backup.
    3. Verifies the backup copy after writing (size matching + metadata / json parse).
    4. If backup creation or verification fails, cleans up any incomplete backup and
       raises RuntimeError to abort replacement before the active file is touched.
    """
    if not os.path.isfile(src_path):
        return ""

    if commit_id:
        ts = commit_id
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    cand = f"{src_path}.bak.{ts}"
    idx = 1
    while os.path.exists(cand):
        cand = f"{src_path}.bak.{ts}_{idx}"
        idx += 1
    bak_path = cand

    try:
        shutil.copy2(src_path, bak_path)

        if not os.path.isfile(bak_path):
            raise IOError(f"Backup file was not created: {bak_path}")

        src_size = os.path.getsize(src_path)
        bak_size = os.path.getsize(bak_path)
        if src_size != bak_size:
            raise ValueError(f"Backup size mismatch: expected {src_size} bytes, got {bak_size} bytes")

        if is_parquet:
            src_is_valid = False
            try:
                with open(src_path, "rb") as f_src:
                    src_meta = pq.read_metadata(f_src)
                src_is_valid = True
            except Exception:
                pass

            if src_is_valid:
                with open(bak_path, "rb") as f_bak:
                    bak_meta = pq.read_metadata(f_bak)
                if bak_meta.num_rows != src_meta.num_rows:
                    raise ValueError(
                        f"Backup row count mismatch: expected {src_meta.num_rows}, got {bak_meta.num_rows}"
                    )

        if is_json:
            src_is_valid = False
            try:
                with open(src_path, "r", encoding="utf-8") as f_src:
                    json.load(f_src)
                src_is_valid = True
            except Exception:
                pass

            if src_is_valid:
                with open(bak_path, "r", encoding="utf-8") as f_bak:
                    json.load(f_bak)

        # Compatibility copy: if unversioned .bak does not exist, populate it without overwriting
        unversioned_bak = src_path + ".bak"
        if not os.path.exists(unversioned_bak):
            try:
                shutil.copy2(bak_path, unversioned_bak)
            except OSError:
                pass

    except Exception as e:
        if os.path.isfile(bak_path):
            try:
                os.remove(bak_path)
            except OSError:
                pass
        raise RuntimeError(f"Backup creation/verification failed for '{src_path}' -> '{bak_path}': {e}") from e

    return bak_path


def safe_rollback(
    project_dir: str,
    location: str,
    model: str,
    target_existed_before_pilot: Optional[bool] = None,
) -> bool:
    """
    Executes a safe rollback for a targeted location/model partition.

    Safety rules:
    - Never deletes a partition merely because its .bak is absent.
    - If target_existed_before_pilot is None (unknown): STOP. Refuse to delete or modify.
    - If target_existed_before_pilot is True: Target existed before pilot. Must restore from a verified backup pair
      corresponding to the state immediately before the relevant commit.
      If no verified consistent backup pair is available: STOP. Refuse to delete or modify.
    - If target_existed_before_pilot is False: Target was newly created by pilot. Safe to remove target partition,
      provided no contradictory pre-existing backup file exists.
    - Restores manifest.json from verified paired manifest.json.bak.
    """
    part_path = os.path.join(project_dir, "cache", "partitions", location, f"{model}.parquet")
    manifest_path = os.path.join(project_dir, "cache", "manifest.json")

    # Clean up any leftover temporary files first
    for tmp in [part_path + ".tmp", manifest_path + ".tmp"]:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    # Rule 1: Unknown original target state -> HALT
    if target_existed_before_pilot is None:
        print(
            f"[ERROR] Rollback aborted for {location}/{model}: original target state is UNKNOWN. "
            f"Cannot determine whether partition existed prior to pilot. "
            f"Refusing to delete or overwrite without verified state.",
            file=sys.stderr,
        )
        return False

    # Rule 2: Target existed before pilot -> MUST restore from verified consistent backup pair
    if target_existed_before_pilot is True:
        part_backups = find_backups(part_path)
        if not part_backups:
            print(
                f"[ERROR] Rollback aborted for {location}/{model}: target partition existed before pilot, "
                f"but no backup file was found at {part_path}.bak*. "
                f"Refusing to delete or overwrite without a verified recovery copy.",
                file=sys.stderr,
            )
            return False

        # Check backup journal first for coordinated commit pair
        journal_entries = load_backup_journal(project_dir)
        matched_entry = None
        for entry in reversed(journal_entries):
            if entry.get("location") == location and entry.get("model") == model:
                p_bak = entry.get("partition_backup")
                if p_bak and os.path.isfile(p_bak):
                    matched_entry = entry
                    break

        if matched_entry:
            cand_part_bak = matched_entry["partition_backup"]
            cand_manifest_bak = matched_entry.get("manifest_backup")
        else:
            # Fallback: candidate partition backup is the latest by embedded timestamp
            cand_part_bak = part_backups[0]
            cand_manifest_bak = None
            part_ts = extract_backup_timestamp(cand_part_bak)

            # Find matching manifest backup by timestamp coordination
            manifest_backups = find_backups(manifest_path)
            if part_ts:
                # 1. Exact timestamp match
                for mb in manifest_backups:
                    if extract_backup_timestamp(mb) == part_ts:
                        cand_manifest_bak = mb
                        break
                # 2. If no exact match, find closest manifest backup immediately at or prior to partition timestamp
                if not cand_manifest_bak:
                    for mb in manifest_backups:
                        m_ts = extract_backup_timestamp(mb)
                        if m_ts and m_ts <= part_ts:
                            cand_manifest_bak = mb
                            break

            if not cand_manifest_bak and manifest_backups:
                cand_manifest_bak = manifest_backups[0]

        # Verify candidate partition backup
        try:
            with open(cand_part_bak, "rb") as f_bak:
                meta_bak = pq.read_metadata(f_bak)
            part_rows = meta_bak.num_rows
        except Exception as e:
            print(
                f"[ERROR] Rollback aborted for {location}/{model}: partition backup {cand_part_bak} is invalid: {e}. "
                f"Refusing to modify active partition.",
                file=sys.stderr,
            )
            return False

        # Verify candidate manifest backup and PAIR CONSISTENCY
        manifest_data = None
        if cand_manifest_bak and os.path.isfile(cand_manifest_bak):
            try:
                with open(cand_manifest_bak, "r", encoding="utf-8") as f_m:
                    manifest_data = json.load(f_m)
            except Exception as e:
                print(
                    f"[ERROR] Rollback aborted for {location}/{model}: manifest backup {cand_manifest_bak} is invalid JSON: {e}. "
                    f"Refusing to restore an unverified manifest.",
                    file=sys.stderr,
                )
                return False

            # Verify that manifest backup contains this partition and that row counts match
            part_in_manifest = [
                p for p in manifest_data.get("partitions", [])
                if p.get("location") == location and p.get("model") == model
            ]
            if not part_in_manifest:
                print(
                    f"[ERROR] Rollback aborted for {location}/{model}: manifest backup {cand_manifest_bak} does not contain an entry for {location}/{model}. "
                    f"The partition and manifest backups do NOT form a consistent pair. Refusing to restore inconsistent state.",
                    file=sys.stderr,
                )
                return False

            manifest_rows = part_in_manifest[0].get("row_count")
            if manifest_rows != part_rows:
                print(
                    f"[ERROR] Rollback aborted for {location}/{model}: Inconsistent backup pair! "
                    f"Partition backup has {part_rows} rows, but manifest backup records {manifest_rows} rows. "
                    f"Refusing to restore inconsistent state.",
                    file=sys.stderr,
                )
                return False

        # Both partition and manifest backups are verified and proven consistent!
        shutil.copy2(cand_part_bak, part_path)
        print(f"[ROLLBACK] Restored partition {location}/{model} from verified backup: {cand_part_bak}")

        if cand_manifest_bak and manifest_data:
            shutil.copy2(cand_manifest_bak, manifest_path)
            print(f"[ROLLBACK] Restored manifest from consistent paired backup: {cand_manifest_bak}")

    # Rule 3: Target did NOT exist before pilot -> Remove newly created pilot file and restore manifest
    elif target_existed_before_pilot is False:
        part_backups = find_backups(part_path)
        if part_backups:
            print(
                f"[ERROR] Rollback aborted for {location}/{model}: target was recorded as non-existent before pilot, "
                f"but backup file(s) exist: {part_backups}. Contradictory state. Refusing to delete.",
                file=sys.stderr,
            )
            return False

        if os.path.isfile(part_path):
            os.remove(part_path)
            print(f"[ROLLBACK] Removed pilot-created partition: {part_path}")
            loc_dir = os.path.dirname(part_path)
            try:
                if os.path.isdir(loc_dir) and not os.listdir(loc_dir):
                    os.rmdir(loc_dir)
            except OSError:
                pass

        # Restore manifest from latest verified backup where this partition is NOT present
        manifest_backups = find_backups(manifest_path)
        for mb in manifest_backups:
            try:
                with open(mb, "r", encoding="utf-8") as f_m:
                    m_data = json.load(f_m)
                part_in_m = [
                    p for p in m_data.get("partitions", [])
                    if p.get("location") == location and p.get("model") == model
                ]
                if not part_in_m:
                    # Found the manifest backup prior to partition addition
                    shutil.copy2(mb, manifest_path)
                    print(f"[ROLLBACK] Restored manifest from verified pre-pilot backup: {mb}")
                    break
            except Exception:
                continue

    return True


class OverviewMetadataIndex(dict):
    """
    Robust, ambiguity-safe index for master/overview TSV metadata.
    Maintains dictionary interface for backward-compatibility while supporting:
    - Exact (wav_file, channel) lookup
    - Extension-case normalization (.WAV <-> .wav)
    - Full-path case-insensitivity
    - Omitted location prefix layouts (e.g. Draga-2024_02A/... when on disk it's Draga/Draga-2024_02A/...)
    - Location-prefixed synthetic paths
    - Conflict/ambiguity detection: refuses to assign uncertain metadata when keys collide.
    """
    def __init__(self, tsv_path: Optional[str] = None):
        super().__init__()
        self.tsv_path = tsv_path
        self.exact_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.ext_norm_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.ambiguous_keys: Set[Tuple[str, int]] = set()
        self.ambiguous_details: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self.total_master_rows = 0
        self.exact_duplicates_count = 0
        self.locations_in_master: Set[str] = set()

        if tsv_path and os.path.isfile(tsv_path):
            self._load(tsv_path)

    def _records_conflict(self, r1: Dict[str, Any], r2: Dict[str, Any]) -> bool:
        """Check if two metadata records have conflicting values for essential metadata."""
        check_cols = ["location", "microlocation", "recorder_type", "day", "start_time", "channel_name"]
        for col in check_cols:
            v1 = str(r1.get(col, "")).strip()
            v2 = str(r2.get(col, "")).strip()
            if v1 and v2 and v1 != v2:
                return True
        return False

    def _index_candidate_key(self, key_tuple: Tuple[str, int], record: Dict[str, Any], target_dict: Dict):
        """Index a candidate key, tracking ambiguities if distinct records conflict."""
        norm_key = (key_tuple[0], int(key_tuple[1]))
        lower_key = (norm_key[0].lower(), norm_key[1])

        if lower_key in self.ambiguous_keys:
            self.ambiguous_details[lower_key].append(record)
            return

        if lower_key in target_dict:
            existing = target_dict[lower_key]
            if self._records_conflict(existing, record):
                self.ambiguous_keys.add(lower_key)
                self.ambiguous_details[lower_key] = [existing, record]
                target_dict.pop(lower_key, None)
                return
            # If records do not conflict (identical metadata), keep existing cleanly
        else:
            target_dict[lower_key] = record

    def _load(self, tsv_path: str):
        try:
            df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
            if "wav_file" not in df.columns:
                return
            self.total_master_rows = len(df)
            has_channel = "channel" in df.columns
            has_loc = "location" in df.columns

            cols = list(df.columns)
            for row in df.itertuples(index=False):
                r = dict(zip(cols, row))
                raw_w = str(r["wav_file"])
                w_norm = normalize_posix(raw_w)
                ch_val = r.get("channel")
                ch = int(ch_val) if has_channel and pd.notna(ch_val) and str(ch_val).isdigit() else 1
                loc = str(r.get("location", "")).strip() if has_loc and pd.notna(r.get("location")) else ""
                if loc:
                    self.locations_in_master.add(loc)

                exact_key = (w_norm, ch)
                lower_exact = (w_norm.lower(), ch)
                if exact_key in self.exact_lookup:
                    if self._records_conflict(self.exact_lookup[exact_key], r):
                        self.ambiguous_keys.add(lower_exact)
                        self.ambiguous_details[lower_exact] = [self.exact_lookup[exact_key], r]
                    else:
                        self.exact_duplicates_count += 1
                else:
                    self.exact_lookup[exact_key] = r

                # Dictionary population for backward-compatibility
                self[exact_key] = r

                # Extension-normalized key
                w_ext = normalize_path_ext(w_norm)
                self._index_candidate_key((w_ext, ch), r, self.ext_norm_lookup)

                # If master path does not start with location prefix, index synthetic prefixed key
                if loc and not w_ext.lower().startswith(loc.lower() + "/"):
                    synth_key = f"{loc}/{w_ext}"
                    self._index_candidate_key((synth_key, ch), r, self.ext_norm_lookup)

        except Exception as e:
            print(f"[WARNING] Could not parse overview TSV {tsv_path}: {e}")

    def lookup_metadata(
        self,
        wav_rel: str,
        channel: int = 1,
        loc: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Lookup metadata with precedence and ambiguity protection.
        Returns (meta_dict, status_string).
        """
        ch_int = int(channel)
        w_norm = normalize_posix(wav_rel)

        # 1. Exact match
        exact_k = (w_norm, ch_int)
        lower_k = (w_norm.lower(), ch_int)

        if lower_k in self.ambiguous_keys:
            return None, "AMBIGUOUS"

        if exact_k in self.exact_lookup:
            return self.exact_lookup[exact_k], "EXACT"

        # 2. Extension-normalized & case-normalized lookup
        w_ext = normalize_path_ext(w_norm)
        ext_k = (w_ext.lower(), ch_int)
        if ext_k in self.ambiguous_keys:
            return None, "AMBIGUOUS"
        if ext_k in self.ext_norm_lookup:
            return self.ext_norm_lookup[ext_k], "EXTENSION_CASE"

        # 3. Stripped location prefix lookup
        cand_stripped = None
        if loc and w_norm.lower().startswith(loc.lower() + "/"):
            cand_stripped = w_norm[len(loc) + 1:]
        elif "/" in w_norm:
            parts = w_norm.split("/", 1)
            cand_stripped = parts[1]

        if cand_stripped:
            str_ext = normalize_path_ext(cand_stripped)
            str_k = (str_ext.lower(), ch_int)
            if str_k in self.ambiguous_keys:
                return None, "AMBIGUOUS"
            if str_k in self.ext_norm_lookup:
                rec = self.ext_norm_lookup[str_k]
                rec_loc = str(rec.get("location", "")).strip()
                path_confirms_loc = bool(rec_loc and str_ext.lower().startswith(rec_loc.lower() + "/"))
                if loc and rec_loc and loc.lower() != rec_loc.lower() and not path_confirms_loc:
                    return None, "AMBIGUOUS"
                return rec, "STRIPPED_PREFIX"

        return None, "MISSING"

    def get(self, key, default=None):
        """Override dict.get to support transparent fallback lookup."""
        if isinstance(key, tuple) and len(key) == 2:
            meta, status = self.lookup_metadata(str(key[0]), int(key[1]))
            if meta is not None:
                return meta
            return default
        return super().get(key, default)


def load_overview_metadata(
    project_dir: str,
    overview_tsv_override: Optional[str] = None,
) -> Tuple[OverviewMetadataIndex, Optional[str]]:
    """
    Search for project overview TSV (e.g. zabe+hyla+citimon.tsv, Rok_spring_summer.tsv or zabe+hyla.tsv)
    and return populated OverviewMetadataIndex and found_tsv_path.
    """
    if overview_tsv_override:
        if os.path.isfile(overview_tsv_override):
            return OverviewMetadataIndex(overview_tsv_override), overview_tsv_override
        cand_in_proj = os.path.join(project_dir, overview_tsv_override)
        if os.path.isfile(cand_in_proj):
            return OverviewMetadataIndex(cand_in_proj), cand_in_proj

    candidates = [
        "zabe+hyla+citimon.tsv",
        "Rok_spring_summer.tsv",
        "zabe+hyla.tsv",
        "zabe_combined.tsv",
        "zabe.tsv",
        "dataset_meta.tsv",
    ]
    found_tsv = None
    for cand in candidates:
        p = os.path.join(project_dir, cand)
        if os.path.isfile(p):
            found_tsv = p
            break

    if not found_tsv:
        # Check any *.tsv in project root
        for f in os.listdir(project_dir):
            if f.endswith(".tsv") and not f.endswith("_old.tsv"):
                found_tsv = os.path.join(project_dir, f)
                break

    if not found_tsv:
        return OverviewMetadataIndex(None), None

    idx = OverviewMetadataIndex(found_tsv)
    return idx, found_tsv


def resolve_source_root(project_dir: str) -> str:
    """
    Determine source root for project: recordings/ is the canonical source root.
    mp3/ is only used as a legacy backward-compatibility fallback when recordings/ does not exist.
    """
    rec_root = os.path.join(project_dir, "recordings")
    if os.path.isdir(rec_root):
        return rec_root

    mp3_root = os.path.join(project_dir, "mp3")
    if os.path.isdir(mp3_root):
        return mp3_root

    raise FileNotFoundError(f"Neither 'recordings' nor 'mp3' source directory found under {project_dir}")


DEFAULT_KNOWN_MODELS = [
    "eat-finetuned-top10classes",
    "eat-finetuned-top20classes",
    "eat-finetuned-top50classes",
    "eat-pretrained-on-vibro",
    "proto_passt-CFPT-128head",
    "passt-pcen-C",
    "perchv2",
    "m2d",
]


def parse_and_rank_tsv_filename(
    filename: str,
    known_models: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], int, int, str]:
    """
    Robust parsing of a positions TSV filename into:
        (stem, sep, model, k, rank, rank_reason)

    Ranking hierarchy (Policy D):
        Rank 40: Standard underscore with explicit k ('_<model>_<k>.umap.tsv' or '_<model>_<k>.tsv')
        Rank 30: Underscore missing separator before k ('_<model><k>.umap.tsv' or '_<model><k>.tsv')
        Rank 20: Underscore missing explicit k ('_<model>.umap.tsv' or '_<model>.tsv')
        Rank 15: Custom numeric / non-standard k with underscore
        Rank 10: Hyphen-separated with explicit k ('-<model>_<k>.umap.tsv' or '-<model>_<k>.tsv')
        Rank  5: Hyphen-separated missing explicit k ('-<model>.umap.tsv' or '-<model>.tsv')
        Rank  1: Non-standard suffix (e.g. '_rerun', '_v2')
        Rank  0: Hyphen-separated missing k separator ('-<model><k>.umap.tsv' or '-<model><k>.tsv')
        Rank -1: Unrecognized naming pattern
    """
    models = list(known_models) if known_models else DEFAULT_KNOWN_MODELS
    models_sorted = sorted(models, key=len, reverse=True)

    found_m = None
    m_pos = -1
    for m in models_sorted:
        pos = filename.find(m)
        if pos != -1:
            found_m = m
            m_pos = pos
            break

    if not found_m or m_pos <= 0:
        # Fallback regex for generic model naming: <stem>[_<sep>]<model>[_<k>|<k>]?.(umap.)?tsv
        m_match = re.match(
            r"^(.+?)([_ \-])([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*?)(?:_(\d+)|(\d+))?(?:_rerun)?(?:\.umap)?\.tsv$",
            filename,
        )
        if m_match:
            stem_g, sep_g, found_m_g, k_us_g, k_no_g = m_match.groups()[:5]
            found_m = found_m_g
            m_pos = len(stem_g) + 1
        else:
            return None, None, None, 10, -1, "unrecognized_naming_pattern"

    sep = filename[m_pos - 1]
    stem = filename[: m_pos - 1]
    remainder = filename[m_pos + len(found_m) :]

    if remainder.endswith(".umap.tsv"):
        k_part = remainder[: -len(".umap.tsv")]
    elif remainder.endswith(".tsv"):
        k_part = remainder[: -len(".tsv")]
    else:
        k_part = remainder

    if k_part in ("_10", "_20"):
        k = int(k_part.lstrip("_"))
        rank = 40 if sep == "_" else 10
        reason = "standard_underscore_with_explicit_k" if sep == "_" else "hyphen_separated_with_explicit_k"
    elif k_part in ("10", "20"):
        k = int(k_part)
        rank = 30 if sep == "_" else 0
        reason = "underscore_missing_k_separator" if sep == "_" else "hyphen_separated_missing_k_separator"
    elif k_part == "":
        k = 10
        rank = 20 if sep == "_" else 5
        reason = "underscore_missing_explicit_k" if sep == "_" else "hyphen_missing_explicit_k"
    elif k_part.startswith("_") and k_part.lstrip("_").isnumeric():
        k = int(k_part.lstrip("_"))
        rank = 40 if sep == "_" else 10
        reason = "standard_underscore_with_explicit_k" if sep == "_" else "hyphen_separated_with_explicit_k"
    elif k_part.isnumeric():
        k = int(k_part)
        rank = 30 if sep == "_" else 0
        reason = "underscore_missing_k_separator" if sep == "_" else "hyphen_separated_missing_k_separator"
    else:
        k = 10
        suffix_clean = k_part.lstrip("_-")
        rank = 1
        reason = f"non_standard_suffix_{suffix_clean}"

    return stem, sep, found_m, k, rank, reason


def parse_single_tsv_file(
    tsv_item: Dict[str, Any],
    overview_meta: Optional[Any],
    project: str,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, int]]:
    """
    Fast worker function to parse a single positions TSV file.
    Returns:
        rows: list of clip dictionaries conforming to CANONICAL_PARQUET_SCHEMA
        error: error string if corrupted or malformed, else None
        stats: counts of cataloged, uncataloged, ambiguous, defaulted clips
    """
    rows: List[Dict[str, Any]] = []
    stats = {"cataloged": 0, "uncataloged": 0, "ambiguous": 0, "defaulted": 0}

    tsv_path = tsv_item["full_path"]
    tsv_rel = tsv_item["rel_path"]
    fname = tsv_item["filename"]
    loc = tsv_item["location"]
    dir_rel = tsv_item.get("dir_rel", "")

    cand_model = tsv_item.get("model")
    cluster_num_default = tsv_item.get("k")
    if not cand_model or cand_model == "unknown_model" or cluster_num_default is None:
        _, _, parsed_m, parsed_k, _, _ = parse_and_rank_tsv_filename(fname)
        cand_model = parsed_m or "unknown_model"
        cluster_num_default = parsed_k if parsed_k is not None else extract_cluster_num_from_tsv(fname)

    try:
        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header:
                return [], "TSV file is empty or missing header", stats

            h_map = {h: i for i, h in enumerate(header)}
            for req in ["file_name", "clip_time", "embedding"]:
                if req not in h_map:
                    return [], f"Missing required column '{req}' in header", stats

            fn_idx = h_map.get("file_name")
            ch_idx = h_map.get("channel")
            ct_idx = h_map.get("clip_time")
            dur_idx = h_map.get("clip_duration")
            emb_idx = h_map.get("embedding")
            m_idx = h_map.get("model_name")
            cid_idx = h_map.get("cluster_id")
            k_idx = h_map.get("num_clusters")
            micro_idx = h_map.get("microlocation")

            for r in reader:
                if not r or len(r) <= ct_idx:
                    continue
                raw_fn = r[fn_idx] if fn_idx is not None and fn_idx < len(r) else ""
                base_fn = os.path.basename(raw_fn)

                try:
                    ch = int(float(r[ch_idx])) if ch_idx is not None and ch_idx < len(r) and r[ch_idx] else 1
                except ValueError:
                    ch = 1

                try:
                    ct = float(r[ct_idx]) if ct_idx < len(r) and r[ct_idx] else 0.0
                except ValueError:
                    ct = 0.0

                dur = 0.0
                if dur_idx is not None and dur_idx < len(r) and r[dur_idx]:
                    try:
                        dur = float(r[dur_idx])
                    except ValueError:
                        dur = 0.0

                model = r[m_idx] if m_idx is not None and m_idx < len(r) and r[m_idx] else cand_model

                cid = -1
                if cid_idx is not None and cid_idx < len(r) and r[cid_idx]:
                    try:
                        cid = int(float(r[cid_idx]))
                    except ValueError:
                        cid = -1

                k = cluster_num_default
                if k_idx is not None and k_idx < len(r) and r[k_idx]:
                    try:
                        k = int(float(r[k_idx]))
                    except ValueError:
                        k = cluster_num_default

                emb_str = r[emb_idx] if emb_idx is not None and emb_idx < len(r) else ""
                try:
                    emb_clean = emb_str.strip("[]")
                    parts = emb_clean.split(",")
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                except (ValueError, IndexError):
                    x = 0.0
                    y = 0.0

                wav_rel = f"{dir_rel}/{base_fn}" if dir_rel else base_fn

                meta = None
                status = "MISSING"
                if overview_meta:
                    if hasattr(overview_meta, "lookup_metadata"):
                        meta, status = overview_meta.lookup_metadata(wav_rel, ch, loc=loc)
                    else:
                        meta = overview_meta.get((wav_rel, ch))
                        status = "EXACT" if meta else "MISSING"

                if meta:
                    stats["cataloged"] += 1
                    recorder_type = str(meta.get("recorder_type", "unknown"))
                    sunrise = str(meta.get("sunrise", "unknown"))
                    sunset = str(meta.get("sunset", "unknown"))
                    channel_name = str(meta.get("channel_name", str(ch)))
                    row_loc = str(meta.get("location", loc))
                    day_val = meta.get("day")
                    start_time_val = meta.get("start_time")
                    raw_micro = meta.get("microlocation")
                    if raw_micro is not None and pd.notna(raw_micro) and str(raw_micro).strip() and str(raw_micro).strip().lower() not in ["default", "none", "nan", "unknown"]:
                        microlocation = str(raw_micro).strip()
                    else:
                        microlocation = row_loc
                else:
                    stats["defaulted"] += 1
                    if status == "AMBIGUOUS":
                        stats["ambiguous"] += 1
                    else:
                        stats["uncataloged"] += 1

                    recorder_type = "unknown"
                    sunrise = "unknown"
                    sunset = "unknown"
                    channel_name = str(ch)
                    row_loc = loc
                    if overview_meta and hasattr(overview_meta, "locations_in_master") and overview_meta.locations_in_master:
                        if loc in overview_meta.locations_in_master:
                            row_loc = loc
                        elif loc.split("_")[0] in overview_meta.locations_in_master:
                            row_loc = loc.split("_")[0]
                        elif loc.replace("_", " ") in overview_meta.locations_in_master:
                            row_loc = loc.replace("_", " ")
                        elif len(overview_meta.locations_in_master) == 1:
                            row_loc = next(iter(overview_meta.locations_in_master))
                    day_val = None
                    start_time_val = None

                    raw_micro = (
                        r[micro_idx]
                        if micro_idx is not None and micro_idx < len(r) and r[micro_idx]
                        else None
                    )
                    if raw_micro is not None and pd.notna(raw_micro) and str(raw_micro).strip() and str(raw_micro).strip().lower() not in ["default", "none", "nan", "unknown"]:
                        microlocation = str(raw_micro).strip()
                    else:
                        microlocation = row_loc

                if not day_val or not start_time_val:
                    d_fn, t_fn = parse_filename_timestamp(base_fn)
                    if not day_val:
                        day_val = d_fn
                    if not start_time_val:
                        start_time_val = t_fn

                day_dt = None
                day_str = None
                if day_val:
                    try:
                        dt = datetime.datetime.strptime(str(day_val), "%Y-%m-%d")
                        day_dt = dt
                        day_str = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass

                start_dt = None
                time_of_day = None
                start_hour_float = None
                if start_time_val:
                    try:
                        t_p = datetime.datetime.strptime(str(start_time_val), "%H:%M:%S")
                        start_dt = datetime.datetime(1900, 1, 1, t_p.hour, t_p.minute, t_p.second)
                        time_of_day = start_dt + datetime.timedelta(seconds=ct)
                        start_hour_float = (
                            time_of_day.hour
                            + time_of_day.minute / 60.0
                            + (time_of_day.second + time_of_day.microsecond / 1e6) / 3600.0
                        )
                    except ValueError:
                        pass

                wav_norm = os.path.normpath(wav_rel)
                wav_base = re.sub(r"\.wav$", "", wav_norm, flags=re.IGNORECASE)
                mp3_file = f"{wav_base}_ch{ch}.mp3"
                abs_file_name = mp3_file
                rel_audio_path = mp3_file.replace("\\", "/")
                recording_id = f"{project}::{rel_audio_path}::ch{ch}"
                raw_clip_id = f"{recording_id}::{model}::{round(ct, 3)}"
                semantic_clip_key = f"{recording_id}::{round(ct, 3)}"

                rows.append({
                    "file_name": base_fn,
                    "channel": ch,
                    "clip_time": ct,
                    "clip_duration": dur,
                    "model_name": model,
                    "cluster_id": cid,
                    "wav_file": wav_norm,
                    "cluster_num": k,
                    "location": row_loc,
                    "microlocation": microlocation,
                    "channel_name": channel_name,
                    "recorder_type": recorder_type,
                    "sunrise": sunrise,
                    "sunset": sunset,
                    "day_dt": day_dt,
                    "start_dt": start_dt,
                    "x": x,
                    "y": y,
                    "time_of_day": time_of_day,
                    "start_hour_float": start_hour_float,
                    "day_str": day_str,
                    "mp3_file": mp3_file,
                    "abs_file_name": abs_file_name,
                    "source_file_name": raw_fn,
                    "source_tsv_path": tsv_rel,
                    "rel_audio_path": rel_audio_path,
                    "recording_id": recording_id,
                    "raw_clip_id": raw_clip_id,
                    "semantic_clip_key": semantic_clip_key,
                })
        return rows, None, stats
    except Exception as e:
        return [], str(e), stats


class SingleFileCacheBuilder:
    """
    Ingests all source locations and models into a single, complete final_data.parquet cache
    using bounded-memory streaming batches.
    """

    def __init__(
        self,
        project: str,
        base_dir: str = "data",
        batch_size: int = 50000,
        num_workers: int = 8,
        force: bool = False,
        dry_run: bool = False,
        overview_tsv: Optional[str] = None,
    ):
        self.project = project
        self.base_dir = base_dir
        self.batch_size = max(1, batch_size)
        self.num_workers = max(1, num_workers)
        self.force = force
        self.dry_run = dry_run

        self.project_dir = os.path.join(base_dir, project)
        if not os.path.isdir(self.project_dir):
            raise FileNotFoundError(f"Project directory does not exist: {self.project_dir}")

        self.source_root = resolve_source_root(self.project_dir)
        self.cache_dir = os.path.join(self.project_dir, "cache")
        self.final_path = os.path.join(self.cache_dir, "final_data.parquet")
        self.tmp_output_path = self.final_path + ".tmp"

        self.overview_meta, self.overview_tsv_path = load_overview_metadata(self.project_dir, overview_tsv_override=overview_tsv)
        self.diagnostics: List[Tuple[str, str]] = []
        self.duplicate_manifest: Dict[str, Any] = {}

    def log(self, level: str, msg: str):
        self.diagnostics.append((level, msg))
        if level == "ERROR":
            print(f"[ERROR] {msg}", file=sys.stderr)
        elif level == "WARNING":
            print(f"[WARNING] {msg}")
        else:
            print(f"[{level}] {msg}")

    def discover_all_tsvs(self) -> List[Dict[str, Any]]:
        """
        Recursively discover all locations and leaf positions/*.tsv files under source_root.
        Uses fast string slicing and directory pruning for sub-second traversal.
        Discovered items are sorted deterministically by relative path.
        """
        src_posix = normalize_posix(self.source_root).rstrip("/")
        locations = sorted([
            d for d in os.listdir(self.source_root)
            if os.path.isdir(os.path.join(self.source_root, d))
            and not d.startswith(".")
            and d not in {"cache", "temp", "tmp", "scratch", "positions"}
        ])

        has_flat_positions = os.path.isdir(os.path.join(self.source_root, "positions"))
        has_flat_audio = any(
            f.lower().endswith((".mp3", ".wav", ".flac"))
            for f in os.listdir(self.source_root)
            if os.path.isfile(os.path.join(self.source_root, f))
        )
        if not locations and (has_flat_positions or has_flat_audio):
            locations = [self.project]

        discovered: List[Dict[str, Any]] = []
        for loc in locations:
            if loc == self.project and not os.path.isdir(os.path.join(self.source_root, loc)):
                loc_path = self.source_root
            else:
                loc_path = os.path.join(self.source_root, loc)

            for root, dirs, files in os.walk(loc_path):
                if os.path.basename(root) == "positions":
                    dirs[:] = []  # Prune recursion into positions subdirectories
                    root_posix = normalize_posix(root)
                    rel_root = root_posix[len(src_posix):].lstrip("/")
                    parts = rel_root.split("/")
                    dir_rel = "/".join(parts[:-1]) if len(parts) > 1 else ""
                    for f in sorted(files):
                        if f.endswith(".tsv") and not f.endswith(".tmp"):
                            tsv_full = os.path.join(root, f)
                            tsv_rel = f"{rel_root}/{f}"
                            discovered.append({
                                "location": loc,
                                "full_path": tsv_full,
                                "rel_path": tsv_rel,
                                "filename": f,
                                "dir_rel": dir_rel,
                            })
        discovered.sort(key=lambda item: item["rel_path"])
        return discovered

    def select_canonical_tsvs(
        self,
        tsvs: List[Dict[str, Any]],
        known_models: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Groups discovered TSVs into logical recording families and applies Policy D
        canonical ranking to select exactly one canonical TSV per family.
        Records all duplicate decisions and any unresolved ambiguities in a provenance manifest.
        """
        families: Dict[Tuple[str, str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
        unrecognized: List[Dict[str, Any]] = []

        for item in tsvs:
            stem, sep, model, k, rank, reason = parse_and_rank_tsv_filename(item["filename"], known_models=known_models)
            cand = dict(item)
            cand.update({
                "stem": stem or item["filename"],
                "sep": sep or "_",
                "model": model or "unknown_model",
                "k": k,
                "rank": rank,
                "reason": reason,
            })
            if rank < 0:
                self.log("WARNING", f"Unrecognized positions TSV filename naming pattern: {item['filename']}")
                unrecognized.append(cand)
                continue

            family_key = (cand["location"], cand.get("dir_rel", ""), cand["stem"], cand["model"], cand["k"])
            families[family_key].append(cand)

        manifest_families: List[Dict[str, Any]] = []
        canonical_tsvs: List[Dict[str, Any]] = []

        for fam_key, candidates in families.items():
            # Deterministic candidate sorting:
            # 1. Primary: descending rank (-rank)
            # 2. Secondary (tie-breaker): ascending relative path (rel_path)
            sorted_candidates = sorted(candidates, key=lambda c: (-c["rank"], c["rel_path"]))
            top_c = sorted_candidates[0]
            canonical_tsvs.append(top_c)

            if len(sorted_candidates) > 1:
                second_c = sorted_candidates[1]
                if top_c["rank"] == second_c["rank"]:
                    try:
                        are_identical = filecmp.cmp(top_c["full_path"], second_c["full_path"], shallow=False)
                    except Exception:
                        are_identical = False

                    if are_identical:
                        is_ambiguous = False
                        ambiguity_reason = "Tied rank between identical file duplicates (byte-for-byte identical)"
                    else:
                        is_ambiguous = True
                        ambiguity_reason = "Tied rank with differing file contents; resolved by deterministic path tie-breaker"
                else:
                    is_ambiguous = False
                    ambiguity_reason = None

                manifest_families.append({
                    "logical_unit": {
                        "location": top_c["location"],
                        "dir_rel": top_c.get("dir_rel", ""),
                        "recording_stem": top_c["stem"],
                        "model": top_c["model"],
                        "k": top_c["k"],
                    },
                    "selected_path": top_c["rel_path"],
                    "selected_rank": top_c["rank"],
                    "selected_reason": top_c["reason"],
                    "discarded_paths": [c["rel_path"] for c in sorted_candidates[1:]],
                    "discarded_ranks": [c["rank"] for c in sorted_candidates[1:]],
                    "discarded_reasons": [c["reason"] for c in sorted_candidates[1:]],
                    "candidate_paths": [c["rel_path"] for c in sorted_candidates],
                    "candidates": [
                        {
                            "rel_path": c["rel_path"],
                            "rank": c["rank"],
                            "reason": c["reason"],
                        }
                        for c in sorted_candidates
                    ],
                    "is_ambiguous": is_ambiguous,
                    "unresolved_ambiguity": ambiguity_reason,
                })

        for unrec in unrecognized:
            canonical_tsvs.append(unrec)

        canonical_tsvs.sort(key=lambda c: c["rel_path"])

        manifest_families.sort(key=lambda f: (
            f["logical_unit"]["location"],
            f["logical_unit"]["dir_rel"],
            f["logical_unit"]["recording_stem"],
            f["logical_unit"]["model"],
            f["logical_unit"]["k"],
        ))

        manifest = {
            "project": self.project,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "policy": "Policy D: Deterministic Canonical Filename Selection",
            "total_duplicate_families": len(manifest_families),
            "total_candidate_files": sum(len(f["candidates"]) for f in manifest_families),
            "total_selected_files": len(manifest_families),
            "total_discarded_files": sum(len(f["discarded_paths"]) for f in manifest_families),
            "total_ambiguous_families": sum(1 for f in manifest_families if f["is_ambiguous"]),
            "families": manifest_families,
        }

        return canonical_tsvs, manifest

    def save_duplicate_manifest(self, manifest_path: Optional[str] = None):
        """
        Saves duplicate runs provenance manifest atomically to cache directory.
        """
        target = manifest_path or os.path.join(self.cache_dir, "duplicate_runs_manifest.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp_path = target + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.duplicate_manifest, f, indent=2)
        os.replace(tmp_path, target)

    def _validate_candidate(self, candidate_path: str, expected_rows: int):
        """
        Perform rigorous pre-commit validation checks on the candidate Parquet file.
        """
        if not os.path.isfile(candidate_path):
            raise RuntimeError(f"Candidate file does not exist: {candidate_path}")
        if os.path.getsize(candidate_path) == 0:
            raise RuntimeError(f"Candidate file is empty: {candidate_path}")

        meta = pq.read_metadata(candidate_path)
        if meta.num_rows != expected_rows:
            raise RuntimeError(f"Row count mismatch in candidate: expected {expected_rows}, got {meta.num_rows}")

        # Check required schema columns
        actual_cols = set(meta.schema.names)
        missing_cols = set(CANONICAL_PARQUET_SCHEMA.names) - actual_cols
        if missing_cols:
            raise RuntimeError(f"Missing canonical columns in candidate: {sorted(list(missing_cols))}")

        # Check existing final cache row count safety
        if os.path.isfile(self.final_path) and not self.force:
            try:
                exist_meta = pq.read_metadata(self.final_path)
                if expected_rows < exist_meta.num_rows * 0.9:
                    raise RuntimeError(
                        f"Candidate row count ({expected_rows:,}) is significantly lower than existing cache "
                        f"({exist_meta.num_rows:,}). Aborting replacement. Use --force to override."
                    )
            except Exception as e:
                if "Candidate row count" in str(e):
                    raise

    def migrate_saved_labels(self, candidate_path: str) -> Optional[str]:
        """
        Hard safety check: if saved_labels.parquet exists (legacy label file indexed by row_idx),
        resolve all historical labels against the existing legacy final_data.parquet and map
        them 1-to-1 onto the candidate cache's new row indices.

        Returns the path to the newly written candidate saved_labels parquet file (.tmp),
        or None if no saved_labels.parquet exists in cache_dir.
        Raises RuntimeError if any historical label cannot be resolved.
        """
        saved_labels_path = os.path.join(self.cache_dir, "saved_labels.parquet")
        if not os.path.isfile(saved_labels_path):
            return None

        if not os.path.isfile(self.final_path):
            raise RuntimeError(
                f"Cannot migrate saved_labels.parquet: active cache does not exist at {self.final_path}"
            )

        print("[SAVED LABELS] Hard safety check: migrating historical labels to candidate cache row order...", flush=True)
        labels_df = pq.read_table(saved_labels_path).to_pandas()
        total_labels = len(labels_df)
        if total_labels == 0:
            return None

        # 1. Load legacy records for the labeled row_indices
        legacy_cols = ["row_idx", "file_name", "channel", "clip_time", "clip_duration", "model_name", "cluster_num"]
        legacy_df = pd.read_parquet(self.final_path, columns=legacy_cols)
        legacy_labeled = labels_df.merge(legacy_df, on="row_idx", how="inner")
        if len(legacy_labeled) != total_labels:
            missing_count = total_labels - len(legacy_labeled)
            raise RuntimeError(
                f"Corrupt legacy saved_labels: {missing_count} row_idx values not found in existing final_data.parquet"
            )

        # 2. Read candidate rows matching the audio files in legacy_labeled
        candidate_cols = ["row_idx", "file_name", "channel", "clip_time", "clip_duration", "model_name", "cluster_num"]
        labeled_files = list(legacy_labeled["file_name"].unique())
        try:
            cand_sub = pd.read_parquet(
                candidate_path,
                columns=candidate_cols,
                filters=[("file_name", "in", labeled_files)],
            )
        except Exception:
            cand_df = pd.read_parquet(candidate_path, columns=candidate_cols)
            cand_sub = cand_df[cand_df["file_name"].isin(set(labeled_files))].copy()

        # 3. Map each legacy label to candidate row_idx
        cand_sub["ct_round"] = cand_sub["clip_time"].round(3)
        legacy_labeled["ct_round"] = legacy_labeled["clip_time"].round(3)

        # Fast exact match on (file_name, channel, ct_round, model_name, cluster_num)
        exact_matches = legacy_labeled.merge(
            cand_sub[["row_idx", "file_name", "channel", "ct_round", "model_name", "cluster_num"]],
            on=["file_name", "channel", "ct_round", "model_name", "cluster_num"],
            how="inner",
            suffixes=("_legacy", "_cand"),
        )

        resolved_map: Dict[int, int] = {}
        for _, row in exact_matches.iterrows():
            resolved_map[int(row["row_idx_legacy"])] = int(row["row_idx_cand"])

        # For remaining unmatched rows (e.g. historical superseded clip grids like 3.00s -> 2.88s),
        # resolve via maximum temporal overlap
        unresolved_indices = set(legacy_labeled["row_idx"]) - set(resolved_map.keys())
        if unresolved_indices:
            for unres_idx in unresolved_indices:
                r = legacy_labeled[legacy_labeled["row_idx"] == unres_idx].iloc[0]
                r_start = float(r["clip_time"])
                r_end = r_start + float(r["clip_duration"])

                cands = cand_sub[
                    (cand_sub["file_name"] == r["file_name"])
                    & (cand_sub["channel"] == r["channel"])
                    & (cand_sub["model_name"] == r["model_name"])
                    & (cand_sub["cluster_num"] == r["cluster_num"])
                ].copy()

                if cands.empty:
                    raise RuntimeError(
                        f"Failed to migrate historical label: no candidate rows match "
                        f"file={r['file_name']}, channel={r['channel']}, model={r['model_name']}, cluster_num={r['cluster_num']}"
                    )

                cands["overlap"] = (
                    cands["clip_time"].add(cands["clip_duration"]).clip(upper=r_end)
                    - cands["clip_time"].clip(lower=r_start)
                ).clip(lower=0)

                best = cands.sort_values("overlap", ascending=False).iloc[0]
                if best["overlap"] <= 0:
                    raise RuntimeError(
                        f"Failed to migrate historical label: 0 overlap for clip at t={r['clip_time']}s in {r['file_name']}"
                    )
                resolved_map[int(unres_idx)] = int(best["row_idx"])

        if len(resolved_map) != total_labels:
            raise RuntimeError(
                f"Historical label migration incomplete: resolved {len(resolved_map)} / {total_labels} labels"
            )

        # 4. Construct candidate saved_labels DataFrame maintaining legacy label values
        new_row_indices = [resolved_map[orig_idx] for orig_idx in legacy_labeled["row_idx"]]
        migrated_df = pd.DataFrame({
            "row_idx": new_row_indices,
            "label": legacy_labeled["label"].values,
        })

        tmp_saved_labels_path = saved_labels_path + ".tmp"
        if os.path.isfile(tmp_saved_labels_path):
            try:
                os.remove(tmp_saved_labels_path)
            except OSError:
                pass

        migrated_df.to_parquet(tmp_saved_labels_path, index=False)

        # Validate written tmp file
        meta = pq.read_metadata(tmp_saved_labels_path)
        if meta.num_rows != total_labels:
            raise RuntimeError(
                f"Migrated saved_labels row count mismatch: expected {total_labels}, got {meta.num_rows}"
            )

        print(
            f"[SAVED LABELS] Validated migration of all {total_labels:,} historical labels "
            f"({len(exact_matches)} exact, {len(unresolved_indices)} temporal-overlap fallback).",
            flush=True,
        )
        return tmp_saved_labels_path

    def build(self) -> Dict[str, Any]:
        """
        Execute bounded-memory ingestion and atomic replacement of final_data.parquet.
        """
        t_start = time.time()
        print(f"\n--- BUILDING SINGLE-FILE CACHE: Project '{self.project}' ---", flush=True)
        print(f"Source Root: {self.source_root}", flush=True)
        print(f"Output File: {self.final_path}", flush=True)
        print(f"Batch Size:  {self.batch_size:,} rows | Workers: {self.num_workers}", flush=True)

        # Stage 1: Discover all TSVs
        t0_stage1 = time.time()
        raw_tsvs = self.discover_all_tsvs()
        total_raw_tsvs = len(raw_tsvs)
        t_stage1 = time.time() - t0_stage1
        print(f"Stage 1 (TSV Discovery): {total_raw_tsvs:,} position TSV files discovered in {t_stage1:.2f}s.", flush=True)

        # Stage 2: Select canonical TSVs and generate duplicate-runs manifest (Policy D)
        t0_stage2 = time.time()
        canonical_tsvs, manifest = self.select_canonical_tsvs(raw_tsvs)
        self.duplicate_manifest = manifest
        total_canonical = len(canonical_tsvs)
        total_discarded = total_raw_tsvs - total_canonical
        total_dup_families = len(manifest.get("families", []))
        t_stage2 = time.time() - t0_stage2

        if total_dup_families > 0:
            print(
                f"Stage 2 (Canonical Selection - Policy D): Resolved {total_dup_families:,} duplicate run families in {t_stage2:.2f}s "
                f"(retained {total_canonical:,} canonical TSVs, discarded {total_discarded:,} alternate TSVs).",
                flush=True,
            )
        else:
            print(f"Stage 2 (Canonical Selection): {total_canonical:,} canonical TSVs retained in {t_stage2:.2f}s.", flush=True)

        if total_raw_tsvs == 0:
            if self.dry_run:
                return {
                    "total_rows": 0,
                    "deduplicated_count": 0,
                    "discovered_tsvs": 0,
                    "canonical_tsvs": 0,
                    "discarded_duplicate_tsvs": 0,
                    "duplicate_families_count": 0,
                    "ambiguous_families_count": 0,
                    "duplicate_manifest": manifest,
                    "dry_run": True,
                }
            raise RuntimeError(f"No position TSV files discovered in {self.source_root}")

        if self.dry_run:
            print("\n[DRY RUN] Validation completed without writing files.", flush=True)
            return {
                "total_rows": 0,
                "deduplicated_count": 0,
                "discovered_tsvs": total_raw_tsvs,
                "canonical_tsvs": total_canonical,
                "discarded_duplicate_tsvs": total_discarded,
                "duplicate_families_count": total_dup_families,
                "ambiguous_families_count": manifest.get("total_ambiguous_families", 0),
                "duplicate_manifest": manifest,
                "dry_run": True,
            }

        os.makedirs(self.cache_dir, exist_ok=True)
        if os.path.isfile(self.tmp_output_path):
            try:
                os.remove(self.tmp_output_path)
            except OSError:
                pass

        writer = pq.ParquetWriter(
            self.tmp_output_path,
            CANONICAL_PARQUET_SCHEMA,
            compression="snappy",
        )

        tsvs = canonical_tsvs
        total_tsvs = len(tsvs)
        seen_keys: Set[int] = set()
        deduplicated_count = 0
        total_emitted_rows = 0
        current_batch = {f.name: [] for f in CANONICAL_PARQUET_SCHEMA}

        total_cataloged = 0
        total_uncataloged = 0
        total_ambiguous = 0
        total_defaulted = 0
        all_locations = set()
        all_models = set()
        failed_tsvs = []

        chunk_size = max(10, min(1000, total_tsvs // self.num_workers if total_tsvs > self.num_workers else total_tsvs))
        processed_tsvs = 0
        last_progress_time = time.time()
        t0_stage3 = time.time()

        tmp_saved_labels = None

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                for i in range(0, total_tsvs, chunk_size):
                    chunk = tsvs[i : i + chunk_size]
                    results = list(
                        pool.map(
                            lambda item: parse_single_tsv_file(
                                item, self.overview_meta, self.project
                            ),
                            chunk,
                        )
                    )

                    for item, (rows, err, stats) in zip(chunk, results):
                        processed_tsvs += 1
                        if err:
                            failed_tsvs.append((item["rel_path"], err))
                            continue

                        total_cataloged += stats["cataloged"]
                        total_uncataloged += stats["uncataloged"]
                        total_ambiguous += stats["ambiguous"]
                        total_defaulted += stats["defaulted"]

                        for r in rows:
                            all_locations.add(r["location"])
                            all_models.add(r["model_name"])

                            dedup_key = hash((
                                r["wav_file"],
                                r["channel"],
                                round(r["clip_time"], 3),
                                r["model_name"],
                                r["cluster_num"],
                            ))
                            if dedup_key in seen_keys:
                                deduplicated_count += 1
                                continue
                            seen_keys.add(dedup_key)

                            # Assign sequential row_idx
                            r["row_idx"] = total_emitted_rows
                            total_emitted_rows += 1

                            for col_name in CANONICAL_PARQUET_SCHEMA.names:
                                current_batch[col_name].append(r[col_name])

                            if len(current_batch["file_name"]) >= self.batch_size:
                                table = pa.Table.from_pydict(current_batch, schema=CANONICAL_PARQUET_SCHEMA)
                                writer.write_table(table)
                                current_batch = {f.name: [] for f in CANONICAL_PARQUET_SCHEMA}

                    now = time.time()
                    if now - last_progress_time >= 5.0 or processed_tsvs == total_tsvs:
                        elapsed = now - t_start
                        pct = (processed_tsvs / total_tsvs) * 100.0
                        rate = processed_tsvs / elapsed if elapsed > 0 else 0
                        eta_sec = (total_tsvs - processed_tsvs) / rate if rate > 0 else 0
                        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
                        cur_mb, peak_mb = get_process_memory_mb()
                        mem_str = f" | RSS: {cur_mb:.1f}MB" if cur_mb is not None else ""
                        print(
                            f"  [PROGRESS] {processed_tsvs:,}/{total_tsvs:,} TSVs ({pct:.1f}%) | "
                            f"{total_emitted_rows:,} rows emitted | {deduplicated_count:,} duplicates skipped | "
                            f"elapsed: {elapsed:.1f}s | ETA: {eta_str}{mem_str}",
                            flush=True,
                        )
                        last_progress_time = now

            # Flush remaining rows in batch
            if len(current_batch["file_name"]) > 0:
                table = pa.Table.from_pydict(current_batch, schema=CANONICAL_PARQUET_SCHEMA)
                writer.write_table(table)
                current_batch = {f.name: [] for f in CANONICAL_PARQUET_SCHEMA}

            writer.close()
            writer = None
            t_stage3 = time.time() - t0_stage3

            # Abort if any TSVs failed parsing
            if failed_tsvs:
                raise RuntimeError(
                    f"Build failed validation: {len(failed_tsvs)} TSV(s) failed parsing/validation. "
                    f"First failure: {failed_tsvs[0][0]} ({failed_tsvs[0][1]})"
                )

            if total_emitted_rows == 0:
                raise RuntimeError("Build produced 0 rows. Aborting replacement.")

            # Stage 4: Validate candidate Parquet file before replacement
            t0_stage4 = time.time()
            self._validate_candidate(self.tmp_output_path, total_emitted_rows)
            t_stage4 = time.time() - t0_stage4

            # Stage 5: Hard safety check: migrate saved_labels.parquet mapping if it exists
            t0_stage5 = time.time()
            tmp_saved_labels = self.migrate_saved_labels(self.tmp_output_path)
            t_stage5 = time.time() - t0_stage5

            # Stage 6: Atomic commit with verified backups
            t0_stage6 = time.time()
            if os.path.isfile(self.final_path):
                create_safe_backup(self.final_path, is_parquet=True)

            saved_labels_path = os.path.join(self.cache_dir, "saved_labels.parquet")
            if tmp_saved_labels and os.path.isfile(saved_labels_path):
                create_safe_backup(saved_labels_path, is_parquet=True)

            os.replace(self.tmp_output_path, self.final_path)

            if tmp_saved_labels:
                os.replace(tmp_saved_labels, saved_labels_path)

            # Always save duplicate runs manifest atomically alongside final_data.parquet so stale manifests never linger
            self.save_duplicate_manifest()
            t_stage6 = time.time() - t0_stage6

        except Exception as e:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            if os.path.isfile(self.tmp_output_path):
                try:
                    os.remove(self.tmp_output_path)
                except OSError:
                    pass
            if tmp_saved_labels and os.path.isfile(tmp_saved_labels):
                try:
                    os.remove(tmp_saved_labels)
                except OSError:
                    pass
            raise RuntimeError(f"Single-file build aborted: {e}") from e

        cur_mb, peak_mb = get_process_memory_mb()
        elapsed_total = time.time() - t_start
        print(f"\nSuccessfully built single-file cache: {self.final_path}", flush=True)
        print(f"Total Rows:     {total_emitted_rows:,}", flush=True)
        print(f"Deduplicated:   {deduplicated_count:,}", flush=True)
        print(f"Locations:      {len(all_locations)} {sorted(list(all_locations))}", flush=True)
        print(f"Models:         {len(all_models)} {sorted(list(all_models))}", flush=True)
        print(f"Cataloged:      {total_cataloged:,}", flush=True)
        print(f"Uncataloged:    {total_uncataloged:,}", flush=True)
        print(f"Ambiguous:      {total_ambiguous:,}", flush=True)
        print(f"Defaulted:      {total_defaulted:,}", flush=True)
        print(f"Elapsed Time:   {elapsed_total:.2f}s", flush=True)
        if peak_mb is not None:
            print(f"Peak Memory:    {peak_mb:.1f} MB (Current: {cur_mb:.1f} MB)", flush=True)
        print("Stage Timings:", flush=True)
        print(f"  Stage 1 (TSV Discovery):        {t_stage1:.2f}s", flush=True)
        print(f"  Stage 2 (Canonical Selection):  {t_stage2:.2f}s", flush=True)
        print(f"  Stage 3 (Streaming Ingestion):  {t_stage3:.2f}s", flush=True)
        print(f"  Stage 4 (Candidate Validation): {t_stage4:.2f}s", flush=True)
        print(f"  Stage 5 (Label Migration):      {t_stage5:.2f}s", flush=True)
        print(f"  Stage 6 (Atomic Replacement):   {t_stage6:.2f}s", flush=True)

        return {
            "total_rows": total_emitted_rows,
            "deduplicated_count": deduplicated_count,
            "discovered_tsvs": total_raw_tsvs,
            "canonical_tsvs": total_canonical,
            "discarded_duplicate_tsvs": total_discarded,
            "duplicate_families_count": total_dup_families,
            "ambiguous_families_count": manifest.get("total_ambiguous_families", 0),
            "cataloged_recordings": total_cataloged,
            "uncataloged_recordings": total_uncataloged,
            "ambiguous_matches": total_ambiguous,
            "defaulted_records": total_defaulted,
            "locations": sorted(list(all_locations)),
            "models": sorted(list(all_models)),
            "elapsed_sec": elapsed_total,
            "peak_memory_mb": peak_mb,
            "stage_timings": {
                "tsv_discovery_sec": t_stage1,
                "canonical_selection_sec": t_stage2,
                "streaming_ingestion_sec": t_stage3,
                "candidate_validation_sec": t_stage4,
                "label_migration_sec": t_stage5,
                "atomic_replacement_sec": t_stage6,
            },
            "output_path": self.final_path,
            "duplicate_manifest": manifest,
            "migrated_saved_labels": tmp_saved_labels is not None,
        }


class CacheBuilder:
    def __init__(
        self,
        project: str,
        base_dir: str = "data",
        location_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        force: bool = False,
        overview_tsv: Optional[str] = None,
    ):
        self.project = project
        self.base_dir = base_dir
        self.location_filter = location_filter
        self.model_filter = model_filter
        self.force = force

        self.project_dir = os.path.join(base_dir, project)
        if not os.path.isdir(self.project_dir):
            raise FileNotFoundError(f"Project directory does not exist: {self.project_dir}")

        self.source_root = resolve_source_root(self.project_dir)
        self.cache_dir = os.path.join(self.project_dir, "cache")
        self.partitions_dir = os.path.join(self.cache_dir, "partitions")
        self.manifest_path = os.path.join(self.cache_dir, "manifest.json")

        self.overview_meta, self.overview_tsv_path = load_overview_metadata(self.project_dir, overview_tsv_override=overview_tsv)
        self.diagnostics: List[Tuple[str, str]] = []  # (level, message)

    def log(self, level: str, msg: str):
        self.diagnostics.append((level, msg))
        if level == "ERROR":
            print(f"[ERROR] {msg}", file=sys.stderr)
        elif level == "WARNING":
            print(f"[WARNING] {msg}")
        else:
            print(f"[{level}] {msg}")

    def discover(self) -> Dict[str, Any]:
        """
        Traverse source_root and discover:
        - Locations
        - Leaf positions/ dirs
        - Audio files (.mp3, .wav)
        - Position TSVs
        - Associations between audio and TSVs
        """
        locations = sorted([
            d for d in os.listdir(self.source_root)
            if os.path.isdir(os.path.join(self.source_root, d))
            and not d.startswith(".")
            and d not in {"cache", "temp", "tmp", "scratch", "positions"}
        ])

        has_flat_positions = os.path.isdir(os.path.join(self.source_root, "positions"))
        has_flat_audio = any(
            f.lower().endswith((".mp3", ".wav", ".flac"))
            for f in os.listdir(self.source_root)
            if os.path.isfile(os.path.join(self.source_root, f))
        )
        if not locations and (has_flat_positions or has_flat_audio):
            locations = [self.project]

        if self.location_filter:
            if self.location_filter not in locations:
                self.log("ERROR", f"Requested location '{self.location_filter}' not found in {self.source_root}")
                return {}
            locations = [self.location_filter]

        discovery_tree: Dict[str, Any] = {}

        for loc in locations:
            if loc == self.project and not os.path.isdir(os.path.join(self.source_root, loc)):
                loc_path = self.source_root
            else:
                loc_path = os.path.join(self.source_root, loc)
            pos_dirs: List[str] = []
            audio_files: Dict[str, str] = {}  # base_name -> audio_rel_path
            tsvs_by_base: Dict[str, List[Dict[str, Any]]] = {}

            for root, dirs, files in os.walk(loc_path):
                if os.path.basename(root) == "positions":
                    pos_dirs.append(root)
                    # Inspect TSVs
                    for f in files:
                        if f.endswith(".tsv"):
                            tsv_full = os.path.join(root, f)
                            tsv_rel = normalize_posix(os.path.relpath(tsv_full, self.source_root))
                            # Determine wav base_name
                            # Patterns: <base_name>_<model>_<cluster>.tsv or <base_name>_<model>_<cluster>.umap.tsv
                            # Or <base_name>_<model>.umap.tsv
                            # Strip .tsv, .umap
                            clean = f.replace(".tsv", "").replace(".umap", "")
                            # Notice base_name might contain dots/dates
                            # Typically the prefix matching audio base name
                            parent_dir = os.path.dirname(root)
                            tsvs_by_base.setdefault(parent_dir, []).append({
                                "filename": f,
                                "full_path": tsv_full,
                                "rel_path": tsv_rel,
                                "size": os.path.getsize(tsv_full),
                                "mtime_ns": os.stat(tsv_full).st_mtime_ns,
                            })
                else:
                    for f in files:
                        if f.lower().endswith((".mp3", ".wav", ".flac")):
                            audio_full = os.path.join(root, f)
                            audio_rel = normalize_posix(os.path.relpath(audio_full, self.source_root))
                            base = os.path.splitext(f)[0]
                            # Remove _chX if present for association key
                            base_no_ch = re.sub(r"_ch\d+$", "", base, flags=re.IGNORECASE)
                            key = (root, base_no_ch)
                            audio_files[key] = audio_rel

            discovery_tree[loc] = {
                "loc_path": loc_path,
                "pos_dirs": pos_dirs,
                "audio_files": audio_files,
                "tsvs_by_parent": tsvs_by_base,
            }

        return discovery_tree

    def validate(self) -> bool:
        """Execute validation-only audit without writing files."""
        print(f"\n--- VALIDATION MODE: Project '{self.project}' ---")
        print(f"Source Root: {self.source_root}")
        tree = self.discover()
        if not tree:
            self.log("ERROR", "Discovery failed or returned empty.")
            return False

        total_audio = 0
        total_tsvs = 0
        total_pos_dirs = 0

        # Audit master overview metadata
        print(f"\n--- Master Metadata Enrichment Audit (Master TSV: {self.overview_tsv_path or 'None'}) ---")
        if self.overview_tsv_path and hasattr(self.overview_meta, "total_master_rows"):
            print(f"  Master TSV loaded: {self.overview_meta.total_master_rows} rows across {len(self.overview_meta.locations_in_master)} locations.")
            if self.overview_meta.ambiguous_keys:
                self.log("ERROR", f"Master TSV has {len(self.overview_meta.ambiguous_keys)} ambiguous/colliding keys!")
        elif not self.overview_tsv_path:
            self.log("WARNING", "No master overview TSV found for project. All partitions will use fallback metadata.")

        for loc, data in tree.items():
            pos_dirs = data["pos_dirs"]
            audio_files = data["audio_files"]
            tsvs_by_parent = data["tsvs_by_parent"]

            loc_tsvs = sum(len(ts) for ts in tsvs_by_parent.values())
            loc_audio = len(audio_files)
            total_audio += loc_audio
            total_tsvs += loc_tsvs
            total_pos_dirs += len(pos_dirs)

            self.log("INFO", f"Location '{loc}': {len(pos_dirs)} positions dirs, {loc_audio} audio groups, {loc_tsvs} TSVs.")

            if loc_tsvs == 0:
                self.log("WARNING", f"Location '{loc}' has 0 position TSVs.")

            # Sample TSVs for model detection and metadata check
            loc_meta_sampled = 0
            loc_meta_matched = 0
            loc_meta_missing = 0
            loc_meta_ambiguous = 0
            missing_sample_files = []
            ambiguous_sample_files = []

            for parent, tsv_list in tsvs_by_parent.items():
                for tsv_info in tsv_list[:5]:
                    tsv_path = tsv_info["full_path"]
                    tsv_rel = tsv_info["rel_path"]
                    try:
                        sample_df = pd.read_csv(tsv_path, sep="\t", nrows=2)
                        if sample_df.empty:
                            self.log("WARNING", f"Empty TSV found: {tsv_info['rel_path']}")
                            continue
                        # Check required fields
                        for req in ["file_name", "clip_time", "embedding"]:
                            if req not in sample_df.columns:
                                self.log("ERROR", f"Missing required column '{req}' in {tsv_info['rel_path']}")

                        # Validate metadata lookup
                        if self.overview_meta:
                            raw_fn = os.path.basename(str(sample_df["file_name"].iloc[0]))
                            parts = tsv_rel.split("/")
                            dir_rel = "/".join(parts[:-2])
                            wav_rel = f"{dir_rel}/{raw_fn}" if dir_rel else raw_fn
                            ch = int(sample_df["channel"].iloc[0]) if "channel" in sample_df.columns and pd.notna(sample_df["channel"].iloc[0]) else 1

                            loc_meta_sampled += 1
                            if hasattr(self.overview_meta, "lookup_metadata"):
                                meta, status = self.overview_meta.lookup_metadata(wav_rel, ch, loc=loc)
                            else:
                                meta = self.overview_meta.get((wav_rel, ch))
                                status = "EXACT" if meta else "MISSING"

                            if meta:
                                loc_meta_matched += 1
                            elif status == "AMBIGUOUS":
                                loc_meta_ambiguous += 1
                                if len(ambiguous_sample_files) < 3:
                                    ambiguous_sample_files.append(f"{wav_rel} (ch={ch})")
                            else:
                                loc_meta_missing += 1
                                if len(missing_sample_files) < 3:
                                    missing_sample_files.append(f"{wav_rel} (ch={ch})")
                    except Exception as e:
                        self.log("ERROR", f"Failed to parse TSV {tsv_info['rel_path']}: {e}")

            # Report location metadata audit
            if loc_meta_sampled > 0:
                pct = (loc_meta_matched / loc_meta_sampled) * 100.0
                loc_known = hasattr(self.overview_meta, "locations_in_master") and loc in self.overview_meta.locations_in_master
                if loc_meta_ambiguous > 0:
                    self.log("ERROR", f"Location '{loc}': {loc_meta_ambiguous}/{loc_meta_sampled} sampled recordings have ambiguous metadata (samples: {ambiguous_sample_files})")
                elif loc_meta_matched == 0 and loc_known:
                    self.log("ERROR", f"Location '{loc}': 0/{loc_meta_sampled} metadata matches despite location existing in master TSV (samples: {missing_sample_files})")
                elif loc_meta_missing > 0 and loc_known:
                    self.log("WARNING", f"Location '{loc}': {loc_meta_missing}/{loc_meta_sampled} sampled recordings missing from master TSV (samples: {missing_sample_files})")
                elif loc_meta_missing > 0 and not loc_known:
                    self.log("INFO", f"Location '{loc}': {loc_meta_sampled} sampled recordings not in master TSV (new uncataloged recordings)")
                else:
                    self.log("INFO", f"Location '{loc}': {loc_meta_matched}/{loc_meta_sampled} metadata matches ({pct:.1f}%)")

        print(f"\nDiscovery Summary:")
        print(f"  Locations: {len(tree)}")
        print(f"  Leaf 'positions/' directories: {total_pos_dirs}")
        print(f"  Audio recording units: {total_audio}")
        print(f"  Position TSV files: {total_tsvs}")

        errors = [d for d in self.diagnostics if d[0] == "ERROR"]
        warnings = [d for d in self.diagnostics if d[0] == "WARNING"]
        print(f"\nValidation Result: {len(errors)} Errors, {len(warnings)} Warnings")
        return len(errors) == 0

    def ingest_location_model(
        self,
        loc: str,
        model_name: str,
        tsv_records: List[Dict[str, Any]],
        output_partition_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Transform and write a single partitioned Parquet file for (location, model).
        Returns partition metadata dict for manifest.json.
        """
        dfs: List[pd.DataFrame] = []
        failed_tsvs: List[Tuple[str, str]] = []
        defaulted_metadata_count = 0
        ambiguous_metadata_count = 0
        defaulted_samples: List[str] = []
        ambiguous_samples: List[str] = []

        for item in tsv_records:
            tsv_path = item["full_path"]
            tsv_rel = item["rel_path"]
            fname = item["filename"]

            try:
                df = pd.read_csv(tsv_path, sep="\t")
                if df.empty:
                    failed_tsvs.append((tsv_rel, "File is empty"))
                    self.log("ERROR", f"Empty TSV '{tsv_rel}'")
                    continue

                # Validate required columns
                required_cols = ["file_name", "channel", "clip_time", "embedding", "model_name", "cluster_id"]
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    reason = f"Missing required column(s): {missing_cols}"
                    failed_tsvs.append((tsv_rel, reason))
                    self.log("ERROR", f"Malformed TSV '{tsv_rel}': {reason}")
                    continue

                if "num_clusters" in df.columns:
                    df["cluster_num"] = df["num_clusters"].fillna(10).astype(int)
                else:
                    cluster_num = extract_cluster_num_from_tsv(fname)
                    df["cluster_num"] = cluster_num

                # Determine relative wav path and channel
                parts = tsv_rel.split("/")
                dir_rel = "/".join(parts[:-2])
                raw_source_fn = str(df["file_name"].iloc[0])
                raw_fn = os.path.basename(raw_source_fn)

                # Check path ambiguity: if raw_source_fn contains directory components,
                # verify that its directory suffix aligns with dir_rel
                norm_source = normalize_posix(raw_source_fn)
                if "/" in norm_source:
                    source_dir = os.path.dirname(norm_source)
                    leaf_dir = os.path.basename(dir_rel)
                    if not (dir_rel.endswith(source_dir) or source_dir.endswith(dir_rel) or source_dir.endswith(leaf_dir)):
                        reason = f"Ambiguous source path '{raw_source_fn}' does not align with directory '{dir_rel}'"
                        failed_tsvs.append((tsv_rel, reason))
                        self.log("ERROR", f"Path ambiguity in '{tsv_rel}': {reason}")
                        continue

                wav_rel = f"{dir_rel}/{raw_fn}" if dir_rel else raw_fn
                df["source_file_name"] = df["file_name"]
                df["file_name"] = df["file_name"].astype(str).apply(os.path.basename)

                # In some TSVs, metadata columns (e.g. day, start_time, channel_name) are int64.
                # Convert to object dtype so string assignment from overview metadata or defaults works cleanly.
                for meta_col in ["day", "start_time", "channel_name", "location", "microlocation", "recorder_type", "sunrise", "sunset"]:
                    if meta_col in df.columns:
                        df[meta_col] = df[meta_col].astype(object)

                unique_channels = df["channel"].unique() if "channel" in df.columns else [1]
                for ch_val in unique_channels:
                    ch_int = int(ch_val)
                    if hasattr(self.overview_meta, "lookup_metadata"):
                        meta, status = self.overview_meta.lookup_metadata(wav_rel, ch_int, loc=loc)
                    else:
                        meta = self.overview_meta.get((wav_rel, ch_int))
                        status = "EXACT" if meta else "MISSING"

                    mask = (df["channel"] == ch_val) if "channel" in df.columns else slice(None)
                    if meta:
                        for k, v in meta.items():
                            if k not in ["channel"]:
                                df.loc[mask, k] = v
                    else:
                        defaulted_metadata_count += 1
                        if status == "AMBIGUOUS":
                            ambiguous_metadata_count += 1
                            if len(ambiguous_samples) < 5:
                                ambiguous_samples.append(f"{wav_rel} (ch={ch_int})")
                        else:
                            if len(defaulted_samples) < 5:
                                defaulted_samples.append(f"{wav_rel} (ch={ch_int})")

                        day_parsed, time_parsed = parse_filename_timestamp(raw_fn)
                        df.loc[mask, "location"] = loc
                        df.loc[mask, "microlocation"] = loc
                        df.loc[mask, "channel_name"] = str(ch_int)
                        df.loc[mask, "recorder_type"] = "unknown"
                        df.loc[mask, "sunrise"] = "unknown"
                        df.loc[mask, "sunset"] = "unknown"
                        df.loc[mask, "day"] = day_parsed if day_parsed else "unknown"
                        df.loc[mask, "start_time"] = time_parsed if time_parsed else "00:00:00"

                df["wav_file"] = wav_rel
                df["source_tsv_path"] = tsv_rel

                dfs.append(df)

            except Exception as e:
                failed_tsvs.append((tsv_rel, str(e)))
                self.log("ERROR", f"Error processing TSV {tsv_rel}: {e}")

        # If any TSV failed validation or was malformed, reject the partition
        if failed_tsvs:
            self.log("ERROR", f"Partition {loc}/{model_name} rejected: {len(failed_tsvs)} TSV(s) failed validation")
            return None

        if not dfs:
            return None

        if ambiguous_metadata_count > 0:
            self.log(
                "WARNING",
                f"Partition {loc}/{model_name}: {ambiguous_metadata_count} clip(s) had ambiguous/conflicting master metadata and were assigned safe defaults (samples: {ambiguous_samples})",
            )
        if defaulted_metadata_count > ambiguous_metadata_count:
            missing_only = defaulted_metadata_count - ambiguous_metadata_count
            self.log(
                "INFO",
                f"Partition {loc}/{model_name}: {missing_only} clip(s) had no master metadata and were assigned safe defaults (samples: {defaulted_samples})",
            )

        try:
            big_df = pd.concat(dfs, ignore_index=True)

            # Datetime transformations
            big_df["day_dt"] = pd.to_datetime(big_df["day"], format="%Y-%m-%d", errors="coerce")
            big_df["start_dt"] = pd.to_datetime(big_df["start_time"], format="%H:%M:%S", errors="coerce")

            # Parse Embeddings
            if "embedding" in big_df.columns:
                temp_df = big_df["embedding"].astype(str).str.strip("[]").str.split(", ", expand=True)
                big_df["x"] = pd.to_numeric(temp_df[0], errors="coerce")
                big_df["y"] = pd.to_numeric(temp_df[1], errors="coerce")
            else:
                big_df["x"] = 0.0
                big_df["y"] = 0.0

            # Exact Timestamp
            big_df["time_of_day"] = big_df["start_dt"] + pd.to_timedelta(big_df["clip_time"], unit="s")
            big_df["start_hour_float"] = (
                big_df["time_of_day"].dt.hour
                + big_df["time_of_day"].dt.minute / 60.0
                + big_df["time_of_day"].dt.second / 3600.0
            )
            big_df["day_str"] = big_df["day_dt"].dt.strftime("%Y-%m-%d")

            # MP3 file path
            base_name = big_df["wav_file"].astype(str).str.replace(r"\.wav$", "", case=False, regex=True)
            big_df["mp3_file"] = (base_name + "_ch" + big_df["channel"].astype(str) + ".mp3").apply(os.path.normpath)
            big_df["abs_file_name"] = big_df["mp3_file"]
            if "cluster_id" in big_df.columns:
                big_df["cluster_id"] = pd.to_numeric(big_df["cluster_id"], errors="coerce").fillna(-1).astype(int)
            else:
                big_df["cluster_id"] = -1

            # Require model_name in TSV data
            if "model_name" not in big_df.columns:
                raise ValueError("Missing required column 'model_name' in TSV data")

            # Deduplicate multiple sweeps if duplicate TSVs exist for same recordings
            if set(['wav_file', 'channel', 'clip_time', 'model_name', 'cluster_num']).issubset(big_df.columns):
                big_df.drop_duplicates(subset=['wav_file', 'channel', 'clip_time', 'model_name', 'cluster_num'], inplace=True)

            # Canonical identities established in Phase 1.14B
            big_df["rel_audio_path"] = big_df["mp3_file"].astype(str).apply(lambda s: s.replace("\\", "/"))
            big_df["recording_id"] = f"{self.project}::" + big_df["rel_audio_path"] + "::ch" + big_df["channel"].astype(str)
            big_df["raw_clip_id"] = (
                big_df["recording_id"]
                + "::"
                + big_df["model_name"]
                + "::"
                + big_df["clip_time"].round(3).astype(str)
            )
            big_df["semantic_clip_key"] = (
                big_df["recording_id"] + "::" + big_df["clip_time"].round(3).astype(str)
            )

            # Clean columns to preserve exact schema
            columns_to_drop = ["day", "start_time", "num_clusters", "embedding"]
            big_df.drop(columns=[col for col in columns_to_drop if col in big_df.columns], inplace=True, errors="ignore")

            # Ensure text metadata columns have uniform string dtype to avoid pyarrow object inference errors
            for col in ["location", "microlocation", "channel_name", "recorder_type", "sunrise", "sunset"]:
                if col in big_df.columns:
                    big_df[col] = big_df[col].fillna("unknown").astype(str)

            # Rule: If no microlocation exists or it's 'default', use the physical location
            if "microlocation" in big_df.columns and "location" in big_df.columns:
                bad_micro = big_df["microlocation"].isna() | (
                    big_df["microlocation"].astype(str).str.strip().str.lower().isin(["", "default", "none", "nan", "unknown"])
                )
                if bad_micro.any():
                    big_df.loc[bad_micro, "microlocation"] = big_df.loc[bad_micro, "location"]

            # Stable row index
            big_df["row_idx"] = range(len(big_df))
        except Exception as e:
            self.log("ERROR", f"Error transforming data for {loc} / {model_name}: {e}")
            return None

        # Safe commit: write to temporary partition file first, validate, then atomic replace
        os.makedirs(os.path.dirname(output_partition_path), exist_ok=True)
        tmp_output_path = output_partition_path + ".tmp"
        try:
            big_df.to_parquet(tmp_output_path, compression="snappy", index=False)

            # Validate written partition immediately before committing
            with open(tmp_output_path, "rb") as f_tmp:
                meta_val = pq.read_metadata(f_tmp)
                col_names = set(meta_val.schema.names)
                if meta_val.num_rows != len(big_df):
                    raise ValueError(
                        f"Row count mismatch in written partition: expected {len(big_df)}, got {meta_val.num_rows}"
                    )
                for col in ["x", "y", "model_name", "channel", "clip_time", "raw_clip_id"]:
                    if col not in col_names:
                        raise ValueError(f"Required column '{col}' missing from written partition")

            if os.path.isfile(output_partition_path):
                create_safe_backup(output_partition_path, commit_id=getattr(self, "session_commit_id", None), is_parquet=True)

            os.replace(tmp_output_path, output_partition_path)
        except Exception as e:
            if os.path.isfile(tmp_output_path):
                try:
                    os.remove(tmp_output_path)
                except OSError:
                    pass
            self.log("ERROR", f"Failed committing partition {output_partition_path}: {e}")
            return None

        file_size = os.path.getsize(output_partition_path)
        valid_dates = big_df["day_str"].dropna()
        min_date = str(valid_dates.min()) if not valid_dates.empty else None
        max_date = str(valid_dates.max()) if not valid_dates.empty else None

        return {
            "location": loc,
            "model": model_name,
            "partition_rel_path": normalize_posix(os.path.relpath(output_partition_path, self.project_dir)),
            "row_count": len(big_df),
            "file_size_bytes": file_size,
            "source_files_count": len(tsv_records),
            "skipped_files_count": len(failed_tsvs),
            "defaulted_metadata_count": defaulted_metadata_count,
            "ambiguous_metadata_count": ambiguous_metadata_count,
            "defaulted_samples": defaulted_samples,
            "ambiguous_samples": ambiguous_samples,
            "date_extent": [min_date, max_date],
        }

    def build(self) -> Dict[str, Any]:
        """Execute partitioned cache build and generate manifest.json."""
        t_start = time.time()
        self.session_commit_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        print(f"\n--- BUILDING PARTITIONED CACHE: Project '{self.project}' ---")
        print(f"Source Root: {self.source_root}")
        print(f"Output Root: {self.partitions_dir}")

        tree = self.discover()
        if not tree:
            self.log("ERROR", "No data discovered to build.")
            return {}

        # Load existing manifest for incremental signature checks
        old_manifest = {}
        if os.path.isfile(self.manifest_path) and not self.force:
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    old_manifest = json.load(f)
            except Exception:
                pass

        manifest_partitions: List[Dict[str, Any]] = []
        loc_models_map: Dict[str, List[str]] = {}

        for loc, data in tree.items():
            tsvs_by_parent = data["tsvs_by_parent"]

            # Group TSVs by model_name
            tsvs_by_model: Dict[str, List[Dict[str, Any]]] = {}
            for parent, tsv_list in tsvs_by_parent.items():
                for tsv_item in tsv_list:
                    fname = tsv_item["filename"]

                    clean = fname.replace(".tsv", "").replace(".umap", "")
                    cand_model = None
                    if "perchv2" in clean:
                        cand_model = "perchv2"
                    elif "m2d" in clean:
                        cand_model = "m2d"
                    elif "passt-pcen-C" in clean:
                        if clean.endswith("passt-pcen-C"):
                            continue
                        cand_model = "passt-pcen-C"
                    elif "proto_passt-CFPT-128head" in clean:
                        cand_model = "proto_passt-CFPT-128head"
                    elif "eat-finetuned-top10classes" in clean:
                        cand_model = "eat-finetuned-top10classes"
                    elif "eat-finetuned-top20classes" in clean:
                        cand_model = "eat-finetuned-top20classes"
                    elif "eat-finetuned-top50classes" in clean:
                        cand_model = "eat-finetuned-top50classes"
                    elif "eat-pretrained-on-vibro" in clean:
                        cand_model = "eat-pretrained-on-vibro"
                    else:
                        # Fallback: inspect 1 line
                        try:
                            d_peek = pd.read_csv(tsv_item["full_path"], sep="\t", nrows=1)
                            if "model_name" in d_peek.columns:
                                cand_model = d_peek["model_name"].iloc[0]
                        except Exception:
                            pass

                    if not cand_model:
                        cand_model = "unknown_model"

                    if self.model_filter and cand_model != self.model_filter:
                        continue

                    tsvs_by_model.setdefault(cand_model, []).append(tsv_item)

            loc_models_map[loc] = sorted(list(tsvs_by_model.keys()))

            for model_name, tsv_records in tsvs_by_model.items():
                out_path = os.path.join(self.partitions_dir, loc, f"{model_name}.parquet")

                # Incremental signature computation
                overview_sig = ""
                if self.overview_tsv_path and os.path.isfile(self.overview_tsv_path):
                    st_ov = os.stat(self.overview_tsv_path)
                    overview_sig = f"{st_ov.st_size}_{st_ov.st_mtime_ns % 1000000}"

                source_sig = f"{len(tsv_records)}_{sum(t['size'] for t in tsv_records)}_{sum(t['mtime_ns'] % 1000000 for t in tsv_records)}"
                if overview_sig:
                    source_sig += f"_{overview_sig}"

                old_part_info = None
                if old_manifest and "partitions" in old_manifest:
                    for p in old_manifest["partitions"]:
                        if p.get("location") == loc and p.get("model") == model_name:
                            old_part_info = p
                            break

                can_reuse = False
                if (
                    not self.force
                    and old_part_info
                    and old_part_info.get("signature") == source_sig
                    and os.path.isfile(out_path)
                    and old_manifest.get("builder_version") == BUILDER_VERSION
                ):
                    try:
                        meta = pq.read_metadata(out_path)
                        if meta.num_rows == old_part_info.get("row_count"):
                            can_reuse = True
                    except Exception:
                        can_reuse = False

                if can_reuse:
                    print(f"  [REUSE] Partition {loc}/{model_name}.parquet (unchanged)")
                    manifest_partitions.append(old_part_info)
                    continue

                print(f"  [BUILD] Partition {loc}/{model_name}.parquet ({len(tsv_records)} source TSVs)...")
                t0_part = time.time()
                part_info = self.ingest_location_model(loc, model_name, tsv_records, out_path)
                if part_info:
                    part_info["signature"] = source_sig
                    manifest_partitions.append(part_info)
                    print(f"          -> {part_info['row_count']} rows in {time.time() - t0_part:.2f}s")
                    p_baks = find_backups(out_path)
                    p_bak_path = p_baks[0] if p_baks else None
                    m_baks = find_backups(self.manifest_path)
                    m_bak_path = m_baks[0] if m_baks else None
                    record_backup_journal_entry(
                        self.project_dir,
                        self.session_commit_id,
                        loc,
                        model_name,
                        p_bak_path,
                        m_bak_path,
                        part_info["row_count"],
                    )

        # Targeted build preservation: if location_filter or model_filter was active,
        # carry forward valid partitions from old_manifest that were outside the filter scope
        if (self.location_filter or self.model_filter) and old_manifest and "partitions" in old_manifest:
            for old_p in old_manifest.get("partitions", []):
                p_loc = old_p.get("location")
                p_mod = old_p.get("model")
                in_scope = True
                if self.location_filter and p_loc != self.location_filter:
                    in_scope = False
                if self.model_filter and p_mod != self.model_filter:
                    in_scope = False

                if not in_scope:
                    p_rel = old_p.get("partition_rel_path")
                    p_disk = os.path.join(self.project_dir, *p_rel.split("/")) if p_rel else None
                    if p_disk and os.path.isfile(p_disk):
                        try:
                            pq.read_metadata(p_disk)
                            manifest_partitions.append(old_p)
                            loc_models_map.setdefault(p_loc, []).append(p_mod)
                        except Exception:
                            pass

        # Cleanup stale partition files on full build
        if not self.location_filter and not self.model_filter:
            active_rel_paths = {p["partition_rel_path"] for p in manifest_partitions}
            if os.path.isdir(self.partitions_dir):
                for root, dirs, files in os.walk(self.partitions_dir):
                    for f in files:
                        if f.endswith(".parquet") and not f.endswith(".tmp"):
                            full_p = os.path.join(root, f)
                            rel_p = normalize_posix(os.path.relpath(full_p, self.project_dir))
                            if rel_p not in active_rel_paths:
                                self.log("INFO", f"Removing stale unreferenced partition file: {rel_p}")
                                try:
                                    os.remove(full_p)
                                except OSError:
                                    pass

        # Derive manifest locations and models cleanly from committed partitions
        models_by_loc_clean: Dict[str, Set[str]] = {}
        for p in manifest_partitions:
            models_by_loc_clean.setdefault(p["location"], set()).add(p["model"])
        manifest_locations = sorted(list(models_by_loc_clean.keys()))
        models_by_loc_dict = {k: sorted(list(v)) for k, v in sorted(models_by_loc_clean.items())}

        # Deterministic manifest generation
        manifest_data = {
            "manifest_version": MANIFEST_SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "project": self.project,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "locations": manifest_locations,
            "models_by_location": models_by_loc_dict,
            "total_partitions": len(manifest_partitions),
            "total_rows": sum(p["row_count"] for p in manifest_partitions),
            "partitions": sorted(manifest_partitions, key=lambda p: (p["location"], p["model"])),
        }

        # Safe commit of manifest.json via temporary file and atomic replace
        tmp_manifest = self.manifest_path + ".tmp"
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        try:
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, sort_keys=True)
            with open(tmp_manifest, "r", encoding="utf-8") as f:
                json.load(f)
            if os.path.isfile(self.manifest_path):
                create_safe_backup(self.manifest_path, commit_id=getattr(self, "session_commit_id", None), is_json=True)

            os.replace(tmp_manifest, self.manifest_path)
        except Exception as e:
            if os.path.isfile(tmp_manifest):
                try:
                    os.remove(tmp_manifest)
                except OSError:
                    pass
            self.log("ERROR", f"Failed committing manifest to {self.manifest_path}: {e}")
            return {}

        print(f"\nManifest written to: {self.manifest_path}")
        print(f"Total rows in generated partitions: {manifest_data['total_rows']}")
        print(f"Total build time: {time.time() - t_start:.2f}s")
        return manifest_data


def main():
    parser = argparse.ArgumentParser(description="VibroVis Cache Ingestion Engine")
    parser.add_argument("--project", default=None, help="Project name (e.g. vibro, zabe)")
    parser.add_argument("--all", action="store_true", help="Build all projects (vibro and zabe)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for streaming Parquet writes (default: 50,000)")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads for parallel TSV parsing (default: 8)")
    parser.add_argument("--validate", action="store_true", help="Discovery/validation audit mode; write nothing")
    parser.add_argument("--dry-run", action="store_true", help="Dry run validation; write nothing")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists or size checks fail")
    parser.add_argument("--partitioned", action="store_true", help="Prototype mode: build partitioned cache instead of single final_data.parquet")
    parser.add_argument("--location", default=None, help="Optional location restriction (for --partitioned)")
    parser.add_argument("--model", default=None, help="Optional model restriction (for --partitioned)")
    parser.add_argument("--rollback", action="store_true", help="Execute safe rollback for specified location/model")
    parser.add_argument("--overview-tsv", default=None, help="Explicit path or filename for master overview TSV (e.g. data/zabe/zabe+hyla+citimon.tsv)")
    parser.add_argument(
        "--target-existed-before-pilot",
        choices=["true", "false", "unknown"],
        default="unknown",
        help="Explicitly declare if target partition existed before pilot: 'true', 'false', or 'unknown'",
    )

    args = parser.parse_args()

    try:
        if args.rollback:
            if not args.project:
                print("[ERROR] --rollback requires --project", file=sys.stderr)
                sys.exit(1)
            project_dir = os.path.join("data", args.project)
            if not args.location or not args.model:
                print("[ERROR] --rollback requires both --location and --model", file=sys.stderr)
                sys.exit(1)
            target_existed = (
                True if args.target_existed_before_pilot == "true"
                else (False if args.target_existed_before_pilot == "false" else None)
            )
            ok = safe_rollback(
                project_dir=project_dir,
                location=args.location,
                model=args.model,
                target_existed_before_pilot=target_existed,
            )
            sys.exit(0 if ok else 1)

        if args.partitioned:
            if not args.project:
                print("[ERROR] --partitioned requires --project", file=sys.stderr)
                sys.exit(1)
            builder = CacheBuilder(
                project=args.project,
                location_filter=args.location,
                model_filter=args.model,
                force=args.force,
                overview_tsv=args.overview_tsv,
            )
            if args.validate or args.dry_run:
                ok = builder.validate()
                sys.exit(0 if ok else 1)
            else:
                manifest = builder.build()
                sys.exit(0 if manifest else 1)

        # Default Single-File Cache Ingestion Workflow
        projects = []
        if args.all:
            projects = ["vibro", "zabe"]
        elif args.project:
            projects = [args.project]
        else:
            print("[ERROR] Please specify --project <name> or --all", file=sys.stderr)
            sys.exit(1)

        for proj in projects:
            s_builder = SingleFileCacheBuilder(
                project=proj,
                base_dir="data",
                batch_size=args.batch_size,
                num_workers=args.workers,
                force=args.force,
                dry_run=args.dry_run or args.validate,
                overview_tsv=args.overview_tsv,
            )
            summary = s_builder.build()
            if not summary:
                sys.exit(1)

    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
