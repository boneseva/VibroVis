"""
VibroVis — Historical Manual Labels Migration Utility

Idempotently migrates historical manual labels from `data/vibro/cache/saved_labels.parquet`
into the active runtime label storage (`MANUAL_LABELS_CACHE` and `CLIP_DURATION_CACHE`).

Safety guarantees:
1. Resolves stable physical clip identities: (location, microlocation, file_basename, channel, round(clip_time * 10)).
2. Checks for multi-model label concordance (zero conflicts among historical duplicates).
3. Detects recording_id ambiguity and skips any ambiguous entries.
4. Never overwrites conflicting runtime labels.
5. Idempotent: running repeatedly produces no duplicates or changes.
6. Creates timestamped backups before applying writes to production storage.
7. Defaults to --dry-run mode.
"""
import os
import sys
import shutil
import time
import argparse
import datetime
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import diskcache as dc

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def create_cache_backup(source_dir: str, backup_base_dir: str = "cache/backups") -> Optional[str]:
    """Creates a timestamped copy of a diskcache directory."""
    if not os.path.isdir(source_dir):
        return None
    os.makedirs(backup_base_dir, exist_ok=True)
    dir_name = os.path.basename(os.path.normpath(source_dir))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_base_dir, f"{dir_name}_{timestamp}")
    shutil.copytree(source_dir, backup_path)
    return backup_path


def migrate_historical_labels(
    saved_labels_parquet_path: str = "data/vibro/cache/saved_labels.parquet",
    final_data_parquet_path: str = "data/vibro/cache/final_data.parquet",
    manual_labels_cache=None,
    clip_duration_cache=None,
    dry_run: bool = True,
    create_backups: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Reads saved_labels.parquet and maps entries into active runtime storage.
    """
    if not os.path.isfile(saved_labels_parquet_path):
        raise FileNotFoundError(f"Saved labels parquet not found at: {saved_labels_parquet_path}")
    if not os.path.isfile(final_data_parquet_path):
        raise FileNotFoundError(f"Final data parquet not found at: {final_data_parquet_path}")

    # 1. Load historical label mappings and active dataset
    saved_df = pd.read_parquet(saved_labels_parquet_path)
    if "row_idx" not in saved_df.columns or "label" not in saved_df.columns:
        raise ValueError("saved_labels.parquet must contain 'row_idx' and 'label' columns")

    final_cols = [
        "row_idx", "file_name", "mp3_file", "channel", "clip_time",
        "clip_duration", "model_name", "location", "microlocation", "recording_id"
    ]
    final_df = pd.read_parquet(final_data_parquet_path, columns=final_cols)

    total_historical = len(saved_df)
    if verbose:
        print(f"[MIGRATION] Loaded {total_historical} historical label records from {saved_labels_parquet_path}")

    # Check row_idx resolution in current final_data.parquet
    missing_idx = saved_df[~saved_df["row_idx"].isin(final_df.index)]
    if not missing_idx.empty:
        raise RuntimeError(f"Corrupt mapping: {len(missing_idx)} row_idx values not in {final_data_parquet_path}")

    merged = saved_df.merge(final_df, on="row_idx", how="inner")

    # 2. Extract physical clip identities and validate
    merged["f_base"] = merged["mp3_file"].apply(
        lambda x: str(x).replace("\\", "/").split("/")[-1] if pd.notna(x) else ""
    )
    fallback_mask = merged["f_base"] == ""
    if fallback_mask.any():
        merged.loc[fallback_mask, "f_base"] = merged.loc[fallback_mask, "file_name"].apply(
            lambda x: str(x).replace("\\", "/").split("/")[-1]
        )

    merged["ct_key"] = (merged["clip_time"].astype(float) * 10).round().astype(int)

    # 3. Group by physical key: (loc, micro, f_base, chan, ct_key)
    physical_groups: Dict[Tuple[str, str, str, int, int], List[Dict[str, Any]]] = {}
    for _, row in merged.iterrows():
        loc = sys.intern(str(row["location"]))
        micro = sys.intern(str(row["microlocation"]))
        f_base = sys.intern(str(row["f_base"]))
        chan = int(row["channel"])
        ct_key = int(row["ct_key"])
        key = (loc, micro, f_base, chan, ct_key)

        entry = {
            "row_idx": int(row["row_idx"]),
            "label": str(row["label"]),
            "clip_time": float(row["clip_time"]),
            "clip_duration": float(row["clip_duration"]),
            "model_name": str(row["model_name"]),
            "recording_id": str(row["recording_id"]) if pd.notna(row["recording_id"]) else "",
        }
        physical_groups.setdefault(key, []).append(entry)

    # 4. Resolve caches
    if manual_labels_cache is None:
        from callbacks.callbacks_constants import MANUAL_LABELS_CACHE
        manual_labels_cache = MANUAL_LABELS_CACHE

    if clip_duration_cache is None:
        from callbacks.callbacks_constants import CLIP_DURATION_CACHE
        clip_duration_cache = CLIP_DURATION_CACHE

    # 5. Evaluate mappings, concordance, ambiguities, and runtime conflicts
    to_import: List[Tuple[Tuple[str, str, str, int, int], str, float, str]] = []
    already_present: List[Tuple[Tuple, str]] = []
    runtime_conflicts: List[Tuple[Tuple, str, str]] = []
    ambiguous_labels: List[Tuple[Tuple, List[str]]] = []
    ambiguous_recordings: List[Tuple[Tuple, List[str]]] = []

    for key, entries in physical_groups.items():
        # Check label agreement across models
        labels_assigned = set(e["label"] for e in entries)
        if len(labels_assigned) > 1:
            ambiguous_labels.append((key, list(labels_assigned)))
            continue

        chosen_label = list(labels_assigned)[0]

        # Check recording_id ambiguity
        rec_ids = set(e["recording_id"] for e in entries if e["recording_id"])
        if len(rec_ids) > 1:
            ambiguous_recordings.append((key, list(rec_ids)))
            continue

        # Duration & model metadata (prefer longest duration / primary entry)
        best_entry = sorted(entries, key=lambda e: e["clip_duration"], reverse=True)[0]
        dur = best_entry["clip_duration"]
        model = best_entry["model_name"]

        # Check current runtime label store
        existing_label = manual_labels_cache.get(key)
        if existing_label is not None:
            if str(existing_label) == chosen_label:
                already_present.append((key, chosen_label))
            else:
                runtime_conflicts.append((key, chosen_label, str(existing_label)))
        else:
            to_import.append((key, chosen_label, dur, model))

    summary = {
        "total_historical_rows": total_historical,
        "unique_physical_clips": len(physical_groups),
        "to_import_count": len(to_import),
        "already_present_count": len(already_present),
        "runtime_conflicts_count": len(runtime_conflicts),
        "ambiguous_labels_count": len(ambiguous_labels),
        "ambiguous_recordings_count": len(ambiguous_recordings),
        "dry_run": dry_run,
        "backup_paths": [],
    }

    if verbose:
        print("\n" + "=" * 60)
        print(" historical labels migration audit / dry-run summary ".upper().center(60, "="))
        print(f"Total Historical Rows:        {total_historical}")
        print(f"Unique Physical Clips:        {len(physical_groups)}")
        print(f"Proposed for Import:          {len(to_import)}")
        print(f"Already Present (identical):  {len(already_present)}")
        print(f"Runtime Conflicts:            {len(runtime_conflicts)}")
        print(f"Ambiguous Labels:             {len(ambiguous_labels)}")
        print(f"Ambiguous Recordings:         {len(ambiguous_recordings)}")
        print("=" * 60)

        if to_import:
            label_dist = pd.Series([t[1] for t in to_import]).value_counts().to_dict()
            print(f"Proposed Import Distribution: {label_dist}")

        if runtime_conflicts:
            print("\n[WARNING] Runtime Conflicts detected (will NOT overwrite):")
            for k, hist_lbl, run_lbl in runtime_conflicts[:5]:
                print(f"  Key {k}: Historical '{hist_lbl}' vs Runtime '{run_lbl}'")

        if ambiguous_labels:
            print("\n[WARNING] Multi-model label discordance (quarantined):")
            for k, lbls in ambiguous_labels:
                print(f"  Key {k}: conflicting labels {lbls}")

    # 6. Apply migration if not dry_run
    if not dry_run:
        if create_backups:
            from callbacks.callbacks_constants import cache_dir, dur_cache_dir
            b1 = create_cache_backup(cache_dir)
            b2 = create_cache_backup(dur_cache_dir)
            if b1: summary["backup_paths"].append(b1)
            if b2: summary["backup_paths"].append(b2)
            if verbose:
                print(f"\n[BACKUP] Created backups prior to write:")
                if b1: print(f"  - {b1}")
                if b2: print(f"  - {b2}")

        imported_count = 0
        from callbacks.callbacks_constants import MANUAL_LABELS_LOCK
        with MANUAL_LABELS_LOCK:
            for key, label, dur, model in to_import:
                manual_labels_cache[key] = label
                clip_duration_cache[key] = (float(dur), str(model))
                imported_count += 1

        summary["imported_count"] = imported_count
        if verbose:
            print(f"\n[SUCCESS] Successfully imported {imported_count} labels into runtime storage.")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Migrate historical labels to runtime cache")
    parser.add_argument("--execute", action="store_true", help="Execute write (default is dry-run)")
    parser.add_argument("--no-backup", action="store_true", help="Disable automatic backups")
    args = parser.parse_args()

    is_dry_run = not args.execute
    migrate_historical_labels(
        dry_run=is_dry_run,
        create_backups=not args.no_backup,
        verbose=True
    )


if __name__ == "__main__":
    main()
