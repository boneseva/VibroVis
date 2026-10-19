"""
cache/reader.py — Phase 1.16 Partitioned & Legacy Cache Reader Adapter

Provides unified reader abstractions:
    - PartitionedCacheReader: Reads partition files indexed by manifest.json
    - LegacyCacheReader: Reads legacy monolithic final_data.parquet
    - CacheReaderAdapter: Dispatches between them, preferring valid partitioned
      caches and explicitly falling back to legacy final_data.parquet.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq

import hierarchy
import read_data


logger = logging.getLogger(__name__)
class CacheIntegrityError(Exception):
    """Raised when a partition listed in manifest.json is corrupted or missing on disk."""
    pass


REQUIRED_PARTITION_COLUMNS = {"x", "y", "model_name", "channel", "clip_time"}


@dataclass
class ProjectMetadata:
    """Lightweight metadata descriptor for a project's cache availability."""
    project_id: str
    source: str  # "partitioned" or "legacy"
    locations: List[str]
    models_by_location: Dict[str, List[str]]
    partitions: List[Dict[str, Any]] = field(default_factory=list)
    manifest_version: Optional[str] = None
    builder_version: Optional[str] = None
    total_rows: int = 0
    total_partitions: int = 0
    date_extents: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)
    has_partitioned_cache: bool = False
    has_legacy_cache: bool = False


def normalize_posix(path: str) -> str:
    """Normalize filesystem path to POSIX forward-slash format."""
    return os.path.normpath(str(path)).replace("\\", "/")


class BaseCacheReader(ABC):
    """Abstract base class defining the cache reader contract."""

    @abstractmethod
    def get_project_metadata(self, project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> Optional[ProjectMetadata]:
        """Retrieve project cache metadata without loading full DataFrames."""
        pass

    @abstractmethod
    def load_location_model_data(
        self,
        project_id: str,
        location_id: str,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> pd.DataFrame:
        """Load DataFrame for given project, location, and model."""
        pass


class PartitionedCacheReader(BaseCacheReader):
    """
    Reader for the Phase 1.15 partitioned cache:
        data/<project>/cache/manifest.json
        data/<project>/cache/partitions/<location>/<model>.parquet
    """

    def __init__(self):
        # In-memory manifest cache: path -> (mtime, manifest_dict)
        self._manifest_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get_manifest_path(self, project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> str:
        """Return the manifest.json path for a project."""
        return os.path.join(base_dir, project_id, "cache", "manifest.json")

    def is_manifest_present(self, project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> bool:
        """Check if manifest.json exists on disk (regardless of whether it parses)."""
        return os.path.isfile(self.get_manifest_path(project_id, base_dir))

    def read_manifest(self, project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> Optional[Dict[str, Any]]:
        """
        Reads and parses manifest.json for a project.
        Uses mtime-aware in-memory caching for sub-millisecond repeated lookups.
        Returns parsed dict if valid, or None if missing or malformed.
        """
        m_path = self.get_manifest_path(project_id, base_dir)
        if not os.path.isfile(m_path):
            return None

        try:
            mtime = os.path.getmtime(m_path)
            cached = self._manifest_cache.get(m_path)
            if cached is not None and cached[0] == mtime:
                return cached[1]

            with open(m_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("[cache] Malformed manifest at %s (root is not dict)", m_path)
                return None

            # Verify minimal manifest schema
            if "manifest_version" not in data or "partitions" not in data:
                logger.warning("[cache] Malformed manifest at %s (missing required schema fields)", m_path)
                return None

            self._manifest_cache[m_path] = (mtime, data)
            return data

        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[cache] Error reading manifest at %s: %s", m_path, e)
            return None

    def is_available(self, project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> bool:
        """Check if a valid partitioned cache manifest exists for the project."""
        manifest = self.read_manifest(project_id, base_dir)
        return manifest is not None and len(manifest.get("partitions", [])) > 0

    def get_project_metadata(self, project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> Optional[ProjectMetadata]:
        """
        Construct ProjectMetadata purely from manifest.json without scanning partition files.
        """
        manifest = self.read_manifest(project_id, base_dir)
        if manifest is None:
            return None

        locations = sorted(list(manifest.get("locations", [])))
        raw_models = manifest.get("models_by_location", {})
        models_by_loc = {k: sorted(list(v)) for k, v in raw_models.items()}
        partitions = manifest.get("partitions", [])

        date_extents = {}
        for p in partitions:
            loc = p.get("location")
            mod = p.get("model")
            ext = p.get("date_extent")
            if loc and mod and ext and isinstance(ext, (list, tuple)) and len(ext) >= 2:
                date_extents[f"{loc}::{mod}"] = (ext[0], ext[1])

        return ProjectMetadata(
            project_id=manifest.get("project", project_id),
            source="partitioned",
            locations=locations,
            models_by_location=models_by_loc,
            partitions=partitions,
            manifest_version=manifest.get("manifest_version"),
            builder_version=manifest.get("builder_version"),
            total_rows=manifest.get("total_rows", 0),
            total_partitions=manifest.get("total_partitions", len(partitions)),
            date_extents=date_extents,
            has_partitioned_cache=True,
            has_legacy_cache=False,
        )

    def find_partition_entry(
        self,
        project_id: str,
        location_id: str,
        model_name: str,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> Optional[Dict[str, Any]]:
        """
        Find partition entry in manifest.json.
        Matches location_id exactly or via folder prefix/clean location matching.
        """
        manifest = self.read_manifest(project_id, base_dir)
        if not manifest:
            return None

        partitions = manifest.get("partitions", [])
        # 1. Exact match on location and model
        for p in partitions:
            if p.get("location") == location_id and p.get("model") == model_name:
                return p

        # 2. Case-insensitive exact match
        for p in partitions:
            if (p.get("location", "").lower() == location_id.lower() and
                    p.get("model", "").lower() == model_name.lower()):
                return p

        # 3. Match using selected_location (e.g. location_id='Bistra_spring', selected_location='Bistra')
        if selected_location:
            for p in partitions:
                if (p.get("location") == selected_location or p.get("location", "").startswith(selected_location)) and p.get("model") == model_name:
                    return p

        # 4. Prefix match (e.g. location_id was 'Bistra', and manifest partition location is 'Bistra_spring')
        prefix_matches = [
            p for p in partitions
            if (p.get("location", "").startswith(location_id) or location_id.startswith(p.get("location", "")))
            and p.get("model") == model_name
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        return None

    def find_partition_entries(
        self,
        project_id: str,
        location_id: Optional[str] = None,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> List[Dict[str, Any]]:
        """
        Find all partition entries matching the criteria.
        If location_id is provided, returns exact/prefix match for that location.
        If location_id is None/empty, returns all partitions matching model_name
        (and optionally selected_location if matched).
        """
        manifest = self.read_manifest(project_id, base_dir)
        if not manifest:
            return []

        partitions = manifest.get("partitions", [])
        if not partitions:
            return []

        if location_id:
            entry = self.find_partition_entry(project_id, location_id, model_name or "", selected_location, base_dir)
            return [entry] if entry else []

        if model_name:
            matched = [p for p in partitions if p.get("model") == model_name or p.get("model", "").lower() == model_name.lower()]
            if selected_location:
                loc_matched = [
                    p for p in matched
                    if p.get("location") == selected_location
                    or p.get("location", "").startswith(selected_location)
                    or selected_location.startswith(p.get("location", ""))
                ]
                if loc_matched:
                    return loc_matched
            return matched

        return partitions

    def has_partition_entry(
        self,
        project_id: str,
        location_id: str,
        model_name: str,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> bool:
        """Checks whether a partition is indexed in manifest.json."""
        return self.find_partition_entry(project_id, location_id, model_name, selected_location, base_dir) is not None

    def has_partition(
        self,
        project_id: str,
        location_id: str,
        model_name: str,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> bool:
        """
        Checks whether a partition is listed in manifest.json AND physically exists on disk.
        """
        entry = self.find_partition_entry(project_id, location_id, model_name, selected_location, base_dir)
        if not entry:
            return False

        rel_path = entry.get("partition_rel_path")
        if not rel_path:
            return False

        # Support POSIX paths on Windows
        disk_path = os.path.join(base_dir, project_id, *rel_path.split("/"))
        return os.path.isfile(disk_path)

    def load_location_model_data(
        self,
        project_id: str,
        location_id: str,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Loads the exact partition:
            <project>/cache/partitions/<location>/<model>.parquet
        Applies standard process_loaded_model_dataframe transformations.
        Raises CacheIntegrityError if manifest entry exists but file is missing or corrupt.
        """
        if not model_name:
            return pd.DataFrame()

        entry = self.find_partition_entry(project_id, location_id, model_name, selected_location, base_dir)
        if not entry:
            # Check direct filesystem convention fallback:
            cand_path = os.path.join(base_dir, project_id, "cache", "partitions", location_id, f"{model_name}.parquet")
            if os.path.isfile(cand_path):
                partition_full = cand_path
            else:
                return pd.DataFrame()
        else:
            rel_path = entry.get("partition_rel_path", "")
            partition_full = os.path.join(base_dir, project_id, *rel_path.split("/"))

        if not os.path.isfile(partition_full):
            raise CacheIntegrityError(
                f"Partition listed in manifest for project='{project_id}', location='{location_id}', "
                f"model='{model_name}' but file is missing on disk at: {partition_full}"
            )

        try:
            model_df = pd.read_parquet(partition_full)
        except Exception as e:
            raise CacheIntegrityError(f"Failed to read parquet partition at {partition_full}: {e}") from e

        # Schema validation
        missing_cols = REQUIRED_PARTITION_COLUMNS - set(model_df.columns)
        if missing_cols:
            raise CacheIntegrityError(
                f"Partition at {partition_full} has incompatible schema: missing required columns {sorted(list(missing_cols))}"
            )

        # Metadata agreement validation (Case 8)
        if entry is not None and "row_count" in entry:
            manifest_row_count = int(entry["row_count"])
            actual_row_count = len(model_df)
            if manifest_row_count != actual_row_count:
                msg = (f"Manifest metadata mismatch for {partition_full}: "
                       f"manifest row_count={manifest_row_count} != partition row_count={actual_row_count}")
                if strict:
                    raise CacheIntegrityError(msg)
                else:
                    logger.warning("[cache][metadata_mismatch] %s", msg)

        # If selected_location was provided and is present in 'location' column, filter if necessary
        if selected_location and "location" in model_df.columns:
            if (model_df["location"] == selected_location).any():
                model_df = model_df[model_df["location"] == selected_location]

        # Standard post-processing (day_dt_str, date_time_str, cluster_id_str, clip_count, categories)
        return read_data.process_loaded_model_dataframe(model_df)

    def load_project_model_data(
        self,
        project_id: str,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Loads all matching partitions for a project and model (e.g. across multiple source folders like Bistra_spring and Bistra_summer).
        Applies standard process_loaded_model_dataframe transformations.
        """
        if not model_name:
            return pd.DataFrame()

        entries = self.find_partition_entries(
            project_id=project_id,
            location_id=None,
            model_name=model_name,
            selected_location=selected_location,
            base_dir=base_dir,
        )
        if not entries:
            return pd.DataFrame()

        dfs = []
        for entry in entries:
            rel_path = entry.get("partition_rel_path", "")
            partition_full = os.path.join(base_dir, project_id, *rel_path.split("/"))
            if not os.path.isfile(partition_full):
                raise CacheIntegrityError(
                    f"Partition listed in manifest for project='{project_id}', "
                    f"model='{model_name}' but file is missing on disk at: {partition_full}"
                )
            try:
                part_df = pd.read_parquet(partition_full)
            except Exception as e:
                raise CacheIntegrityError(f"Failed to read parquet partition at {partition_full}: {e}") from e

            missing_cols = REQUIRED_PARTITION_COLUMNS - set(part_df.columns)
            if missing_cols:
                raise CacheIntegrityError(
                    f"Partition at {partition_full} has incompatible schema: missing required columns {sorted(list(missing_cols))}"
                )
            dfs.append(part_df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if selected_location and "location" in combined.columns:
            if (combined["location"] == selected_location).any():
                combined = combined[combined["location"] == selected_location]

        return read_data.process_loaded_model_dataframe(combined)



class LegacyCacheReader(BaseCacheReader):
    """
    Reader for the legacy monolithic cache:
        data/<project>/cache/final_data.parquet
    """

    def get_legacy_parquet_path(
        self,
        project_id: str,
        location_id: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> Optional[str]:
        """Resolves the legacy final_data.parquet file path."""
        candidates = [
            os.path.join(base_dir, project_id, "cache", "final_data.parquet"),
            os.path.join(base_dir, project_id, "final_data.parquet"),
            os.path.join(base_dir, "cache", "partitions", project_id, "perchv2.parquet"),
            os.path.join(base_dir, "cache", "partitions", project_id.capitalize(), "perchv2.parquet"),
            os.path.join(base_dir, "cache", "partitions", project_id.lower(), "perchv2.parquet"),
            os.path.join(base_dir, "cache", f"{project_id}.parquet"),
            os.path.join(base_dir, "cache", f"{project_id.lower()}.parquet"),
        ]
        if location_id:
            candidates.extend([
                os.path.join(base_dir, project_id, location_id, "cache", "final_data.parquet"),
                os.path.join(base_dir, project_id, location_id, "final_data.parquet"),
                os.path.join(base_dir, project_id, "recordings", location_id, "cache", "final_data.parquet"),
                os.path.join(base_dir, project_id, "mp3", location_id, "cache", "final_data.parquet"),
            ])

        for cand in candidates:
            if os.path.isfile(cand):
                return cand

        # Fallback via hierarchy path discovery
        if location_id:
            p_info = hierarchy.get_location_paths(project_id, location_id, base_dir=base_dir)
            if p_info.get("parquet") and os.path.isfile(p_info["parquet"]):
                return p_info["parquet"]

        return None

    def is_available(
        self,
        project_id: str,
        location_id: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> bool:
        """Check if legacy final_data.parquet is available."""
        p = self.get_legacy_parquet_path(project_id, location_id, base_dir)
        return p is not None and os.path.isfile(p)

    def get_project_metadata(
        self,
        project_id: str,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> Optional[ProjectMetadata]:
        """
        Extracts locations and models by reading metadata columns from legacy parquet.
        """
        p_path = self.get_legacy_parquet_path(project_id, base_dir=base_dir)
        if not p_path:
            return None

        try:
            schema = pq.read_schema(p_path)
            avail = set(schema.names)
            cols = [c for c in ["location", "model_name"] if c in avail]
            if not cols:
                return None

            df = pd.read_parquet(p_path, columns=cols)
            locations = sorted(list(df["location"].dropna().unique())) if "location" in df.columns else []
            models_by_loc = {}
            if "location" in df.columns and "model_name" in df.columns:
                for loc in locations:
                    models_by_loc[loc] = sorted(list(df[df["location"] == loc]["model_name"].dropna().unique()))
            elif "model_name" in df.columns:
                models_by_loc["default"] = sorted(list(df["model_name"].dropna().unique()))

            return ProjectMetadata(
                project_id=project_id,
                source="legacy",
                locations=locations,
                models_by_location=models_by_loc,
                total_rows=len(df),
                has_partitioned_cache=False,
                has_legacy_cache=True,
            )
        except Exception as e:
            logger.warning("[cache] Error reading legacy metadata from %s: %s", p_path, e)
            return None

    def load_location_model_data(
        self,
        project_id: str,
        location_id: str,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> pd.DataFrame:
        """
        Loads model/location filtered data from the legacy monolithic Parquet file.
        Correctly distinguishes between:
        - Physical location: location_id (e.g. 'Bistra_spring' or 'Cerknica')
        - Metadata location: selected_location (e.g. 'Bistra')
        """
        parquet_path = self.get_legacy_parquet_path(project_id, location_id, base_dir=base_dir)
        if not parquet_path or not os.path.exists(parquet_path):
            return pd.DataFrame()

        # Determine location filter:
        eff_location = selected_location

        filters = []
        if model_name is not None and eff_location is not None:
            filters = [[('model_name', '==', model_name), ('location', '==', eff_location)]]
        elif model_name is not None and location_id:
            filters = [[('model_name', '==', model_name), ('location', '==', location_id)]]
        elif model_name is not None:
            filters = [[('model_name', '==', model_name)]]
        elif eff_location is not None:
            filters = [[('location', '==', eff_location)]]
        elif location_id:
            filters = [[('location', '==', location_id)]]
        else:
            filters = None

        try:
            schema = pq.read_schema(parquet_path)
            available_cols = set(schema.names)
            cols_to_request = [c for c in read_data.FULL_DATA_COLUMNS if c in available_cols]
            if not cols_to_request:
                return pd.DataFrame()

            model_df = pd.read_parquet(
                parquet_path,
                filters=filters,
                columns=cols_to_request
            )

            # Fallback 1: if no rows matched location_id (e.g. location_id was a folder name like
            # Bistra_spring or ds, rather than a value in the 'location' column):
            if model_df.empty and (eff_location is None and location_id):
                fallback_filters = [[('model_name', '==', model_name)]] if model_name else None
                fallback_df = pd.read_parquet(
                    parquet_path,
                    filters=fallback_filters,
                    columns=cols_to_request
                )
                if not fallback_df.empty:
                    if 'wav_file' in fallback_df.columns:
                        sub = fallback_df[fallback_df['wav_file'].astype(str).str.startswith(location_id)]
                        if not sub.empty:
                            model_df = sub
                        else:
                            model_df = fallback_df
                    else:
                        model_df = fallback_df

            # Fallback 2: if eff_location was provided and matched a broader site (e.g. Bistra),
            # while location_id is a specific folder prefix (e.g. Bistra_spring):
            if not model_df.empty and location_id and 'wav_file' in model_df.columns:
                folder_matches = model_df['wav_file'].astype(str).str.startswith(location_id)
                if folder_matches.any() and not folder_matches.all():
                    model_df = model_df[folder_matches]

            return read_data.process_loaded_model_dataframe(model_df)

        except Exception as e:
            logger.warning("[cache] Legacy load error: %s", e)
            return pd.DataFrame()

    def load_project_model_data(
        self,
        project_id: str,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> pd.DataFrame:
        """Loads model filtered data for a project from legacy cache."""
        return self.load_location_model_data(
            project_id=project_id,
            location_id="",
            model_name=model_name,
            selected_location=selected_location,
            base_dir=base_dir,
        )



class CacheReaderAdapter(BaseCacheReader):
    """
    Unified Dispatcher and Adapter:
    - Prefers the new partitioned cache when valid and available.
    - Explicitly and deterministically falls back to legacy final_data.parquet.
    - Preserves single-source isolation per (Project, Location, Model) selection.
    - Tracks and logs which backend was used ('partitioned' vs 'legacy').
    - Supports strict_mode for testing and validation to prevent concealing stale/broken data.
    """

    def __init__(self, strict_mode: bool = False):
        self.partitioned_reader = PartitionedCacheReader()
        self.legacy_reader = LegacyCacheReader()
        self.strict_mode = strict_mode
        self.last_source_used: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None

    def get_last_source_used(self) -> Optional[str]:
        """Returns the source used by the most recent load operation ('partitioned', 'legacy', or None)."""
        return self.last_source_used

    def get_project_metadata(
        self,
        project_id: str,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    ) -> Optional[ProjectMetadata]:
        """
        Discovers project metadata. Prefers partitioned cache manifest when valid.
        Populates has_partitioned_cache and has_legacy_cache flags.
        """
        has_part = self.partitioned_reader.is_available(project_id, base_dir)
        has_leg = self.legacy_reader.is_available(project_id, None, base_dir)

        if has_part:
            meta = self.partitioned_reader.get_project_metadata(project_id, base_dir)
            if meta is not None:
                meta.has_partitioned_cache = True
                meta.has_legacy_cache = has_leg
                return meta

        if has_leg:
            meta = self.legacy_reader.get_project_metadata(project_id, base_dir)
            if meta is not None:
                meta.has_partitioned_cache = False
                meta.has_legacy_cache = True
                return meta

        return None

    def load_location_model_data(
        self,
        project_id: str,
        location_id: Optional[str] = None,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
        dataset_id: Optional[str] = None,
        selected_model: Optional[str] = None,
        strict: Optional[bool] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load location and model data using single-source deterministic selection:
        Handles all 8 fallback and cache-integrity cases with explicit logging and deterministic isolation:
        1. Valid manifest and valid partition -> loads partition (source=partitioned).
        2. Missing manifest with valid legacy -> loads legacy (source=legacy).
        3. Malformed manifest with valid legacy -> in strict mode raises error, else falls back to legacy.
        4. Manifest entry missing -> in strict mode raises error, else falls back to legacy.
        5. Manifest references missing partition -> in strict mode raises error, else falls back to legacy with warning.
        6. Partition corrupt / incompatible schema -> in strict mode raises error, else falls back to legacy with warning.
        7. Neither backend available -> in strict mode raises KeyError, else returns empty DataFrame (source=none).
        8. Manifest and partition metadata disagree -> in strict mode raises error, else logs warning and returns partition.
        """
        eff_loc = location_id if location_id is not None else dataset_id
        eff_model = model_name if model_name is not None else selected_model
        if not eff_loc:
            return self.load_project_model_data(
                project_id=project_id,
                model_name=eff_model,
                selected_location=selected_location,
                base_dir=base_dir,
                strict=strict,
                **kwargs,
            )
        location_id = eff_loc or ""
        model_name = eff_model
        is_strict = self.strict_mode if strict is None else strict

        manifest_present = self.partitioned_reader.is_manifest_present(project_id, base_dir)
        manifest_valid = self.partitioned_reader.is_available(project_id, base_dir)

        # Case 3: Manifest exists on disk but is malformed
        if manifest_present and not manifest_valid:
            if is_strict:
                raise CacheIntegrityError(
                    f"Malformed manifest at {self.partitioned_reader.get_manifest_path(project_id, base_dir)}"
                )
            logger.warning("[cache] Malformed partitioned cache manifest for project '%s'.", project_id)

        # Step 1: Check partitioned cache
        if model_name and manifest_valid:
            has_entry = self.partitioned_reader.has_partition_entry(
                project_id=project_id,
                location_id=location_id,
                model_name=model_name,
                selected_location=selected_location,
                base_dir=base_dir,
            )

            if not has_entry:
                # Case 4: Partition not in manifest
                if is_strict:
                    raise CacheIntegrityError(
                        f"Partition not listed in manifest for project='{project_id}', location='{location_id}', model='{model_name}'"
                    )
            else:
                # Partition entry IS in manifest. Attempt to load partition file.
                try:
                    df = self.partitioned_reader.load_location_model_data(
                        project_id=project_id,
                        location_id=location_id,
                        model_name=model_name,
                        selected_location=selected_location,
                        base_dir=base_dir,
                        strict=is_strict,
                    )
                    if df is not None and not df.empty:
                        self.last_source_used = "partitioned"
                        self.last_fallback_reason = None
                        print(f"[cache] project={project_id} location={location_id} model={model_name} source=partitioned")
                        return df
                except CacheIntegrityError as cie:
                    # Cases 5, 6, 8 (integrity failures)
                    if is_strict:
                        raise
                    print(f"[cache][integrity] {cie}. Falling back to legacy cache.")

        # Step 2: Fallback to legacy final_data.parquet
        if self.legacy_reader.is_available(project_id, location_id, base_dir):
            df = self.legacy_reader.load_location_model_data(
                project_id=project_id,
                location_id=location_id,
                model_name=model_name,
                selected_location=selected_location,
                base_dir=base_dir,
            )
            if df is not None and not df.empty:
                self.last_source_used = "legacy"
                if not manifest_present:
                    reason = "no partitioned cache"
                elif not manifest_valid:
                    reason = "malformed partitioned cache manifest"
                elif model_name and not self.partitioned_reader.has_partition_entry(project_id, location_id, model_name, selected_location, base_dir):
                    reason = "partition not in partitioned cache"
                else:
                    reason = "integrity fallback: partition missing or corrupt"
                self.last_fallback_reason = reason
                print(f"[cache] project={project_id} location={location_id} model={model_name} source=legacy ({reason})")
                return df

        # Step 3: Neither available (Case 7)
        if is_strict:
            raise KeyError(
                f"No cache backend could serve project='{project_id}', location='{location_id}', model='{model_name}'"
            )

        self.last_source_used = None
        self.last_fallback_reason = "not found in any cache"
        print(f"[cache] project={project_id} location={location_id} model={model_name} source=none")
        return pd.DataFrame()

    def load_project_model_data(
        self,
        project_id: str,
        model_name: Optional[str] = None,
        selected_location: Optional[str] = None,
        base_dir: str = hierarchy.DEFAULT_DATA_DIR,
        strict: Optional[bool] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Loads project-scoped model data across partitions or falls back to legacy cache.
        """
        is_strict = self.strict_mode if strict is None else strict

        manifest_present = self.partitioned_reader.is_manifest_present(project_id, base_dir)
        manifest_valid = self.partitioned_reader.is_available(project_id, base_dir)

        if manifest_present and not manifest_valid:
            if is_strict:
                raise CacheIntegrityError(
                    f"Malformed manifest at {self.partitioned_reader.get_manifest_path(project_id, base_dir)}"
                )
            logger.warning("[cache] Malformed partitioned cache manifest for project '%s'.", project_id)

        # Step 1: Check partitioned cache
        if model_name and manifest_valid:
            entries = self.partitioned_reader.find_partition_entries(
                project_id=project_id,
                location_id=None,
                model_name=model_name,
                selected_location=selected_location,
                base_dir=base_dir,
            )
            if entries:
                try:
                    df = self.partitioned_reader.load_project_model_data(
                        project_id=project_id,
                        model_name=model_name,
                        selected_location=selected_location,
                        base_dir=base_dir,
                        strict=is_strict,
                    )
                    if df is not None and not df.empty:
                        self.last_source_used = "partitioned"
                        self.last_fallback_reason = None
                        print(f"[cache] project={project_id} model={model_name} source=partitioned")
                        return df
                except CacheIntegrityError as cie:
                    if is_strict:
                        raise
                    print(f"[cache][integrity] {cie}. Falling back to legacy cache.")

        # Step 2: Fallback to legacy final_data.parquet
        if self.legacy_reader.is_available(project_id, location_id="", base_dir=base_dir):
            df = self.legacy_reader.load_project_model_data(
                project_id=project_id,
                model_name=model_name,
                selected_location=selected_location,
                base_dir=base_dir,
            )
            if df is not None and not df.empty:
                self.last_source_used = "legacy"
                self.last_fallback_reason = "legacy fallback"
                print(f"[cache] project={project_id} model={model_name} source=legacy")
                return df

        if is_strict:
            raise KeyError(
                f"No cache backend could serve project='{project_id}', model='{model_name}'"
            )

        self.last_source_used = None
        self.last_fallback_reason = "not found in any cache"
        print(f"[cache] project={project_id} model={model_name} source=none")
        return pd.DataFrame()


# Global singleton adapter instance
_DEFAULT_ADAPTER = CacheReaderAdapter()


def get_default_adapter() -> CacheReaderAdapter:
    """Return the shared CacheReaderAdapter singleton."""
    return _DEFAULT_ADAPTER


def get_project_metadata(project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> Optional[ProjectMetadata]:
    """Module-level convenience API: get project metadata."""
    return _DEFAULT_ADAPTER.get_project_metadata(project_id, base_dir=base_dir)


def load_location_model_data(
    project_id: str,
    location_id: Optional[str] = None,
    model_name: Optional[str] = None,
    selected_location: Optional[str] = None,
    base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    dataset_id: Optional[str] = None,
    selected_model: Optional[str] = None,
    strict: Optional[bool] = None,
    **kwargs,
) -> pd.DataFrame:
    """Module-level convenience API: load location model data."""
    return _DEFAULT_ADAPTER.load_location_model_data(
        project_id=project_id,
        location_id=location_id,
        model_name=model_name,
        selected_location=selected_location,
        base_dir=base_dir,
        dataset_id=dataset_id,
        selected_model=selected_model,
        strict=strict,
        **kwargs,
    )


def load_project_model_data(
    project_id: str,
    model_name: Optional[str] = None,
    selected_location: Optional[str] = None,
    base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    strict: Optional[bool] = None,
    **kwargs,
) -> pd.DataFrame:
    """Module-level convenience API: load project model data."""
    return _DEFAULT_ADAPTER.load_project_model_data(
        project_id=project_id,
        model_name=model_name,
        selected_location=selected_location,
        base_dir=base_dir,
        strict=strict,
        **kwargs,
    )


def get_last_source_used() -> Optional[str]:
    """Module-level convenience API: get source used by last load operation."""
    return _DEFAULT_ADAPTER.get_last_source_used()

