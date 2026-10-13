"""
Filesystem hierarchy discovery and abstraction layer for VibroVis.

Production structure:
    data/
        <project>/
            recordings/
                ...
            cache/
                manifest.json
                partitions/
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


DEFAULT_DATA_DIR = "data"
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg')

IGNORED_DIR_NAMES = {
    "__pycache__",
    "cache",
    "spectrogram_cache",
    "positions",
    "mp3",
    "recordings",
    "temp",
    "scratch",
    "tmp",
    ".git",
    ".venv",
    ".idea",
}


@dataclass(frozen=True)
class Project:
    """Lightweight metadata representation of a Project."""
    id: str
    name: str
    path: str


@dataclass(frozen=True)
class Location:
    """Lightweight metadata representation of a Location within a Project."""
    id: str
    name: str
    project_id: str
    path: str
    has_microlocations: bool = False


@dataclass(frozen=True)
class Microlocation:
    """Lightweight metadata representation of an optional Microlocation within a Location."""
    id: str
    name: str
    location_id: str
    project_id: str
    path: str


# Backward-compatibility alias for Dataset -> Location during transition
Dataset = Location


def _read_manifest_name(directory_path: str, manifest_filename: str, fallback_name: str) -> str:
    """
    Reads an optional JSON manifest to extract a human-readable display name.
    Falls back to `fallback_name` if absent, malformed, or missing 'name'.
    """
    manifest_path = os.path.join(directory_path, manifest_filename)
    if not os.path.isfile(manifest_path):
        return fallback_name

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            name = data.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    except Exception:
        pass

    return fallback_name


def _is_valid_dir(name: str) -> bool:
    """Check if a directory name is valid and not ignored."""
    if not name or name.startswith("."):
        return False
    if name in IGNORED_DIR_NAMES:
        return False
    return True


def get_available_projects(base_dir: str = DEFAULT_DATA_DIR) -> List[Project]:
    """
    Discovers all valid projects directly beneath base_dir.
    Deterministic, sorted by project ID.
    """
    if not os.path.isdir(base_dir):
        return []

    projects = []
    try:
        entries = sorted(os.listdir(base_dir))
    except OSError:
        return []

    for entry in entries:
        if not _is_valid_dir(entry):
            continue

        project_path = os.path.join(base_dir, entry)
        if not os.path.isdir(project_path):
            continue

        name = _read_manifest_name(project_path, "project.json", fallback_name=entry)
        projects.append(Project(id=entry, name=name, path=project_path))

    return projects


def get_project(project_id: str, base_dir: str = DEFAULT_DATA_DIR) -> Optional[Project]:
    """Lookup a specific project by ID."""
    for p in get_available_projects(base_dir=base_dir):
        if p.id == project_id:
            return p
    return None


def get_locations_for_project(project_id: str, base_dir: str = DEFAULT_DATA_DIR) -> List[Location]:
    """
    Discovers all valid locations for a project.
    Canonical rule:
        data/<project>/recordings/<location>/
    Locations are the immediate directories under <project>/recordings/.
    Falls back to <project>/mp3/ or direct project child directories for legacy/development compatibility.
    Deterministic, sorted by location ID.
    """
    project = get_project(project_id, base_dir=base_dir)
    if project is None or not os.path.isdir(project.path):
        return []

    recordings_root = os.path.join(project.path, "recordings")
    mp3_root = os.path.join(project.path, "mp3")

    # Prefer canonical recordings/ directory, fallback to mp3/
    if os.path.isdir(recordings_root):
        source_root = recordings_root
    elif os.path.isdir(mp3_root):
        source_root = mp3_root
    else:
        source_root = None

    locations = []

    if source_root is not None:
        try:
            entries = sorted(os.listdir(source_root))
        except OSError:
            entries = []

        for entry in entries:
            if not _is_valid_dir(entry):
                continue
            location_path = os.path.join(source_root, entry)
            if not os.path.isdir(location_path):
                continue

            # Determine whether location has child directories (microlocations / recording groups)
            has_micros = False
            try:
                sub_entries = os.listdir(location_path)
                for se in sub_entries:
                    if _is_valid_dir(se) and os.path.isdir(os.path.join(location_path, se)):
                        has_micros = True
                        break
            except OSError:
                pass

            name = _read_manifest_name(location_path, "location.json", fallback_name=entry)
            if name == entry:
                name = _read_manifest_name(location_path, "dataset.json", fallback_name=entry)

            locations.append(Location(
                id=entry,
                name=name,
                project_id=project_id,
                path=location_path,
                has_microlocations=has_micros
            ))
        return locations

    # Fallback for synthetic/legacy test structures where locations are direct children
    try:
        entries = sorted(os.listdir(project.path))
    except OSError:
        return []

    for entry in entries:
        if not _is_valid_dir(entry):
            continue

        location_path = os.path.join(project.path, entry)
        if not os.path.isdir(location_path):
            continue

        name = _read_manifest_name(location_path, "location.json", fallback_name=entry)
        if name == entry:
            name = _read_manifest_name(location_path, "dataset.json", fallback_name=entry)

        locations.append(Location(
            id=entry,
            name=name,
            project_id=project_id,
            path=location_path,
            has_microlocations=False
        ))

    return locations


def get_location(project_id: str, location_id: str, base_dir: str = DEFAULT_DATA_DIR) -> Optional[Location]:
    """Lookup a specific location by project_id and location_id."""
    for loc in get_locations_for_project(project_id, base_dir=base_dir):
        if loc.id == location_id:
            return loc
    return None


def get_microlocations(project_id: str, location_id: str, base_dir: str = DEFAULT_DATA_DIR) -> List[Microlocation]:
    """
    Discovers optional microlocations for a given location.
    
    Rule:
    1. If location directory contains child directories (excluding internal 'positions'),
       those child directories are Microlocations.
    2. If location directory contains only direct audio files (or internal 'positions' directory),
       no explicit microlocations exist (returns empty list).
    3. If location directory contains both direct audio files and child directories,
       raises ValueError for ambiguous structure.
    """
    loc = get_location(project_id, location_id, base_dir=base_dir)
    if loc is None or not os.path.isdir(loc.path):
        return []

    try:
        sub_items = sorted(os.listdir(loc.path))
    except OSError:
        return []

    audio_files = []
    child_dirs = []

    for item in sub_items:
        item_path = os.path.join(loc.path, item)
        if not _is_valid_dir(item):
            continue
        if os.path.isfile(item_path):
            if item.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                audio_files.append(item)
        elif os.path.isdir(item_path):
            child_dirs.append(item)

    # Ambiguity check: cannot have both direct audio files and subdirectories
    if audio_files and child_dirs:
        raise ValueError(
            f"Ambiguous location structure in '{loc.path}': "
            f"contains both {len(audio_files)} direct audio files and {len(child_dirs)} subdirectories."
        )

    microlocations = []
    for cd in child_dirs:
        cd_path = os.path.join(loc.path, cd)
        name = _read_manifest_name(cd_path, "microlocation.json", fallback_name=cd)
        microlocations.append(Microlocation(
            id=cd,
            name=name,
            location_id=location_id,
            project_id=project_id,
            path=cd_path
        ))

    return microlocations


def get_location_paths(
    project_id: str,
    location_id: str,
    microlocation_id: Optional[str] = None,
    base_dir: str = DEFAULT_DATA_DIR,
    allow_legacy_positions: bool = False
) -> Dict[str, Any]:
    """
    Resolves filesystem paths for a location and checks availability of key assets.
    Supports both single-level locations and nested microlocations.

    Returns:
        dict containing:
            - root: absolute or resolved path to location directory
            - parquet: path to final_data.parquet if present, else None
            - mp3: path to mp3 audio directory if present, else None
            - positions: path to positions directory if present, else None
            - metadata_tsv: path to metadata TSV if present, else None
            - has_parquet: bool
            - has_audio: bool
            - has_positions: bool
    """
    project_dir = os.path.join(base_dir, project_id)
    location_rec_dir = os.path.join(project_dir, "recordings", location_id)
    location_mp3_dir = os.path.join(project_dir, "mp3", location_id)
    legacy_location_dir = os.path.join(project_dir, location_id)

    if os.path.isdir(location_rec_dir):
        loc_root = location_rec_dir
    elif os.path.isdir(location_mp3_dir):
        loc_root = location_mp3_dir
    else:
        loc_root = legacy_location_dir

    target_audio_dir = os.path.join(loc_root, microlocation_id) if microlocation_id else loc_root

    # 1. Parquet resolution: checks project cache first, then project root, then legacy location cache
    parquet_path = None
    candidate_parquets = [
        os.path.join(project_dir, "cache", "final_data.parquet"),
        os.path.join(project_dir, "final_data.parquet"),
        os.path.join(legacy_location_dir, "cache", "final_data.parquet"),
        os.path.join(legacy_location_dir, "final_data.parquet"),
    ]
    for cand in candidate_parquets:
        if os.path.isfile(cand):
            parquet_path = cand
            break

    # 2. Audio directory resolution
    mp3_path = None
    candidate_audio_dirs = [
        os.path.join(loc_root, "recordings"),
        os.path.join(loc_root, "mp3"),
        os.path.join(loc_root, "audio"),
        target_audio_dir,
    ]
    for cand in candidate_audio_dirs:
        if os.path.isdir(cand):
            # Check if this dir or its children have audio files, or if it is an explicit audio folder
            if cand.endswith("mp3") or cand.endswith("audio") or cand.endswith("recordings"):
                mp3_path = cand
                break
            # Otherwise check if it has audio files or subdirectories with audio
            has_audio_files = False
            try:
                for root, _, files in os.walk(cand):
                    if any(f.lower().endswith(AUDIO_EXTENSIONS) for f in files):
                        has_audio_files = True
                        break
            except OSError:
                pass
            if has_audio_files:
                mp3_path = cand
                break
    if mp3_path is None and os.path.isdir(loc_root):
        # Fallback only if loc_root itself or subdirs have audio
        try:
            for root, _, files in os.walk(loc_root):
                if any(f.lower().endswith(AUDIO_EXTENSIONS) for f in files):
                    mp3_path = loc_root
                    break
        except OSError:
            pass

    # 3. Positions directory resolution
    # In the VM structure (recordings/), positions are located inside recording folders:
    #   <project>/recordings/<location>/[<subfolder>/]positions
    # In legacy/synthetic mp3 structures:
    #   <project>/positions/<location>/[<microlocation>] (project-level canonical)
    #   <project>/mp3/<location>/.../positions (legacy positions)
    positions_path = None
    legacy_positions_path = None
    project_pos_dir = os.path.join(project_dir, "positions")
    is_recordings_source = (location_rec_dir and os.path.isdir(location_rec_dir))

    canonical_pos_cand = None
    if is_recordings_source:
        # In VM structure under recordings/, positions is at the leaf level
        if microlocation_id:
            leaf_pos = os.path.join(target_audio_dir, "positions")
            if os.path.isdir(leaf_pos):
                canonical_pos_cand = leaf_pos
            else:
                try:
                    for root, dirs, _ in os.walk(target_audio_dir):
                        if 'positions' in dirs and os.path.isdir(os.path.join(root, 'positions')):
                            canonical_pos_cand = os.path.join(root, 'positions')
                            break
                except OSError:
                    pass
        else:
            leaf_pos = os.path.join(loc_root, "positions")
            if os.path.isdir(leaf_pos):
                canonical_pos_cand = leaf_pos
            else:
                try:
                    for root, dirs, _ in os.walk(loc_root):
                        if 'positions' in dirs and os.path.isdir(os.path.join(root, 'positions')):
                            canonical_pos_cand = os.path.join(root, 'positions')
                            break
                except OSError:
                    pass
        legacy_candidates = [
            os.path.join(project_pos_dir, location_id, microlocation_id) if microlocation_id else os.path.join(project_pos_dir, location_id),
            os.path.join(project_pos_dir, microlocation_id) if microlocation_id else None,
        ]
    else:
        # In mp3/ or legacy direct-location structure
        if microlocation_id:
            canonical_pos_cand = os.path.join(project_pos_dir, location_id, microlocation_id)
            legacy_candidates = [
                os.path.join(project_pos_dir, microlocation_id),
                os.path.join(target_audio_dir, "positions"),
                os.path.join(loc_root, "positions"),
            ]
        else:
            canonical_pos_cand = os.path.join(project_pos_dir, location_id)
            legacy_candidates = [
                os.path.join(loc_root, "positions"),
                os.path.join(legacy_location_dir, "positions"),
            ]

    # Check canonical path first
    if canonical_pos_cand and os.path.isdir(canonical_pos_cand):
        positions_path = canonical_pos_cand
    else:
        # Check if a legacy positions directory exists
        for l_cand in legacy_candidates:
            if l_cand and os.path.isdir(l_cand):
                legacy_positions_path = l_cand
                break
        if allow_legacy_positions and legacy_positions_path:
            positions_path = legacy_positions_path



    # 4. Metadata TSV resolution
    metadata_tsv = None
    for cand_dir in (legacy_location_dir, loc_root, project_dir):
        if not os.path.isdir(cand_dir):
            continue
        for candidate_tsv_name in ("location_meta.tsv", "dataset_meta.tsv", "overview.tsv"):
            cand_tsv = os.path.join(cand_dir, candidate_tsv_name)
            if os.path.isfile(cand_tsv):
                metadata_tsv = cand_tsv
                break
        if metadata_tsv is not None:
            break
        try:
            for f in sorted(os.listdir(cand_dir)):
                if f.endswith(".tsv") and os.path.isfile(os.path.join(cand_dir, f)):
                    metadata_tsv = os.path.join(cand_dir, f)
                    break
        except OSError:
            pass
        if metadata_tsv is not None:
            break

    return {
        "root": loc_root,
        "parquet": parquet_path,
        "mp3": mp3_path,
        "positions": positions_path,
        "canonical_positions": canonical_pos_cand if (canonical_pos_cand and os.path.isdir(canonical_pos_cand)) else None,
        "legacy_positions": legacy_positions_path,
        "metadata_tsv": metadata_tsv,
        "has_parquet": parquet_path is not None,
        "has_audio": mp3_path is not None,
        "has_positions": positions_path is not None,
    }


# Backward-compatibility aliases for Phase 1.10 - 1.13 code and tests
get_datasets_for_project = get_locations_for_project
get_dataset = get_location


def get_dataset_paths(project_id: str, dataset_id: str, microlocation_id: Optional[str] = None, base_dir: str = DEFAULT_DATA_DIR, allow_legacy_positions: bool = True) -> Dict[str, Any]:
    """Backward-compatibility shim for Phase 1.10 - 1.13 dataset path callers."""
    return get_location_paths(
        project_id=project_id,
        location_id=dataset_id,
        microlocation_id=microlocation_id,
        base_dir=base_dir,
        allow_legacy_positions=allow_legacy_positions
    )


def get_project_paths(project_id: str, base_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Resolves filesystem paths for a project root and its assets."""
    return get_location_paths(project_id, "", base_dir=base_dir, allow_legacy_positions=True)


