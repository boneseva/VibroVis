import pathlib
import re
import os
import ast
import pandas as pd
from tqdm import tqdm

# Hardcoded data directory as requested
BASE_DATA_DIR = "data"
# BASE_DATA_DIR = "data_multimon"

DATA_DIR = os.path.abspath(os.path.join(BASE_DATA_DIR, "mp3"))
OVERVIEW_TSV = os.path.join(BASE_DATA_DIR, "Rok_spring_summer.tsv")
SAVE_PATH = os.path.join(BASE_DATA_DIR, "cache", "final_data.parquet")

# OVERVIEW_TSV = os.path.join(BASE_DATA_DIR, "zabe+hyla.tsv")

# Load the overview TSV as reference (if it exists, to avoid import-time crash)
if os.path.exists(OVERVIEW_TSV):
    wav_meta = pd.read_csv(OVERVIEW_TSV, sep='\t', dtype={6: str})
    wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)
else:
    wav_meta = pd.DataFrame()


def load_positions_tsv_optimized(wav_meta=OVERVIEW_TSV, data_dir=DATA_DIR, save_path=SAVE_PATH):
    """
    Optimized data loading function.
    - Gathers all file paths first.
    - Reads all files into a list of DataFrames.
    - Concatenates them all at once.
    - Performs transformations in a vectorized manner on the full DataFrame.
    """
    all_files_to_process = []

    wav_meta = pd.read_csv(wav_meta, sep='\t', dtype={6: str})
    wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)
    if 'channel' in wav_meta.columns:
        wav_meta['channel'] = wav_meta['channel'].astype(int)
    
    unique_wavs = wav_meta['wav_file'].unique()

    # 1. Collect all file paths uniquely
    for wav_path in tqdm(unique_wavs, desc="Finding TSV files"):
        wav_path_norm = wav_path.replace("\\", "/")
        positions_dir = os.path.join(os.path.dirname(wav_path_norm), 'positions')
        base_name = os.path.splitext(os.path.basename(wav_path_norm))[0]

        full_positions_dir = os.path.join(data_dir, positions_dir)
        if not os.path.exists(full_positions_dir):
            continue

        pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')
        try:
            for file in os.listdir(full_positions_dir):
                if pattern.match(file):
                    tsv_path = os.path.join(full_positions_dir, file)
                    all_files_to_process.append({'path': tsv_path, 'wav_file': wav_path, 'file_name': file})
        except OSError:
            continue

    if not all_files_to_process:
        return pd.DataFrame()

    # 2. Read all TSV files into a list of DataFrames
    df_list = []
    for file_info in tqdm(all_files_to_process, desc="Reading TSV files"):
        try:
            dff = pd.read_csv(file_info['path'], sep='\t')
        except Exception:
            continue

        dff['wav_file'] = file_info['wav_file']

        # Add cluster number from the file contents if available
        if 'num_clusters' in dff.columns:
            # Handle potential NaNs or floats safely
            dff['cluster_num'] = dff['num_clusters'].fillna(10).astype(int)
        else:
            c = file_info['file_name'].split('.')[-2].split('_')[-1]
            dff['cluster_num'] = int(c) if c.isnumeric() else 10

        df_list.append(dff)

    if not df_list:
        return pd.DataFrame()

    # 3. Concatenate everything at once and merge metadata safely by channel
    df = pd.concat(df_list, ignore_index=True)
    if 'channel' in df.columns:
        df['channel'] = df['channel'].astype(int)
        
    df.drop(columns=['day', 'start_time'], errors='ignore', inplace=True)

    # Clean up exact duplicate clustering entries caused by duplicate TSV sweeps in directory
    if set(['wav_file', 'channel', 'clip_time', 'model_name', 'cluster_num']).issubset(df.columns):
        df.drop_duplicates(subset=['wav_file', 'channel', 'clip_time', 'model_name', 'cluster_num'], inplace=True)
        
    df = df.merge(wav_meta, on=['wav_file', 'channel'], how='left')

    # 4. Perform all transformations in a vectorized way on the full DataFrame
    df['day_dt'] = pd.to_datetime(df['day'], format="%Y-%m-%d")
    df['start_dt'] = pd.to_datetime(df['start_time'], format='%H:%M:%S')

    # Optimized embedding parsing
    if 'embedding' in df.columns:
        temp_df = df['embedding'].str.strip('[]').str.split(', ', expand=True)
        if len(temp_df.columns) >= 2:
            df['x'] = pd.to_numeric(temp_df[0], errors='coerce')
            df['y'] = pd.to_numeric(temp_df[1], errors='coerce')
        else:
            df['x'] = 0
            df['y'] = 0

    if 'time_of_day' not in df.columns:
        df['time_of_day'] = df['start_dt'] + pd.to_timedelta(df['clip_time'], unit='s')
        
    df['start_hour_float'] = (
            df['time_of_day'].dt.hour +
            df['time_of_day'].dt.minute / 60 +
            df['time_of_day'].dt.second / 3600
    )
    df['day_str'] = df['day_dt'].dt.strftime('%Y-%m-%d')

    base_name = df['wav_file'].str.removesuffix('.wav')
    base_name = base_name.str.removesuffix('.WAV')
    df['mp3_file'] = base_name + '_ch' + df['channel'].astype(str) + '.mp3'
    df['mp3_file'] = df['mp3_file'].apply(os.path.normpath)

    df['abs_file_name'] = df['mp3_file']
    df['cluster_id'] = df['cluster_id'].astype(int)
    df['row_idx'] = range(len(df))

    # Clean up columns
    columns_to_drop = [
        'day', 'start_time', 'num_clusters', 'embedding'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Ensure target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the processed data
    df.to_parquet(save_path)

    return df


from typing import Optional, List
import hierarchy

# Canonical column definitions shared between legacy and hierarchical loading
INITIAL_LAYOUT_COLUMNS: List[str] = [
    'location', 'microlocation', 'model_name', 'channel', 'cluster_num',
    'cluster_id', 'day_dt', 'start_hour_float'
]

FULL_DATA_COLUMNS: List[str] = [
    'x', 'y', 'location', 'microlocation', 'model_name', 'channel',
    'cluster_num', 'cluster_id', 'day_dt', 'start_hour_float',
    'file_name', 'clip_time', 'clip_duration', 'mp3_file', 'row_idx',
    'start_dt', 'time_of_day', 'recorder_type', 'wav_file'
]


def process_loaded_model_dataframe(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard post-processing transformations to a loaded model DataFrame:
    - resets index
    - generates day_dt_str formatted as YYYY-MM-DD
    - precomputes date_time_str formatted as DD-MM-YYYY HH:MM
    - converts cluster_id to cluster_id_str
    - ensures clip_count column is initialized to 1
    - casts file_name and channel to categorical for memory efficiency
    """
    if model_df is None:
        return pd.DataFrame()

    model_df = model_df.reset_index(drop=True)

    if model_df.empty:
        if 'day_dt' in model_df.columns and 'day_dt_str' not in model_df.columns:
            model_df['day_dt_str'] = pd.Series(dtype='object')
        if 'date_time_str' not in model_df.columns:
            model_df['date_time_str'] = pd.Series(dtype='object')
        if 'cluster_id' in model_df.columns and 'cluster_id_str' not in model_df.columns:
            model_df['cluster_id_str'] = pd.Series(dtype='object')
        if 'clip_count' not in model_df.columns:
            model_df['clip_count'] = pd.Series(dtype='int64')
        return model_df

    if 'day_dt' in model_df.columns:
        model_df['day_dt_str'] = pd.to_datetime(model_df['day_dt']).dt.strftime('%Y-%m-%d')

    if 'day_dt' in model_df.columns and 'start_hour_float' in model_df.columns:
        d_part = model_df['day_dt_str'].str[8:10] + '-' + model_df['day_dt_str'].str[5:7] + '-' + model_df['day_dt_str'].str[0:4]
        h = pd.to_numeric(model_df['start_hour_float'], errors='coerce').fillna(0).clip(lower=0, upper=24)
        h_int = h.astype(int)
        m_int = ((h % 1) * 60).astype(int)
        model_df['date_time_str'] = d_part + ' ' + h_int.astype(str).str.zfill(2) + ':' + m_int.astype(str).str.zfill(2)

    if 'cluster_id' in model_df.columns:
        model_df['cluster_id_str'] = model_df['cluster_id'].astype(str)

    if 'clip_count' not in model_df.columns:
        model_df['clip_count'] = 1

    if 'file_name' in model_df.columns and model_df['file_name'].dtype != 'category':
        model_df['file_name'] = model_df['file_name'].astype('category')
    if 'channel' in model_df.columns and model_df['channel'].dtype != 'category':
        model_df['channel'] = model_df['channel'].astype('category')

    # Rule: If a microlocation exists: use it. If no microlocation exists: use physical location. Never show 'default'.
    if 'microlocation' in model_df.columns and 'location' in model_df.columns:
        bad_micro = model_df['microlocation'].isna() | (
            model_df['microlocation'].astype(str).str.strip().str.lower().isin(['', 'default', 'none', 'nan', 'unknown'])
        )
        if bad_micro.any():
            model_df.loc[bad_micro, 'microlocation'] = model_df.loc[bad_micro, 'location']

    return model_df


def get_location_parquet_path(project_id: str, location_id: Optional[str] = None, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> Optional[str]:
    """
    Resolves the Parquet file path for a given project and location via hierarchy discovery.
    Returns path string if file exists, else None.
    """
    paths = hierarchy.get_location_paths(project_id, location_id or "", base_dir=base_dir)
    return paths.get("parquet")


# Alias for backward compatibility
get_dataset_parquet_path = get_location_parquet_path


def get_project_parquet_path(project_id: str, base_dir: str = hierarchy.DEFAULT_DATA_DIR) -> Optional[str]:
    """Resolves the Parquet file path for a given project via hierarchy discovery."""
    return get_location_parquet_path(project_id, None, base_dir=base_dir)


def get_initial_data_for_location(
    project_id: str,
    location_id: Optional[str] = None,
    base_dir: str = hierarchy.DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Loads lightweight metadata columns for a specific hierarchical project and location.
    If location_id is provided, filters to that location (and matching wav_file prefix if applicable).
    Returns an empty DataFrame with standard INITIAL_LAYOUT_COLUMNS if parquet is missing or unreadable.
    """
    parquet_path = get_location_parquet_path(project_id, location_id, base_dir=base_dir)
    if not parquet_path or not os.path.exists(parquet_path):
        # Fallback: check partitioned cache if monolithic final_data.parquet is absent
        from cache import reader
        part_reader = reader.PartitionedCacheReader()
        if part_reader.is_available(project_id, base_dir=base_dir):
            meta = part_reader.get_project_metadata(project_id, base_dir=base_dir)
            if meta and meta.partitions:
                target_loc = location_id or ""
                matching = [
                    p for p in meta.partitions
                    if not target_loc or p.get("location") == target_loc or p.get("location", "").startswith(target_loc) or target_loc.startswith(p.get("location", ""))
                ]
                p_dfs = []
                for mp in matching:
                    p_full = os.path.join(base_dir, project_id, *mp["partition_rel_path"].split("/"))
                    if os.path.isfile(p_full):
                        try:
                            import pyarrow.parquet as pq_m
                            schema = pq_m.read_schema(p_full)
                            req = [c for c in INITIAL_LAYOUT_COLUMNS if c in set(schema.names)]
                            if req:
                                p_df = pd.read_parquet(p_full, columns=req)
                                p_dfs.append(p_df)
                        except Exception:
                            pass
                if p_dfs:
                    combined = pd.concat(p_dfs, ignore_index=True)
                    if 'microlocation' in combined.columns and 'location' in combined.columns:
                        bad_micro = combined['microlocation'].isna() | (
                            combined['microlocation'].astype(str).str.strip().str.lower().isin(['', 'default', 'none', 'nan', 'unknown'])
                        )
                        if bad_micro.any():
                            combined.loc[bad_micro, 'microlocation'] = combined.loc[bad_micro, 'location']
                    for col in INITIAL_LAYOUT_COLUMNS:
                        if col not in combined.columns:
                            combined[col] = pd.Series(dtype='object')
                    return combined[INITIAL_LAYOUT_COLUMNS]
        return pd.DataFrame({col: [] for col in INITIAL_LAYOUT_COLUMNS})

    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(parquet_path)
        available_cols = set(schema.names)
        cols_to_request = [c for c in INITIAL_LAYOUT_COLUMNS if c in available_cols]
        if not cols_to_request:
            return pd.DataFrame({col: [] for col in INITIAL_LAYOUT_COLUMNS})

        filters = None
        if location_id and 'location' in available_cols:
            filters = [[('location', '==', location_id)]]

        df = pd.read_parquet(parquet_path, columns=cols_to_request, filters=filters)
        
        # If no rows returned by location column filter (e.g. location_id was a folder prefix like Bistra_spring),
        # fallback to reading without filter if empty
        if df.empty and location_id and 'location' in available_cols:
            df_all = pd.read_parquet(parquet_path, columns=cols_to_request)
            if not df_all.empty:
                df = df_all

        if 'microlocation' in df.columns and 'location' in df.columns:
            bad_micro = df['microlocation'].isna() | (
                df['microlocation'].astype(str).str.strip().str.lower().isin(['', 'default', 'none', 'nan', 'unknown'])
            )
            if bad_micro.any():
                df.loc[bad_micro, 'microlocation'] = df.loc[bad_micro, 'location']

        for col in INITIAL_LAYOUT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series(dtype='object')
        return df[INITIAL_LAYOUT_COLUMNS]
    except Exception:
        return pd.DataFrame({col: [] for col in INITIAL_LAYOUT_COLUMNS})


def get_initial_data_for_project(
    project_id: str,
    base_dir: str = hierarchy.DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """Loads lightweight metadata columns for a project root."""
    return get_initial_data_for_location(project_id, location_id=None, base_dir=base_dir)


# Alias for backward compatibility
get_initial_data_for_dataset = get_initial_data_for_location


def load_project_model_data(
    project_id: str,
    selected_model: Optional[str] = None,
    selected_location: Optional[str] = None,
    base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    strict: Optional[bool] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Loads model/location filtered data for a project across all source partitions
    or fallback to legacy cache.
    """
    eff_model = selected_model if selected_model is not None else kwargs.get("model_name")
    from cache import reader
    return reader.load_project_model_data(
        project_id=project_id,
        model_name=eff_model,
        selected_location=selected_location,
        base_dir=base_dir,
        strict=strict,
        **kwargs
    )


def load_location_model_data(
    project_id: str,
    location_id: Optional[str] = None,
    selected_model: Optional[str] = None,
    selected_location: Optional[str] = None,
    base_dir: str = hierarchy.DEFAULT_DATA_DIR,
    dataset_id: Optional[str] = None,
    model_name: Optional[str] = None,
    strict: Optional[bool] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Loads model/location filtered data from a hierarchical project/location.
    Uses the cache reader adapter (preferring partitioned cache when available,
    falling back to legacy final_data.parquet).
    Guarantees isolation: reads strictly from the resolved project/location.
    Applies standard process_loaded_model_dataframe transformations.
    """
    eff_loc = location_id if location_id is not None else dataset_id
    eff_model = selected_model if selected_model is not None else model_name
    from cache import reader
    return reader.load_location_model_data(
        project_id=project_id,
        location_id=eff_loc or "",
        model_name=eff_model,
        selected_location=selected_location,
        base_dir=base_dir,
        strict=strict,
        **kwargs
    )


# Alias for backward compatibility
load_dataset_model_data = load_location_model_data


def get_initial_data_for_layout():
    """
    Loads only the columns necessary to build the initial UI layout.
    This is much more memory-efficient than loading the entire dataset.
    Maintains 100% backward compatibility for the existing application startup.
    """
    if os.path.exists(SAVE_PATH):
        try:
            return pd.read_parquet(SAVE_PATH, columns=INITIAL_LAYOUT_COLUMNS)
        except Exception:
            pass

    # Check available projects to find default project initial data
    try:
        available_projects = hierarchy.get_available_projects()
        if available_projects:
            df_proj = get_initial_data_for_project(available_projects[0].id)
            if df_proj is not None and not df_proj.empty:
                return df_proj
    except Exception:
        pass

    return pd.DataFrame({col: [] for col in INITIAL_LAYOUT_COLUMNS})


if __name__ == '__main__':
    load_positions_tsv_optimized()