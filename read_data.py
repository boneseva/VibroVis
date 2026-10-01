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


def get_initial_data_for_layout():
    """
    Loads only the columns necessary to build the initial UI layout.
    This is much more memory-efficient than loading the entire dataset.
    """
    if not os.path.exists(SAVE_PATH):
        # If cache doesn't exist, return an empty structure
        return pd.DataFrame({
            'location': [], 'model_name': [], 'channel': [],
            'cluster_num': [], 'cluster_id': [], 'day_dt': [],
            'start_hour_float': []
        })

    # Read only the specific columns needed for the filters and sliders
    cols_to_load = [
        'location', 'microlocation', 'model_name', 'channel', 'cluster_num', 
        'cluster_id', 'day_dt', 'start_hour_float'
    ]
    try:
        initial_df = pd.read_parquet(SAVE_PATH, columns=cols_to_load)
        return initial_df
    except Exception as e:
        # Error reading initial data from Parquet file: {e}
        # Return an empty df on error to prevent app crash
        return pd.DataFrame({col: [] for col in cols_to_load})

if __name__ == '__main__':
    load_positions_tsv_optimized()