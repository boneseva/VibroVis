import pathlib
import re
import os
import ast
import pandas as pd
from tqdm import tqdm

# Allow configuration via environment variable, default to 'data'
BASE_DATA_DIR = os.getenv('DATA_DIR', 'data')
DATA_DIR = os.path.abspath(os.path.join(BASE_DATA_DIR, "mp3"))
OVERVIEW_TSV = os.path.join(BASE_DATA_DIR, "Rok_spring_summer.tsv")
SAVE_PATH = os.path.join(BASE_DATA_DIR, "cache", "final_data.parquet")
LABELS_SAVE_PATH = pathlib.Path(os.path.join(BASE_DATA_DIR, "cache", "saved_labels.parquet"))

# Load the overview TSV as reference (if it exists, to avoid import-time crash)
if os.path.exists(OVERVIEW_TSV):
    wav_meta = pd.read_csv(OVERVIEW_TSV, sep='\t')
    wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)
else:
    wav_meta = pd.DataFrame()
    # print(f"WARNING: Overview file not found at {OVERVIEW_TSV}")


def load_positions_tsv_optimized(wav_meta, data_dir, save_path=SAVE_PATH):
    """
    Optimized data loading function.
    - Gathers all file paths first.
    - Reads all files into a list of DataFrames.
    - Concatenates them all at once.
    - Performs transformations in a vectorized manner on the full DataFrame.
    """
    all_files_to_process = []

    # 1. Collect all file paths and their associated metadata first
    for idx, row in wav_meta.iterrows():
        wav_path = row['wav_file'].replace("\\", "/")
        positions_dir = os.path.join(os.path.dirname(wav_path), 'positions')
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        full_positions_dir = os.path.join(data_dir, positions_dir)
        if not os.path.exists(full_positions_dir):
            continue

        pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')
        try:
            for file in os.listdir(full_positions_dir):
                if pattern.match(file):
                    tsv_path = os.path.join(full_positions_dir, file)
                    all_files_to_process.append({'path': tsv_path, 'meta': row.to_dict(), 'file_name': file})
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

        # Add metadata from the overview file to each row
        for key, value in file_info['meta'].items():
            dff[key] = value

        # Add cluster number from the filename
        c = file_info['file_name'].split('.')[-3].split('_')[-1]
        dff['cluster_num'] = int(c) if c.isnumeric() else 10

        df_list.append(dff)

    if not df_list:
        return pd.DataFrame()

    # 3. Concatenate everything at once
    df = pd.concat(df_list, ignore_index=True)

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
