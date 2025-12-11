import pathlib
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 1. Configuration & Paths ---
DATA_DIR = os.path.abspath("data_multimon/mp3")
OVERVIEW_TSV = os.path.join("data_multimon", "zabe_combined.tsv")
SAVE_PATH = "data_multimon/cache/final_data_FIXED.parquet"
LABELS_SAVE_PATH = pathlib.Path("data_multimon/cache/saved_labels.parquet")

if __name__ == "__main__":
    if os.path.exists(OVERVIEW_TSV):
        print(f"Loading Overview TSV from: {OVERVIEW_TSV}")
        wav_meta = pd.read_csv(OVERVIEW_TSV, sep='\t')
        wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)
    else:
        print(f"WARNING: Overview TSV not found at {OVERVIEW_TSV}")
        wav_meta = pd.DataFrame()


# --- 3. Data Loading Functions ---

def load_positions_tsv_optimized(wav_meta, data_dir, save_path=SAVE_PATH):
    all_files_to_process = []

    # 1. Collect file paths
    print("Collecting file paths...")
    if wav_meta.empty:
        return pd.DataFrame()

    for idx, row in wav_meta.iterrows():
        if idx % 1000 == 0: print(f" {idx}", end="", flush=True)

        wav_path = row['wav_file'].replace("\\", "/")
        positions_dir = os.path.join(os.path.dirname(wav_path), 'positions')
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        full_positions_dir = os.path.join(data_dir, positions_dir)

        if not os.path.exists(full_positions_dir): continue

        pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')
        try:
            for file in os.listdir(full_positions_dir):
                if pattern.match(file):
                    tsv_path = os.path.join(full_positions_dir, file)
                    # We store 'row.to_dict()' here, which contains the start_time from Overview
                    all_files_to_process.append({
                        'path': tsv_path,
                        'meta': row.to_dict(),
                        'file_name': file
                    })
        except OSError:
            continue

    if not all_files_to_process:
        print("\nNo position files found.")
        return pd.DataFrame()

    # 2. Read TSVs and Merge Metadata
    df_list = []
    print(f"\nReading {len(all_files_to_process)} TSV files...")

    for file_info in tqdm(all_files_to_process, desc="Reading TSV files"):
        try:
            dff = pd.read_csv(file_info['path'], sep='\t')
            if dff.empty: continue

            # --- CRITICAL FIX FOR MISSING START TIME ---
            # This loop takes everything from the Overview TSV (including start_time)
            # and injects it into the current rows.
            for key, value in file_info['meta'].items():
                dff[key] = value

            c = file_info['file_name'].replace('.tsv', '').split('_')[-1]
            dff['cluster_num'] = int(c) if c.isnumeric() else 10
            df_list.append(dff)

        except pd.errors.EmptyDataError:
            pass
        except Exception as e:
            print(f"Error reading {file_info['path']}: {e}")

    if not df_list: return pd.DataFrame()

    # 3. Concatenate
    print("Concatenating DataFrames...")
    df = pd.concat(df_list, ignore_index=True)

    # 4. Transformations (Calculating Start Hour)
    print("Performing vectorized transformations...")
    try:
        # Format the start_time (which we just ensured exists via the merge above)
        df['day_dt'] = pd.to_datetime(df['day'], format="%Y-%m-%d")
        df['start_dt'] = pd.to_datetime(df['start_time'], format='%H:%M:%S')

        # Parse Embeddings safely
        temp_df = df['embedding'].astype(str).str.strip('[]').str.split(', ', expand=True)
        df['x'] = pd.to_numeric(temp_df[0], errors='coerce')
        df['y'] = pd.to_numeric(temp_df[1], errors='coerce')

        # --- CALCULATE EXACT TIME ---
        # start_dt (from Overview) + clip_time (from individual file) = Actual Time
        df['time_of_day'] = df['start_dt'] + pd.to_timedelta(df['clip_time'], unit='s')

        # Calculate the float hour for your visualization
        df['start_hour_float'] = (
                df['time_of_day'].dt.hour +
                df['time_of_day'].dt.minute / 60 +
                df['time_of_day'].dt.second / 3600
        )
        df['day_str'] = df['day_dt'].dt.strftime('%Y-%m-%d')

        # --- GENERATE TEMPERATURE (if not already present) ---
        # Generate realistic temperature values based on time/date if temperature column doesn't exist
        if 'temperature' not in df.columns:
            # Base temperature varies by month (seasonal effect)
            # Assuming Northern Hemisphere: colder in winter (Dec-Feb), warmer in summer (Jun-Aug)
            month = df['day_dt'].dt.month
            seasonal_base = 15 + 10 * np.sin((month - 3) * np.pi / 6)  # Range roughly 5-25°C
            
            # Diurnal variation: colder at night, warmer during day
            hour = df['time_of_day'].dt.hour
            diurnal_variation = 8 * np.sin((hour - 6) * np.pi / 12)  # Peak around 2 PM, minimum around 2 AM
            
            # Add some random variation (±2°C)
            np.random.seed(42)  # For reproducibility
            random_variation = np.random.normal(0, 1.5, len(df))
            
            # Combine all effects
            df['temperature'] = seasonal_base + diurnal_variation + random_variation
            # Clip to reasonable range (0-35°C)
            df['temperature'] = df['temperature'].clip(0, 35)
            df['temperature'] = df['temperature'].round(1)

        # Fix Filenames
        base_name = df['wav_file'].astype(str).str.replace(r'\.wav$', '', case=False, regex=True)
        df['mp3_file'] = base_name + '_ch' + df['channel'].astype(str) + '.mp3'
        df['mp3_file'] = df['mp3_file'].apply(os.path.normpath)

        df['abs_file_name'] = df['mp3_file']
        df['cluster_id'] = df['cluster_id'].astype(int)
        df['row_idx'] = range(len(df))

        # Cleanup
        columns_to_drop = ['day', 'start_time', 'num_clusters', 'embedding']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
        print(f"Success! Wrote {len(df)} rows to {save_path}")

    except KeyError as e:
        print(f"CRITICAL ERROR: Missing column: {e}")
        return df
    except Exception as e:
        print(f"Unexpected error: {e}")
        return df

    return df


def get_initial_data_for_layout():
    if not os.path.exists(SAVE_PATH):
        return pd.DataFrame(
            {'location': [], 'model_name': [], 'channel': [], 'cluster_num': [], 'cluster_id': [], 'day_dt': [],
             'start_hour_float': []})

    cols_to_load = ['location', 'microlocation', 'model_name', 'channel', 'cluster_num', 'cluster_id', 'day_dt',
                    'start_hour_float']
    try:
        return pd.read_parquet(SAVE_PATH, columns=cols_to_load)
    except:
        return pd.DataFrame({col: [] for col in cols_to_load})


if not os.path.exists(SAVE_PATH):
    print("Cache not found. Running robust data loader...")
    df = load_positions_tsv_optimized(wav_meta, DATA_DIR, save_path=SAVE_PATH)
else:
    print(f"Loading cached data from {SAVE_PATH}...")
    try:
        df = pd.read_parquet(SAVE_PATH)
        print(f"Loaded {len(df)} rows.")
    except:
        print("Cache corrupted. Regenerating...")
        df = load_positions_tsv_optimized(wav_meta, DATA_DIR, save_path=SAVE_PATH)