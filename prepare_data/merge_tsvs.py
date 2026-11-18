import pathlib
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# --- 1. Configuration & Paths ---
# Ensure these match your actual folder structure
DATA_DIR = os.path.abspath("data_multimon/mp3")
OVERVIEW_TSV = os.path.join("data_multimon", "zabe_combined.tsv")
SAVE_PATH = "data_multimon/cache/final_data_FIXED.parquet"


# --- 2. The Main Processing Function ---
def load_positions_tsv_optimized(wav_meta, data_dir, save_path=SAVE_PATH, limit=None):
    """
    Loads and processes TSV files.
    :param limit: Integer. If set, stops collecting files after this many are found.
                  Useful for testing the pipeline quickly.
    """
    start_time_global = time.time()
    all_files_to_process = []

    print(f"Looking for data in: {data_dir}")
    print("Collecting file paths...")

    if wav_meta.empty:
        print("CRITICAL: Overview TSV is empty.")
        return pd.DataFrame()

    # 1. Match Overview rows to Position files
    start_collect = time.time()
    for idx, row in wav_meta.iterrows():
        # --- PRINT PROGRESS EVERY 1000 ROWS ---
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(wav_meta)}...", end='\r')

        # --- LIMIT CHECK ---
        if limit is not None and len(all_files_to_process) >= limit:
            print(f"\n--- DEBUG: Reached limit of {limit} files. Stopping collection. ---")
            break
        # -------------------

        # Normalize paths for Windows/Linux compatibility
        wav_path = row['wav_file'].replace("\\", "/")
        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # Construct path to the 'positions' folder for this file
        # Assumption: positions folder is inside the same folder as the wav file
        relative_dir = os.path.dirname(wav_path)
        positions_dir = os.path.join(data_dir, relative_dir, 'positions')

        if not os.path.exists(positions_dir):
            continue

        # Find the specific TSV for this wav file
        pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')

        try:
            for file in os.listdir(positions_dir):
                if pattern.match(file):
                    tsv_path = os.path.join(positions_dir, file)

                    # Store the path AND the metadata
                    all_files_to_process.append({
                        'path': tsv_path,
                        'meta': row.to_dict(),
                        'file_name': file
                    })

                    # Check limit inside inner loop too for precision
                    if limit is not None and len(all_files_to_process) >= limit:
                        break
        except OSError as e:
            print(f"Error accessing {positions_dir}: {e}")
            continue

    end_collect = time.time()
    print(
        f"\nStep 1: File collection complete. Found {len(all_files_to_process)} files. Time taken: {end_collect - start_collect:.2f} seconds.")

    if not all_files_to_process:
        print("\nCRITICAL: No position TSV files matched the Overview file.")
        return pd.DataFrame()

    # 2. Read TSVs and Inject Metadata
    df_list = []
    print(f"Step 2: Reading {len(all_files_to_process)} TSV files...")
    start_read = time.time()

    for file_info in tqdm(all_files_to_process, desc="Reading Files"):
        try:
            dff = pd.read_csv(file_info['path'], sep='\t')
            if dff.empty: continue

            # --- CRITICAL STEP: INJECT OVERVIEW METADATA ---
            for key, value in file_info['meta'].items():
                dff[key] = value

            # Extract Cluster Number (k)
            try:
                c_str = file_info['file_name'].replace('.tsv', '').split('_')[-1]
                dff['cluster_num'] = int(c_str) if c_str.isnumeric() else 10
            except:
                dff['cluster_num'] = 10

            df_list.append(dff)

        except Exception as e:
            print(f"Error reading {file_info['path']}: {e}")

    end_read = time.time()
    print(f"Step 2: Reading complete. Time taken: {end_read - start_read:.2f} seconds.")

    if not df_list:
        print("Error: All found TSV files were empty or unreadable.")
        return pd.DataFrame()

    # 3. Concatenate into one big DataFrame
    print("Step 3: Concatenating DataFrames...")
    start_concat = time.time()
    df = pd.concat(df_list, ignore_index=True)
    end_concat = time.time()
    print(f"Step 3: Concatenation complete. Rows: {len(df)}. Time taken: {end_concat - start_concat:.2f} seconds.")

    # 4. Vectorized Transformations (The "Heavy Lifting")
    print("Step 4: Performing vectorized calculations...")
    start_transform = time.time()

    try:
        # Convert Day and Start Time strings to Datetime objects
        df['day_dt'] = pd.to_datetime(df['day'], format="%Y-%m-%d", errors='coerce')
        df['start_dt'] = pd.to_datetime(df['start_time'], format='%H:%M:%S', errors='coerce')

        # Parse Embeddings
        temp_df = df['embedding'].astype(str).str.strip('[]').str.split(', ', expand=True)
        df['x'] = pd.to_numeric(temp_df[0], errors='coerce')
        df['y'] = pd.to_numeric(temp_df[1], errors='coerce')

        # --- CALCULATE EXACT TIMESTAMP FOR EACH CLIP ---
        df['time_of_day'] = df['start_dt'] + pd.to_timedelta(df['clip_time'], unit='s')

        # Calculate Float Hour
        df['start_hour_float'] = (
                df['time_of_day'].dt.hour +
                df['time_of_day'].dt.minute / 60 +
                df['time_of_day'].dt.second / 3600
        )

        df['day_str'] = df['day_dt'].dt.strftime('%Y-%m-%d')

        # Generate MP3 Filepath
        base_name = df['wav_file'].astype(str).str.replace(r'\.wav$', '', case=False, regex=True)
        df['mp3_file'] = base_name + '_ch' + df['channel'].astype(str) + '.mp3'
        df['mp3_file'] = df['mp3_file'].apply(os.path.normpath)

        # Ensure cluster_id is integer
        df['cluster_id'] = pd.to_numeric(df['cluster_id'], errors='coerce').fillna(-1).astype(int)

        # Create a stable Row Index
        df['row_idx'] = range(len(df))

        # 5. Cleanup & Save
        columns_to_drop = ['day', 'start_time', 'num_clusters', 'embedding']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

        end_transform = time.time()
        print(f"Step 4: Transformations complete. Time taken: {end_transform - start_transform:.2f} seconds.")

        print(f"Step 5: Saving to {save_path}...")
        start_save = time.time()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
        end_save = time.time()
        print(f"Step 5: Save complete. Time taken: {end_save - start_save:.2f} seconds.")
        print(f"DONE. Total rows processed: {len(df)}. Total time: {end_save - start_time_global:.2f} seconds.")

    except KeyError as e:
        print(f"CRITICAL ERROR: Missing expected column in data: {e}")
        print("Columns found:", df.columns.tolist())
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return pd.DataFrame()

    return df


# --- 3. Helper function for App.py ---
def get_initial_data_for_layout():
    """
    Fast loader for app startup. Reads only columns needed for dropdowns.
    """
    if not os.path.exists(SAVE_PATH):
        return pd.DataFrame({
            'location': [], 'model_name': [], 'channel': [],
            'cluster_num': [], 'cluster_id': [], 'day_dt': [],
            'start_hour_float': []
        })

    cols_to_load = [
        'location', 'microlocation', 'model_name', 'channel', 'cluster_num',
        'cluster_id', 'day_dt', 'start_hour_float'
    ]
    try:
        return pd.read_parquet(SAVE_PATH, columns=cols_to_load)
    except Exception as e:
        print(f"Error reading cache for layout: {e}")
        return pd.DataFrame({col: [] for col in cols_to_load})


# --- 4. Script Execution Block ---
# Removed the if __name__ == "__main__": guard so it runs on import/execution
print("--- Starting Offline Data Processing ---")

if os.path.exists(OVERVIEW_TSV):
    print(f"Loading Overview TSV from: {OVERVIEW_TSV}")
    wav_meta_df = pd.read_csv(OVERVIEW_TSV, sep='\t')

    # Normalize overview paths immediately
    wav_meta_df['wav_file'] = wav_meta_df['wav_file'].apply(os.path.normpath)

    # Run the processor WITH A LIMIT for testing
    # Change limit=100 to limit=None when you are ready for the full run!
    load_positions_tsv_optimized(wav_meta_df, DATA_DIR, save_path=SAVE_PATH)
else:
    print(f"ERROR: Overview TSV not found at {OVERVIEW_TSV}")