import pathlib
import re
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc

# --- CONFIGURATION ---
DATA_DIR = os.path.abspath("data_multimon/mp3")
OVERVIEW_TSV = os.path.join("data_multimon", "zabe_combined.tsv")
TEMP_LIST_PATH = pathlib.Path("data_multimon/cache/temp_files_to_fix.npy")


# --- HELPER: Time Extractor ---
def extract_time_from_filename(abs_file_name_series):
    base_name = abs_file_name_series.apply(os.path.basename)
    pattern = r'(\d{8})_(\d{6})'
    extracted = base_name.str.extract(pattern, expand=True)
    full_datetime_str = extracted[0] + extracted[1]
    return pd.to_datetime(full_datetime_str, format='%Y%m%d%H%M%S', errors='coerce')


# --- LOADER (Test Version) ---
def load_positions_tsv_optimized(wav_meta, data_dir, save_file=False):
    # 1. Collect paths
    all_files_to_process = []
    print(f"  > Scanning {len(wav_meta)} source files...")
    for idx, row in wav_meta.iterrows():
        wav_path = row['wav_file'].replace("\\", "/")
        positions_dir = os.path.join(os.path.dirname(wav_path), 'positions')
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        full_positions_dir = os.path.join(data_dir, positions_dir)

        if os.path.exists(full_positions_dir):
            pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')
            for file in os.listdir(full_positions_dir):
                if pattern.match(file):
                    tsv_path = os.path.join(full_positions_dir, file)
                    all_files_to_process.append({'path': tsv_path, 'meta': row.to_dict(), 'file_name': file})

    if not all_files_to_process: return pd.DataFrame()

    # 2. Read
    df_list = []
    for file_info in all_files_to_process:
        try:
            dff = pd.read_csv(file_info['path'], sep='\t')
            if dff.empty: continue
            for key, value in file_info['meta'].items(): dff[key] = value
            c = file_info['file_name'].replace('.tsv', '').split('_')[-1]
            dff['cluster_num'] = int(c) if c.isnumeric() else 10
            df_list.append(dff)
        except Exception:
            continue

    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    # 3. Process Columns
    base_name = df['wav_file'].astype(str).str.replace(r'\.wav$', '', case=False, regex=True)
    df['mp3_file'] = base_name + '_ch' + df['channel'].astype(str) + '.mp3'
    df['mp3_file'] = df['mp3_file'].apply(os.path.normpath)
    df['abs_file_name'] = df['mp3_file']

    # --- THE FIX LOGIC ---
    tsv_timestamp_str = df['day'].astype(str) + ' ' + df['start_time'].astype(str)
    df['base_dt'] = pd.to_datetime(tsv_timestamp_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    nan_mask = df['base_dt'].isna()
    if nan_mask.any():
        # EXTRACT TIME
        filename_dt = extract_time_from_filename(df.loc[nan_mask, 'abs_file_name'])
        df.loc[nan_mask, 'base_dt'] = filename_dt

    # Calculate final time
    df['time_of_day'] = df['base_dt'] + pd.to_timedelta(df['clip_time'], unit='s', errors='coerce')

    df['start_hour_float'] = (
            df['time_of_day'].dt.hour +
            df['time_of_day'].dt.minute / 60 +
            df['time_of_day'].dt.second / 3600
    )

    return df


# --- TEST EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- 🕵️‍♂️ FAILURE INSPECTION RUN (First 50 Files) 🕵️‍♂️ ---")

    wav_meta = pd.read_csv(OVERVIEW_TSV, sep='\t')
    wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)

    files_to_fix = np.load(TEMP_LIST_PATH, allow_pickle=True).tolist()
    test_subset_files = files_to_fix[:50]

    test_meta = wav_meta[wav_meta['wav_file'].isin(test_subset_files)]
    df_result = load_positions_tsv_optimized(test_meta, DATA_DIR, save_file=False)

    # FILTER FOR FAILURES
    failed_rows = df_result[df_result['start_hour_float'].isna()]

    if not failed_rows.empty:
        print(f"\n❌ FOUND {len(failed_rows)} FAILED ROWS.")
        print("Here is a sample of what they look like:")
        print("=" * 80)
        # Show File Name, Extracted Time (base_dt), and Clip Time
        print(failed_rows[['abs_file_name', 'base_dt', 'clip_time']].head(15).to_markdown(index=False))
        print("=" * 80)
        print("\nAnalyze the table above:")
        print("1. If 'base_dt' is NaT -> The filename format is different.")
        print("2. If 'base_dt' is OK but result is NaN -> 'clip_time' is the problem.")
    else:
        print("\n✅ Weirdly, no failures found in this run?")