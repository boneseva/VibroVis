import re
import os
import ast
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.abspath("data/mp3")
OVERVIEW_TSV = os.path.join("data", "Rok_spring_summer.tsv")
SAVE_PATH = "data/cache/final_data.parquet"
BATCH_SIZE = 10000

# Load the overview TSV as reference
wav_meta = pd.read_csv(OVERVIEW_TSV, sep='\t')
wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)

def load_positions_tsv(wav_meta, data_dir, save_path=SAVE_PATH, batch_size=BATCH_SIZE):
    df = pd.DataFrame()
    
    last_saved_len = 0
    buffer = []

    for idx, row in tqdm(wav_meta.iterrows(), total=len(wav_meta), desc="Loading positions TSV files"):
        wav_path = row['wav_file'].replace("\\", "/")
        channel = row['channel']
        mp3_wav_path = wav_path.replace('.wav', f'_ch{channel}.mp3')
      #  mp3_wav_path = "mp3/" + mp3_wav_path
        positions_dir = os.path.join(os.path.dirname(wav_path), 'positions')
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            position_files = os.listdir(os.path.join(DATA_DIR, positions_dir))
        except FileNotFoundError:
            continue
        pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')

        for file in position_files:
            if not pattern.match(file):
                continue
            tsv_path = os.path.join(DATA_DIR, positions_dir, file)
            if not os.path.isfile(tsv_path):
                continue

            dff = pd.read_csv(tsv_path, sep='\t')
            dff['day_dt'] = pd.to_datetime(row['day'], format="%Y-%m-%d", errors='raise')
            dff[['x', 'y']] = dff['embedding'].apply(lambda s: pd.Series(ast.literal_eval(s)))
            dff['start_dt'] = pd.to_datetime(row['start_time'], format='%H:%M:%S', errors='raise')
            dff['time_of_day'] = dff['start_dt'] + pd.to_timedelta(dff['clip_time'], unit='s')
            dff['start_hour_float'] = (
                dff['time_of_day'].dt.hour +
                dff['time_of_day'].dt.minute / 60 +
                dff['time_of_day'].dt.second / 3600
            )

            dff['wav_file'] = row['wav_file']
            dff['mp3_file'] = os.path.normpath(mp3_wav_path)
            dff['channel'] = channel

            c = file.split('.')[-3].split('_')[-1]
            try:
                dff['cluster_num'] = int(c)
            except (ValueError, TypeError):
                dff['cluster_num'] = 10

            dff['cluster_id'] = dff['cluster_id'].astype(int)
            dff['sunrise'] = row['sunrise']
            dff['sunset'] = row['sunset']
            dff['location'] = row['location']
            dff['microlocation'] = row['microlocation']
            dff['recorder_type'] = row['recorder_type']
            dff['channel_name'] = row['channel_name']

            dff['abs_file_name'] = mp3_wav_path

            dff = dff.merge(row.to_frame().T, how='left', on=['wav_file', 'channel'], suffixes=('', '_meta'))
            df = pd.concat([df, dff], ignore_index=True)

            if len(df) - last_saved_len >= batch_size:
                df.to_parquet(save_path)
                last_saved_len = len(df)
                print(f"Wrote {len(df)} rows to {save_path}")

    if len(df) > last_saved_len:
        df.to_parquet(save_path)

    # Remove unused columns as before
    columns_to_drop = [
        'day', 'start_time', 'num_clusters', 'embedding', 'day_meta', 'start_time_meta',
        'location_meta', 'microlocation_meta', 'channel_meta', 'channel_name_meta',
        'recorder_type_meta', 'sunrise_meta', 'sunset_meta'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

if not os.path.exists(SAVE_PATH):
    print("Loading positions TSV files...")
    df = load_positions_tsv(wav_meta, DATA_DIR, save_path=SAVE_PATH, batch_size=BATCH_SIZE)
else:
    df = pd.read_parquet(SAVE_PATH)
    print(f"Loaded cached data with {len(df)} rows.")