import re

import pandas as pd
import os
import ast

from tqdm import tqdm

DATA_DIR = os.path.abspath("data")
OVERVIEW_TSV = os.path.join(DATA_DIR, "Rok_spring_summer.tsv")

# Load the overview TSV as reference
wav_meta = pd.read_csv(OVERVIEW_TSV, sep='\t')
wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)

def load_positions_tsv(wav_meta, data_dir):
    df = pd.DataFrame()
    for idx, row in tqdm(wav_meta.iterrows(), total=len(wav_meta), desc="Loading positions TSV files"):
        wav_path = row['wav_file']
        channel = row['channel']
        mp3_wav_path = wav_path.replace('.wav', f'_ch{channel}.mp3')
        mp3_wav_path = "mp3/" + mp3_wav_path
        positions_dir = os.path.join(os.path.dirname(wav_path), 'positions')
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            position_files = os.listdir(os.path.join(DATA_DIR, positions_dir))
        except FileNotFoundError:
            continue
        pattern = re.compile(rf'^{re.escape(base_name)}.*\.tsv$')

        for file in position_files:
            if pattern.match(file):
                tsv_path = os.path.join(DATA_DIR, positions_dir, file)
                if not os.path.isfile(tsv_path):
                    continue

                dff = pd.read_csv(tsv_path, sep='\t')
                dff['day_dt'] = pd.to_datetime(row['day'], format="%Y-%m-%d", errors='raise')
                dff[['x', 'y']] = dff['embedding'].apply(lambda s: pd.Series(ast.literal_eval(s)))
                dff['start_dt'] = pd.to_datetime(row['start_time'], format='%H:%M:%S', errors='raise')
                dff['time_of_day'] = dff['start_dt'] + pd.to_timedelta(dff['clip_time'], unit='s')
                dff['start_hour_float'] = dff['time_of_day'].dt.hour + dff['time_of_day'].dt.minute / 60 + dff['time_of_day'].dt.second / 3600

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

                dff['manual_label'] = row['manual_label'] if 'manual_label' in dff.columns else 'unlabeled'

                dff['abs_file_name'] = mp3_wav_path

                dff = dff.merge(row.to_frame().T, how='left', on=['wav_file', 'channel'], suffixes=('', '_meta'))
                df = pd.concat([df, dff], ignore_index=True)

    # remove some columns that are not needed
    columns_to_drop = ['day', 'start_time', 'num_clusters', 'embedding', 'day_meta', 'start_time_meta', 'location_meta', 'microlocation_meta', 'channel_meta', 'channel_name_meta', 'recorder_type_meta', 'sunrise_meta', 'sunset_meta']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return df

if not os.path.exists("data/cache/final_data.parquet"):
    print("Loading positions TSV files...")
    df = load_positions_tsv(wav_meta, DATA_DIR)
    num = len(df)
    df.to_parquet(f"data/cache/final_data_{num}.parquet")
else:
    df = pd.read_parquet("data/cache/final_data.parquet")
    print(f"Loaded cached data with {len(df)} rows.")

