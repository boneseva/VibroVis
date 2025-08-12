import pandas as pd
import os
import ast

DATA_DIR = os.path.abspath("data")
OVERALL_TSV = os.path.join(DATA_DIR, "Rok_spring_summer.tsv")

wav_meta = pd.read_csv(OVERALL_TSV, sep='\t')
wav_meta['wav_file'] = wav_meta['wav_file'].apply(lambda x: os.path.normpath(x))

def load_positions_tsv(tsv_dir, wav_meta):
    df = pd.DataFrame()
    tsv_root = os.path.normpath(tsv_dir)
    for root, dirs, files in os.walk(tsv_dir):
        for dir in dirs:
            if dir == 'positions':
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith('.tsv'):
                        tsv_path = os.path.join(root, dir, file)
                        print(".")
                        dff = pd.read_csv(tsv_path, sep='\t')
                        dff['day_str'] = dff['day'].astype(str).str.zfill(6)
                        dff['day_dt'] = pd.to_datetime(dff['day_str'], format='%y%m%d')
                        dff[['x', 'y']] = dff['embedding'].apply(lambda s: pd.Series(ast.literal_eval(s)))
                        dff['abs_file_name'] = dff['file_name'].apply(lambda x: os.path.relpath(os.path.normpath(os.path.join(root, x)), tsv_root))
                        dff['start_dt'] = pd.to_datetime(dff['start_time'], format='%H%M%S')
                        dff['time_of_day'] = dff['start_dt'] + pd.to_timedelta(dff['clip_time'], unit='s')
                        c = file.split('.')[-3].split('_')[-1]
                        dff['start_hour_float'] = dff['time_of_day'].dt.hour + dff['time_of_day'].dt.minute / 60 + dff['time_of_day'].dt.second / 3600
                        dff['cluster_num'] = c if c.isnumeric() else 10
                        dff['cluster_id'] = dff['cluster_id'].astype(int)
                        dff['wav_file'] = dff['file_name'].apply(lambda x: os.path.normpath(os.path.join(root, x)))
                        dff = dff.merge(wav_meta, how='left', on=['wav_file', 'channel'], suffixes=('', '_meta'))
                        df = pd.concat([df, dff], ignore_index=True)
    return df

df = load_positions_tsv(DATA_DIR, wav_meta)