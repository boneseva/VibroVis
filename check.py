# go through Rok_spring_summer and check if all mp3 files are present
import re

import copy_positions
import os
import pandas as pd
import wav_to_mp3

OVERVIEW_TSV = os.path.join(r"C:\Users\evabo\OneDrive - Univerza v Ljubljani\Documents\Repos\VibroVis\data", "Rok_spring_summer.tsv")
DATA = r"\\voz.fri.uni-lj.si\Databases\Vibroscape\Nonannotated recordings\Rok spring summer\16k\mp3"

def check_mp3_files(overview_tsv):
    wav_meta = pd.read_csv(overview_tsv, sep='\t')
    wav_meta['wav_file'] = wav_meta['wav_file'].apply(os.path.normpath)

    missing_files = []
    for idx, row in wav_meta.iterrows():
        wav_path = os.path.join(DATA,row['wav_file'])
        channel = row['channel']
        mp3_wav_path = wav_path.replace('.wav', f'_ch{channel}.mp3')

        if not os.path.isfile(mp3_wav_path):
            missing_files.append(mp3_wav_path)

    return missing_files

if __name__ == '__main__':
    FIX = False

    missing_mp3_files = check_mp3_files(OVERVIEW_TSV)
    if missing_mp3_files:
        print("Missing MP3 files:")
        for file in missing_mp3_files:
            print(file)
    else:
        print("All MP3 files are present.")

    # try to wav to mp3 the missing files
    if FIX:
        for file in missing_mp3_files:
            wav_file = re.sub(r'_ch\d+', '.wav', file).replace('.mp3', '')
            wav_file = wav_file.replace('\\mp3', '')
            if os.path.isfile(wav_file):
                print(f"Converting {wav_file} to MP3...")
                wav_to_mp3.split_multichannel_wav_to_mp3_ffmpeg(wav_file, output_dir=os.path.dirname(file))
            else:
                print(f"WAV file not found for {file}: {wav_file}")