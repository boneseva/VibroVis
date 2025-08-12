import os
import subprocess

def split_multichannel_wav_to_mp3_ffmpeg(wav_path, output_dir='mp3'):
    os.makedirs(output_dir, exist_ok=True)

    # Get the number of channels using ffprobe
    ffprobe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=channels',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        wav_path
    ]
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ffprobe error: {result.stderr}")
        return

    channels = int(result.stdout.strip())
    filter_labels = ''.join([f'[ch{i + 1}]' for i in range(channels)])
    filter_complex = get_channel_split_option(channels, filter_labels)

    name = os.path.splitext(os.path.basename(wav_path))[0]
    cmd = ['ffmpeg', '-y', '-i', wav_path, '-filter_complex', filter_complex]
    for i in range(channels):
        mp3_filename = os.path.join(output_dir, f'{name}_ch{i + 1}.mp3')
        cmd += ['-map', f'[ch{i + 1}]', '-c:a', 'libmp3lame', mp3_filename]

    subprocess.run(cmd)

def get_channel_split_option(channels, filter_labels):
    # Map common channel counts to recognized names where possible; else use channels=
    known_layouts = {
        1: 'mono', 2: 'stereo', 4: 'quad', 5: '5.1',
        6: '5.1', 7: '7.1', 8: '7.1'
    }
    return (f"channelsplit=channel_layout={known_layouts[channels]}{filter_labels}"
            if channels in known_layouts
            else f"channelsplit=channels={channels}{filter_labels}")

if __name__ == '__main__':
    datapath = r"C:\Users\evabo\OneDrive - Univerza v Ljubljani\Documents\Repos\VibroVis\data_test"
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                print(f"Processing {wav_path}...")
                rootmp3 = root.replace("\\data_test\\", "\\data_test\\mp3\\")
                split_multichannel_wav_to_mp3_ffmpeg(wav_path, output_dir=rootmp3)