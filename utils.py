# Updated utils.py
# (This file now has a memory-efficient audio loader)

import soundfile as sf
import numpy as np
import os
from scipy.signal import spectrogram
import math
from collections import Counter
import read_data  # Import read_data to access DATA_DIR
import librosa  # Make sure librosa is imported
from callbacks.callbacks_constants import MANUAL_LABELS_CACHE


# --- *** NEW, MEMORY-EFFICIENT FUNCTION *** ---
def load_audio_segment(mp3_file_relative_path, clip_time, clip_duration, channel=0, padding_s=1.0):
    """
    Loads a specific audio segment from a file *without* loading the
    entire file into memory.
    """
    mp3_file_relative_path = mp3_file_relative_path.replace("\\", "/")
    full_path = os.path.join(read_data.DATA_DIR, mp3_file_relative_path)
    if not os.path.exists(full_path):
        print(f"Warning: Audio file not found at {full_path}")
        return None, None

    try:
        # Suppress stderr from mpg123 decoder (redirects to devnull)
        with open(os.devnull, 'w') as devnull:
            # Note: direct C-level stderr redirection requires more complex handling (os.dup2)
            # but this might catch Python-level propagated errors or if library uses sys.stderr
            # For pure C libraries like libmpg123, we often need C-level redirection.
            # However, we'll restore the previous user attempt first.
            from contextlib import redirect_stderr
            with redirect_stderr(devnull):
                 # 1. Get file info without loading data
                info = sf.info(full_path)
                samplerate = info.samplerate
                total_frames = info.frames
                num_channels = info.channels

                # 2. Calculate start and end samples with padding
                start_sample = int(max(0, (clip_time - padding_s)) * samplerate)
                end_sample = int(min(total_frames, (clip_time + clip_duration + padding_s) * samplerate))

                # 3. Read *only* the required segment from disk
                # We read as float32 for consistency with librosa
                segment, _ = sf.read(full_path, start=start_sample, stop=end_sample, dtype='float32')

        # 4. Handle mono vs multi-channel *after* loading the small segment
        if segment.ndim > 1:
            if channel < num_channels:
                channel_data = segment[:, channel]
            else:  # Fallback to first channel
                channel_data = segment[:, 0]
        else:
            channel_data = segment

        return channel_data, samplerate

    except Exception as e:
        print(f"Error loading with soundfile: {e}. Trying librosa...")
        try:
             # Fallback to librosa (loads entire file, slower but more robust)
             # librosa.load returns (y, sr)
             # We load with original sr to match behavior, or just let librosa decide
             y, sr = librosa.load(full_path, sr=None) 
             
             # Calculate samples
             start_sample = int(max(0, (clip_time - padding_s)) * sr)
             end_sample = int(min(len(y), (clip_time + clip_duration + padding_s) * sr))
             
             segment = y[start_sample:end_sample]
             
             # Handle multi-channel if librosa loaded it (librosa usually loads mono by default unless mono=False)
             # Let's force load as mono for simplicity in fallback, or handle channels if needed.
             # VibroVis seems to rely on channels, so let's try reading multi-channel.
             y_multi, sr_multi = librosa.load(full_path, sr=None, mono=False)
             
             if y_multi.ndim > 1:
                  if channel < y_multi.shape[0]:
                       segment = y_multi[channel, start_sample:end_sample]
                  else:
                       segment = y_multi[0, start_sample:end_sample]
             else:
                  segment = y_multi[start_sample:end_sample]
                  
             return segment, sr_multi

        except Exception as e2:
             print(f"Error loading audio segment from {full_path} with librosa: {e2}")
             return None, None
# --- *** END OF NEW FUNCTION *** ---


def compute_spectrogram(segment, samplerate, scale='log', fft_window_size=1024, window_overlap=0.5,
                        window_type='hann', min_freq=50, max_freq=5000, num_bins=256, db_floor=-100, **kwargs):
    """
    Computes the spectrogram with selectable frequency scales (Linear, Log, Mel).
    (This function is unchanged)
    """
    if segment is None or len(segment) == 0:
        # Segment empty, returning empty.
        return np.array([]), np.array([]), np.array([])

    if np.isnan(segment).any():
        # NaN values found in audio segment! Replacing with zeros.
        segment = np.nan_to_num(segment)
    
    # Calculate step size from overlap
    try:
        fft_step_size = int(fft_window_size * (1 - window_overlap))
        # Calculate nfft as the next power of 2 for efficiency
        nfft = int(2 ** np.ceil(np.log2(fft_window_size)))
    except Exception as e:
        # Error calculating FFT params: {e}
        return np.array([]), np.array([]), np.array([])

    # --- Mel Scale Spectrogram ---
    if scale == 'mel':
        try:
            # Use librosa for Mel scale
            Sxx_mel = librosa.feature.melspectrogram(
                y=segment, sr=samplerate, n_fft=nfft,
                hop_length=fft_step_size, win_length=fft_window_size, window=window_type,
                n_mels=int(num_bins), fmin=float(min_freq), fmax=float(max_freq)
            )
            Sxx_db = librosa.power_to_db(Sxx_mel, ref=np.max)
            Sxx_db = np.maximum(Sxx_db, db_floor)

            # Get the time and frequency axes for plotting
            t = librosa.times_like(Sxx_db, sr=samplerate, hop_length=fft_step_size, n_fft=nfft)
            f = librosa.mel_frequencies(n_mels=int(num_bins), fmin=float(min_freq), fmax=float(max_freq))
            return f, t, Sxx_db
        except Exception as e:
            print(f"ERROR in librosa.feature.melspectrogram: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), np.array([])

    # --- Linear & Log Scale Spectrograms ---
    f, t, Sxx = spectrogram(
        segment, fs=samplerate, nperseg=fft_window_size,
        noverlap=(fft_window_size - fft_step_size), nfft=nfft, window=window_type
    )
    print(f"DEBUG: Scipy spectrogram done. Sxx shape={Sxx.shape}")

    # Filter by frequency range
    freq_slice = np.where((f >= min_freq) & (f <= max_freq))
    f = f[freq_slice]
    Sxx = Sxx[freq_slice, :]
    
    # Sxx might allow squeezing if frequency bins are 1, but we usually want 2D.
    # If Sxx is 3D due to some weirdness, squeeze. But standard spectorgram is 2D.
    # The previous code had [0] which took the first ROW (first frequency bin), discarding the rest.
    if Sxx.ndim > 2:
        Sxx = Sxx.squeeze()

    if Sxx.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    f_out = f # Default assignment

    # --- Log Scale (with interpolation) ---
    if scale == 'log':
        if min_freq <= 0: min_freq = 1  # Avoid log(0)
        log_f_bins = np.logspace(np.log10(min_freq), np.log10(max_freq), num=num_bins)
        Sxx_log_interpolated = np.zeros((num_bins, Sxx.shape[1]))
        
        for i in range(Sxx.shape[1]):
            Sxx_log_interpolated[:, i] = np.interp(log_f_bins, f, Sxx[:, i])

        Sxx_db = 10 * np.log10(Sxx_log_interpolated + 1e-10)
    
        Sxx_db -= Sxx_db.max()
        f_out = log_f_bins

    # --- Linear Scale (no interpolation) ---
    else:  # scale == 'linear'
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        Sxx_db -= Sxx_db.max()
        f_out = f

    # Normalize to 0 dB max (matches librosa behavior for Mel)
    if Sxx_db.size > 0:
        Sxx_db -= np.max(Sxx_db)

    Sxx_db = np.maximum(Sxx_db, db_floor)
    return f_out, t, Sxx_db


# --- *** MANUAL LABELING HELPERS *** ---
def apply_manual_labels_efficiently(dff):
    """
    Apply manual labels to dff using an inverted loop strategy.
    Instead of iterating all rows, iterate likely labeled groups.
    MOVED TO UTILS to be shared between callbacks.
    """
    if not MANUAL_LABELS_CACHE or dff.empty:
        dff['manual_label'] = 'Unlabeled'
        return dff
        
    # Initialize with Unlabeled
    dff['manual_label'] = 'Unlabeled'
    
    # Fast optimization: pre-calculate label map
    label_map = {}
    if MANUAL_LABELS_CACHE:
        # Key format: (location, microlocation, f_base, chan, sec)
        for key_tuple, label in MANUAL_LABELS_CACHE.items():
            if len(key_tuple) == 5:
                loc, micro, f_base, chan, sec = key_tuple
                
                # Group by (loc, micro, f_base, chan)
                group_key = (loc, micro, f_base, chan)
                if group_key not in label_map:
                    label_map[group_key] = {}
                label_map[group_key][sec] = label
            # Add backwards compatibility check if needed, or simply let old keys fail/drop
    
    if not label_map:
        dff['manual_label'] = 'Unlabeled'
        return dff
        
    # Initialize with Unlabeled
    dff['manual_label'] = 'Unlabeled'
    
    # Iterate over labeled files only
    for (loc, micro, f_base, chan), sec_map in label_map.items():
        # Fast filter: channel is usually integer or char
        try:
             # Mask: Channel AND File Basename AND Location AND Microlocation
             
             # Channel match
             if dff['channel'].dtype.name == 'category':
                  mask = (dff['channel'].astype(int) == int(chan))
             else:
                  mask = (dff['channel'] == int(chan))
             
             # Location match (if column exists, usually does)
             if 'location' in dff.columns:
                 # Handle NaN vs Unknown
                 # dff is a copy, safe to modify or just use fillna in comparison
                 mask &= (dff['location'].fillna('Unknown') == loc)
             
             # Microlocation match
             if 'microlocation' in dff.columns:
                 mask &= (dff['microlocation'].fillna('Unknown') == micro)
             
             # File Basename match
             mask &= dff['mp3_file'].astype(str).str.endswith(f_base)
             
             # Get indices
             indices = dff.index[mask]
             
             if len(indices) == 0:
                 continue
                 
             # Iterate only the relevant rows
             # This is much faster (e.g. 100 rows vs 700k)
             for idx in indices:
                 row_start = float(dff.at[idx, 'clip_time'])
                 row_dur = float(dff.at[idx, 'clip_duration']) if 'clip_duration' in dff.columns else 5.0
                 
                 # Majority vote using the sec_map directly
                 start_second = math.floor(row_start)
                 end_second = math.ceil(row_start + row_dur)
                 
                 found_labels = []
                 for sec in range(start_second, end_second):
                     l = sec_map.get(sec)
                     if l and l != 'Unlabeled':
                         found_labels.append(l)
                 
                 if found_labels:
                     # Filter valid (ignore 'Unlabeled' if it snuck in)
                     valid_labels = [l for l in found_labels if l != 'Unlabeled']
                     
                     # NEW LOGIC: Require >50% overlap
                     if len(valid_labels) >= (0.5 * row_dur):
                         # Majority vote
                         final_label = Counter(valid_labels).most_common(1)[0][0]
                         dff.at[idx, 'manual_label'] = final_label
                     else:
                          # Explicitly Unlabeled if threshold not met
                          dff.at[idx, 'manual_label'] = 'Unlabeled'
                     
        except Exception as e:
            # Fallback or ignore errors in optimization to allow other rows to proceed
            print(f"Error applying labels for {f_base}: {e}")
            continue

    return dff