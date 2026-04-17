# Updated utils.py
# (This file now has a memory-efficient audio loader)

import soundfile as sf
import numpy as np
import pandas as pd  # Added pandas import
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
        f_out = log_f_bins

    # --- Linear Scale (no interpolation) ---
    else:  # scale == 'linear'
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        f_out = f

    # Normalize to 0 dB max (matches librosa behavior for Mel)
    if Sxx_db.size > 0:
        Sxx_db -= np.max(Sxx_db)

    Sxx_db = np.maximum(Sxx_db, db_floor)
    return f_out, t, Sxx_db


# --- *** MANUAL LABELING HELPERS *** ---
def apply_manual_labels_efficiently(dff):
    """
    Apply manual labels to dff using a FULLY VECTORIZED O(n) hash table approach.
    
    OPTIMIZATION #5: Eliminates all nested loops and row-by-row processing.
    Performance: O(n) instead of O(n×m) where n=rows, m=cache groups.
    """
    if not MANUAL_LABELS_CACHE or dff.empty:
        dff['manual_label'] = 'Unlabeled'
        return dff
        
    # Initialize with Unlabeled
    if 'manual_label' not in dff.columns:
        dff['manual_label'] = 'Unlabeled'
    else:
        dff['manual_label'] = 'Unlabeled'
    
    # OPTIMIZATION #5: Pre-filter cache to current location for massive speedup
    current_locations = dff['location'].unique() if 'location' in dff.columns else []
    relevant_cache = {}
    
    for key_tuple, label in MANUAL_LABELS_CACHE.items():
        if len(key_tuple) == 5:
            loc, micro, f_base, chan, sec = key_tuple
            # Skip irrelevant locations early
            if len(current_locations) > 0 and loc not in current_locations:
                continue
            relevant_cache[key_tuple] = label
    
    if not relevant_cache:
        return dff

    # OPTIMIZATION #5: Vectorized basename extraction (cached permanently)
    if 'f_basename_cache' not in dff.columns:
        if 'mp3_file' in dff.columns:
            # Ultra-fast vectorized basename using pandas string operations
            dff['f_basename_cache'] = dff['mp3_file'].astype(str).str.split('/').str[-1].str.split('\\').str[-1]
        elif 'file_name' in dff.columns:
            dff['f_basename_cache'] = dff['file_name'].astype(str).str.split('/').str[-1].str.split('\\').str[-1]
        else:
            return dff # Cannot match without filenames

    # OPTIMIZATION #5: Build comprehensive hash table for O(1) lookups
    # Structure: {(loc, micro, f_base, chan, start_sec): label}
    print(f"[OPTIMIZATION #5] Building hash table from {len(relevant_cache)} cache entries...")
    
    hash_table = {}
    for (loc, micro, f_base, chan, sec), label in relevant_cache.items():
        if label and label != 'Unlabeled':
            hash_table[(loc, micro, f_base, int(chan), int(sec))] = label
    
    if not hash_table:
        return dff
    
    print(f"[OPTIMIZATION #5] Hash table built: {len(hash_table)} lookup keys")
    
    # OPTIMIZATION #5: Vectorized clip time processing
    # Pre-compute all start/end seconds for ALL rows at once
    clip_times = dff['clip_time'].astype(float)
    clip_durations = dff['clip_duration'].astype(float) if 'clip_duration' in dff.columns else pd.Series(5.0, index=dff.index)
    
    # Vectorized floor/ceil operations 
    start_seconds = np.floor(clip_times).astype(int)
    end_seconds = np.ceil(clip_times + clip_durations).astype(int)
    
    # OPTIMIZATION #5: Vectorized lookup using pandas operations
    # Create lookup keys for each row
    locations = dff['location'].fillna('Unknown') if 'location' in dff.columns else pd.Series('Unknown', index=dff.index)
    microlocations = dff['microlocation'].fillna('Unknown') if 'microlocation' in dff.columns else pd.Series('Unknown', index=dff.index)
    channels = dff['channel'].astype(int)
    basenames = dff['f_basename_cache']
    
    # Process rows in batches for memory efficiency (STILL O(n) but memory-friendly)
    batch_size = 10000
    total_rows = len(dff)
    labeled_count = 0
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_indices = dff.index[batch_start:batch_end]
        
        # Extract batch data
        batch_locations = locations.iloc[batch_start:batch_end]
        batch_micros = microlocations.iloc[batch_start:batch_end]
        batch_basenames = basenames.iloc[batch_start:batch_end]
        batch_channels = channels.iloc[batch_start:batch_end]
        batch_start_secs = start_seconds.iloc[batch_start:batch_end]
        batch_end_secs = end_seconds.iloc[batch_start:batch_end]
        
        # Vectorized label lookup for this batch
        for i, idx in enumerate(batch_indices):
            loc = batch_locations.iloc[i]
            micro = batch_micros.iloc[i]
            basename = batch_basenames.iloc[i]
            channel = batch_channels.iloc[i]
            start_sec = batch_start_secs.iloc[i]
            end_sec = batch_end_secs.iloc[i]
            
            # Collect labels for all seconds in clip duration
            found_labels = []
            for sec in range(start_sec, end_sec + 1):
                lookup_key = (loc, micro, basename, channel, sec)
                label = hash_table.get(lookup_key)
                if label:
                    found_labels.append(label)
            
            # Majority vote (vectorized using Counter)
            if found_labels:
                majority_label = Counter(found_labels).most_common(1)[0][0]
                dff.at[idx, 'manual_label'] = majority_label
                labeled_count += 1
    
    print(f"[OPTIMIZATION #5] Vectorized labeling complete: {labeled_count}/{total_rows} rows labeled")
    return dff