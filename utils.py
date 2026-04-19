# Updated utils.py
# Optimized for Random Access - Selective segment loading without full-file caching

import soundfile as sf
import numpy as np
import os
from scipy.signal import spectrogram
from collections import Counter
import read_data  # Import read_data to access DATA_DIR
import librosa  # Make sure librosa is imported


# --- *** OPTIMIZED SELECTIVE AUDIO LOADING *** ---
def load_audio_segment(mp3_file_relative_path, clip_time, clip_duration, channel=0, padding_s=1.0):
    """
    Loads a specific audio segment using optimized selective loading.
    Perfect for random access across many different files.
    Uses soundfile/librosa offset parameters to skip file decoding.
    """
    mp3_file_relative_path = mp3_file_relative_path.replace("\\", "/")
    full_path = os.path.join(read_data.DATA_DIR, mp3_file_relative_path)
    
    if not os.path.exists(full_path):
        print(f"Warning: Audio file not found at {full_path}")
        return None, None

    try:
        # Suppress stderr from mpg123 decoder to prevent console clutter
        import sys
        import contextlib
        
        @contextlib.contextmanager
        def suppress_stderr():
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr

        with suppress_stderr():
            # OPTIMIZED: Get file info without loading data
            info = sf.info(full_path)
            samplerate = info.samplerate
            total_frames = info.frames
            num_channels = info.channels

            # Calculate start and end samples with padding
            start_sample = int(max(0, (clip_time - padding_s) * samplerate))
            end_sample = int(min(total_frames, (clip_time + clip_duration + padding_s) * samplerate))

            # EFFICIENCY TWEAK: Use start/stop parameters for selective loading
            # This allows the C-engine to skip decoding most of the file
            segment, _ = sf.read(full_path, start=start_sample, stop=end_sample, dtype='float32')

        # Handle mono vs multi-channel after loading the small segment
        if segment.ndim > 1:
            if channel < num_channels:
                channel_data = segment[:, channel]
            else:
                # Fallback to first channel
                channel_data = segment[:, 0]
        else:
            channel_data = segment

        return channel_data, samplerate

    except Exception as e:
        # Fallback to librosa with selective loading
        try:
            with suppress_stderr():
                # Calculate offset and duration for librosa
                offset_seconds = max(0, clip_time - padding_s)
                duration_seconds = clip_duration + (2 * padding_s)
                
                # EFFICIENCY TWEAK: Use offset and duration for selective loading
                y, sr = librosa.load(full_path, sr=None, offset=offset_seconds, 
                                   duration=duration_seconds, mono=False)
                
                # Handle multi-channel selection
                if y.ndim > 1:
                    if channel < y.shape[0]:
                        segment = y[channel, :]
                    else:
                        segment = y[0, :]  # Fallback to first channel
                else:
                    segment = y
                    
                return segment, sr
                
        except Exception as e2:
            print(f"Error loading audio segment from {full_path}: {e2}")
            return None, None


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
def apply_manual_labels_efficiently(dff, manual_labels_cache=None, manual_labels_lock=None):
    """
    Apply manual labels to dff using a high-performance vectorized approach.
    Optimized to handle 1M+ rows without redundant string scans or console flooding.
    """
    # Import here to avoid circular dependency
    if manual_labels_cache is None:
        from callbacks.callbacks_constants import MANUAL_LABELS_CACHE
        manual_labels_cache = MANUAL_LABELS_CACHE
    
    if manual_labels_lock is None:
        from callbacks.callbacks_constants import MANUAL_LABELS_LOCK
        manual_labels_lock = MANUAL_LABELS_LOCK
    
    # THREAD SAFETY: Use lock to prevent concurrent access during read
    with manual_labels_lock:
        if not manual_labels_cache or dff.empty:
            dff['manual_label'] = 'Unlabeled'
            return dff
        
        # Create a snapshot of the cache to work with outside the lock
        # For diskcache, we need to safely iterate and handle missing keys
        cache_snapshot = {}
        try:
            # Use list() to create a snapshot of keys first to avoid concurrent modification
            cache_keys = list(manual_labels_cache.iterkeys())
            for key in cache_keys:
                try:
                    cache_snapshot[key] = manual_labels_cache[key]
                except KeyError:
                    # Key was deleted during iteration, skip it
                    continue
        except Exception:
            # If anything fails, return the dataframe with unlabeled entries
            dff['manual_label'] = 'Unlabeled'
            return dff

    # Initialize with Unlabeled
    if 'manual_label' not in dff.columns:
        dff['manual_label'] = 'Unlabeled'
    else:
        dff['manual_label'] = 'Unlabeled'

    # 0. Pre-filter the cache to ONLY those relevant to the current location/microlocation
    # Most dffs in the app only represent one location at a time.
    current_locations = dff['location'].unique() if 'location' in dff.columns else []

    # 1. Fast optimization: group labels by (loc, micro, f_base, chan)
    label_map = {}
    for key_tuple in cache_snapshot:
        label = cache_snapshot[key_tuple]
        if len(key_tuple) == 5:
            loc, micro, f_base, chan, sec = key_tuple

            # Skip if not in current data view (heavy optimization)
            if len(current_locations) > 0 and loc not in current_locations:
                continue

            group_key = (loc, micro, f_base, chan)
            if group_key not in label_map:
                label_map[group_key] = {}
            label_map[group_key][sec] = label

    if not label_map:
        return dff

    # 2. Vectorized extraction of basenames (DO IT ONCE, NOT IN A LOOP)
    # This is the most expensive part on 1M rows, avoid doing it for every label.
    if 'f_basename_cache' not in dff.columns:
        if 'mp3_file' in dff.columns:
            # Cross-platform basename extraction
            dff['f_basename_cache'] = dff['mp3_file'].astype(str).str.replace('\\', '/').str.split('/').str[-1]
        elif 'file_name' in dff.columns:
            dff['f_basename_cache'] = dff['file_name'].astype(str).str.replace('\\', '/').str.split('/').str[-1]
        else:
            return dff # Cannot match without filenames

    # 3. Iterate over the SMALL set of labeled groups (e.g. 10 groups vs 1M rows)
    for (loc, micro, f_base, chan), sec_map in label_map.items():
        try:
             # Fast vectorized mask preparation
             # Channel match (fastest)
             chan_val = int(chan)
             if dff['channel'].dtype.name == 'category':
                  mask = (dff['channel'].astype(int) == chan_val)
             else:
                  mask = (dff['channel'] == chan_val)

             # Group comparison (Avoid loc/micro check if they are already unique in dff to save time)
             if 'location' in dff.columns:
                 mask &= (dff['location'].fillna('Unknown') == loc)
             if 'microlocation' in dff.columns:
                 mask &= (dff['microlocation'].fillna('Unknown') == micro)

             # Basename match (Now O(1) string check per row using cached column)
             mask &= (dff['f_basename_cache'] == f_base)

             # Get matching indices
             indices = dff.index[mask]
             if indices.empty:
                 continue

             # Only iterate the tiny subset of matching rows
             for idx in indices:
                 row_start = float(dff.at[idx, 'clip_time'])
                 row_dur = float(dff.at[idx, 'clip_duration']) if 'clip_duration' in dff.columns else 5.0

                 # CRITICAL FIX: Use round() to match callbacks_plots.py key generation
                 start_sec = round(row_start)
                 end_sec = round(row_start + row_dur) - 1

                 # Handle edge case where very short clips might have end_sec < start_sec
                 if end_sec < start_sec:
                     end_sec = start_sec

                 found = []
                 for s in range(start_sec, end_sec + 1):
                     l = sec_map.get(int(s))  # Ensure int type for key consistency
                     if l and l != 'Unlabeled':
                         found.append(l)

                 if found:
                     final_label = Counter(found).most_common(1)[0][0]
                     dff.at[idx, 'manual_label'] = final_label

        except Exception:
            continue

    return dff