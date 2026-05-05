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

    Performance tip: pass a pre-built plain ``dict`` as *manual_labels_cache* to skip
    lock acquisition and diskcache I/O entirely.  Build it once per callback with::

        with MANUAL_LABELS_LOCK:
            snapshot = dict(MANUAL_LABELS_CACHE)          # or iterate iterkeys()
        dff = apply_manual_labels_efficiently(dff, snapshot)
    """
    # Fast path: caller already provided a plain-dict snapshot — no lock, no disk I/O.
    if isinstance(manual_labels_cache, dict):
        if not manual_labels_cache or dff.empty:
            dff['manual_label'] = 'Unlabeled'
            return dff
        cache_snapshot = manual_labels_cache
    else:
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

    # 1. Import the duration cache (plain dict, no lock needed).
    #    Keys are the same 5-tuples as MANUAL_LABELS_CACHE.
    #    Values are float clip durations written at label-save time.
    #    Falls back gracefully to an empty dict if not available.
    try:
        from callbacks.callbacks_constants import CLIP_DURATION_CACHE as _dur_cache
    except Exception:
        _dur_cache = {}

    # 2. Group labels by (loc, micro, f_base, chan).
    #    Key precision: round(clip_time * 10) → 0.1 s units (stored as int).
    #    Each entry: clip_time_x10 → (label, clip_end_x10_or_None, labeled_model_or_None)
    #    clip_end_x10 / labeled_model are None for legacy entries with no duration recorded.
    label_map: dict = {}
    for key_tuple in cache_snapshot:
        label = cache_snapshot[key_tuple]
        if len(key_tuple) == 5:
            loc, micro, f_base, chan, clip_time_key = key_tuple

            # Skip if not relevant to current data view
            if len(current_locations) > 0 and loc not in current_locations:
                continue

            ct_x10 = int(clip_time_key)
            dur_entry = _dur_cache.get(key_tuple)
            if isinstance(dur_entry, tuple) and len(dur_entry) == 2:
                dur, labeled_model = dur_entry
                clip_end_x10 = ct_x10 + round(dur * 10)
            else:
                clip_end_x10 = None
                labeled_model = None

            group_key = (loc, micro, f_base, chan)
            if group_key not in label_map:
                label_map[group_key] = {}
            label_map[group_key][ct_x10] = (label, clip_end_x10, labeled_model)

    if not label_map:
        return dff

    # 3. Vectorized extraction of basenames (done once, not per label group)
    if 'f_basename_cache' not in dff.columns:
        if 'mp3_file' in dff.columns:
            # Cross-platform basename extraction
            dff['f_basename_cache'] = dff['mp3_file'].astype(str).str.replace('\\', '/').str.split('/').str[-1]
        elif 'file_name' in dff.columns:
            dff['f_basename_cache'] = dff['file_name'].astype(str).str.replace('\\', '/').str.split('/').str[-1]
        else:
            return dff # Cannot match without filenames

    # 4. Iterate over the SMALL set of labeled groups, match candidates by the rule:
    #
    #   • Same model as the label was created in  →  EXACT clip_time_x10 match only.
    #     Prevents bleeding to adjacent/overlapping clips within the same model.
    #
    #   • Different model (cross-model transfer)  →  OVERLAP-based match.
    #     A candidate receives the label when:
    #       overlap / duration_of_SHORTER_clip  >=  OVERLAP_THRESHOLD (50%)
    #
    #   • Legacy entry (no duration/model stored) →  exact clip_time_x10 match (safe fallback).
    OVERLAP_THRESHOLD = 0.5

    # Determine the model currently loaded in dff (used for same/different model check)
    current_model = str(dff['model_name'].iloc[0]) if 'model_name' in dff.columns and not dff.empty else None

    # Micro-optimization: bucket labeled clips into coarse time bins so each candidate
    # only compares with nearby labeled clips. This avoids N*L worst-case behavior
    # when labels are widely spread but candidates are concentrated.
    BIN_SIZE_X10 = 10  # 10 units = 1.0 second at the x10 scale (0.1s resolution)

    for (loc, micro, f_base, chan), clip_map in label_map.items():
        try:
            chan_val = int(chan)
            if dff['channel'].dtype.name == 'category':
                mask = (dff['channel'].astype(int) == chan_val)
            else:
                mask = (dff['channel'] == chan_val)

            if 'location' in dff.columns:
                mask &= (dff['location'].fillna('Unknown') == loc)
            if 'microlocation' in dff.columns:
                mask &= (dff['microlocation'].fillna('Unknown') == micro)

            mask &= (dff['f_basename_cache'] == f_base)

            indices = dff.index[mask]
            if indices.empty:
                continue

            # Vectorize candidate starts/ends for this group
            cand_cts = (dff.loc[indices, 'clip_time'].astype(float) * 10).round().astype(int).to_numpy()
            cand_cds = (dff.loc[indices, 'clip_duration'].astype(float) * 10).round().astype(int).to_numpy()
            cand_starts = cand_cts
            cand_ends = cand_cts + cand_cds

            # Build labeled items list and bucket them into BINs
            labeled_items = []
            max_labeled_span = 0
            for labeled_start, (lbl, labeled_end, labeled_model) in clip_map.items():
                span = (labeled_end - labeled_start) if (labeled_end is not None) else 0
                if span > max_labeled_span:
                    max_labeled_span = span
                labeled_items.append((labeled_start, lbl, labeled_end, labeled_model))

            # If there are no labeled items (should not happen), continue
            if not labeled_items:
                continue

            # Create bins: int -> list of labeled_items indices
            bin_map = {}
            for item in labeled_items:
                lstart = item[0]
                bin_idx = lstart // BIN_SIZE_X10
                bin_map.setdefault(bin_idx, []).append(item)

            # Number of neighboring bins to check based on max span (at least 1)
            neighbor_bins = max(1, int(np.ceil(max_labeled_span / BIN_SIZE_X10)))

            # Micro-prefilter: limit candidates to those that overlap the overall
            # labeled span expanded by max_labeled_span. This avoids checking
            # candidates that are far away in time from any labeled clip.
            labeled_starts = [it[0] for it in labeled_items]
            labeled_ends = [it[2] if it[2] is not None else it[0] for it in labeled_items]
            min_labeled_start = min(labeled_starts)
            max_labeled_end = max(labeled_ends)
            margin = max_labeled_span

            # Boolean mask (on the candidates arrays) selecting only nearby candidates
            cand_keep_mask = (cand_ends >= (min_labeled_start - margin)) & (cand_starts <= (max_labeled_end + margin))
            if not cand_keep_mask.any():
                continue

            # Reduce the indices and candidate arrays to the prefiltered subset
            filtered_indices = indices[cand_keep_mask]
            filtered_starts = cand_starts[cand_keep_mask]
            filtered_ends = cand_ends[cand_keep_mask]

            # Build mapping from labeled items → best candidate (single-best per labeled clip)
            # and then resolve conflicts so each candidate receives at most one label (highest overlap wins).
            num_cands = len(filtered_indices)
            if num_cands == 0:
                continue

            cand_st = np.array(filtered_starts, dtype=int)
            cand_en = np.array(filtered_ends, dtype=int)
            cand_span = cand_en - cand_st

            # Candidate assignment map: pos_in_filtered -> (label, overlap_frac)
            candidate_assignments = {}

            # For each labeled item, find the single best candidate (by overlap fraction)
            for labeled_start, lbl, labeled_end, labeled_model in labeled_items:
                labeled_start = int(labeled_start)
                if labeled_end is None:
                    # Legacy entry: require exact clip_time match
                    labeled_end = labeled_start
                labeled_end = int(labeled_end)

                same_model = (labeled_model is not None and current_model is not None and labeled_model == current_model)

                if labeled_end == labeled_start or same_model:
                    # Exact match only — find candidate(s) with identical start
                    matches = np.nonzero(cand_st == labeled_start)[0]
                    if matches.size > 0:
                        # Choose the first matching candidate (stable) — treat as full overlap
                        pos = int(matches[0])
                        # Mark assignment with overlap fraction 1.0 to take precedence
                        prev = candidate_assignments.get(pos)
                        if prev is None or 1.0 > prev[1]:
                            candidate_assignments[pos] = (lbl, 1.0)
                    # If no exact match, nothing to do for this labeled clip
                    continue

                # Cross-model labeled clip: compute overlap fractions against filtered candidates
                # overlap_x10 = max(0, min(cand_end, labeled_end) - max(cand_start, labeled_start))
                overlap = np.maximum(0, np.minimum(cand_en, labeled_end) - np.maximum(cand_st, labeled_start))
                if overlap.max() <= 0:
                    continue

                labeled_span = labeled_end - labeled_start
                shorter = np.minimum(labeled_span, cand_span)
                valid_mask = shorter > 0
                if not valid_mask.any():
                    continue

                frac = np.zeros_like(overlap, dtype=float)
                frac[valid_mask] = overlap[valid_mask] / shorter[valid_mask]

                # Pick the candidate index with maximum fraction
                best_pos = int(np.argmax(frac))
                best_frac = float(frac[best_pos])
                if best_frac >= OVERLAP_THRESHOLD:
                    prev = candidate_assignments.get(best_pos)
                    # If candidate already assigned, prefer the label with higher overlap fraction
                    if prev is None or best_frac > prev[1]:
                        candidate_assignments[best_pos] = (lbl, best_frac)

            # Apply assignments to dataframe
            for pos, (lbl, frac) in candidate_assignments.items():
                if lbl and lbl != 'Unlabeled':
                    dff.at[filtered_indices[pos], 'manual_label'] = lbl

        except Exception:
            continue

    return dff