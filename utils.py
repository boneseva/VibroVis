import soundfile as sf
import numpy as np
import os
from scipy.signal import spectrogram
import read_data  # Import read_data to access DATA_DIR


def load_audio_segment(mp3_file_relative_path, clip_time, clip_duration, channel=0, padding_s=1.0):
    """
    Loads a specific audio segment from an MP3 file with padding.
    """
    mp3_file_relative_path = mp3_file_relative_path.replace("\\", "/")
    full_path = os.path.join(read_data.DATA_DIR, mp3_file_relative_path)
    if not os.path.exists(full_path):
        print(f"Warning: Audio file not found at {full_path}")
        return None, None

    try:
        data, samplerate = sf.read(full_path)

        # Handle mono vs multi-channel
        if data.ndim > 1:
            if channel < data.shape[1]:
                channel_data = data[:, channel]
            else:  # Fallback to first channel if specified channel is out of bounds
                channel_data = data[:, 0]
        else:
            channel_data = data

        # Calculate start and end samples with padding
        start_sample = int(max(0, (clip_time - padding_s)) * samplerate)
        end_sample = int(min(len(channel_data), (clip_time + clip_duration + padding_s) * samplerate))

        segment = channel_data[start_sample:end_sample]
        return segment, samplerate

    except Exception as e:
        print(f"Error loading audio segment from {full_path}: {e}")
        return None, None


# In utils.py
import librosa
from scipy.signal import spectrogram
import numpy as np


def compute_spectrogram(segment, samplerate, scale='log', fft_window_size=1024, window_overlap=0.5,
                        window_type='hann', min_freq=50, max_freq=5000, num_bins=256, db_floor=-100, **kwargs):
    """
    Computes the spectrogram with selectable frequency scales (Linear, Log, Mel).
    """
    if segment is None or len(segment) == 0:
        return np.array([]), np.array([]), np.array([])

    # Calculate step size from overlap
    fft_step_size = int(fft_window_size * (1 - window_overlap))
    # Calculate nfft as the next power of 2 for efficiency
    nfft = int(2 ** np.ceil(np.log2(fft_window_size)))

    # --- Mel Scale Spectrogram ---
    if scale == 'mel':
        # Use librosa for Mel scale
        Sxx_mel = librosa.feature.melspectrogram(
            y=segment, sr=samplerate, n_fft=nfft,
            hop_length=fft_step_size, win_length=fft_window_size, window=window_type,
            n_mels=num_bins, fmin=min_freq, fmax=max_freq
        )
        Sxx_db = librosa.power_to_db(Sxx_mel, ref=np.max)
        Sxx_db = np.maximum(Sxx_db, db_floor)

        # Get the time and frequency axes for plotting
        t = librosa.times_like(Sxx_db, sr=samplerate, hop_length=fft_step_size, n_fft=nfft)
        f = librosa.mel_frequencies(n_mels=num_bins, fmin=min_freq, fmax=max_freq)
        return f, t, Sxx_db

    # --- Linear & Log Scale Spectrograms ---
    f, t, Sxx = spectrogram(
        segment, fs=samplerate, nperseg=fft_window_size,
        noverlap=(fft_window_size - fft_step_size), nfft=nfft, window=window_type
    )

    # Filter by frequency range
    freq_slice = np.where((f >= min_freq) & (f <= max_freq))
    f = f[freq_slice]
    Sxx = Sxx[freq_slice, :][0]

    if Sxx.size == 0:
        return np.array([]), np.array([]), np.array([])

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

    Sxx_db = np.maximum(Sxx_db, db_floor)
    return f_out, t, Sxx_db