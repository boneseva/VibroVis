import sys
import traceback

try:
    import numpy as np
    import soundfile as sf
    import os
    import librosa
    # Use explicit relative import or path hack if needed, but standard import should work if CWD is root
    sys.path.append(os.getcwd())
    import utils
except Exception:
    with open('debug_error.log', 'w') as f:
        f.write(traceback.format_exc())
    print("Import failed. See debug_error.log")
    sys.exit(1)

def create_dummy_audio(filename='dummy.wav', duration=2.0, sr=22050):
    t = np.linspace(0, duration, int(duration * sr))
    # Chirp from 100Hz to 1000Hz
    freqs = np.linspace(100, 1000, len(t))
    y = 0.5 * np.sin(2 * np.pi * freqs * t)
    # Add some noise
    y += 0.01 * np.random.normal(size=len(t))
    sf.write(filename, y, sr)
    return y, sr

def test_scales():
    y, sr = create_dummy_audio()
    
    print(f"Audio stats: Min={y.min():.4f}, Max={y.max():.4f}, Mean={y.mean():.4f}")
    
    scales = ['mel', 'linear', 'log']
    
    for scale in scales:
        print(f"\n--- Testing Scale: {scale} ---")
        f, t, Sxx_db = utils.compute_spectrogram(
            segment=y, samplerate=sr, scale=scale,
            fft_window_size=1024, window_overlap=0.5,
            min_freq=0, max_freq=sr/2, num_bins=128, db_floor=-120
        )
        
        if Sxx_db.size == 0:
            print("EMPTY RESULT")
            continue
            
        print(f"Shape: {Sxx_db.shape}")
        print(f"Freq range: {f.min():.1f} - {f.max():.1f}")
        print(f"DB Stats -> Max: {Sxx_db.max():.2f}")
        print(f"DB Stats -> Min: {Sxx_db.min():.2f}")
        print(f"DB Stats -> Mean: {Sxx_db.mean():.2f}")
        print(f"DB Stats -> 5th %: {np.percentile(Sxx_db, 5):.2f}")
        print(f"DB Stats -> 95th %: {np.percentile(Sxx_db, 95):.2f}")

    if os.path.exists('dummy.wav'):
        os.remove('dummy.wav')

if __name__ == "__main__":
    test_scales()
