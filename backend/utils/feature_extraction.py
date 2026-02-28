"""
Enhanced feature extraction for Speech Emotion Recognition.
Supports: Log-Mel Spectrogram, MFCC, Chroma, Spectral Contrast, ZCR, RMS.
Includes noise reduction, normalization, silence trimming.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

# Optional: noise reduction
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("⚠️  noisereduce not installed. Noise reduction disabled. Install with: pip install noisereduce")


# ═══════════════════════════════════════════════════════════════
# Audio Preprocessing
# ═══════════════════════════════════════════════════════════════

def preprocess_audio(audio_path, target_sr=16000, apply_noise_reduction=True):
    """
    Load and preprocess audio:
      1. Resample to target_sr
      2. Convert to mono
      3. Apply high-pass filter (remove hum)
      4. Apply pre-emphasis filter (boost high frequencies)
      5. Apply noise reduction (spectral gating)
      6. Trim silence
      7. Normalize amplitude (robustly)
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        if len(audio) == 0:
            print(f"⚠️  Empty audio file: {audio_path}")
            return None, None

        # 1. High-pass filter (remove low-freq hum < 80Hz)
        from scipy import signal
        b, a = signal.butter(4, 80 / (sr / 2), 'high')
        audio = signal.filtfilt(b, a, audio)

        # 2. Pre-emphasis filter (y[n] = x[n] - 0.97 * x[n-1])
        # This makes the spectral energy more balanced for MFCC
        audio = librosa.effects.preemphasis(audio, coef=0.97)

        # 3. Noise reduction (spectral gating)
        if apply_noise_reduction and HAS_NOISEREDUCE:
            # Stationary=True helps if the background noise is consistent
            # prop_decrease=0.7 is a good balance for speech
            audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.7, stationary=True)

        # 4. Trim silence (top_db=30)
        audio, _ = librosa.effects.trim(audio, top_db=30)

        # 5. Ensure minimum length (at least 0.8 seconds for better feature alignment)
        min_samples = int(0.8 * sr)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

        # 6. Robust normalization
        # Instead of peak normalization (which is sensitive to one loud spike)
        # We use RMS-based normalization to bring the average volume to a target level
        target_db = -20.0
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            curr_db = 20 * np.log10(rms)
            gain = 10**((target_db - curr_db) / 20)
            audio = audio * gain
        
        # Clip to avoid artifacts
        audio = np.clip(audio, -1.0, 1.0)

        return audio, sr
    except Exception as e:
        print(f"Error preprocessing audio {audio_path}: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# Feature Extraction Methods
# ═══════════════════════════════════════════════════════════════

def extract_log_mel_spectrogram(audio, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """Extract Log-Mel Spectrogram features."""
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    except Exception as e:
        print(f"Error extracting Mel Spectrogram: {e}")
        return None


def extract_combined_features(audio_path, target_sr=16000, n_mfcc=40,
                               n_fft=2048, hop_length=512, fixed_length=200):
    """
    Extract a rich combined feature set per frame, then pad/truncate to fixed_length.

    Features per frame (141 total):
      - MFCC (40) + delta-MFCC (40) + delta2-MFCC (40)
      - Chroma (12)
      - Spectral Contrast (7)
      - ZCR (1)
      - RMS Energy (1)

    Returns: np.ndarray of shape (141, fixed_length) or None
    """
    audio, sr = preprocess_audio(audio_path, target_sr=target_sr)
    if audio is None:
        return None

    try:
        # MFCC + deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr,
                                              n_fft=n_fft, hop_length=hop_length)

        # Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)

        # RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)

        # Concatenate all features: (141, T)
        combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2,
                              chroma, spec_contrast, zcr, rms])

        # Pad or truncate to fixed_length
        if combined.shape[1] < fixed_length:
            pad_width = fixed_length - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :fixed_length]

        return combined

    except Exception as e:
        print(f"Error extracting combined features from {audio_path}: {e}")
        return None


def extract_features(audio_path, target_sr=16000, fixed_length=200):
    """
    Main feature extraction — uses combined features (MFCC + Chroma + ZCR + RMS etc.)
    Returns: np.ndarray of shape (141, fixed_length) or None
    """
    return extract_combined_features(
        audio_path, target_sr=target_sr, fixed_length=fixed_length
    )


# ═══════════════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════════════

def add_noise(audio, noise_factor=0.005):
    """Add Gaussian noise."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_stretch(audio, rate=1.0):
    """Time-stretch audio."""
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr, n_steps=2):
    """Pitch-shift audio."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def extract_features_with_augmentation(audio_path, target_sr=16000, fixed_length=200):
    """
    Extract features from original + augmented versions.
    Returns list of feature arrays for training data expansion.
    """
    audio, sr = preprocess_audio(audio_path, target_sr=target_sr)
    if audio is None:
        return []

    def _extract(aug_audio):
        """Helper to extract combined features from an audio array."""
        try:
            n_mfcc, n_fft, hop_length = 40, 2048, 512

            mfcc = librosa.feature.mfcc(y=aug_audio, sr=sr, n_mfcc=n_mfcc,
                                         n_fft=n_fft, hop_length=hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            chroma = librosa.feature.chroma_stft(y=aug_audio, sr=sr,
                                                  n_fft=n_fft, hop_length=hop_length)
            spec_contrast = librosa.feature.spectral_contrast(
                y=aug_audio, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
            zcr = librosa.feature.zero_crossing_rate(aug_audio, hop_length=hop_length)
            rms = librosa.feature.rms(y=aug_audio, hop_length=hop_length)

            combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2,
                                  chroma, spec_contrast, zcr, rms])

            if combined.shape[1] < fixed_length:
                combined = np.pad(combined, ((0, 0), (0, fixed_length - combined.shape[1])),
                                  mode='constant')
            else:
                combined = combined[:, :fixed_length]
            return combined
        except Exception:
            return None

    augmented = []

    # Original
    feat = _extract(audio)
    if feat is not None:
        augmented.append(feat)

    # Noise (low + medium)
    for nf in [0.003, 0.008]:
        feat = _extract(add_noise(audio, noise_factor=nf))
        if feat is not None:
            augmented.append(feat)

    # Time stretch
    for rate in [0.85, 1.15]:
        try:
            feat = _extract(time_stretch(audio, rate=rate))
            if feat is not None:
                augmented.append(feat)
        except Exception:
            pass

    # Pitch shift
    for steps in [2, -2]:
        try:
            feat = _extract(pitch_shift(audio, sr, n_steps=steps))
            if feat is not None:
                augmented.append(feat)
        except Exception:
            pass

    return augmented


# ═══════════════════════════════════════════════════════════════
# Audio Format Conversion
# ═══════════════════════════════════════════════════════════════

def convert_to_wav(file_stream, output_path):
    """Convert WebM or other formats to WAV using pydub."""
    try:
        audio = AudioSegment.from_file(file_stream)
        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        return False


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        features = extract_features(audio_file)
        if features is not None:
            print(f"✅ Extracted features with shape: {features.shape}")
        else:
            print("❌ Feature extraction failed")
