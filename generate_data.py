import numpy as np
import soundfile as sf
import os
import random
from scipy import signal

# Emotion map with HUMAN VOICE frequencies
# Male: 85-180 Hz, Female: 165-255 Hz
# We will mix ranges to cover both.
EMOTIONS = {
    '01': {'name': 'neutral', 'freq': 120, 'speed': 1.0, 'amp': 0.5, 'var': 5},
    '02': {'name': 'calm', 'freq': 110, 'speed': 0.9, 'amp': 0.4, 'var': 2},
    '03': {'name': 'happy', 'freq': 220, 'speed': 1.2, 'amp': 0.7, 'var': 30}, # Higher pitch, variable
    '04': {'name': 'sad', 'freq': 100, 'speed': 0.7, 'amp': 0.3, 'var': 5},    # Low pitch, flat
    '05': {'name': 'angry', 'freq': 250, 'speed': 1.5, 'amp': 0.9, 'var': 50}, # High pitch, loud, erratic
    '06': {'name': 'fearful', 'freq': 280, 'speed': 1.3, 'amp': 0.6, 'var': 40},
    '07': {'name': 'disgust', 'freq': 130, 'speed': 0.8, 'amp': 0.5, 'var': 10},
    '08': {'name': 'surprised', 'freq': 300, 'speed': 1.4, 'amp': 0.6, 'var': 60}
}

OUTPUT_DIR = r"d:\learnthonproject\speech-emotion-analysis\dataset\RAVDESS"
SR = 22050
DURATION = 3.0

def generate_voice_like_wave(freq, duration, sr, amp=0.5, variance=0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Pitch modulation (intonation)
    if variance > 0:
        # Add random slow modulation to simulate speech contour
        mod_freq = random.uniform(2, 5) # Syllable rate
        modulation = variance * np.sin(2 * np.pi * mod_freq * t)
        freq_t = freq + modulation
        
        # Integrate frequency to get phase
        phase = 2 * np.pi * np.cumsum(freq_t) / sr
    else:
        phase = 2 * np.pi * freq * t

    # Use Sawtooth wave (closer to vocal cords than sine)
    audio = amp * signal.sawtooth(phase)
    
    # Add harmonics (formants simulation - very basic)
    audio += (amp * 0.5) * np.sin(phase * 2)
    audio += (amp * 0.25) * np.sin(phase * 3)
    
    # Add noise (breathiness)
    noise = np.random.normal(0, 0.05, audio.shape)
    audio = audio + noise
    
    # Apply envelope (attack/decay) slightly
    return audio

def create_synthetic_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"🎵 Generating HUMAN-LIKE synthetic dataset in {OUTPUT_DIR}...")
    
    # Create 20 actors
    for actor in range(1, 21):
        actor_dir = os.path.join(OUTPUT_DIR, f"Actor_{actor:02d}")
        os.makedirs(actor_dir, exist_ok=True)
        
        # Create 2 samples per emotion
        for code, params in EMOTIONS.items():
            for rep in range(1, 3):
                # Filename: 03-01-XX-01-01-01-XX.wav
                filename = f"03-01-{code}-01-01-01-{actor:02d}.wav"
                filepath = os.path.join(actor_dir, filename)
                
                # Vary parameters slightly
                base_freq = params['freq'] * random.uniform(0.9, 1.1)
                amp = params['amp'] * random.uniform(0.8, 1.2)
                var = params['var']
                
                audio = generate_voice_like_wave(base_freq, DURATION, SR, amp, var)
                
                sf.write(filepath, audio, SR)
                
    print("✅ Human-like synthetic dataset created successfully!")

if __name__ == "__main__":
    create_synthetic_dataset()
