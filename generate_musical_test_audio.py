import numpy as np
import scipy.io.wavfile as wav
import os
from scipy import signal

def generate_wave(freq, duration, sr=16000, wave_type='sine'):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'sawtooth':
        return signal.sawtooth(2 * np.pi * freq * t)
    elif wave_type == 'square':
        return signal.square(2 * np.pi * freq * t)
    return np.zeros_like(t)

def apply_envelope(audio, attack=0.05, release=0.1, sr=16000):
    n = len(audio)
    a_samples = int(attack * sr)
    r_samples = int(release * sr)
    
    envelope = np.ones(n)
    if a_samples > 0:
        envelope[:a_samples] = np.linspace(0, 1, a_samples)
    if r_samples > 0:
        envelope[-r_samples:] = np.linspace(1, 0, r_samples)
    
    return audio * envelope

def create_musical_sample(name, notes, bpm, wave_type='sine', attack=0.05, release=0.1, sr=16000):
    beat_duration = 60.0 / bpm
    pattern = []
    
    for freq in notes:
        # Each note is a quarter note
        note_audio = generate_wave(freq, beat_duration, sr, wave_type)
        note_audio = apply_envelope(note_audio, attack, release, sr)
        pattern.append(note_audio)
    
    full_audio = np.concatenate(pattern)
    # Normalize
    if np.max(np.abs(full_audio)) > 0:
        full_audio = full_audio / np.max(np.abs(full_audio))
    return full_audio

def main():
    sr = 16000
    output_dir = r'd:\learnthonproject\speech-emotion-analysis\musical_test_audio'
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    import random

    # Frequencies for common notes
    notes_lib = {
        'C3': 130.81, 'Db3': 138.59, 'D3': 146.83, 'Eb3': 155.56, 'E3': 164.81, 'F3': 174.61, 'Gb3': 185.00, 'G3': 196.00, 'Ab3': 207.65, 'A3': 220.00, 'Bb3': 233.08, 'B3': 246.94,
        'C4': 261.63, 'Db4': 277.18, 'D4': 293.66, 'Eb4': 311.13, 'E4': 329.63, 'F4': 349.23, 'Gb4': 369.99, 'G4': 392.00, 'Ab4': 415.30, 'A4': 440.00, 'Bb4': 466.16, 'B4': 493.88,
        'C5': 523.25
    }

    # Emotion configurations with multiple scale options for diversity
    emo_configs = {
        "Happy": {
            "scales": [
                [notes_lib['C4'], notes_lib['E4'], notes_lib['G4'], notes_lib['C5']], # C Major
                [notes_lib['F4'], notes_lib['A4'], notes_lib['C5'], notes_lib['F4']], # F Major
                [notes_lib['G4'], notes_lib['B4'], notes_lib['D4'], notes_lib['G4']]  # G Major
            ],
            "bpm_range": (140, 180),
            "waves": ['sine', 'triangle'],
            "attack_range": (0.01, 0.05),
            "release_range": (0.05, 0.1)
        },
        "Sad": {
            "scales": [
                [notes_lib['C3'], notes_lib['Eb3'], notes_lib['G3'], notes_lib['Bb3']], # C Minor
                [notes_lib['A3'], notes_lib['C4'], notes_lib['E4'], notes_lib['G4']],  # A Minor
                [notes_lib['D3'], notes_lib['F3'], notes_lib['A3'], notes_lib['C4']]   # D Minor
            ],
            "bpm_range": (50, 75),
            "waves": ['sine'],
            "attack_range": (0.2, 0.5),
            "release_range": (0.4, 0.8)
        },
        "Angry": {
            "scales": [
                [notes_lib['C4'], notes_lib['Db4'], notes_lib['Gb4'], notes_lib['F4']], # Dissonant
                [notes_lib['E3'], notes_lib['F3'], notes_lib['Bb3'], notes_lib['B3']], # Tritonish
                [notes_lib['G3'], notes_lib['Ab3'], notes_lib['Db4'], notes_lib['D4']] 
            ],
            "bpm_range": (170, 210),
            "waves": ['sawtooth', 'square'],
            "attack_range": (0.005, 0.015),
            "release_range": (0.01, 0.03)
        },
        "Neutral": {
            "scales": [
                [notes_lib['C4'], notes_lib['G4']], # Fifth
                [notes_lib['D4'], notes_lib['A4']],
                [notes_lib['E4'], notes_lib['B4']]
            ],
            "bpm_range": (90, 115),
            "waves": ['sawtooth', 'triangle'],
            "attack_range": (0.05, 0.15),
            "release_range": (0.1, 0.25)
        }
    }

    for emo, config in emo_configs.items():
        print(f"Generating 20 highly diverse samples for {emo}...")
        emo_dir = os.path.join(output_dir, emo)
        os.makedirs(emo_dir, exist_ok=True)
        
        for i in range(20):
            # 1. Randomize Scale
            scale = random.choice(config["scales"])
            # 2. Randomize Sequence
            var_notes = [random.choice(scale) for _ in range(8)]
            # 3. Randomize Tempo
            var_bpm = random.randint(*config["bpm_range"])
            # 4. Randomize Wave
            var_wave = random.choice(config["waves"])
            # 5. Randomize Envelope
            var_attack = random.uniform(*config["attack_range"])
            var_release = random.uniform(*config["release_range"])
            
            audio = create_musical_sample(emo, var_notes, bpm=var_bpm, 
                                        wave_type=var_wave, 
                                        attack=var_attack, 
                                        release=var_release)
            
            if emo == "Angry":
                # Add extra harmonics and grit
                audio = np.clip(audio * 1.8, -1, 1) + np.random.normal(0, 0.08, len(audio))
                audio = np.clip(audio, -1, 1)
            
            fname = f"{emo.lower()}_var_{i+1:02d}.wav"
            wav.write(os.path.join(emo_dir, fname), sr, (audio * 32767).astype(np.int16))

    print(f"\n✅ SUCCESS: 80 unique musical samples generated in: {output_dir}")

if __name__ == "__main__":
    main()
