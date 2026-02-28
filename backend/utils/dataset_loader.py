"""
Unified dataset loader for RAVDESS, TESS, and CREMA-D datasets.
Maps all datasets to 4 core emotions: Angry, Happy, Neutral, Sad.
"""

import os
import glob

# Core emotion indices (must match config.EMOTIONS order)
EMOTION_MAP = {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3}


def load_ravdess(dataset_path):
    """
    Load RAVDESS dataset.
    Filename format: 03-01-XX-01-01-01-YY.wav
    XX = emotion code:
      01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    ravdess_map = {
        '01': 'Neutral',
        '02': 'Neutral',   # Calm -> Neutral
        '03': 'Happy',
        '04': 'Sad',
        '05': 'Angry',
        # 06=fearful, 07=disgust, 08=surprised -> skip (not in 4-class set)
    }

    files, labels = [], []
    audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)

    for filepath in audio_files:
        filename = os.path.basename(filepath)
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion_name = ravdess_map.get(emotion_code)
            if emotion_name is not None:
                files.append(filepath)
                labels.append(EMOTION_MAP[emotion_name])

    print(f"  📂 RAVDESS: loaded {len(files)} files from {dataset_path}")
    return files, labels


def load_tess(dataset_path):
    """
    Load TESS (Toronto Emotional Speech Set) dataset.
    Folder structure: OAF_angry/, OAF_happy/, YAF_sad/, etc.
    Or filename contains emotion: OAF_back_angry.wav
    """
    tess_map = {
        'angry': 'Angry',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        # pleasant_surprise, fear, disgust -> skip
    }

    files, labels = [], []
    audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)

    for filepath in audio_files:
        filename = os.path.basename(filepath).lower()
        folder = os.path.basename(os.path.dirname(filepath)).lower()

        # Try to find emotion from filename or folder name
        matched_emotion = None
        for keyword, emotion_name in tess_map.items():
            if keyword in filename or keyword in folder:
                matched_emotion = emotion_name
                break

        if matched_emotion is not None:
            files.append(filepath)
            labels.append(EMOTION_MAP[matched_emotion])

    print(f"  📂 TESS: loaded {len(files)} files from {dataset_path}")
    return files, labels


def load_cremad(dataset_path):
    """
    Load CREMA-D dataset.
    Filename format: 1001_DFA_ANG_XX.wav
    ANG=angry, HAP=happy, NEU=neutral, SAD=sad, FEA=fear, DIS=disgust
    """
    cremad_map = {
        'ANG': 'Angry',
        'HAP': 'Happy',
        'NEU': 'Neutral',
        'SAD': 'Sad',
        # FEA, DIS -> skip
    }

    files, labels = [], []
    audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)

    for filepath in audio_files:
        filename = os.path.basename(filepath)
        parts = filename.replace('.wav', '').split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion_name = cremad_map.get(emotion_code)
            if emotion_name is not None:
                files.append(filepath)
                labels.append(EMOTION_MAP[emotion_name])

    print(f"  📂 CREMA-D: loaded {len(files)} files from {dataset_path}")
    return files, labels


def load_dataset(name, dataset_path):
    """Load a single dataset by name."""
    loaders = {
        'ravdess': load_ravdess,
        'tess': load_tess,
        'cremad': load_cremad,
    }
    loader = loaders.get(name.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loader(dataset_path)


def load_all_datasets(ravdess_path, tess_path=None, cremad_path=None):
    """
    Load all available datasets and combine them.
    Returns: (all_files, all_labels, dataset_info)
    """
    all_files, all_labels = [], []
    dataset_info = {}

    # RAVDESS (always available)
    if os.path.exists(ravdess_path):
        f, l = load_ravdess(ravdess_path)
        dataset_info['RAVDESS'] = {'count': len(f), 'start_idx': len(all_files)}
        all_files.extend(f)
        all_labels.extend(l)

    # TESS
    if tess_path and os.path.exists(tess_path):
        f, l = load_tess(tess_path)
        dataset_info['TESS'] = {'count': len(f), 'start_idx': len(all_files)}
        all_files.extend(f)
        all_labels.extend(l)

    # CREMA-D
    if cremad_path and os.path.exists(cremad_path):
        f, l = load_cremad(cremad_path)
        dataset_info['CREMA-D'] = {'count': len(f), 'start_idx': len(all_files)}
        all_files.extend(f)
        all_labels.extend(l)

    print(f"\n  📊 Total loaded: {len(all_files)} files across {len(dataset_info)} dataset(s)")
    for name, info in dataset_info.items():
        print(f"     - {name}: {info['count']} files")

    return all_files, all_labels, dataset_info
