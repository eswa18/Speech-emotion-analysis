"""
Download TESS and CREMA-D datasets from public sources.
Uses direct download links to avoid Kaggle API key requirement.
"""

import os
import sys
import zipfile
import urllib.request
import ssl
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')


class DownloadProgress:
    def __init__(self, desc):
        self.desc = desc
        self.last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            pct = int(block_num * block_size * 100 / total_size)
            if pct != self.last_pct and pct % 10 == 0:
                self.last_pct = pct
                print(f"    {self.desc}: {pct}%")


def download_file(url, dest_path, desc="file"):
    """Download a file with progress."""
    print(f"  📥 Downloading {desc}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path, DownloadProgress(desc))
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  ✅ Downloaded {desc} ({size_mb:.1f} MB)")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  📦 Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  ✅ Extracted")


def download_tess():
    """Download TESS dataset from Kaggle via direct link."""
    tess_dir = os.path.join(DATASET_DIR, 'TESS')
    
    if os.path.exists(tess_dir):
        import glob
        wav_files = glob.glob(os.path.join(tess_dir, '**', '*.wav'), recursive=True)
        if len(wav_files) > 0:
            print(f"\n✅ TESS already exists ({len(wav_files)} files)")
            return True

    os.makedirs(tess_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Downloading TESS Dataset")
    print(f"{'='*50}")
    
    # Try kaggle CLI
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'kaggle', 'datasets', 'download', 
             '-d', 'ejlok1/toronto-emotional-speech-set-tess', 
             '-p', DATASET_DIR, '--unzip'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            # Kaggle may extract into a subfolder, reorganize if needed
            _reorganize_tess(tess_dir)
            print(f"  ✅ TESS downloaded via Kaggle")
            return True
        else:
            print(f"  ⚠️  Kaggle failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"  ⚠️  Kaggle not available: {e}")
    
    print(f"\n  ❌ Automatic download requires Kaggle credentials.")
    print(f"  📋 Manual download instructions:")
    print(f"     1. Go to: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
    print(f"     2. Click 'Download' (you'll need a free Kaggle account)")
    print(f"     3. Extract the ZIP into: {tess_dir}")
    print(f"     4. The folder should contain subfolders like OAF_angry, OAF_happy, etc.")
    return False


def download_cremad():
    """Download CREMA-D dataset."""
    cremad_dir = os.path.join(DATASET_DIR, 'CREMA-D')
    
    if os.path.exists(cremad_dir):
        import glob
        wav_files = glob.glob(os.path.join(cremad_dir, '**', '*.wav'), recursive=True)
        if len(wav_files) > 0:
            print(f"\n✅ CREMA-D already exists ({len(wav_files)} files)")
            return True

    os.makedirs(cremad_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Downloading CREMA-D Dataset")
    print(f"{'='*50}")
    
    # Try kaggle CLI
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'kaggle', 'datasets', 'download',
             '-d', 'ejlok1/cremad',
             '-p', DATASET_DIR, '--unzip'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            _reorganize_cremad(cremad_dir)
            print(f"  ✅ CREMA-D downloaded via Kaggle")
            return True
        else:
            print(f"  ⚠️  Kaggle failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"  ⚠️  Kaggle not available: {e}")
    
    print(f"\n  ❌ Automatic download requires Kaggle credentials.")
    print(f"  📋 Manual download instructions:")
    print(f"     1. Go to: https://www.kaggle.com/datasets/ejlok1/cremad")
    print(f"     2. Click 'Download'")
    print(f"     3. Extract the ZIP into: {cremad_dir}")
    print(f"     4. The folder should contain WAV files like 1001_DFA_ANG_XX.wav")
    return False


def _reorganize_tess(tess_dir):
    """Handle Kaggle extraction which sometimes creates nested folders."""
    import glob
    # Check if files are in a nested subfolder
    for subfolder_name in ['TESS Toronto emotional speech set data', 'tess toronto emotional speech set data']:
        nested = os.path.join(DATASET_DIR, subfolder_name)
        if os.path.exists(nested):
            for item in os.listdir(nested):
                src = os.path.join(nested, item)
                dst = os.path.join(tess_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(nested, ignore_errors=True)


def _reorganize_cremad(cremad_dir):
    """Handle Kaggle extraction for CREMA-D."""
    import glob
    # Check for nested AudioWAV folder
    nested = os.path.join(DATASET_DIR, 'AudioWAV')
    if os.path.exists(nested):
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(cremad_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)


def main():
    print("=" * 60)
    print("  🎤 Speech Emotion Recognition – Dataset Manager")
    print("=" * 60)

    tess_ok = download_tess()
    cremad_ok = download_cremad()

    # Check RAVDESS
    import glob
    ravdess_dir = os.path.join(DATASET_DIR, 'RAVDESS')
    ravdess_count = len(glob.glob(os.path.join(ravdess_dir, '**', '*.wav'), recursive=True))
    
    print(f"\n{'='*60}")
    print(f"  Dataset Status Summary")
    print(f"{'='*60}")
    print(f"  RAVDESS  : {'✅ Ready' if ravdess_count > 0 else '❌ Missing'} ({ravdess_count} files)")
    
    tess_count = len(glob.glob(os.path.join(DATASET_DIR, 'TESS', '**', '*.wav'), recursive=True))
    print(f"  TESS     : {'✅ Ready' if tess_count > 0 else '❌ Missing'} ({tess_count} files)")
    
    cremad_count = len(glob.glob(os.path.join(DATASET_DIR, 'CREMA-D', '**', '*.wav'), recursive=True))
    print(f"  CREMA-D  : {'✅ Ready' if cremad_count > 0 else '❌ Missing'} ({cremad_count} files)")

    if ravdess_count > 0:
        print(f"\n  ✅ You can train now with what's available:")
        print(f"     cd backend")
        print(f"     python model/train_model.py --dataset ravdess --epochs 60")
    
    if tess_count > 0 or cremad_count > 0:
        print(f"\n  ✅ To compare datasets:")
        print(f"     python model/train_model.py --compare")


if __name__ == '__main__':
    main()
