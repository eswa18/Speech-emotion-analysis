"""
Download RAVDESS dataset for emotion recognition training.

This script downloads the RAVDESS (Ryerson Audio-Visual Database of Emotional 
Speech and Song) dataset from Zenodo.
"""

import os
import urllib.request
import zipfile
import sys

def download_file(url, destination):
    """Download a file with progress indication."""
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded * 100) / total_size)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)
        sys.stdout.write(f'\r[{bar}] {percent:.1f}%')
        sys.stdout.flush()
    
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print("\n✅ Download complete!")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"\n📦 Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("=" * 60)
    print("📥 RAVDESS Dataset Downloader")
    print("=" * 60)
    
    # RAVDESS Audio Speech Actors (smaller subset for faster download)
    # Full dataset: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
    
    print("\n⚠️  NOTE: Full RAVDESS dataset is ~1GB")
    print("For demo purposes, you can also manually download from:")
    print("https://zenodo.org/record/1188976")
    print("\nAlternatively, use sample emotion audio files for testing.\n")
    
    # Option: Download from Kaggle or Zenodo
    # Since direct Zenodo download requires authentication, providing instructions
    
    print("📋 Manual Download Instructions:")
    print("1. Visit: https://zenodo.org/record/1188976")
    print("2. Download: Audio_Speech_Actors_01-24.zip")
    print(f"3. Place in: {dataset_dir}")
    print("4. Run this script again to extract")
    
    # Check if zip already exists
    zip_path = os.path.join(dataset_dir, "Audio_Speech_Actors_01-24.zip")
    
    if os.path.exists(zip_path):
        print(f"\n✅ Found {os.path.basename(zip_path)}")
        extract_to = os.path.join(dataset_dir, "RAVDESS")
        extract_zip(zip_path, extract_to)
        
        # Count files
        audio_files = []
        for root, dirs, files in os.walk(extract_to):
            audio_files.extend([f for f in files if f.endswith('.wav')])
        
        print(f"\n✅ Successfully extracted {len(audio_files)} audio files")
        print(f"📁 Dataset location: {extract_to}")
        print("\n🎯 Next step: Run model training")
        print("   cd backend")
        print("   .\\venv\\Scripts\\python model\\train_model.py")
        
    else:
        print(f"\n❌ Zip file not found at: {zip_path}")
        print("\nPlease download manually and place in the dataset folder.")
        print("\n💡 Alternative: Use sample audio files from:")
        print("   - Kaggle: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio")
        print("   - Or create small test set with emotion-labeled audio")


if __name__ == '__main__':
    main()
