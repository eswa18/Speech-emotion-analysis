import os
import requests
import zipfile
import io

DATASET_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
ZIP_PATH = os.path.join(DATASET_DIR, 'Audio_Speech_Actors_01-24.zip')
EXTRACT_PATH = os.path.join(DATASET_DIR, 'RAVDESS')

def download_file(url, filename):
    print(f"📥 Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB
    wrote = 0
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            wrote += len(data)
            f.write(data)
            done = int(50 * wrote / total_size)
            print(f"\rProgress: [{'=' * done}{' ' * (50-done)}] {wrote//1024//1024}MB / {total_size//1024//1024}MB", end='')
    print("\n✅ Download complete!")

def extract_zip(zip_path, extract_to):
    print(f"📦 Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete!")

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    try:
        if not os.path.exists(EXTRACT_PATH):
            if not os.path.exists(ZIP_PATH):
                download_file(DATASET_URL, ZIP_PATH)
            
            extract_zip(ZIP_PATH, EXTRACT_PATH)
            
            # Count files
            count = 0
            for root, dirs, files in os.walk(EXTRACT_PATH):
                for file in files:
                    if file.endswith(".wav"):
                        count += 1
            print(f"📊 Found {count} audio files")
            print("🎯 Dataset ready for training!")
        else:
            print("✅ Dataset already exists!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
