import zipfile
import os

zip_path = r"d:\learnthonproject\speech-emotion-analysis\dataset\Audio_Speech_Actors_01-24.zip"
extract_path = r"d:\learnthonproject\speech-emotion-analysis\dataset\RAVDESS"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

print(f"📦 Extracting {zip_path}...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extraction complete!")
except Exception as e:
    print(f"❌ Error: {e}")
