import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'emotion_model.pth')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Database configuration
DATABASE_PATH = os.path.join(BASE_DIR, 'emotions.db')

# API configuration
API_HOST = '0.0.0.0'
API_PORT = int(os.environ.get('PORT', 5000))
DEBUG = True

# ── Emotion Labels ──────────────────────────────────────────────
# Core 4 emotions (used for training and prediction)
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad']

# Set this to True if using the synthetic demo model
IS_DEMO = True

# CORS configuration
CORS_ORIGINS = ["*", "http://localhost:5173", "http://127.0.0.1:5173"]

# ── Dataset Paths ───────────────────────────────────────────────
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
RAVDESS_DIR = os.path.join(DATASET_DIR, 'RAVDESS')
TESS_DIR = os.path.join(DATASET_DIR, 'TESS')
CREMAD_DIR = os.path.join(DATASET_DIR, 'CREMA-D')

# ── Preprocessing Constants ────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
FIXED_LENGTH = 200    # Time-frames for spectrogram padding/truncation
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40

# ── Feature Config ──────────────────────────────────────────────
# Total combined feature vector size per frame:
#   MFCC(40) + delta(40) + delta2(40) + Chroma(12) + SpectralContrast(7) + ZCR(1) + RMS(1) = 141
COMBINED_FEATURE_DIM = 141

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
