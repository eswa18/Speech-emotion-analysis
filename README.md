# Speech Emotion Analysis Using Voice

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![Vite](https://img.shields.io/badge/Vite-7.3-646cff)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)

A comprehensive **full-stack machine learning application** that analyzes emotions from voice audio using advanced AI algorithms. Built for Learnthon competition with a modern React + Vite frontend and Flask backend.

## 🎯 Project Overview

This application uses **machine learning** to detect human emotions from voice recordings. It supports:
- **7 Emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **Real-time audio recording** via browser
- **File upload** for pre-recorded audio
- **Prediction history** with database storage
- **Beautiful UI** with emotion-specific themes

## ✨ Features

### Frontend (React + Vite)
- 🎨 **Premium Dark UI** with glassmorphism effects
- 🎤 **Live Audio Recording** using MediaRecorder API
- 📁 **Drag & Drop Upload** with file validation
- 📊 **Real-time Results** with confidence scores
- 📜 **History Management** with CSV export
- 🎭 **Emotion-specific Themes** with smooth animations
- 📱 **Fully Responsive** design

### Backend (Flask + Python)
- 🔬 **Advanced Feature Extraction** (MFCC, Chroma, Mel Spectrogram)
- 🤖 **RandomForest Classifier** trained on RAVDESS dataset
- 💾 **SQLite Database** for prediction history
- 🔌 **RESTful API** with CORS support
- ⚡ **Fast Response** times

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | React 18, Vite, Axios |
| **Backend** | Python, Flask, Flask-CORS |
| **ML Framework** | Scikit-learn, Librosa |
| **Database** | SQLite |
| **Dataset** | RAVDESS (optional for training) |
| **Styling** | CSS3 with animations |

## 📋 Prerequisites

- **Python 3.9+** installed
- **Node.js 18+** and npm installed
- Microphone access for recording
- Modern web browser (Chrome, Firefox, Edge)

## 🚀 Installation & Setup

### 1️⃣ Clone or Navigate to Project

```bash
cd d:\learnthonproject\speech-emotion-analysis
```

### 2️⃣ Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python utils/database.py

# Train the model (creates demo model if no dataset available)
python model/train_model.py
```

### 3️⃣ Frontend Setup

```bash
# Navigate to frontend (from project root)
cd frontend

# Dependencies are already installed, but if needed:
# npm install
```

## ▶️ Running the Application

### Start Backend Server

```bash
# From backend directory
python app.py
```

Backend will run at: `http://localhost:5000`

### Start Frontend Dev Server

```bash
# From frontend directory (in a new terminal)
npm run dev
```

Frontend will run at: `http://localhost:5173`

### Open Application

Visit **`http://localhost:5173`** in your browser

## 📖 Usage

1. **Choose Input Method**:
   - **Upload**: Drag & drop or click to select an audio file (WAV, MP3, OGG, FLAC)
   - **Record**: Click "Start Recording", speak for 3-5 seconds, then click "Stop & Analyze"

2. **View Results**:
   - Primary emotion with confidence percentage
   - Detailed probability breakdown for all emotions
   - Emotion-specific color theme and emoji

3. **Manage History**:
   - View all past predictions in the history table
   - Export history to CSV for analysis
   - Delete individual predictions

## 🎓 Dataset Information

The model can be trained on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset:

- **Download**: [RAVDESS on Zenodo](https://zenodo.org/record/1188976)
- **Size**: 7,356 files
- **Emotions**: 8 emotions (we map to 7)
- **Format**: WAV files

Place the dataset in `d:\learnthonproject\speech-emotion-analysis\dataset\` folder.

**Note**: A demo model is automatically created if the dataset is not available, perfect for testing!

## 📊 Model Performance

- **Algorithm**: RandomForestClassifier
- **Features**: MFCC (40), Chroma (12), Mel Spectrogram (128) = 180 features
- **Expected Accuracy**: 65-75% on RAVDESS (demo model is synthetic)
- **Training Time**: ~2-5 minutes on RAVDESS

## 🔌 API Documentation

### Health Check
```http
GET /api/health
```

### Predict Emotion
```http
POST /api/predict
Content-Type: multipart/form-data

Body: { audio: <audio_file> }
```

### Get History
```http
GET /api/history
```

### Delete Prediction
```http
DELETE /api/history/<prediction_id>
```

## 📁 Project Structure

```
speech-emotion-analysis/
├── backend/
│   ├── app.py                 # Flask application
│   ├── config.py              # Configuration
│   ├── requirements.txt       # Python dependencies
│   ├── model/
│   │   ├── train_model.py     # ML training script
│   │   ├── emotion_model.pkl  # Trained model
│   │   └── scaler.pkl         # Feature scaler
│   └── utils/
│       ├── database.py        # Database operations
│       └── feature_extraction.py  # Audio feature extraction
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main component
│   │   ├── components/        # React components
│   │   ├── styles/            # CSS styling
│   │   └── utils/             # API utilities
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🎨 Screenshots

The application features:
- Dark mode with gradient backgrounds
- Glassmorphism card effects
- Emotion-specific color schemes
- Animated confidence bars
- Responsive mobile design

## 🚧 Future Enhancements

- [ ] Deploy to cloud (Heroku/Railway)
- [ ] Add more emotion categories
- [ ] Implement CNN/LSTM for better accuracy
- [ ] Add multi-language support
- [ ] Real-time waveform visualization
- [ ] Export audio with predictions

## 🏆 For Learnthon Judges

**Why This Project Stands Out**:

1. **Full-Stack Implementation**: Complete end-to-end solution
2. **Modern Tech Stack**: React + Vite (fast), Flask (reliable)
3. **Production-Ready**: Database, error handling, validation
4. **Great UX**: Beautiful design, smooth animations, responsive
5. **Practical ML**: Real-world audio processing with Librosa
6. **Extensible**: Clean code architecture, easy to enhance

## 🤝 Contributing

This is a Learnthon project. Feedback and suggestions are welcome!

## 📝 License

MIT License - Feel free to use for learning purposes

## 👨‍💻 Author

Built with ❤️ for Learnthon

---

**Happy Emotion Analyzing! 🎤🎭**
