"""
FastAPI backend for Speech Emotion Analysis.
Uses CNN-BiLSTM model with enhanced feature extraction.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import torch
import numpy as np

# Local imports
import config
from utils.database import init_database, save_prediction, get_all_predictions, delete_prediction
from utils.feature_extraction import extract_features, convert_to_wav
from model.model_pytorch import get_model

app = FastAPI(title="Speech Emotion Analysis API", version="3.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_database()

# Global model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
EMOTIONS = config.EMOTIONS


def load_pytorch_model():
    """Load the trained PyTorch model."""
    global model
    model_path = config.MODEL_PATH
    if os.path.exists(model_path):
        # input_size matches COMBINED_FEATURE_DIM from config
        model = get_model(num_classes=len(EMOTIONS), input_size=config.COMBINED_FEATURE_DIM)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ PyTorch model loaded on {device} ({len(EMOTIONS)} emotions)")
    else:
        print(f"⚠️ Model not found at {model_path}. Please train the model first.")
        print(f"   Run: python model/train_model.py --dataset ravdess")


@app.on_event("startup")
async def startup_event():
    load_pytorch_model()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "emotions": EMOTIONS,
    }


@app.post("/api/predict")
async def predict_emotion(audio: UploadFile = File(...)):
    """Predict emotion from uploaded audio file."""
    if not model:
        raise HTTPException(status_code=500,
                            detail="Model not loaded. Train the model first: "
                                   "python model/train_model.py --dataset ravdess")

    # Create a unique filename
    file_id = str(uuid.uuid4())
    temp_dir = config.UPLOAD_FOLDER
    os.makedirs(temp_dir, exist_ok=True)

    filename = audio.filename
    temp_path = os.path.join(temp_dir, f"{file_id}_{filename}")

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        final_wav_path = temp_path

        # If it's WebM or non-WAV, convert to WAV
        if filename.endswith(".webm") or audio.content_type == "audio/webm":
            wav_path = os.path.join(temp_dir, f"{file_id}.wav")
            if convert_to_wav(temp_path, wav_path):
                final_wav_path = wav_path
                os.remove(temp_path)
            else:
                raise HTTPException(status_code=400, detail="Failed to convert audio to WAV")

        # Extract features (combined: MFCC + Chroma + ZCR + RMS etc.)
        # This applies noise reduction, normalization, silence trimming automatically
        features = extract_features(final_wav_path, fixed_length=config.FIXED_LENGTH)

        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from audio")

        # Inference
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        prediction_idx = int(np.argmax(probabilities))
        emotion = EMOTIONS[prediction_idx]
        confidence = float(probabilities[prediction_idx]) * 100

        # Save to database
        prediction_id = save_prediction(emotion, round(confidence, 2), filename)

        # Cleanup
        if os.path.exists(final_wav_path):
            os.remove(final_wav_path)

        return {
            "success": True,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "prediction_id": prediction_id,
            "all_probabilities": {
                EMOTIONS[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(EMOTIONS))
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history():
    predictions = get_all_predictions()
    return {
        "success": True,
        "predictions": predictions,
        "count": len(predictions),
    }


@app.delete("/api/history/{prediction_id}")
async def delete_history_item(prediction_id: int):
    delete_prediction(prediction_id)
    return {"success": True, "message": f"Prediction {prediction_id} deleted"}


@app.get("/api/emotions")
async def get_emotions():
    return {"success": True, "emotions": EMOTIONS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
