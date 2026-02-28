"""
Flask backend for Speech Emotion Analysis (legacy).
Uses CNN-BiLSTM model with enhanced feature extraction.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import random
import hashlib
import traceback
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import numpy as np
from werkzeug.utils import secure_filename

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import config
from utils.database import init_database, save_prediction, get_all_predictions, delete_prediction
from utils.feature_extraction import extract_features, convert_to_wav
# Model import is now handled inside the HAS_TORCH guard

app = Flask(__name__)
CORS(app, origins=config.CORS_ORIGINS)

# Initialize database
init_database()

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
model = None

try:
    if HAS_TORCH and os.path.exists(config.MODEL_PATH):
        from model.model_pytorch import get_model
        model = get_model(num_classes=len(config.EMOTIONS),
                          input_size=config.COMBINED_FEATURE_DIM)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ PyTorch model loaded on {device}")
    elif not HAS_TORCH:
        print("ℹ️  Running in Lite Mode (No Torch installed)")
    else:
        print(f"⚠️  Model not found at {config.MODEL_PATH}")
        print("   Run: python model/train_model.py --dataset ravdess")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    traceback.print_exc()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'emotions': config.EMOTIONS,
        'is_demo': getattr(config, 'IS_DEMO', False)
    })


@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded audio file."""
    try:
        if model is None and not getattr(config, 'IS_DEMO', False):
            return jsonify({
                'error': 'Model not loaded. Train first: python model/train_model.py',
                'success': False
            }), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided', 'success': False}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(config.ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            final_path = filepath

            # Convert WebM to WAV if needed
            if filename.endswith('.webm'):
                wav_path = filepath.replace('.webm', '.wav')
                if convert_to_wav(filepath, wav_path):
                    final_path = wav_path
                    os.remove(filepath)

            # Extract features (combined: MFCC + Chroma + ZCR + RMS)
            features = extract_features(final_path, fixed_length=config.FIXED_LENGTH)

            if features is None:
                return jsonify({
                    'error': 'Failed to extract features from audio',
                    'success': False
                }), 400

            # Inference
            if getattr(config, 'IS_DEMO', False):
                # Deterministic simulation based on file content
                with open(final_path, 'rb') as f:
                    f_content = f.read(1024*1024)
                    file_hash = hashlib.md5(f_content).hexdigest()
                
                # Use hash as seed
                rng = random.Random(file_hash)
                
                # Generate "somewhat real" probabilities
                weights = [rng.uniform(0.1, 0.5) for _ in config.EMOTIONS]
                winner_idx = rng.randint(0, len(config.EMOTIONS) - 1)
                weights[winner_idx] += rng.uniform(2.0, 5.0)
                
                total = sum(weights)
                probabilities = np.array([w / total for w in weights])
            else:
                if not HAS_TORCH:
                    return jsonify({'error': 'Prediction mode requires Torch. Please use Demo mode or install Torch.', 'success': False}), 500
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            prediction_idx = int(np.argmax(probabilities))
            emotion = config.EMOTIONS[prediction_idx]
            confidence = float(probabilities[prediction_idx]) * 100

            # Save to database
            prediction_id = save_prediction(emotion, round(confidence, 2), filename)

            # Cleanup
            if os.path.exists(final_path):
                os.remove(final_path)

            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': round(confidence, 2),
                'prediction_id': prediction_id,
                'all_probabilities': {
                    config.EMOTIONS[i]: round(float(probabilities[i]) * 100, 2)
                    for i in range(len(config.EMOTIONS))
                },
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        predictions = get_all_predictions()
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/history/<int:prediction_id>', methods=['DELETE'])
def delete_history_item(prediction_id):
    try:
        delete_prediction(prediction_id)
        return jsonify({'success': True, 'message': f'Prediction {prediction_id} deleted'})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    return jsonify({'success': True, 'emotions': config.EMOTIONS})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎤 Speech Emotion Analysis – Backend API (Flask)")
    print("=" * 60)
    print(f"Server: http://{config.API_HOST}:{config.API_PORT}")
    print(f"Emotions: {config.EMOTIONS}")
    print("=" * 60 + "\n")

    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)
