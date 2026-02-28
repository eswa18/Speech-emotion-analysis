import joblib
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import config

print(f"Checking model at: {config.MODEL_PATH}")
print(f"Checking scaler at: {config.SCALER_PATH}")

if os.path.exists(config.MODEL_PATH):
    print("Model file exists.")
    try:
        model = joblib.load(config.MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file DOES NOT exist.")

if os.path.exists(config.SCALER_PATH):
    print("Scaler file exists.")
    try:
        scaler = joblib.load(config.SCALER_PATH)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
else:
    print("Scaler file DOES NOT exist.")
