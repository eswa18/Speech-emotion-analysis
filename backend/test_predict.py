import requests
import numpy as np
import scipy.io.wavfile as wav
import io

# Generate a 1-second sine wave at 440Hz
sr = 22050
t = np.linspace(0, 1, sr, endpoint=False)
x = 0.5 * np.sin(2 * np.pi * 440 * t)

# Save to in-memory buffer
buffer = io.BytesIO()
wav.write(buffer, sr, (x * 32767).astype(np.int16))
buffer.seek(0)

# Send request
url = 'http://localhost:5000/api/predict'
files = {'audio': ('test.wav', buffer, 'audio/wav')}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
