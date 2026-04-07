from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv, io, soundfile as sf, resampy

app = Flask(__name__)
CORS(app)

# ─── Load YAMNet model ───────────────────────────────────────────────
print("Loading YAMNet model...")
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

def class_names_from_csv(text):
    class_map_csv = io.StringIO(text)
    names = [d for (_, _, d) in csv.reader(class_map_csv)]
    return names[1:]

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(
    tf.io.read_file(class_map_path).numpy().decode('utf-8')
)
print("✅ Model loaded!\n")

# ─── Map YAMNet → your threat categories ────────────────────────────
THREAT_MAP = {
    'chainsaw': ['Chainsaw', 'Power tool', 'Sawing'],
    'gunshot':  ['Gunshot', 'Gunfire', 'Explosion', 'Bang'],
    'vehicle':  ['Car', 'Engine', 'Truck', 'Motor vehicle'],
    'animal':   ['Bird', 'Dog', 'Animal', 'Frog', 'Insect'],
}

def map_to_threat(top_class):
    for threat, keywords in THREAT_MAP.items():
        if any(k.lower() in top_class.lower() for k in keywords):
            return threat
    return 'ambient'

# ─── Predict endpoint ────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Read and preprocess audio
    waveform, sr = sf.read(io.BytesIO(file.read()), dtype='float32')
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)       # stereo → mono
    if sr != 16000:
        waveform = resampy.resample(waveform, sr, 16000)

    # Run model
    scores, _, _ = model(waveform)
    mean_scores = scores.numpy().mean(axis=0)
    top_idx = mean_scores.argmax()
    top_class = class_names[top_idx]
    confidence = float(mean_scores[top_idx])
    threat = map_to_threat(top_class)

    return jsonify({
        'raw_class':   top_class,
        'threat_type': threat,
        'confidence':  round(confidence * 100, 1)
    })

# ─── Health check ────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return "✅ YAMNet API is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5001)