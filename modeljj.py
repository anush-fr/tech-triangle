import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io

# ─── Load YAMNet model ───────────────────────────────────────────────
print("Loading YAMNet model...")
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# ─── Load class names ────────────────────────────────────────────────
def class_names_from_csv(class_map_csv_text):
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    return class_names[1:]  # Skip CSV header

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
print("Model loaded successfully!\n")

# ─── Load a WAV file ─────────────────────────────────────────────────
def load_wav(file_path):
    raw = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(raw, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)  # shape: (samples,)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        print(f"  ⚠ Sample rate is {sample_rate}, resampling to 16000Hz")
        waveform = tf.signal.resample(waveform, int(len(waveform) * 16000 / sample_rate))

    return waveform.numpy().astype(np.float32)

# ─── Run YAMNet on one file ──────────────────────────────────────────
def predict(file_path):
    print(f"Testing: {file_path}")
    waveform = load_wav(file_path)
    scores, embeddings, log_mel_spectrogram = model(waveform)

    mean_scores = scores.numpy().mean(axis=0)
    top5_indices = mean_scores.argsort()[-5:][::-1]

    print(f"  🎯 Top detected sounds:")
    for i, idx in enumerate(top5_indices):
        print(f"     {i+1}. {class_names[idx]:40s} (score: {mean_scores[idx]:.4f})")
    print()

# ─── Your 5 audio files ──────────────────────────────────────────────
# 👇 Put your actual filenames here
audio_files = [
    r"C:\Users\ADMIN\Desktop\forest proj\audio1.wav",
    r"C:\Users\ADMIN\Desktop\forest proj\audio2.wav",
    r"C:\Users\ADMIN\Desktop\forest proj\audio3.wav",
    r"C:\Users\ADMIN\Desktop\forest proj\audio4.wav",
    r"C:\Users\ADMIN\Desktop\forest proj\audio5.wav",
]

# ─── Run on all files ────────────────────────────────────────────────
for f in audio_files:
    try:
        predict(f)
    except Exception as e:
        print(f"  ❌ Error on {f}: {e}\n")
        