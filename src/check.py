import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib


# ============================
# CONFIG
# ============================

MODEL_PATH = "best_model.keras"   # use best model
SCALER_PATH = "scaler.save"       # created during training

SR = 16000
N_MFCC = 40

# Decision threshold (can tune later)
THRESHOLD = 0.5


# ============================
# LOAD MODEL
# ============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


# ============================
# LOAD SCALER
# ============================

if not os.path.exists(SCALER_PATH):
    print("❌ scaler.save not found!")
    print("Retrain model first.")
    exit()

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded.")


# ============================
# FIND FILE
# ============================

def find_audio(name):

    for root, dirs, files in os.walk("."):
        if name in files:
            return os.path.join(root, name)

    return None


# ============================
# FEATURE EXTRACTION
# ============================

def extract_mfcc(path):

    y, _ = librosa.load(path, sr=SR)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC
    )

    feat = np.mean(mfcc.T, axis=0)

    return feat


# ============================
# PREDICTION
# ============================

def predict(file_name):

    path = find_audio(file_name)

    if path is None:
        print("❌ File not found:", file_name)
        exit()

    print("Using file:", path)

    # Feature
    feat = extract_mfcc(path)

    # Normalize (same as training)
    feat = scaler.transform([feat])

    # Predict
    prob = model.predict(feat, verbose=0)[0][0]

    if prob >= THRESHOLD:
        label = "FAKE (Spoof)"
        conf = prob
    else:
        label = "REAL (Bonafide)"
        conf = 1 - prob

    return label, conf


# ============================
# MAIN
# ============================

if len(sys.argv) != 2:

    print("Usage:")
    print("  python src/check.py audio_file")
    exit()


file = sys.argv[1]

label, confidence = predict(file)

print("\n==============================")
print(" Prediction:", label)
print(" Confidence:", round(confidence * 100, 2), "%")
print("==============================\n")
