import os
import librosa
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ============================
# BASE DIR
# ============================

BASE = os.path.dirname(os.path.abspath(__file__))

LA_DIR = os.path.join(BASE, "LA")


# ============================
# PATHS
# ============================

AUDIO_DIR = os.path.join(
    LA_DIR,
    "ASVspoof2019_LA_train",
    "flac"
)
# ============================
# FIND PROTOCOL FILE
# ============================

PROTO_DIR = os.path.join(
    LA_DIR,
    "ASVspoof2019_LA_cm_protocols"
)

proto_files = [f for f in os.listdir(PROTO_DIR)
               if "train" in f.lower() and f.endswith(".txt")]

if len(proto_files) == 0:
    raise FileNotFoundError("No train protocol file found!")

PROTOCOL = os.path.join(PROTO_DIR, proto_files[0])

print("Using protocol file:", PROTOCOL)



# ============================
# CONFIG
# ============================

SR = 16000
N_MFCC = 40


# ============================
# MFCC
# ============================

def extract_mfcc(path):

    y, sr = librosa.load(path, sr=SR)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return np.mean(mfcc.T, axis=0)


# ============================
# LOAD DATA
# ============================

def load_data():

    X, y = [], []

    print("Loading LA train data...")

    with open(PROTOCOL) as f:

        for line in tqdm(f):

            parts = line.strip().split()

            file_id = parts[1]
            label = parts[-1]

            wav = os.path.join(AUDIO_DIR, file_id + ".flac")

            if not os.path.exists(wav):
                continue

            feat = extract_mfcc(wav)

            X.append(feat)

            y.append(0 if label=="bonafide" else 1)

    return np.array(X), np.array(y)


# ============================
# PREPARE
# ============================

X, y = load_data()

print("Samples:", X.shape)


scaler = StandardScaler()
X = scaler.fit_transform(X)


Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# MODEL
# ============================

model = Sequential([

    Dense(256, activation="relu", input_shape=(N_MFCC,)),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================
# TRAIN
# ============================

print("Training started...")

model.fit(
    Xtr, ytr,
    validation_data=(Xte, yte),
    epochs=25,
    batch_size=64
)


# ============================
# SAVE
# ============================

model.save(os.path.join(BASE, "model_LA.h5"))

print("Model saved!")
