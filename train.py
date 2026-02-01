import os
import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================
# CONFIG
# ==========================

BASE = "LA"

TRAIN_AUDIO = os.path.join(
    BASE,
    "ASVspoof2019_LA_train",
    "flac"
)

DEV_AUDIO = os.path.join(
    BASE,
    "ASVspoof2019_LA_dev",
    "flac"
)

PROTO_DIR = os.path.join(
    BASE,
    "ASVspoof2019_LA_cm_protocols"
)

TRAIN_PROTO = os.path.join(
    PROTO_DIR,
    "ASVspoof2019.LA.cm.train.trn.txt"
)

DEV_PROTO = os.path.join(
    PROTO_DIR,
    "ASVspoof2019.LA.cm.dev.txt"
)


SR = 16000
N_MFCC = 40


EPOCHS = 50
BATCH = 64


# ==========================
# FEATURE
# ==========================

def extract(path):

    y, _ = librosa.load(path, sr=SR)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC
    )

    return np.mean(mfcc.T, axis=0)



# ==========================
# LOAD SET
# ==========================

def load_set(audio_dir, proto):

    X = []
    y = []

    with open(proto) as f:

        lines = f.readlines()

    print("Loading:", os.path.basename(proto))

    for line in tqdm(lines):

        p = line.strip().split()

        fid = p[1]
        lab = p[-1]

        path = os.path.join(audio_dir, fid + ".flac")

        if not os.path.exists(path):
            continue

        feat = extract(path)

        X.append(feat)

        y.append(0 if lab=="bonafide" else 1)

    return np.array(X), np.array(y)



# ==========================
# LOAD DATA
# ==========================

print("Loading training data...")
Xtr, Ytr = load_set(TRAIN_AUDIO, TRAIN_PROTO)

print("Loading dev data...")
Xdev, Ydev = load_set(DEV_AUDIO, DEV_PROTO)



# ==========================
# BALANCE CLASSES
# ==========================

print("Balancing data...")

idx0 = np.where(Ytr==0)[0]
idx1 = np.where(Ytr==1)[0]

m = min(len(idx0), len(idx1))

idx = np.concatenate([
    np.random.choice(idx0, m, False),
    np.random.choice(idx1, m, False)
])

np.random.shuffle(idx)

Xtr = Xtr[idx]
Ytr = Ytr[idx]



# ==========================
# NORMALIZE
# ==========================

scaler = StandardScaler()

Xtr = scaler.fit_transform(Xtr)
Xdev = scaler.transform(Xdev)



# ==========================
# MODEL
# ==========================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(N_MFCC,)),

    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()



# ==========================
# CALLBACKS
# ==========================

cb = [

    tf.keras.callbacks.EarlyStopping(
        patience=8,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True
    )
]


# ==========================
# TRAIN
# ==========================

print("Training...")

model.fit(
    Xtr, Ytr,
    validation_data=(Xdev, Ydev),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=cb,
    shuffle=True
)



# ==========================
# SAVE
# ==========================

model.save("final_model.keras")

print("Training finished!")
print("Saved: final_model.keras")
