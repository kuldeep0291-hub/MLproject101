MLproject101 – Deepfake Voice Detection Using MFCC and Deep Learning
📌 Overview

MLproject101 is a machine learning project focused on detecting deepfake (spoofed) voices using audio feature extraction and deep learning techniques. The project uses Mel-Frequency Cepstral Coefficients (MFCCs) for feature extraction and a neural network model for classification.

The goal of this project is to classify audio samples as:

✅ Real (Bonafide)

❌ Fake (Spoofed)

This project is suitable for beginners who want to learn about audio processing, MFCCs, and applying deep learning to real-world problems.

✨ Features

Audio preprocessing using MFCC

Dataset download scripts (Linux & Windows)

Model training pipeline

Model evaluation

Prediction on new audio files

Pre-trained models included

📁 Project Structure
MLproject101/
│
├── download_linux.py       # Dataset download (Linux)
├── download_windows.py     # Dataset download (Windows)
├── train.py                # Training script
├── check.py                # Prediction / testing script
├── best_model.keras        # Best saved model
├── final_model.keras       # Final trained model
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
🧠 Technologies Used

Python 3.x

TensorFlow / Keras

NumPy

Librosa

Scikit-learn

KaggleHub

📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/kuldeep0291-hub/MLproject101.git
cd MLproject101
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt is missing, install manually:

pip install tensorflow numpy librosa scikit-learn kagglehub tqdm soundfile
📊 Dataset

This project uses the ASVspoof 2019 Dataset for training and evaluation.

Download Dataset

For Linux:

python download_linux.py

For Windows:

python download_windows.py

Make sure you have your Kaggle API credentials configured properly.

⚙️ Preprocessing

Each audio file is processed as follows:

Load audio using Librosa

Resample to target sampling rate

Apply framing and windowing

Extract MFCC features

Normalize features

Store in NumPy arrays

These features are then used as input to the neural network.

🚀 Training the Model

To train the model, run:

python train.py

This script will:

Load and preprocess data

Split into train/test sets

Train a neural network

Save the best model

Display training metrics

Saved models:

best_model.keras

final_model.keras

🔍 Model Evaluation

During training, the following metrics are monitored:

Accuracy

Loss

Validation Accuracy

Validation Loss

You can extend this with:

Confusion Matrix

Precision / Recall / F1 Score

🎯 Making Predictions

To test the model on new audio files:

python check.py --file path_to_audio.wav

Output example:

Prediction: REAL
Confidence: 94.2%

Supported format: .wav

📈 Sample Results (Example)
Metric	Value
Accuracy	91%
Loss	0.28

Results may vary depending on hardware and dataset split.

🛠️ Customization

You can modify the following parameters in train.py:

Number of MFCCs

Learning rate

Batch size

Number of epochs

Model architecture

Example:

EPOCHS = 50
BATCH_SIZE = 32
N_MFCC = 13
⚠️ Common Issues
1. Librosa Installation Error
pip install librosa --upgrade
2. CUDA / GPU Not Detected

Make sure you have installed:

CUDA Toolkit

cuDNN

Compatible TensorFlow version

Check with:

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
3. Dataset Not Found

Ensure dataset path is correct in scripts.

📌 Future Improvements

Add CNN/RNN models

Use spectrogram images

Add web interface

Improve evaluation metrics

Support more datasets

Hyperparameter tuning

🤝 Contributing

Contributions are welcome!

Steps:

Fork the repo

Create a new branch

Commit your changes

Push to your branch

Open a Pull Request

📄 License

This project is licensed under the MIT License. See LICENSE file for details.

👤 Author

Kuldeep Bishnoi

GitHub: https://github.com/kuldeep0291-hub

⭐ Acknowledgements

ASVspoof Dataset Team

Librosa Developers

TensorFlow Community

If you find this project useful, please consider giving it a ⭐ on GitHub!
