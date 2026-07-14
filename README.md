# Sign Language Recognition

A real-time American Sign Language (ASL) Recognition System that converts hand gestures into text and speech using Computer Vision, Machine Learning, and a custom Tkinter GUI.

---

# Overview

This project recognizes static ASL hand gestures captured through a webcam and converts them into text in real time. It uses MediaPipe to detect hand landmarks and a trained MLPClassifier model from Scikit-learn to classify gestures.

The application also supports sentence formation, speech output, and an interactive graphical user interface.

---

# Features

- Real-time ASL alphabet recognition using a webcam
- Hand landmark detection using MediaPipe
- Gesture classification using a trained MLPClassifier
- Sentence formation from predicted letters
- Text-to-Speech support
- Add Letter, Space, Backspace, and Clear functions
- Interactive Tkinter graphical user interface
- Light/Dark theme toggle
- Toggleable ASL gesture guide
- Smooth real-time prediction

---

# Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Tkinter
- Joblib
- pyttsx3

---

# Project Structure

```
sign_language_recognition/
│
├── app/
│   └── final_app_tkinter.py
│
├── data/
│   ├── x_data.npy
│   ├── y_labels.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   ├── landmark_data.npy
│   └── landmark_labels.npy
│
├── dataset/
│   └── asl_alphabet_test/
│
├── model/
│   ├── sign_language_model.pkl
│   ├── scaler.pkl
│   ├── label_map.json
│   └── model_metadata.json
│
├── scripts/
│   ├── extract_landmarks_from_asl_dataset.py
│   ├── generate_landmark_data.py
│   ├── train_model.py
│   ├── test_model.py
│   ├── predict_letters.py
│   └── predict_to_sentence.py
│
├── tests/
├── feature_utils.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/AmruthaGirishkumar/sign_language_recognition.git

cd sign_language_recognition
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the application

```bash
cd app

python final_app_tkinter.py
```

---

# Model Training (Optional)

To retrain the model:

## Step 1

Download the ASL Alphabet Dataset from Kaggle.

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Step 2

Extract it to the following directory:

```
dataset/
└── asl_alphabet_train/
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

## Step 3

Run the following scripts in order:

```
extract_landmarks_from_asl_dataset.py

generate_landmark_data.py

train_model.py

test_model.py
```

---

# Controls

| Key | Function |
|------|----------|
| Enter | Add predicted letter |
| Space | Insert space |
| Backspace | Delete last character |
| C | Clear sentence |
| S | Speak sentence |
| G | Toggle gesture guide |
| T | Toggle theme |
| Q | Quit application |

---

# Model Information

| Parameter | Value |
|-----------|-------|
| Model | MLPClassifier |
| Input Features | 63 MediaPipe Hand Landmark values |
| Preprocessing | StandardScaler |
| Framework | Scikit-learn |
| Dataset | ASL Alphabet Dataset |
| Test Accuracy | Approximately 99% on the prepared test dataset |

> Note: Real-world prediction accuracy depends on lighting conditions, camera quality, hand positioning, and gesture consistency.

---

# Usage Guidelines

- Use the application in a well-lit environment.
- Ensure only one hand is visible to the webcam.
- Hold each gesture steadily for accurate prediction.
- Use the gesture guide if required.
- The sentence can be spoken using the built-in Text-to-Speech functionality.

---

# Future Improvements

- Dynamic sign recognition
- Word-level prediction
- Sentence auto-completion
- Deep learning models (CNN/LSTM/Transformer)
- Web application deployment
- Mobile application support

---

# Author

**Amrutha Girishkumar**

# License

This project is intended for educational and research purposes.
