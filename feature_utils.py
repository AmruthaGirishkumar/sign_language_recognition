import json
import os
import numpy as np

RAW_LANDMARKS_PER_HAND = 21
FEATURE_DIM = 63


def flatten_landmarks(hand_landmarks):
    """Convert a MediaPipe hand result into a flat list of x/y/z values."""
    return [value for landmark in hand_landmarks.landmark for value in (landmark.x, landmark.y, landmark.z)]


def preprocess_landmarks(landmarks):
    """Convert raw MediaPipe landmarks into scale- and position-invariant features."""
    values = np.asarray(landmarks, dtype=np.float32).reshape(-1)
    if values.size != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} values, got {values.size}")

    pts = values.reshape(RAW_LANDMARKS_PER_HAND, 3)
    wrist = pts[0]
    relative = pts - wrist

    palm_vector = pts[9] - wrist
    scale = np.linalg.norm(palm_vector)
    if scale < 1e-6:
        scale = 1.0

    relative = relative / scale
    return relative.reshape(-1)


def save_label_map(path, labels):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)


def load_label_map(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
