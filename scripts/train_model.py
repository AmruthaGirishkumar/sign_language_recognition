import argparse
import json
import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from feature_utils import FEATURE_DIM, flatten_landmarks, preprocess_landmarks, save_label_map


def load_dataset(dataset_dir, max_images_per_class=500):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    features = []
    labels = []
    total_images = 0
    detected_hands = 0

    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"  Processing class: {class_name}")
        class_images = 0
        class_detected = 0

        for image_name in sorted(os.listdir(class_dir)):
            if class_images >= max_images_per_class:
                break
            total_images += 1
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            class_images += 1
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                continue

            class_detected += 1
            detected_hands += 1
            raw_landmarks = flatten_landmarks(results.multi_hand_landmarks[0])
            if len(raw_landmarks) != FEATURE_DIM:
                continue

            features.append(raw_landmarks)
            labels.append(class_name)

        print(f"    -> {class_detected}/{class_images} images had detectable hands")

    print(f"\nTotal: {detected_hands}/{total_images} images had detectable hands")
    if not features:
        raise ValueError(f"No usable hand landmarks extracted from {dataset_dir}")

    return np.array(features, dtype=float), np.array(labels, dtype=object)


def train_model(dataset_dir, output_dir, max_images_per_class=500):
    raw_features, labels = load_dataset(dataset_dir, max_images_per_class)
    X_features = np.array([preprocess_landmarks(sample) for sample in raw_features], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "sign_language_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    label_map_path = os.path.join(output_dir, "label_map.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    save_label_map(label_map_path, model.classes_.tolist())

    metadata = {
        "accuracy": float(accuracy),
        "feature_dim": FEATURE_DIM,
        "preprocessing": "relative_to_wrist_and_palm_scale",
        "dataset_dir": dataset_dir,
        "labels": model.classes_.tolist(),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Metadata saved to: {metadata_path}")


def find_dataset_dir():
    """Auto-detect the dataset location."""
    candidates = [
        os.path.join(ROOT_DIR, "dataset", "asl_alphabet_train"),
        os.path.join(ROOT_DIR, "dataset", "archive", "asl_alphabet_train", "asl_alphabet_train"),
        os.path.join(ROOT_DIR, "dataset", "archive", "asl_alphabet_train"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            # Check if it has letter subdirectories
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            if any(d in subdirs for d in ["A", "B", "C"]):
                print(f"Auto-detected dataset at: {candidate}")
                return candidate
    
    # Default if not found
    return os.path.join(ROOT_DIR, "dataset", "asl_alphabet_train")


def main():
    parser = argparse.ArgumentParser(description="Train a sign language recognition model")
    parser.add_argument(
        "--dataset-dir",
        default=find_dataset_dir(),
        help="Path to the downloaded ASL alphabet dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "model"),
        help="Directory for saving the trained model and metadata",
    )
    args = parser.parse_args()

    train_model(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
