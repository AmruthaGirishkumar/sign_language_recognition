import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Load the trained model
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, '../model/sign_language_model.pkl') 
with open(model_path, 'rb') as f: 
    model = pickle.load(f)
    
# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

current_prediction = ""

print("🔠 Starting letter prediction... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
                print(np.min(landmarks), np.max(landmarks), np.mean(landmarks))
            
            if len(landmarks) == model.n_features_in_:
                prediction = model.predict([landmarks])[0]
                current_prediction = prediction

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show prediction on the screen
    cv2.putText(frame, f'Prediction: {current_prediction}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('ASL Letter Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
