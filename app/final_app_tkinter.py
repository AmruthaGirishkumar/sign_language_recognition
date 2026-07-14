import os
import sys
import ctypes
import queue
import threading
import tkinter as tk
from PIL import Image, ImageTk

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_utils import load_label_map, preprocess_landmarks

base_path = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_path, "..", "model")
model = joblib.load(os.path.join(model_dir, "sign_language_model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
label_map = list(model.classes_) if hasattr(model, "classes_") else load_label_map(os.path.join(model_dir, "label_map.json"))
if not label_map:
    label_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def predict_letter(landmarks, debug=False):
    if len(landmarks) != 63:
        return ("", 0.0, {}) if debug else ("", 0.0)
    try:
        features = preprocess_landmarks(landmarks).reshape(1, -1)
        probabilities = model.predict_proba(scaler.transform(features))[0]
        index = int(np.argmax(probabilities))
        letter = str(model.classes_[index]) if hasattr(model, "classes_") else label_map[index]
        confidence = float(probabilities[index])
        if debug:
            top_indices = np.argsort(probabilities)[-5:][::-1]
            return letter, confidence, {"top_5": [(str(model.classes_[i]), float(probabilities[i])) for i in top_indices]}
        return letter, confidence
    except Exception as error:
        return ("", 0.0, {"error": str(error)}) if debug else ("", 0.0)


speech_queue = queue.Queue()


def speech_worker():
    """Run all SAPI calls on one COM-initialized background thread."""
    engine = None
    com_initialized = False
    try:
        if os.name == "nt":
            ctypes.windll.ole32.CoInitializeEx(None, 0x2)
            com_initialized = True
        while True:
            text = speech_queue.get()
            try:
                if text is None:
                    return
                if engine is None:
                    engine = pyttsx3.init(driverName="sapi5")
                    engine.setProperty("rate", 130)
                engine.say(text)
                engine.runAndWait()
            except Exception as error:
                print(f"Text-to-speech error: {error}")
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                engine = None
            finally:
                speech_queue.task_done()
    finally:
        if com_initialized:
            ctypes.windll.ole32.CoUninitialize()


threading.Thread(target=speech_worker, name="speech-worker", daemon=True).start()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

sentence = ""
current_letter = ""
prediction_buffer = []
show_guide = False
debug_mode = False
dark_mode = False
closing = False
cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("950x700")
root.configure(bg="#EAF6F6")

top_frame = tk.Frame(root, bg="#EAF6F6")
top_frame.pack(side="top", pady=5)
main_frame = tk.Frame(root, bg="#EAF6F6")
main_frame.pack(pady=10)
video_guide_frame = tk.Frame(main_frame, bg="#EAF6F6")
video_guide_frame.pack()
video_label = tk.Label(video_guide_frame, bg="#FFFFFF", width=500, height=400)
video_label.pack(side="left", padx=5)

try:
    guide_image = Image.open(os.path.join(base_path, "gesture_guide.png")).resize((400, 400), Image.Resampling.LANCZOS)
    guide_photo = ImageTk.PhotoImage(guide_image)
    guide_label = tk.Label(video_guide_frame, image=guide_photo, bg="#F0F0F0")
    guide_label.image = guide_photo
except Exception as error:
    print(f"Warning: Could not load gesture guide: {error}")
    guide_label = tk.Label(video_guide_frame, text="ASL Gesture Guide\n(unavailable)", bg="#F0F0F0", fg="#666666", font=("Arial", 12))
guide_label.pack_forget()

prediction_var = tk.StringVar(value="Prediction: ")
sentence_var = tk.StringVar(value="Sentence: ")
prediction_label = tk.Label(root, textvariable=prediction_var, font=("Comic Sans MS", 18, "bold"), fg="#3C91E6", bg="#EAF6F6")
prediction_label.pack(pady=5)
sentence_label = tk.Label(root, textvariable=sentence_var, font=("Times New Roman", 20), fg="#FF6B6B", bg="#EAF6F6")
sentence_label.pack(pady=5)

def speak_sentence():
    if sentence:
        speech_queue.put(sentence)

def clear_sentence():
    global sentence
    sentence = ""
    sentence_var.set("Sentence: ")


def add_letter():
    global sentence
    if current_letter:
        sentence += current_letter
        sentence_var.set(f"Sentence: {sentence}")


def add_space():
    global sentence
    sentence += " "
    sentence_var.set(f"Sentence: {sentence}")


def delete_last():
    global sentence
    if sentence:
        sentence = sentence[:-1]
        sentence_var.set(f"Sentence: {sentence}")


def toggle_gesture_guide():
    global show_guide
    show_guide = not show_guide
    if show_guide:
        guide_label.pack(side="right", padx=5)
    else:
        guide_label.pack_forget()


def toggle_debug_mode():
    global debug_mode
    debug_mode = not debug_mode
    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")


def toggle_theme():
    global dark_mode
    if not dark_mode:
        root.configure(bg="#2E2E2E")
        prediction_label.configure(bg="#2E2E2E", fg="lightgreen")
        sentence_label.configure(bg="#2E2E2E", fg="lightblue")
    else:
        root.configure(bg="#F0F0F0")
        prediction_label.configure(bg="#F0F0F0", fg="darkgreen")
        sentence_label.configure(bg="#F0F0F0", fg="darkblue")
    dark_mode = not dark_mode


def close_application():
    global closing
    if closing:
        return
    closing = True
    speech_queue.put(None)
    if cap.isOpened():
        cap.release()
    hands.close()
    root.destroy()


def key_event(event):
    key = event.char.lower()
    if key == "s":
        speak_sentence()
    elif key == "c":
        clear_sentence()
    elif key == " ":
        add_space()
    elif key == "\r":
        add_letter()
    elif key == "t":
        toggle_theme()
    elif key == "g":
        toggle_gesture_guide()
    elif key == "d":
        toggle_debug_mode()
    elif event.keysym == "BackSpace":
        delete_last()
    elif key == "q":
        close_application()

root.bind("<Key>", key_event)
button_style = {"font": ("Verdana", 10), "bg": "#FFDDC1", "fg": "#333", "padx": 5, "pady": 2}
for text, command in [
    ("Add Letter (Enter)", add_letter), ("Space (Space)", add_space), ("Backspace", delete_last),
    ("Clear (C)", clear_sentence), ("Speak (S)", speak_sentence),
    ("Toggle Guide (G)", toggle_gesture_guide), ("Toggle Theme (T)", toggle_theme), ("Quit (Q)", close_application),
]:
    tk.Button(top_frame, text=text, command=command, **button_style).pack(side="left", padx=5)


def update_video():
    global current_letter
    if closing:
        return
    success, frame = cap.read()
    if not success:
        video_label.after(30, update_video)
        return

    # Do not mirror input to MediaPipe: training used unmirrored images.
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    letter, confidence, debug_info = "", 0.0, {}
    hand_detected = bool(results.multi_hand_landmarks)
    if hand_detected:
        landmarks = [value for landmark in results.multi_hand_landmarks[0].landmark for value in (landmark.x, landmark.y, landmark.z)]
        if debug_mode:
            letter, confidence, debug_info = predict_letter(landmarks, debug=True)
        else:
            letter, confidence = predict_letter(landmarks)

    if letter and confidence > 0.40:
        prediction_buffer.append((letter, confidence))
        if len(prediction_buffer) > 5:
            prediction_buffer.pop(0)
    if prediction_buffer:
        scores = {}
        for buffered_letter, buffered_confidence in prediction_buffer:
            scores[buffered_letter] = scores.get(buffered_letter, 0.0) + buffered_confidence
        current_letter = max(scores, key=scores.get)
        confidence = scores[current_letter] / len(prediction_buffer)
    else:
        current_letter, confidence = "", 0.0
    prediction_var.set(f"Prediction: {current_letter} ({confidence * 100:.1f}%)" if current_letter else "Prediction: ")

    display = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image_array = np.asarray(Image.fromarray(display).resize((500, 400)))
    if debug_mode:
        cv2.putText(image_array, "DEBUG MODE ON (Press D to toggle)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image_array, f"Hand Detected: {hand_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for offset, (top_letter, probability) in enumerate(debug_info.get("top_5", []), start=1):
            cv2.putText(image_array, f"{offset}. {top_letter}: {probability * 100:.1f}%", (10, 75 + 25 * offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    photo = ImageTk.PhotoImage(Image.fromarray(image_array))
    video_label.imgtk = photo
    video_label.configure(image=photo)
    video_label.after(10, update_video)


root.protocol("WM_DELETE_WINDOW", close_application)
update_video()
root.mainloop()