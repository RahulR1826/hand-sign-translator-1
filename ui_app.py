import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
import mediapipe as mp
import threading
import joblib
import os
import mss
import pyttsx3

# Load model and reply dictionary
model = joblib.load("sign_model.joblib")
reply_dict = joblib.load("reply_dict.joblib")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Global status variable and text variable
status_var = None
text_var = None
text_entry = None
engine = pyttsx3.init()

# Extract 21 landmark points (x, y only)
def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x)
        landmarks.append(lm.y)
    return np.array(landmarks)

# Predict from a given frame
def predict_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    prediction = "None"
    reply = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            if landmarks.shape[0] == 42:
                prediction = model.predict([landmarks])[0]
                reply = reply_dict.get(prediction, "...")

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f'{prediction} -> {reply}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Hand Detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return prediction, reply

# Update status label
def update_status(pred, reply):
    if status_var and text_var:
        if pred != "None":
            status_var.set(f"✋ Detected: {pred} → {reply}")
            text_var.set(reply)
        else:
            status_var.set("\U0001F7E1 No hand detected")
            text_var.set("")

def speak_text():
    if text_var:
        text = text_var.get()
        if text:
            engine.say(text)
            engine.runAndWait()

# Webcam recognition thread wrapper
def wrapped_recognize():
    def run():
        cap = cv2.VideoCapture(0)
        print("Press 'r' to play reply sign. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            prediction, reply = predict_frame(frame)
            update_status(prediction, reply)
            cv2.imshow("Webcam Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and reply:
                video_path = f"sign_videos/{reply}.mp4"
                if os.path.exists(video_path):
                    cap_reply = cv2.VideoCapture(video_path)
                    while cap_reply.isOpened():
                        ret_vid, frame_vid = cap_reply.read()
                        if not ret_vid:
                            break
                        cv2.imshow("Reply Sign", frame_vid)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    cap_reply.release()
                    cv2.destroyWindow("Reply Sign")
                else:
                    print(f"Video for reply '{reply}' not found!")

            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        update_status("None", None)

    threading.Thread(target=run).start()

# Screen capture + prediction
def start_screen_capture():
    def capture():
        with mss.mss() as sct:
            screen = sct.grab(sct.monitors[1])
            img = np.array(screen)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            roi = cv2.selectROI("Select Area", img, False, False)
            cv2.destroyWindow("Select Area")

            if roi[2] > 0 and roi[3] > 0:
                monitor = {
                    "top": int(roi[1]),
                    "left": int(roi[0]),
                    "width": int(roi[2]),
                    "height": int(roi[3])
                }

                print("Press 'r' to play reply sign. Press 'q' to quit.")
                while True:
                    img = np.array(sct.grab(monitor))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    prediction, reply = predict_frame(img)
                    update_status(prediction, reply)
                    cv2.imshow("Screen Capture Recognition", img)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('r') and reply:
                        video_path = f"sign_videos/{reply}.mp4"
                        if os.path.exists(video_path):
                            cap_reply = cv2.VideoCapture(video_path)
                            while cap_reply.isOpened():
                                ret_vid, frame_vid = cap_reply.read()
                                if not ret_vid:
                                    break
                                cv2.imshow("Reply Sign", frame_vid)
                                if cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                            cap_reply.release()
                            cv2.destroyWindow("Reply Sign")
                        else:
                            print(f"Video for reply '{reply}' not found!")

                    if key == ord('q'):
                        break

                cv2.destroyAllWindows()
                update_status("None", None)

    threading.Thread(target=capture).start()

# GUI

def create_ui():
    global status_var, text_var, text_entry
    root = tk.Tk()
    root.title("✋ Sign Recognition Interface")
    root.geometry("500x400")
    root.configure(bg="#1e1e2f")

    title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
    button_font = tkfont.Font(family="Helvetica", size=12)

    tk.Label(root, text="Sign Language Recognition", font=title_font,
             bg="#1e1e2f", fg="#f1c40f").pack(pady=(20, 10))

    tk.Button(root, text="🎥 Start Webcam + Gesture", command=wrapped_recognize,
              font=button_font, bg="#27ae60", fg="white",
              activebackground="#2ecc71", relief="flat", padx=10, pady=8).pack(pady=10)

    tk.Button(root, text="🖥️ Screen Capture + Gesture", command=start_screen_capture,
              font=button_font, bg="#2980b9", fg="white",
              activebackground="#3498db", relief="flat", padx=10, pady=8).pack(pady=10)

    tk.Button(root, text="❌ Exit", command=root.quit, font=button_font,
              bg="#c0392b", fg="white", activebackground="#e74c3c",
              relief="flat", padx=10, pady=8).pack(pady=15)

    status_var = tk.StringVar(value="🟢 Ready")
    tk.Label(root, textvariable=status_var, font=("Helvetica", 11),
             bg="#1e1e2f", fg="#ffffff").pack(pady=(5, 5))

    text_var = tk.StringVar(value="")
    text_entry = tk.Entry(root, textvariable=text_var, font=("Helvetica", 11),
                          justify="center", width=40)
    text_entry.pack(pady=(5, 5))

    tk.Button(root, text="🔊 Play Text", command=speak_text,
              font=button_font, bg="#8e44ad", fg="white",
              activebackground="#9b59b6", relief="flat", padx=10, pady=6).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_ui()
