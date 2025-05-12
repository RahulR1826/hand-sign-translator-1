import tkinter as tk
import cv2
import numpy as np
import mediapipe as mp
import threading
import joblib
import os
import mss
import tkinter.ttk as ttk 
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
                if reply:
                    cv2.putText(frame, f'{reply}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "No Hand Detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return prediction, reply


# Webcam recognition
def recognize():
    cap = cv2.VideoCapture(0)

    print("Press 'r' to play reply sign. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction, reply = predict_frame(frame)
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

    threading.Thread(target=capture).start()


# GUI
def create_ui():
    tts_engine = pyttsx3.init()

    root = tk.Tk()
    root.title("Sign Language Translator")
    root.geometry("420x380")
    root.configure(bg="#e9f0f7")  # Light blue-gray background

    # Custom style
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TButton",
                    font=("Segoe UI", 10, "bold"),
                    padding=6,
                    foreground="white",
                    background="#007acc")
    style.map("TButton",
              background=[('active', '#005f99')])

    style.configure("TLabelframe",
                    background="#e9f0f7",
                    foreground="#003366",
                    font=("Segoe UI", 11, "bold"))

    style.configure("TLabel",
                    background="#e9f0f7",
                    foreground="#333")

    # Title
    title_label = tk.Label(root, text="Sign Language Translator",
                           font=("Segoe UI", 16, "bold"),
                           bg="#e9f0f7", fg="#003366")
    title_label.pack(pady=15)

    # Gesture Recognition Section
    gesture_frame = ttk.LabelFrame(root, text="Gesture Recognition")
    gesture_frame.pack(padx=20, pady=10, fill="x")

    webcam_btn = ttk.Button(gesture_frame, text="Start Webcam + Gesture", command=recognize)
    webcam_btn.pack(pady=5, padx=20, fill="x")

    screen_btn = ttk.Button(gesture_frame, text="Start Screen Capture + Gesture", command=start_screen_capture)
    screen_btn.pack(pady=5, padx=20, fill="x")

    # Text-to-Speech Section
    tts_frame = ttk.LabelFrame(root, text="Text-to-Speech")
    tts_frame.pack(padx=20, pady=10, fill="x")

    tts_entry = tk.Entry(tts_frame, font=("Segoe UI", 10), width=40, bg="#ffffff", fg="#000")
    tts_entry.pack(pady=5, padx=10)

    def play_text():
        text = tts_entry.get()
        if text.strip():
            tts_engine.say(text)
            tts_engine.runAndWait()

    play_btn = ttk.Button(tts_frame, text="Play Text", command=play_text)
    play_btn.pack(pady=5, padx=20, fill="x")

    # Exit Button
    quit_btn = ttk.Button(root, text="Exit", command=root.quit)
    quit_btn.pack(pady=20, fill="x", padx=60)

    root.mainloop()


if __name__ == "__main__":
    create_ui()

