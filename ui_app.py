import tkinter as tk
import cv2
import numpy as np
import mediapipe as mp
import threading
import joblib
import os
import mss

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
# Predict from a given frame
def predict_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    prediction = "None"
    reply = None

    # Define replies only for specific word signs
    signs_with_replies = {
        "Hello": "Hi!",
        "Help": "Do you need help?",
        "Yes": "Alright!",
        "No": "No problem",
        "Thank You":"You're Welcome",
        "I Love You": "Love you too",
        "OK/Okay": "Okay!",
        "Peace/Victory": "Peace!",
        "STR": "ATMEN",
    }

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            if landmarks.shape[0] == 42:
                predicted_id = model.predict([landmarks])[0]
                prediction = reply_dict.get(predicted_id, "Unknown")

                reply = signs_with_replies.get(prediction, None)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f'{prediction}' + (f' -> {reply}' if reply else ''),
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    root = tk.Tk()
    root.title("Sign Recognition Interface")
    root.geometry("300x200")

    webcam_btn = tk.Button(root, text="Start Webcam + Gesture", command=recognize)
    webcam_btn.pack(pady=10)

    screen_btn = tk.Button(root, text="Start Screen Capture + Gesture", command=start_screen_capture)
    screen_btn.pack(pady=10)

    quit_btn = tk.Button(root, text="Exit", command=root.quit, width=25)
    quit_btn.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_ui()

