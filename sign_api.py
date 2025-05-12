#sign api realtime
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
import cv2
import numpy as np
import joblib
import mediapipe as mp

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x)
        landmarks.append(lm.y)
    return np.array(landmarks)

def predict_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    prediction = "None"
    reply = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            if landmarks.shape[0] == 42:
                raw = model.predict([landmarks])[0]
                raw_str = str(raw)
                reply = reply_dict.get(raw_str, "...")
                prediction = ''.join(filter(str.isalpha, raw_str))
    return prediction, reply

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
