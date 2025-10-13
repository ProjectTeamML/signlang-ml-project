import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
from utils import compute_hog_features, decode_prediction

# -------------------------------
# 1. Load the trained model (the brain) and HOG params
# -------------------------------
saved = joblib.load("../models/sign_model_500_30_HOG12.pkl")

if isinstance(saved, dict):
    model = saved["model"]
    hog_params = saved["hog_params"]
    scaler = saved.get("scaler")
else:
    model = saved
    hog_params = {
        'orientations': 12,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (2, 2)
    }
    scaler = None

# -------------------------------
# 2. Setup MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# 3. Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
pred_buffer = deque(maxlen=15)  # rolling buffer for smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            # 1️⃣ Get raw bounding box
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # 2️⃣ Center into square
            box_w, box_h = x_max - x_min, y_max - y_min
            size = max(box_w, box_h)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_min = max(x_center - size // 2, 0)
            x_max = min(x_center + size // 2, w)
            y_min = max(y_center - size // 2, 0)
            y_max = min(y_center + size // 2, h)

            # Crop & preprocess
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess: resize & grayscale
            hand_img = cv2.resize(hand_img, (28, 28))
            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

            # Extract HOG features
            features = compute_hog_features(gray, hog_params)

            # Optional scaling
            if scaler is not None:
                features = scaler.transform(features)

            # Skip low-confidence frames
            # if not is_high_confidence(model, features):
            #     continue

            # Prediction + smoothing
            pred_raw = model.predict(features)[0]
            pred = decode_prediction(pred_raw)
            pred_buffer.append(pred)

            if len(pred_buffer) == pred_buffer.maxlen:
                prediction = Counter(pred_buffer).most_common(1)[0][0]
            else:
                prediction = pred  # show raw until buffer fills

            print("Raw:", pred, "Smoothed:", prediction)
            cv2.putText(frame, f"Prediction: {prediction}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
