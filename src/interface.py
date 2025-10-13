import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
from utils import compute_hog_features, decode_prediction, make_square_crop, most_common_prediction
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load trained model + HOG params
# -------------------------------
data = joblib.load("/Users/saniabhandari/Documents/signlang/models/sign_model_asl.pkl")

if isinstance(data, dict):
    model = data["model"]
    hog_params = data["hog_params"]
    scaler = data.get("scaler")
else:
    model = data
    hog_params = {'orientations':12, 'pixels_per_cell':(4,4), 'cells_per_block':(2,2)}
    scaler = None

# -------------------------------
# 2. Setup MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# -------------------------------
# 3. Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
pred_buffer = deque(maxlen=15)  # rolling buffer

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
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Make crop square
            x_min, y_min, x_max, y_max = make_square_crop(x_min, y_min, x_max, y_max, w, h)
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Resize & grayscale
            hand_img = cv2.resize(hand_img, (28,28))
            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

            # Extract HOG features
            features = compute_hog_features(gray, hog_params)

            # Optional: scale features
            if scaler:
                features = scaler.transform(features)

            # Predict + smoothing
            pred_raw = model.predict(features)[0]
            pred = decode_prediction(pred_raw)
            pred_buffer.append(pred)

            if len(pred_buffer) == pred_buffer.maxlen:
                prediction = most_common_prediction(pred_buffer)
            else:
                prediction = pred  # show raw until buffer fills

            # Show prediction
            cv2.putText(frame, f"Prediction: {prediction}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
