import cv2
import mediapipe as mp
import joblib
import numpy as np
from skimage.feature import hog
from collections import Counter # Used for temporal smoothing

# -------------------------------
# A. Crucial Mapping for Sign MNIST
# -------------------------------
# Standard Sign MNIST mapping (0-24, skipping 'J' and 'Z')
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# --- CONFIGURATION ---
# Padding around the hand bounding box (try 30-40)
PADDING = 30 
# Number of frames to average predictions over (5 is a good starting point)
PREDICTION_BUFFER_SIZE = 7 
# ---------------------

# Temporal Smoothing Buffer
prediction_buffer = []

# -------------------------------
# 1. Load the trained model (the brain)
# -------------------------------
try:
    saved = joblib.load("models/sign_model_500_30_HOG12.pkl")

    if isinstance(saved, dict):
        model = saved["model"]       
        scaler = saved.get("scaler") 
    else:
        model = saved
        scaler = None
    
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file not found. Ensure 'models/sign_model_500_30_HOG12.pkl' exists.")
    exit()

# -------------------------------
# 2. Setup MediaPipe Hands (the hand detector)
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
# 3. Start webcam (the eyes)
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for selfie-view (CRUCIAL)
    frame = cv2.flip(frame, 1) 

    # Convert to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_sign = "NO HAND" 
    confidence = 0.0

    # -------------------------------
    # 4. If hand is detected
    # -------------------------------
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Add safe padding
            x_min, y_min = max(x_min - PADDING, 0), max(y_min - PADDING, 0)
            x_max, y_max = min(x_max + PADDING, w), min(y_max + PADDING, h)

            # Draw the bounding box for visual debugging
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Crop hand region 
            hand_img_bgr = frame[y_min:y_max, x_min:x_max]

            if hand_img_bgr.size > 0:
                # -------------------------------
                # 5. Preprocess: resize & grayscale (Match training pipeline!)
                # -------------------------------
                hand_img = cv2.resize(hand_img_bgr, (28, 28))
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

                # Extract HOG features (MUST MATCH TRAINING NOTEBOOK!)
                features = hog(
                    gray,
                    orientations=12,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    visualize=False
                )

                features = features.reshape(1, -1) 

                if scaler is not None:
                    features = scaler.transform(features)

                # -------------------------------
                # 6. Model prediction with Temporal Smoothing
                # -------------------------------
                
                # 6a. Get the immediate prediction from the model
                instant_prediction_index = model.predict(features)[0]
                
                # 6b. Add instant prediction to buffer and maintain size
                prediction_buffer.append(instant_prediction_index)
                if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
                    prediction_buffer.pop(0) 
                
                # 6c. Get the STABLE prediction (the most frequent in the buffer)
                if prediction_buffer:
                    # Counter finds the most common item in the list
                    stable_prediction_index = Counter(prediction_buffer).most_common(1)[0][0]
                else:
                    stable_prediction_index = instant_prediction_index


                # 6d. Calculate confidence based on the stable prediction
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0]
                    if stable_prediction_index < len(probabilities):
                        confidence = probabilities[stable_prediction_index] * 100
                else:
                    confidence = 100.0 # Placeholder
                
                # Map the stable prediction number to the sign letter
                detected_sign = LABEL_MAP.get(stable_prediction_index, f"Idx:{stable_prediction_index} (Check Map)")


    # -------------------------------
    # 7. Show prediction on screen
    # -------------------------------
    # Choose color based on confidence: Green for high, Red for low
    if confidence > 80:
        color = (0, 255, 0) # Green
    elif confidence > 50:
        color = (0, 255, 255) # Yellow
    else:
        color = (0, 0, 255) # Red

    cv2.putText(frame, f"Detected: {detected_sign}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)
    
    cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    # Display the webcam output
    cv2.imshow("Sign Language Recognition", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()