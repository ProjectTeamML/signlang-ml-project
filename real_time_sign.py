import cv2
import mediapipe as mp
import joblib
import numpy as np
from skimage.feature import hog

# -------------------------------
# 1. Load the trained model (the brain)
# -------------------------------
saved = joblib.load("models/sign_model_500_30_HOG12.pkl")

# Handle if model was saved inside a dictionary
if isinstance(saved, dict):
    model = saved["model"]          # actual trained model
    scaler = saved.get("scaler")    # optional, only if used
else:
    model = saved
    scaler = None

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

    # Flip horizontally for selfie-view
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe needs RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

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
            x_min, y_min = max(x_min - 20, 0), max(y_min - 20, 0)
            x_max, y_max = min(x_max + 20, w), min(y_max + 20, h)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # -------------------------------
                # 5. Preprocess: resize & grayscale
                # (Match training pipeline exactly!)
                # -------------------------------
                hand_img = cv2.resize(hand_img, (28, 28))
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

                # Extract HOG features (match training notebook!)
                features = hog(
                    gray,
                    orientations=12,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    visualize=False
                )

                features = features.reshape(1, -1)  # shape will be (1, 1728)

                # Optional: scale features if scaler exists
                if scaler is not None:
                    features = scaler.transform(features)

                # -------------------------------
                # 6. Model prediction
                # -------------------------------
                prediction = model.predict(features)[0]

                # -------------------------------
                # 7. Show prediction on screen
                # -------------------------------
                cv2.putText(frame, f"Detected: {prediction}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

    # Display the webcam output
    cv2.imshow("Sign Language Recognition", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()