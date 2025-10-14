import cv2
import mediapipe as mp
import joblib
import numpy as np
from skimage.feature import hog
from collections import Counter
import time 

# -------------------------------
# A. Crucial Mapping for Sign MNIST
# -------------------------------
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# --- CONFIGURATION (Easily tunable) ---
# Padding around the hand bounding box 
PADDING = 30 
# Number of frames to average predictions over 
PREDICTION_BUFFER_SIZE = 7 
# -------------------------------------

# Temporal Smoothing Buffer and State Management
prediction_buffer = []
current_stable_sign = "NO HAND"
current_confidence = 0.0

# -------------------------------
# 1. Load the trained model and scaler
# -------------------------------
MODEL_PATH = "models/sign_model_500_30_HOG12.pkl"
try:
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict):
        model = saved["model"]       
        scaler = saved.get("scaler") 
    else:
        model = saved
        scaler = None
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found. Ensure '{MODEL_PATH}' exists.")
    exit()

# -------------------------------
# 2. Setup MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6
)

# -------------------------------
# 3. Start webcam and timing
# -------------------------------
cap = cv2.VideoCapture(0)
prev_time = time.time() # For FPS calculation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for selfie-view 
    frame = cv2.flip(frame, 1) 
    
    # CRITICAL FIX: Define h and w here for use by FPS counter and drawing
    h, w, _ = frame.shape 
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    result = hands.process(rgb)

    detected_this_frame = False
    
    # -------------------------------
    # 4. If hand is detected
    # -------------------------------
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        detected_this_frame = True

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
        x_max, y_max = int(max(x_coords) * w), int(max(y_coords) * h)

        # Add safe padding
        x_min, y_min = max(x_min - PADDING, 0), max(y_min - PADDING, 0)
        x_max, y_max = min(x_max + PADDING, w), min(y_max + PADDING, h)

        # Crop hand region 
        hand_img_bgr = frame[y_min:y_max, x_min:x_max]

        if hand_img_bgr.size > 0 and hand_img_bgr.shape[0] > 0 and hand_img_bgr.shape[1] > 0:
            
            # -------------------------------
            # 5. Preprocess & Feature Extraction (CRITICAL ENHANCEMENT)
            # -------------------------------
            hand_img = cv2.resize(hand_img_bgr, (28, 28))
            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

            # 5a. Feature: Histogram Equalization (Boost contrast)
            # This makes lighting more consistent
            gray_enhanced = cv2.equalizeHist(gray) 

            # 5b. Feature: Thresholding + OTSU (Mimic MNIST white-on-black)
            # OTSU automatically calculates the best threshold for the image, isolating the hand.
            # cv2.THRESH_BINARY_INV creates a white hand on a black background, better matching the data.
            _, processed_image = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # DEBUG FEATURE: Show the HOG input image
            cv2.imshow("HOG Input (28x28 Isolated)", processed_image)
            
            # Extract HOG features 
            features = hog(
                processed_image, # Use the isolated image for better feature extraction!
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
            
            instant_prediction_index = model.predict(features)[0]
            
            prediction_buffer.append(instant_prediction_index)
            if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
                prediction_buffer.pop(0) 
            
            stable_prediction_index = Counter(prediction_buffer).most_common(1)[0][0]
            current_stable_sign = LABEL_MAP.get(stable_prediction_index, f"?{stable_prediction_index}")
            
            # Calculate confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                if stable_prediction_index < len(probabilities):
                    current_confidence = probabilities[stable_prediction_index] * 100
            else:
                current_confidence = 100.0

            # Choose color based on confidence 
            if current_confidence > 85:
                color = (0, 255, 0) # Green 
            elif current_confidence > 65:
                color = (0, 255, 255) # Yellow 
            else:
                color = (0, 0, 255) # Red 

            # Re-draw bounding box with the confidence-based color
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Add label INSIDE the bounding box
            cv2.putText(frame, current_stable_sign,
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)


    # -------------------------------
    # 7. Post-Processing and Display
    # -------------------------------

    # Buffer Reset logic
    if not detected_this_frame:
        if prediction_buffer:
            prediction_buffer.clear()
        current_stable_sign = "NO HAND"
        current_confidence = 0.0
        
    # Update the overall prediction text and confidence color
    if current_confidence > 85:
        display_color = (0, 255, 0) 
    elif current_confidence > 65:
        display_color = (0, 255, 255)
    else:
        # Orange for 'NO HAND', Red for low-confidence prediction
        display_color = (0, 165, 255) if current_stable_sign == "NO HAND" else (0, 0, 255) 

    # FPS Counter
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Stable Prediction Display (Top-Left)
    cv2.putText(frame, f"Stable Sign: {current_stable_sign}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, display_color, 2)
    
    cv2.putText(frame, f"Confidence: {current_confidence:.1f}%",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, display_color, 2)

    # Display the webcam output
    cv2.imshow("Sign Language Recognition [Stable]", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()