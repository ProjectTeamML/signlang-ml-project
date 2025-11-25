
"""
hand_graph_pipeline.py

Usage:
  # Collect landmark samples (press letter keys to save samples for that label)
  python hand_graph_pipeline.py collect

  # Train classifier on saved landmark csv files
  python hand_graph_pipeline.py train

  # Run real-time prediction using the trained landmark model
  python hand_graph_pipeline.py predict

Requirements:
  pip install mediapipe opencv-python scikit-learn pandas joblib numpy
"""

import sys
import os
import cv2
import joblib
import time
import numpy as np
import pandas as pd

# sklearn for training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# mediapipe
try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("Install mediapipe: pip install mediapipe") from e

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Config
DATA_DIR = "landmark_data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "landmark_model.pkl")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")  # append mode
SAMPLES_PER_SAVE = 1  # number of frames saved per keypress (1 is fine)
TARGET_LABELS = None   # optional: set to list e.g. ['A','B','C'] to restrict allowed labels
MAX_SAVED = None       # optional: limit saves per label during collect

# MediaPipe config
MP_CONFIG = dict(static_image_mode=False, max_num_hands=1,
                 min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Ensure directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Utility: convert multi-hand landmarks to a fixed-length feature vector.
# We'll use 21 landmarks, each (x,y,z) normalized relative to wrist bounding box.
NUM_LANDMARKS = 21

def landmarks_to_feature(landmarks, image_w, image_h):
    """
    landmarks: list of 21 landmark objects (normalized x,y,z)
    returns: 63-dim vector: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20] normalized
             coordinates are normalized to the hand bbox and centered + scaled to [-1,1]
    """
    # convert to numpy array (N,3) in pixel coords
    pts = np.array([[lm.x * image_w, lm.y * image_h, lm.z] for lm in landmarks], dtype=np.float32)
    # use wrist (landmark 0) as origin optionally; instead use bounding box normalization
    x_min, y_min = np.min(pts[:,0]), np.min(pts[:,1])
    x_max, y_max = np.max(pts[:,0]), np.max(pts[:,1])
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    # center and scale
    cx = x_min + w/2.0
    cy = y_min + h/2.0
    scale = max(w, h) / 2.0  # half-size
    # produce normalized coords [-1,1]
    norm = []
    for (x,y,z) in pts:
        nx = (x - cx) / scale
        ny = (y - cy) / scale
        nz = z  # z is relative depth — keep raw (can also scale)
        norm.extend([float(nx), float(ny), float(nz)])
    return norm  # length = 63

# ---------- COLLECT MODE ----------
def collect_mode():
    """
    Interactive capture:
      - Press a letter key (A-Z) to save the current hand landmarks with that label.
      - Press ESC or 'q' to quit.
      - Press 'p' to pause/unpause.
    Saved CSV columns: label, lx0,ly0,lz0, lx1,ly1,lz1, ... (63 cols)
    """
    print("COLLECT MODE: Press letter keys to save sample with that label. Press 'q' or ESC to quit.")
    print("Optional: define TARGET_LABELS inside script to restrict which labels allowed.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0)")

    header = None
    if not os.path.exists(CSV_PATH):
        # create header
        cols = ["label"] + [f"l{idx}_{coord}" for idx in range(NUM_LANDMARKS) for coord in ("x","y","z")]
        pd.DataFrame(columns=cols).to_csv(CSV_PATH, index=False)
        print("Created new CSV at", CSV_PATH)

    saved_counts = {}  # track counts per label
    paused = False
    with mp_hands.Hands(**MP_CONFIG) as hands:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                h, w = frame.shape[:2]
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)

                display = frame.copy()
                if res.multi_hand_landmarks:
                    # draw first hand
                    for hl in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display, hl, mp_hands.HAND_CONNECTIONS)
                # instructions overlay
                cv2.putText(display, "Press letter key to save sample. 'q' to quit, 'p' pause", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # show counts
                y0 = 40
                for i, (lbl, cnt) in enumerate(sorted(saved_counts.items())):
                    cv2.putText(display, f"{lbl}:{cnt}", (10, y0 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

                cv2.imshow("Collect (press keys)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("Quitting collect mode.")
                break
            if key == ord('p'):
                paused = not paused
                print("Paused:", paused)
                continue

            # if a letter key pressed
            if 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
                lbl = chr(key).upper()
                if TARGET_LABELS and lbl not in TARGET_LABELS:
                    print("Label not allowed:", lbl)
                    continue
                if res is None or not res.multi_hand_landmarks:
                    print("No hand detected — not saved.")
                    continue
                # use first hand
                lm = res.multi_hand_landmarks[0].landmark
                feat = landmarks_to_feature(lm, w, h)  # 63 dims
                row = [lbl] + feat
                # append to CSV
                pd.DataFrame([row]).to_csv(CSV_PATH, mode='a', header=False, index=False)
                saved_counts[lbl] = saved_counts.get(lbl, 0) + 1
                print(f"Saved sample for {lbl}. Total for {lbl}: {saved_counts[lbl]}")

    cap.release()
    cv2.destroyAllWindows()

# ---------- TRAIN MODE ----------
def train_mode():
    """
    Loads CSV of collected landmarks, trains RandomForest, saves model to MODEL_PATH.
    """
    if not os.path.exists(CSV_PATH):
        print("No data file found at", CSV_PATH)
        return
    print("Loading dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    if df.shape[0] < 20:
        print("Not enough samples to train. Need more collected samples.")
        return

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values.astype(str)
    # encode labels as integers via sklearn (let classifier store classes_)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # classifier
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    print("Training classifier...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model + label encoder
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

# ---------- PREDICT MODE ----------
def predict_mode():
    """
    Real-time prediction using saved landmark model.
    Press 'q' to quit, 's' to save a sample 28x28 image (for debugging).
    """
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first with: python hand_graph_pipeline.py train")
        return
    data = joblib.load(MODEL_PATH)
    model = data.get("model") if isinstance(data, dict) else data
    le = data.get("label_encoder", None)
    if le is None:
        print("Label encoder missing. Predictions will return numeric class or raw label from model.classes_.")
    print("Loaded model. classes:", getattr(model, "classes_", None))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    with mp_hands.Hands(**MP_CONFIG) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            display = frame.copy()
            pred_label = None
            conf_text = ""

            if res.multi_hand_landmarks:
                # draw landmarks
                for hl in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display, hl, mp_hands.HAND_CONNECTIONS)
                lm = res.multi_hand_landmarks[0].landmark
                feat = landmarks_to_feature(lm, w, h)
                X = np.array(feat).reshape(1, -1)
                try:
                    pred = model.predict(X)[0]
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)[0]
                        conf = float(np.max(probs))
                        conf_text = f"{conf*100:.1f}%"
                    else:
                        conf = None
                    # decode label
                    if le is not None:
                        pred_label = le.inverse_transform([pred])[0]
                    else:
                        # try model.classes_
                        classes = getattr(model, "classes_", None)
                        if classes is not None:
                            try:
                                pred_label = classes[pred]
                            except Exception:
                                pred_label = str(pred)
                        else:
                            pred_label = str(pred)
                except Exception as e:
                    pred_label = "ERR"
                    conf_text = str(e)

            # overlay
            cv2.putText(display, f"Pred: {pred_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            if conf_text:
                cv2.putText(display, f"Conf: {conf_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
            cv2.putText(display, "Press 'q' to quit. Collect: python hand_graph_pipeline.py collect", (10, display.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("Predict", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # save small debug crop of the first hand (if exists)
                if res.multi_hand_landmarks:
                    # crop bbox from landmarks
                    pts = np.array([[lm.x * w, lm.y * h] for lm in res.multi_hand_landmarks[0].landmark])
                    x_min, y_min = int(pts[:,0].min()), int(pts[:,1].min())
                    x_max, y_max = int(pts[:,0].max()), int(pts[:,1].max())
                    padx = int(0.2*(x_max-x_min))
                    pady = int(0.2*(y_max-y_min))
                    x0 = max(0, x_min-padx); y0 = max(0, y_min-pady)
                    x1 = min(w, x_max+padx); y1 = min(h, y_max+pady)
                    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(crop, (28,28), interpolation=cv2.INTER_AREA)
                    fname = f"debug_sample_{int(time.time())}.png"
                    cv2.imwrite(fname, small)
                    print("Saved debug sample", fname)
    cap.release()
    cv2.destroyAllWindows()

# ---------- CLI ----------
def print_usage():
    print("Usage: python hand_graph_pipeline.py [collect|train|predict]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    mode = sys.argv[1].lower().strip()
    if mode == "collect":
        collect_mode()
    elif mode == "train":
        train_mode()
    elif mode == "predict":
        predict_mode()
    else:
        print_usage()
=======
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

