🖐 Real-Time ASL Alphabet Recognition System

A real-time sign language recognition application built using MediaPipe and machine learning, designed for stable webcam-based inference.

This project focuses on engineering a deployable, real-time computer vision system rather than just training a classifier.

🚀 What This Project Demonstrates

Real-time webcam inference

MediaPipe hand tracking integration

Robust preprocessing under varying lighting

HOG feature engineering pipeline

Temporal smoothing for prediction stability

Confidence-based UI feedback

FPS monitoring

🧠 Real-Time Pipeline Architecture
1️⃣ Hand Detection

MediaPipe detects 21 hand landmarks per frame

Single-hand optimized tracking

Bounding box dynamically computed

2️⃣ Preprocessing (Lighting Robustness)

To improve real-world reliability:

Hand crop with padding

Resize to 28×28

Grayscale conversion

Histogram Equalization

OTSU thresholding (binary inversion)

This ensures consistency across lighting environments.

3️⃣ Feature Extraction

HOG (Histogram of Oriented Gradients):

12 orientations

4×4 pixels per cell

2×2 cells per block

L2-Hys normalization

Designed to capture hand shape and edge structure efficiently.

4️⃣ Prediction Stabilization (Core Engineering Component)

To prevent flickering predictions:

Sliding buffer (size = 7 frames)

Majority voting using Counter

Automatic reset when no hand detected

This significantly improves user experience compared to frame-wise prediction.

5️⃣ Real-Time Feedback

Confidence-based bounding box color:

Green (>85%)

Yellow (>65%)

Red (low confidence)

FPS counter displayed live

Stable prediction display

⚙️ Model

Pre-trained ASL alphabet classifier

Serialized using Joblib

Supports probability output

Real-time inference on CPU

🛠 Tech Stack

Python

OpenCV

MediaPipe

Scikit-learn

NumPy

scikit-image (HOG)

Joblib

📊 Performance

(Update with your real numbers)

Classes: 24 ASL alphabets (J and Z excluded in Sign MNIST)

Real-time performance: ~XX FPS on CPU

Prediction latency: < XX ms/frame
