# Real-Time Sign Language Recognition Using Hand Landmarks
**Overview**

This project implements a real-time sign language recognition system that detects hand gestures through a webcam and converts them into English alphabet letters.
The system uses hand landmark detection and a machine learning classifier to achieve fast and reliable predictions in live video.

The focus of the project is on real-time usability, robust feature representation, and lightweight deployment.

**Tech Stack**

Python

MediaPipe – hand landmark detection

OpenCV – webcam capture & visualization

Scikit-learn – machine learning

Random Forest Classifier

**System Design**
1. Hand Landmark Detection

The system uses MediaPipe Hands to detect 21 hand landmarks per frame.
Each landmark provides (x, y, z) coordinates, resulting in a 63-dimensional feature vector per gesture.

Using landmarks instead of raw images ensures:

Consistency across lighting conditions

Reduced background influence

Faster and more stable real-time inference

**2. Machine Learning Model
**
A Random Forest classifier is trained on the extracted landmark features.

Performs well on structured numerical data

Low inference latency (suitable for real-time systems)

No GPU requirement

The model predicts the corresponding alphabet letter for each frame.

**Workflow**

Webcam captures live video frames

Hand landmarks are detected using MediaPipe

Landmark coordinates are normalized and flattened

Feature vector is passed to the trained model

Predicted letter is displayed in real time

The system runs at approximately 20–30 FPS, enabling smooth interaction.

**Dataset**

Custom dataset collected using a webcam

Multiple samples per alphabet gesture

Data captured from different angles

Dataset can be extended easily for additional gestures

This approach ensures compatibility with real-time input rather than static images.

**Performance**

Test Accuracy: ~98.25%

Real-time Prediction: Yes

Model Size: Lightweight

Compared to an earlier image-based approach, landmark-based learning provided:

Improved stability in live video

Reduced false predictions

Better generalization across users

**Challenges**

Initial image-based model failed in real-time environments

Visually similar signs required more training samples

Live inference needed smoothing and tuning

Dataset balancing and label consistency

These issues were resolved by switching to landmark-based features and expanding the dataset.

**Future Work**

Word and sentence-level recognition

Dynamic gesture support (e.g., J, Z)

Two-hand gesture detection

Mobile or web deployment

Voice output integration

**Conclusion**

This project demonstrates a practical real-time sign language recognition system built using hand landmark features and classical machine learning.
The approach prioritizes reliability, efficiency, and real-world usability using minimal hardware.
