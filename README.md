<div align="center">

# ✋ Real-Time Sign Language Recognition  
### Hand Landmark–Based Gesture Classification

<img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square"/>
<img src="https://img.shields.io/badge/MediaPipe-Hands-green?style=flat-square"/>
<img src="https://img.shields.io/badge/ML-Random%20Forest-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/Accuracy-98.25%25-brightgreen?style=flat-square"/>

</div>

---

## 📌 Overview

This project implements a **real-time sign language recognition system** that detects hand gestures through a webcam and converts them into English alphabet letters.

The system is designed for **live video input**, using **hand landmark geometry** instead of raw images to ensure stable and efficient real-time performance.

---

## 🧠 Approach

### 🔹 Hand Landmark Detection
- Uses **MediaPipe Hands**
- Extracts **21 hand landmarks** per frame  
- Each landmark provides *(x, y, z)* coordinates  
- Total features per gesture: **63**

Landmark-based representation reduces sensitivity to lighting, background noise, and camera variations.

---

### 🔹 Machine Learning Model
- **Random Forest Classifier**
- Trained on normalized landmark features
- Optimized for low-latency inference

Chosen for its strong performance on structured numerical data and suitability for real-time systems.

---

## ⚙️ System Workflow

Webcam Frame
↓
Hand Landmark Detection
↓
Feature Normalization
↓
Random Forest Model
↓
Predicted Alphabet Letter


The system runs at **~20–30 FPS**, enabling smooth real-time interaction.

---

## 📊 Dataset

- Custom dataset collected using a webcam  
- Multiple samples per alphabet sign  
- Captured from different angles and positions  
- Easily extensible for additional gestures  

This ensures compatibility with real-world usage rather than static image data.

---

## 🚀 Performance

| Metric | Value |
|------|------|
| Test Accuracy | **~98.25%** |
| Inference Speed | Real-time |
| Hardware | Standard Webcam |
| GPU Required | ❌ No |

Switching from image-based features to landmark-based learning significantly improved prediction stability and accuracy.

---

## 🧩 Challenges

- Image-based models failed in live environments  
- Visually similar gestures required more samples  
- Early real-time predictions were unstable  

These were resolved by redesigning the feature pipeline around hand landmarks and refining the dataset.

---

## 🔮 Future Enhancements

- Word and sentence-level recognition  
- Dynamic gesture support (J, Z)  
- Two-hand gesture detection  
- Mobile or web deployment  
- Text-to-speech integration  

---

## 📁 Project Structure

├── data/
│ └── landmarks_dataset.csv
├── model/
│ └── random_forest.pkl
├── src/
│ ├── collect_data.py
│ ├── train_model.py
│ └── realtime_predict.py
├── requirements.txt
└── README.md


---

## 🏁 Conclusion

This project demonstrates a **practical real-time sign language recognition pipeline** using hand landmark features and classical machine learning, optimized for efficiency and real-world usability.

---
