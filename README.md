# 🖐️ Real-Time Sign Language Recognition (ASL)

This project implements a **real-time American Sign Language (ASL) recognition system** using **MobileNetV2 + MediaPipe Hands**.  
It detects hand gestures (A–Z, 0–9) from a live webcam feed and converts them into text with confidence scores.

---

## 📌 Features
- Live hand tracking using **MediaPipe Hands**  
- Real-time classification of ASL gestures (A–Z, 0–9)  
- Deep learning model based on **MobileNetV2**  
- Confidence thresholding & smoothing for robust predictions  
- Bounding box & landmarks drawn on video feed for visualization  
- Trained with data augmentation for better generalization  

---

## 🛠 Tools & Libraries Used
- Python 3.10  
- TensorFlow / Keras  
- OpenCV  
- MediaPipe  
- NumPy & Pandas  
- Matplotlib  

---

## 📂 Project Structure
```bash
├── DATASET/ # Dataset containing folders 0–9 and A–Z
├── train_mobilenetv2.py # Training script with augmentation
├── detect_live.py # Live detection script (MediaPipe + MobileNetV2)
├── labels.json # Auto-generated label mapping
├── asl_mobilenetv2.h5 # Trained model (saved after training)
└── README.md
```

---

## 🚀 How to Run
### 1. Clone this repository
```bash
git clone https://github.com/yourusername/Real-Time-Sign-Language-Recognition.git
cd Real-Time-Sign-Language-Recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_mobilenetv2.py
```

### 4. Run live detection
```bash
python detect_live.py
```
Press q to quit live detection.

---

## 📖 Report & Video Demo
Project Report: <a href="https://drive.google.com/file/d/1dwqgQY1vTRWgFNSjKo4nTRZYEzecQXkc/view?usp=sharing">Download Report From Here!</a>

Demo Video: <a href="https://drive.google.com/file/d/1_zXxhFY_tCowQnbMgcpCX_346NLt01M7/view?usp=sharing">See Video Here!</a>

---

## 📌 Future Improvements
- Multi-hand support (detect both hands simultaneously)
- Word-level recognition instead of single characters
- Mobile deployment (TensorFlow Lite / ONNX)
- Integration with speech-to-text for real-time communication

--- 

## 👨‍💻 Author

Praveen Tak - <a href="https://www.linkedin.com/in/praveentak">LinkedIn Profile</a>

---
