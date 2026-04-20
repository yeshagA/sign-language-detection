# ✋ Real-Time Sign Language Detection

A real-time hand gesture recognition system built using **MediaPipe and Machine Learning**.

## 🚀 Features

* Real-time hand tracking using webcam
* Custom dataset collection (A, B, C gestures)
* Machine Learning model (Random Forest)
* Stable predictions using smoothing
* Live gesture-to-word conversion

## 🧠 Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn

## 📂 Project Structure

```
data/          # Gesture datasets (A, B, C)
model/         # Trained model
collect_data.py
train_model.py
predict.py
requirements.txt
```

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run prediction:

```
python predict.py
```

## 📸 Example Output

| Gesture | Output    |
| ------- | --------- |
| A       | HELLO     |
| B       | THANK YOU |
| C       | YES       |

## 🎥 Demo

(Add your demo video here)

## 💼 Author

Yesha
