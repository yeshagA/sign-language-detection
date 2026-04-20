import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# Load trained model
model = joblib.load("model/model.pkl")

# Label → Word mapping
label_map = {
    "A": "HELLO",
    "B": "THANK YOU",
    "C": "YES"
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

print("Warming up camera...")

# Warm-up camera (important for your laptop)
good_frame = False
start = time.time()

while time.time() - start < 5:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    if frame.mean() > 20:
        good_frame = True
        print("Camera ready!")
        break

if not good_frame:
    print("❌ Camera unstable. Try restarting.")
    cap.release()
    exit()

time.sleep(1)

# Store predictions for smoothing
prediction_history = []

# Auto stop timer
start_time = time.time()
duration = 60  # seconds

while True:
    ret, frame = cap.read()

    # Skip bad frames
    if not ret or frame is None or frame.mean() < 20:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Predict
            prediction = model.predict([landmarks])[0]

            # Add to history
            prediction_history.append(prediction)

            # Keep last 10 predictions
            if len(prediction_history) > 10:
                prediction_history.pop(0)

            # Most common prediction (stabilization)
            final_pred = max(set(prediction_history), key=prediction_history.count)

            # Convert to word
            word = label_map.get(final_pred, final_pred)

            # Draw hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show result
            cv2.putText(frame, word, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Prediction", frame)

    # Auto stop
    if time.time() - start_time > duration:
        print("⏱ Auto stopped")
        break

    # Manual stop (q or ESC)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        print("🛑 Stopped manually")
        break

cap.release()
cv2.destroyAllWindows()