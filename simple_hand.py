import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 👉 Use the version that worked before (NO CAP_DSHOW / MSMF)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

start_time = time.time()
duration = 30  # ⏱ runs for 30 seconds

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ Frame not received")
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking", frame)

    # ⏱ AUTO STOP after 30 seconds
    if time.time() - start_time > duration:
        print("⏱ Auto stop after 30 seconds")
        break

    # (optional) press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopped by user")
        break

cap.release()
cv2.destroyAllWindows()