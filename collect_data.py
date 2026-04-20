import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ----- TRY DIFFERENT CAMERA INDEXES IF 0 DOESN'T WORK -----
# Change 0 to 1 or 2 if your camera still shows gray
cap = cv2.VideoCapture(0)

# Set camera resolution explicitly — fixes gray frames on many laptops
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Force MJPG format — helps on Windows webcams
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("Camera not working — try changing VideoCapture(0) to VideoCapture(1)")
    exit()

label = input("Enter label (A/B/C): ")

data = []

# ----- WARM UP THE CAMERA PROPERLY -----
# Read and THROW AWAY frames until we get a real one (not gray)
print("Warming up camera... please wait")

good_frame = False
warmup_start = time.time()

while time.time() - warmup_start < 5:        # try for up to 5 seconds
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    if frame.mean() > 20:                     # real frame has brightness > 20
        good_frame = True
        print("Camera ready!")
        break
    time.sleep(0.1)

if not good_frame:
    print("Could not get a clear frame. Try these fixes:")
    print("  1. Change VideoCapture(0) to VideoCapture(1)")
    print("  2. Close other apps using the camera (Teams, Zoom, etc.)")
    print("  3. On Mac: grant camera permission to Terminal/VS Code")
    cap.release()
    exit()

# ----- COUNTDOWN SO YOU CAN GET YOUR HAND READY -----
for i in range(3, 0, -1):
    ret, frame = cap.read()
    if ret and frame is not None and frame.mean() > 20:
        cv2.putText(frame, f"Starting in {i}...", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1000)

start_time = time.time()
duration = 20

print(f"Recording '{label}' for {duration} seconds... show your hand!")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    # Skip gray/dark frames
    if frame.mean() < 20:
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
            data.append(landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show countdown timer and sample count
    elapsed = time.time() - start_time
    remaining = max(0, duration - elapsed)
    cv2.putText(frame, f"Label: {label}  Samples: {len(data)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Data Collection", frame)

    # Press Q to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if elapsed > duration:
        break

cap.release()
cv2.destroyAllWindows()

if len(data) == 0:
    print("No hand detected! Make sure your hand is visible in the frame.")
else:
    np.save(f"{label}.npy", np.array(data))
    print(f"Saved {len(data)} samples for '{label}' -> {label}.npy")