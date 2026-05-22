"""
collect_data.py
===============
Run this ONCE before training to collect your own hand sign samples.

For each sign (B, C, D, O, Q, U, M, LEFT, RIGHT) the script will:
  1. Show a countdown so you can get your hand ready
  2. Record 200 frames of landmark data while you hold the sign
  3. Save everything to  data/landmarks.csv

Usage:
    python collect_data.py

Requirements:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import csv
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
SIGNS = ["B", "C", "D", "O", "Q", "U", "M", "LEFT", "RIGHT"]
SAMPLES_PER_SIGN = 200          # frames collected per sign  (9 signs total)
COUNTDOWN_SECONDS = 3           # time to get hand ready before recording
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def extract_landmarks(hand_landmarks):
    """
    Return a flat list of 63 values: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
    Coordinates are normalized relative to the wrist (landmark 0)
    so the features are position-invariant.
    """
    lm = hand_landmarks.landmark

    wrist_x = lm[0].x
    wrist_y = lm[0].y
    wrist_z = lm[0].z

    features = []
    for point in lm:
        features.append(point.x - wrist_x)
        features.append(point.y - wrist_y)
        features.append(point.z - wrist_z)

    return features   # length = 21 * 3 = 63


def collect_sign(sign_label, writer):
    """Show countdown then record SAMPLES_PER_SIGN frames for one sign."""

    # ── Countdown ──
    start_time = time.time()
    while time.time() - start_time < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        remaining = COUNTDOWN_SECONDS - int(time.time() - start_time)
        cv2.putText(frame,
                    f"Get ready for sign: [{sign_label}]",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Starting in {remaining}...",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 200, 255), 3)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)

    # ── Recording ──
    collected = 0
    while collected < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            features = extract_landmarks(hand_lm)
            writer.writerow([sign_label] + features)
            collected += 1

        # Progress bar
        progress = int((collected / SAMPLES_PER_SIGN) * 400)
        cv2.rectangle(frame, (30, 420), (430, 445), (50, 50, 50), -1)
        cv2.rectangle(frame, (30, 420), (30 + progress, 445), (0, 200, 100), -1)
        cv2.putText(frame,
                    f"Recording [{sign_label}]:  {collected}/{SAMPLES_PER_SIGN}",
                    (30, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 100), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"\nData Collection - Hand Sign Recognition")
print(f"Signs to collect: {SIGNS}")
print(f"Samples per sign: {SAMPLES_PER_SIGN}")
print(f"Output: {CSV_PATH}\n")
print("Press Q at any time to stop early.\n")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    header = ["label"] + [f"f{i}" for i in range(63)]
    writer.writerow(header)

    for sign in SIGNS:
        print(f"  Collecting sign: {sign} ...")
        ok = collect_sign(sign, writer)
        if not ok:
            print("  Stopped early by user.")
            break
        print(f"  Done - {SAMPLES_PER_SIGN} samples saved.\n")

cap.release()
cv2.destroyAllWindows()
print(f"\nAll done! Data saved to: {CSV_PATH}")
print("Next step: run  python train_model.py")
