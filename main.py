"""
main.py
=======
Real-time hand sign recognition -> Canva keyboard control.

Signs recognized:
  B  ->  B   (Blur/Unblur screen)
  C  ->  C   (Confetti animation)
  D  ->  D   (Drumroll sound effect)
  O  ->  O   (Bubble animation)
  Q  ->  Q   (Quiet emoji animation)
  U  ->  U   (Curtain call animation)
  M  ->  M   (Mic drop animation)
  LEFT  -> Left arrow  (Previous slide)
  RIGHT -> Right arrow (Next slide)

A sign must be held steady for HOLD_FRAMES consecutive frames
before the keyboard action fires. This prevents accidental triggers.

Usage:
    python main.py

Pipeline stages (timed individually for performance report):
  Stage 1 -- Capture & Preprocess
  Stage 2 -- Hand Detection (MediaPipe)
  Stage 3 -- Landmark Extraction & Normalization
  Stage 4 -- Sign Classification (sklearn MLP)
  Stage 5 -- Stability Filter (hold detection)
  Stage 6 -- Keyboard Injection + Live Display
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import joblib
import time
import os
import csv
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "model/sign_model.pkl"
LABEL_PATH     = "model/label_map.txt"
HOLD_FRAMES    = 20
CONF_THRESHOLD = 0.80
COOLDOWN_SEC   = 1.5
TIMING_WINDOW  = 300
LOG_PATH       = "session_log.csv"

# ── Canva keyboard mapping ────────────────────────────────────────────────────
SIGN_ACTIONS = {
    "B":     ("Blur/Unblur screen",     lambda: pyautogui.press("b")),
    "C":     ("Confetti animation",     lambda: pyautogui.press("c")),
    "D":     ("Drumroll sound effect",  lambda: pyautogui.press("d")),
    "O":     ("Bubble animation",       lambda: pyautogui.press("o")),
    "Q":     ("Quiet emoji animation",  lambda: pyautogui.press("q")),
    "U":     ("Curtain call animation", lambda: pyautogui.press("u")),
    "M":     ("Mic drop animation",     lambda: pyautogui.press("m")),
    "LEFT":  ("Previous slide",         lambda: pyautogui.press("left")),
    "RIGHT": ("Next slide",             lambda: pyautogui.press("right")),
}

# ── Colors ────────────────────────────────────────────────────────────────────
CLR_GREEN  = (50, 220, 140)
CLR_AMBER  = (30, 170, 255)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (20, 20, 40)
CLR_PURPLE = (200, 130, 255)


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["encoder"]


def sklearn_predict(model, features):
    """Run one inference. Returns (label_index, confidence)."""
    inp   = np.array([features], dtype=np.float32)
    probs = model.predict_proba(inp)[0]
    idx   = int(np.argmax(probs))
    return idx, float(probs[idx])


# ── Stage 3: Landmark extraction & normalization ──────────────────────────────
def extract_landmarks(hand_landmarks):
    """63 values relative to wrist position (position-invariant)."""
    lm = hand_landmarks.landmark
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    features = []
    for point in lm:
        features.append(point.x - wx)
        features.append(point.y - wy)
        features.append(point.z - wz)
    return features   # len = 63


def load_labels(path):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            idx, name = line.strip().split(",")
            labels[int(idx)] = name
    return labels


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_confidence_bar(img, x, y, w, h, conf, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(conf * w)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(img, f"{conf*100:.0f}%",
                (x + w + 8, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_WHITE, 1)


def draw_hold_arc(img, cx, cy, progress):
    angle = int(360 * progress)
    cv2.ellipse(img, (cx, cy), (32, 32), -90, 0, angle, CLR_GREEN, 4)
    cv2.ellipse(img, (cx, cy), (32, 32), -90, angle, 360, (80, 80, 80), 2)


# ── Session logger ────────────────────────────────────────────────────────────
class SessionLogger:
    def __init__(self, path):
        self.path = path
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "sign", "confidence", "action_fired"])

    def log(self, sign, conf, fired):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [time.strftime("%H:%M:%S"), sign, f"{conf:.3f}",
                 "YES" if fired else "no"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        print("ERROR: Model not found.")
        print("Run  python train_model.py  first.")
        return

    print("Loading model ...")
    clf, _ = load_model(MODEL_PATH)
    label_map = load_labels(LABEL_PATH)
    print(f"  Labels: {label_map}")

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stage_times = {s: deque(maxlen=TIMING_WINDOW) for s in range(1, 7)}

    hold_count    = 0
    last_label    = None
    last_action_t = 0.0
    last_fired    = None
    last_fired_t  = 0.0
    logger        = SessionLogger(LOG_PATH)
    pyautogui.FAILSAFE = False

    print("\nHand Sign Canva Controller - running.")
    print("Press  ESC  to quit.\n")

    while True:
        # ── Stage 1: Capture & Preprocess ────────────────────────────────────
        t1 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(gray)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = frame.shape[:2]
        stage_times[1].append((time.perf_counter() - t1) * 1000)

        # ── Stage 2: Hand Detection ───────────────────────────────────────────
        t2 = time.perf_counter()
        results      = hands.process(rgb)
        hand_detected = results.multi_hand_landmarks is not None
        stage_times[2].append((time.perf_counter() - t2) * 1000)

        pred_label, pred_conf = None, 0.0

        if hand_detected:
            hand_lm = results.multi_hand_landmarks[0]

            # ── Stage 3: Landmark Extraction ─────────────────────────────────
            t3 = time.perf_counter()
            features = extract_landmarks(hand_lm)
            stage_times[3].append((time.perf_counter() - t3) * 1000)

            # ── Stage 4: Classification ───────────────────────────────────────
            t4 = time.perf_counter()
            idx, conf = sklearn_predict(clf, features)
            if conf >= CONF_THRESHOLD:
                pred_label = label_map[idx]
                pred_conf  = conf
            stage_times[4].append((time.perf_counter() - t4) * 1000)

            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
        else:
            stage_times[3].append(0.0)
            stage_times[4].append(0.0)

        # ── Stage 5: Stability Filter ─────────────────────────────────────────
        t5 = time.perf_counter()
        now = time.time()

        if pred_label is not None:
            hold_count = hold_count + 1 if pred_label == last_label else 1
            last_label = pred_label
            if hold_count >= HOLD_FRAMES and now - last_action_t >= COOLDOWN_SEC:
                action = SIGN_ACTIONS.get(pred_label)
                if action:
                    desc, fn = action
                    fn()
                    last_action_t = now
                    last_fired    = pred_label
                    last_fired_t  = now
                    hold_count    = 0
                    logger.log(pred_label, pred_conf, True)
                    print(f"  Fired: {desc}  (conf={pred_conf:.2f})")
            else:
                logger.log(pred_label, pred_conf, False)
        else:
            hold_count = 0
            last_label = None

        stage_times[5].append((time.perf_counter() - t5) * 1000)

        # ── Stage 6: Display ──────────────────────────────────────────────────
        t6 = time.perf_counter()

        draw_rounded_rect(frame, 0, 0, w, 70, CLR_DARK, alpha=0.7)

        if pred_label:
            cv2.putText(frame, pred_label,
                        (20, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, CLR_PURPLE, 3)
            draw_confidence_bar(frame, 100, 30, 200, 18, pred_conf, CLR_GREEN)
            draw_hold_arc(frame, w - 55, 45, min(hold_count / HOLD_FRAMES, 1.0))
            cv2.putText(frame, "HOLD", (w - 72, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1)
        else:
            cv2.putText(frame,
                        "No hand detected" if not hand_detected else "Low confidence",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (120, 120, 160), 2)

        if last_fired and now - last_fired_t < 1.5:
            draw_rounded_rect(frame, 10, 80, w - 10, 130, CLR_GREEN, alpha=0.5)
            cv2.putText(frame, f"ACTION: {SIGN_ACTIONS[last_fired][0]}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_WHITE, 2)

        draw_rounded_rect(frame, 0, h - 105, w, h, CLR_DARK, alpha=0.65)

        if all(len(stage_times[s]) > 10 for s in range(1, 7)):
            means = {s: np.mean(stage_times[s]) for s in range(1, 7)}
            total = sum(means.values())
            fps   = 1000.0 / total if total > 0 else 0

            labels_txt = ["Pre", "Det", "Lmk", "Clf", "Stb", "Vis"]
            for i, (s, lbl) in enumerate(zip(range(1, 7), labels_txt)):
                xp = 10 + i * 105
                cv2.putText(frame, f"S{s} {lbl}", (xp, h - 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_AMBER, 1)
                cv2.putText(frame, f"{means[s]:.1f}ms", (xp, h - 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_WHITE, 1)
            cv2.putText(frame,
                        f"Total: {total:.1f}ms   FPS: {fps:.1f}",
                        (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_GREEN, 1)

        legend_items = [
            ("B", "Blur"),  ("C", "Confetti"), ("D", "Drum"),
            ("O", "Bubble"), ("Q", "Quiet"),   ("U", "Curtain"),
            ("M", "MicDrop"), ("L", "<- Prev"), ("R", "-> Next"),
        ]
        for i, (sign, action) in enumerate(legend_items):
            cv2.putText(frame, f"{sign}:{action}",
                        (10 + i * 90, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, CLR_AMBER, 1)

        stage_times[6].append((time.perf_counter() - t6) * 1000)

        cv2.imshow("Hand Sign Canva Controller", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    # ── Final performance report ───────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("PERFORMANCE REPORT  (averaged over last 300 frames)")
    print("=" * 58)
    stage_names = {
        1: "Capture & Preprocess",
        2: "Hand Detection",
        3: "Landmark Extraction",
        4: "Sign Classification",
        5: "Stability Filter",
        6: "Display & Logging",
    }
    total_mean = 0
    for s in range(1, 7):
        if stage_times[s]:
            arr  = np.array(stage_times[s])
            mean = float(np.mean(arr))
            std  = float(np.std(arr))
            total_mean += mean
            print(f"  Stage {s}  {stage_names[s]:<26}  {mean:6.2f} ms  +/-{std:5.2f} ms")
    print("-" * 58)
    print(f"  {'TOTAL':<32}  {total_mean:6.2f} ms")
    print(f"  {'Est. FPS':<32}  {1000/total_mean:.1f} fps")
    print("=" * 58)
    print(f"\nSession log saved -> {LOG_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
