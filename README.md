# Hand Sign Language Recognition — Canva Controller
**Computer Vision Final Project**

Control Canva presentations using ASL hand signs detected through your PC webcam.

---

## Signs & Actions

| Sign | Canva Action | Shortcut Fired |
|------|-------------|----------------|
| **B** | Blur / Unblur screen | `B` |
| **C** | Confetti animation | `C` |
| **D** | Drumroll sound effect | `D` |
| **O** | Bubble animation | `O` |
| **Q** | Quiet emoji animation | `Q` |
| **U** | Curtain call animation | `U` |
| **M** | Mic drop animation | `M` |
| **LEFT** | Previous slide | `←` |
| **RIGHT** | Next slide | `→` |

---

## Setup

### 1. Install dependencies
```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn matplotlib seaborn pyautogui
```

### 2. Collect training data
```bash
python collect_data.py
```
- For each of the 7 signs, hold the sign in front of your webcam
- The script records 200 frames per sign automatically
- Data is saved to `data/landmarks.csv`

### 3. Train the model
```bash
python train_model.py
```
- Trains an MLP classifier on your collected data
- Saves `model/sign_model.tflite` and `model/label_map.txt`
- Prints classification accuracy and saves `model/confusion_matrix.png`

### 4. Run the live controller
```bash
python main.py
```
- Open Canva in your browser
- Hold a sign steady in front of your webcam for ~1 second
- The corresponding keyboard action fires automatically

---

## Project Structure

```
├── collect_data.py       # Stage 1–3: data collection tool
├── train_model.py        # Stage 4: model training + evaluation
├── main.py               # Stages 1–6: live recognition + keyboard injection
├── data/
│   └── landmarks.csv     # collected training data (created by collect_data.py)
├── model/
│   ├── sign_model.tflite # trained model (created by train_model.py)
│   ├── label_map.txt     # index → sign name mapping
│   └── confusion_matrix.png
└── session_log.csv       # timestamped prediction log (created by main.py)
```

---

## CV Pipeline Stages

| # | Stage | File | Course Week |
|---|-------|------|-------------|
| 1 | Capture & Preprocess (CLAHE, resize) | `main.py` | Wk 1–2 |
| 2 | Hand Detection (MediaPipe Hands) | `main.py` | Wk 6, 8 |
| 3 | Landmark Extraction (21 keypoints, normalized) | `main.py` | Wk 5, 10 |
| 4 | Sign Classification (TFLite MLP) | `main.py` | Wk 8, 10 |
| 5 | Stability Filter (hold 20 frames, cooldown) | `main.py` | Wk 7 |
| 6 | Keyboard Injection + Live Display | `main.py` | Wk 11 |

---

## Tips for Better Accuracy

- **Lighting**: Make sure your hand is well lit. Avoid having a bright light source directly behind you.
- **Background**: A plain, static background helps MediaPipe detect your hand faster.
- **Camera distance**: Keep your hand 40–70 cm from the camera so all 21 landmarks are visible.
- **Collect your own data**: The model trained on your hand will always outperform a generic dataset. Run `collect_data.py` yourself.
- **Hold steady**: The stability filter requires 20 consecutive matching predictions before firing — this prevents accidental triggers. Adjust `HOLD_FRAMES` in `main.py` if needed.

---

## Performance (estimated, Intel Core i5 laptop, no GPU)

| Stage | Target |
|-------|--------|
| Capture & Preprocess | < 10 ms |
| Hand Detection | < 15 ms |
| Landmark Extraction | < 5 ms |
| Sign Classification | < 10 ms |
| Stability Filter | < 2 ms |
| Display & Logging | < 20 ms |
| **Total** | **< 62 ms → ~16 FPS** |
