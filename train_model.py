"""
train_model.py
==============
Reads the landmark CSV collected by collect_data.py,
trains a small MLP classifier, evaluates it, and saves:
  - model/sign_model.pkl      (used by main.py at runtime)
  - model/label_map.txt       (maps index -> sign label)
  - model/confusion_matrix.png (for your project report)

Usage:
    python train_model.py

Requirements:
    pip install numpy scikit-learn matplotlib seaborn joblib
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report, confusion_matrix
from sklearn.neural_network  import MLPClassifier

# -- Config --------------------------------------------------------------------
CSV_PATH   = "data/landmarks.csv"
MODEL_DIR  = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "sign_model.pkl")
LABEL_PATH  = os.path.join(MODEL_DIR, "label_map.txt")
CM_PATH     = os.path.join(MODEL_DIR, "confusion_matrix.png")

os.makedirs(MODEL_DIR, exist_ok=True)

# -- 1. Load Data --------------------------------------------------------------
print("Loading data ...")
labels   = []
features = []

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row["label"])
        features.append([float(row[f"f{i}"]) for i in range(63)])

X = np.array(features, dtype=np.float32)
y_raw = np.array(labels)

print(f"  Total samples: {len(X)}")
print(f"  Signs found:   {sorted(set(y_raw))}")

# -- 2. Encode Labels ----------------------------------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)
print(f"  Number of classes: {num_classes}  -> {list(encoder.classes_)}")

with open(LABEL_PATH, "w") as f:
    for i, name in enumerate(encoder.classes_):
        f.write(f"{i},{name}\n")
print(f"  Label map saved -> {LABEL_PATH}")

# -- 3. Train/Test Split -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} samples   Test: {len(X_test)} samples")

# -- 4. Train MLP --------------------------------------------------------------
print("\nTraining MLP classifier ...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500,
    random_state=42,
    verbose=True,
)
model.fit(X_train, y_train)

# -- 5a. Training Loss Curve ---------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("MLP Training Convergence")
plt.tight_layout()
CURVE_PATH = os.path.join(MODEL_DIR, "training_curve.png")
plt.savefig(CURVE_PATH, dpi=150)
plt.close()
print(f"  Training curve saved -> {CURVE_PATH}")

# -- 5. Evaluate ---------------------------------------------------------------
print("\nEvaluating on test set ...")
acc = model.score(X_test, y_test)
print(f"  Test accuracy: {acc*100:.1f}%")

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# -- 6. Confusion Matrix -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hand Sign Classifier")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
print(f"\nConfusion matrix saved -> {CM_PATH}")

# -- 7. Save Model -------------------------------------------------------------
joblib.dump({"model": model, "encoder": encoder}, MODEL_PATH)
size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"  Model saved -> {MODEL_PATH}  ({size_kb:.1f} KB)")

print("\nTraining complete!")
print("Next step: run  python main.py")
