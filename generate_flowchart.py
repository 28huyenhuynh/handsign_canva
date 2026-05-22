"""
generate_flowchart.py
Generates a clean black-and-white pipeline flowchart.
Output: model/pipeline_flowchart.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(6, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis("off")

stages = [
    # (y_center, stage_num, title, subtitle)
    (14.5, "",   "Webcam Input",           "640 x 480 px"),
    (12.5, "1",  "Capture & Preprocess",   "Flip  |  CLAHE  |  BGR to RGB"),
    (10.5, "2",  "Hand Detection",         "MediaPipe Hands  |  max_hands=1  |  conf >= 0.7"),
    ( 8.5, "3",  "Landmark Extraction",    "21 keypoints  |  wrist-relative normalisation  |  63-D"),
    ( 6.5, "4",  "Sign Classification",    "MLP 128-64  |  ReLU  |  softmax  |  conf >= 0.80"),
    ( 4.5, "5",  "Stability Filter",       "hold >= 20 frames  |  cooldown 1.5 s"),
    ( 2.5, "6",  "Keyboard Injection",     "PyAutoGUI  ->  Canva shortcut"),
    ( 0.7, "",   "Session Log",            "session_log.csv"),
]

BOX_W = 7.0
BOX_H = 1.0
CX    = 5.0

for (yc, snum, title, subtitle) in stages:
    x0 = CX - BOX_W / 2
    y0 = yc - BOX_H / 2

    box = FancyBboxPatch((x0, y0), BOX_W, BOX_H,
                         boxstyle="square,pad=0.0",
                         linewidth=1.2,
                         edgecolor="black",
                         facecolor="white",
                         zorder=3)
    ax.add_patch(box)

    title_x = x0 + 0.25

    ax.text(title_x, yc + 0.18, title,
            ha="left", va="center",
            fontsize=12, fontweight="bold",
            color="black", zorder=5)
    ax.text(title_x, yc - 0.20, subtitle,
            ha="left", va="center",
            fontsize=9.5, color="black",
            style="italic", zorder=5)

# ── Arrows ─────────────────────────────────────────────────────────────────────
for i in range(len(stages) - 1):
    y_start = stages[i][0]   - BOX_H / 2
    y_end   = stages[i+1][0] + BOX_H / 2
    ax.annotate("",
                xy=(CX, y_end + 0.03),
                xytext=(CX, y_start - 0.03),
                arrowprops=dict(arrowstyle="-|>",
                                color="black", lw=1.2,
                                mutation_scale=14),
                zorder=2)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(CX, 15.6, "Hand Sign Recognition — Six-Stage Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold", color="black")

os.makedirs("model", exist_ok=True)
out = "model/pipeline_flowchart.png"
plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved -> {out}")
