"""
generate_flowchart.py
Generates a pipeline flowchart for the hand sign recognition system.
Output: model/pipeline_flowchart.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(8, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 17)
ax.axis("off")

# ── Color palette ──────────────────────────────────────────────────────────────
C_INPUT   = "#4A90D9"   # blue   – input/output
C_STAGE   = "#6C5CE7"   # purple – pipeline stages
C_FILTER  = "#00B894"   # green  – stability filter
C_ACTION  = "#E17055"   # orange – action/output
C_ARROW   = "#636e72"
C_TEXT    = "white"
C_SUB     = "#dfe6e9"   # light  – sub-label bg

stages = [
    # (y_center, color, stage_num, title, subtitle)
    (15.5, C_INPUT,  "",   "WEBCAM INPUT",             "640 × 480 px · 30 FPS"),
    (13.5, C_STAGE,  "S1", "Capture & Preprocess",     "Flip · CLAHE (clip=2.0, tile=8×8) · BGR→RGB"),
    (11.5, C_STAGE,  "S2", "Hand Detection",           "MediaPipe Hands · max_hands=1 · conf≥0.7"),
    ( 9.5, C_STAGE,  "S3", "Landmark Extraction",      "21 keypoints · wrist-relative normalization → 63-D vector"),
    ( 7.5, C_STAGE,  "S4", "Sign Classification",      "MLP  128→64 · ReLU · softmax · conf≥0.80"),
    ( 5.5, C_FILTER, "S5", "Stability Filter",         "hold_count ≥ 20 frames · cooldown 1.5 s"),
    ( 3.5, C_ACTION, "S6", "Keyboard Injection",       "PyAutoGUI.press() → Canva shortcut"),
    ( 1.5, C_INPUT,  "",   "SESSION LOG",              "session_log.csv  (timestamp · sign · conf · fired)"),
]

BOX_W = 7.4
BOX_H = 1.1
CX    = 5.0   # horizontal center

for (yc, color, snum, title, subtitle) in stages:
    x0 = CX - BOX_W / 2
    y0 = yc - BOX_H / 2

    # Main box
    box = FancyBboxPatch((x0, y0), BOX_W, BOX_H,
                         boxstyle="round,pad=0.08",
                         linewidth=1.5,
                         edgecolor="white",
                         facecolor=color,
                         zorder=3)
    ax.add_patch(box)

    # Stage badge
    if snum:
        badge = FancyBboxPatch((x0 + 0.12, yc - 0.28), 0.52, 0.56,
                               boxstyle="round,pad=0.05",
                               linewidth=0,
                               facecolor="white",
                               alpha=0.25,
                               zorder=4)
        ax.add_patch(badge)
        ax.text(x0 + 0.38, yc, snum,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="white", zorder=5)

    # Title
    title_x = x0 + (0.75 if snum else 0.3)
    ax.text(title_x, yc + 0.18, title,
            ha="left", va="center",
            fontsize=11, fontweight="bold",
            color=C_TEXT, zorder=5)

    # Subtitle
    ax.text(title_x, yc - 0.22, subtitle,
            ha="left", va="center",
            fontsize=8, color=C_SUB,
            style="italic", zorder=5)

# ── Arrows between boxes ───────────────────────────────────────────────────────
arrow_ys = []
for i in range(len(stages) - 1):
    y_top    = stages[i][0]   - BOX_H / 2
    y_bottom = stages[i+1][0] + BOX_H / 2
    arrow_ys.append((y_top, y_bottom))

for (y_start, y_end) in arrow_ys:
    ax.annotate("",
                xy=(CX, y_end + 0.04),
                xytext=(CX, y_start - 0.04),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=C_ARROW,
                    lw=2.0,
                    mutation_scale=16,
                ),
                zorder=2)

# ── "No hand" branch label ─────────────────────────────────────────────────────
ax.text(CX + BOX_W / 2 + 0.15, 10.5,
        "No hand\ndetected →\nskip S3–S5",
        ha="left", va="center",
        fontsize=7.5, color="#b2bec3",
        style="italic")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(CX, 16.7,
        "Hand Sign Recognition Pipeline",
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="#2d3436")
ax.text(CX, 16.3,
        "Six-Stage Computer Vision System  ·  ~31.6 ms end-to-end  ·  ~32 FPS",
        ha="center", va="center",
        fontsize=9, color="#636e72")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
out = "model/pipeline_flowchart.png"
plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved -> {out}")
