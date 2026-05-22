"""
generate_lifecycle.py
Generates the system lifecycle diagram (Fig. 2).
Output: model/lifecycle_diagram.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

BOX_W = 3.0
BOX_H = 2.2
BY    = 1.4   # box y bottom

phases = [
    (0.5,  "Data Collection",  "collect_data.py",  ["Webcam captures", "200 frames/sign", "9 signs = 1,800 samples"]),
    (4.5,  "Model Training",   "train_model.py",   ["Load landmarks.csv", "Train MLP 128→64", "Evaluate & save model"]),
    (8.5,  "Live Inference",   "main.py",          ["Load sign_model.pkl", "6-stage CV pipeline", "Keyboard injection"]),
]

for (bx, title, script, bullets) in phases:
    box = FancyBboxPatch((bx, BY), BOX_W, BOX_H,
                         boxstyle="square,pad=0.0",
                         linewidth=1.2,
                         edgecolor="black",
                         facecolor="white",
                         zorder=3)
    ax.add_patch(box)

    # Title bar divider line
    ax.plot([bx, bx + BOX_W], [BY + BOX_H - 0.55, BY + BOX_H - 0.55],
            color="black", lw=1.0, zorder=4)

    # Phase title
    ax.text(bx + BOX_W / 2, BY + BOX_H - 0.27, title,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=5)

    # Script name
    ax.text(bx + BOX_W / 2, BY + BOX_H - 0.72, script,
            ha="center", va="center",
            fontsize=9, color="black", style="italic", zorder=5)

    # Bullet points
    for i, line in enumerate(bullets):
        ax.text(bx + 0.18, BY + BOX_H - 1.05 - i * 0.38, f"• {line}",
                ha="left", va="center",
                fontsize=9, color="black", zorder=5)

# ── Arrows between phases ──────────────────────────────────────────────────────
arrow_xs = [
    (0.5 + BOX_W, 4.5),   # Phase 1 → Phase 2
    (4.5 + BOX_W, 8.5),   # Phase 2 → Phase 3
]
for (x_start, x_end) in arrow_xs:
    ax.annotate("",
                xy=(x_end - 0.03, BY + BOX_H / 2),
                xytext=(x_start + 0.03, BY + BOX_H / 2),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.5, mutation_scale=16),
                zorder=2)

# ── Artifact labels on arrows ──────────────────────────────────────────────────
ax.text(4.0, BY + BOX_H / 2 + 0.18, "landmarks.csv",
        ha="center", va="bottom", fontsize=8.5, style="italic", color="black")

ax.text(8.0, BY + BOX_H / 2 + 0.18, "sign_model.pkl\nlabel_map.txt",
        ha="center", va="bottom", fontsize=8.5, style="italic", color="black")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(6.0, 4.6, "System Lifecycle",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="black")

os.makedirs("model", exist_ok=True)
out = "model/lifecycle_diagram.png"
plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved -> {out}")
