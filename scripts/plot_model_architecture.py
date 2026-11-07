"""
scripts/plot_model_architecture.py
Generates a publication-grade architecture diagram for the Multimodal Fusion Model.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path("experiments/multimodal_fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.axis("off")

# Helper to draw boxes
def draw_box(ax, xy, width, height, title, subtitle, color, edgecolor, text_color="black"):
    x, y = xy
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=color, edgecolor=edgecolor, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2 + 0.12, title,
            ha="center", va="center", fontsize=9.5, weight="bold", color=text_color)
    if subtitle:
        ax.text(x + width/2, y + height/2 - 0.16, subtitle,
                ha="center", va="center", fontsize=8, color=text_color, alpha=0.9)

def draw_arrow(ax, p1, p2, text=None):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.5, mutation_scale=12)
    )
    if text:
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2 + 0.1
        ax.text(mid_x, mid_y, text, ha="center", va="bottom", fontsize=7.5, color="#555555")

# 1. Inputs
draw_box(ax, (0.5, 4.3), 2.2, 1.2, "Sensor Sequence Input", "[B, 24, 27]\n24h Telemetry + Weather", "#e3f2fd", "#1565c0")
draw_box(ax, (0.5, 1.0), 2.2, 1.2, "Camera Image Input", "[B, 3, 224, 224]\nOutdoor RGB Photograph", "#f3e5f5", "#7b1fa2")

# 2. Encoders
draw_box(ax, (3.3, 4.3), 2.2, 1.2, "LSTM Sensor Encoder", "2-Layer Recurrent\n(Hidden Size: 64)", "#bbdefb", "#0d47a1")
draw_box(ax, (3.3, 1.0), 2.2, 1.2, "ResNet-50 Encoder", "ImageNet Conv Backbone\n(Pooled Feats: 2048)", "#e1bee7", "#4a148c")

draw_arrow(ax, (2.7, 4.9), (3.3, 4.9))
draw_arrow(ax, (2.7, 1.6), (3.3, 1.6))

# 3. Projections
draw_box(ax, (6.0, 4.3), 1.8, 1.2, "Sensor Projection", "Linear(64, 64)\nReLU + Dropout", "#90caf9", "#1565c0")
draw_box(ax, (6.0, 1.0), 1.8, 1.2, "Image Projection", "Linear(2048, 64)\nBN + ReLU + Dropout", "#ce93d8", "#6a1b9a")

draw_arrow(ax, (5.5, 4.9), (6.0, 4.9), "[B, 64]")
draw_arrow(ax, (5.5, 1.6), (6.0, 1.6), "[B, 2048]")

# 4. Fusion Concatenation
draw_box(ax, (8.3, 2.5), 2.2, 1.5, "Feature Fusion", "Concatenation: [e_s, e_img]\nFused Vector: [B, 128]", "#b2dfdb", "#00695c")

draw_arrow(ax, (7.8, 4.9), (8.3, 3.6), "e_sensor [B, 64]")
draw_arrow(ax, (7.8, 1.6), (8.3, 2.9), "e_image [B, 64]")

# 5. MLP Head and Output
draw_box(ax, (8.3, 0.4), 2.2, 1.5, "Regression Head & Output", "Linear(128, 64) -> ReLU\nLinear(64, 32) -> Linear(32, 1)\nSoftplus() -> PM2.5 (ug/m3)", "#ffe0b2", "#e65100")

draw_arrow(ax, (9.4, 2.5), (9.4, 1.9), "fused [B, 128]")

# Title banner
plt.suptitle("VayuDrishti Multimodal Fusion Architecture: Feature-Level Joint Estimation",
             fontsize=12.5, weight="bold", y=0.96)
ax.text(5.5, 6.0, "Synchronous 24-Hour Telemetry + Outdoor Camera Imagery with Non-Negative Softplus Output",
        ha="center", va="center", fontsize=9, color="#555555", style="italic")

plt.tight_layout()
out_fig = OUTPUT_DIR / "model_architecture.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"Saved: {out_fig}")
