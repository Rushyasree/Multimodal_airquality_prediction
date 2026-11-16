"""
scripts/generate_ieee_architecture_diagrams.py

Generates compact, publication-quality, minimalist scientific illustrations
strictly adhering to IEEE conference guidelines:
- Compact height (2.3 - 2.7 inches for single column; 3.2 inches for full-width)
  preventing awkward column breaks and large blank spaces.
- Zero truncated text (complete labels, ample box padding).
- Zero text-arrow collisions (clear stage separation).
- Consistent IEEE typography and accessible colors.
- Vector output (SVG, PDF, 300 DPI PNG).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Accessible, publication-grade color palette
C_DARK    = "#0f172a"  # Slate-900: Text and connectors
C_SLATE   = "#334155"  # Slate-700: Subtitles and annotations
C_MUTED   = "#64748b"  # Slate-500: Dimensions and auxiliary notes

# Thematic pastel fills with matching crisp borders
C_BLUE_BG = "#e0f2fe"  # Sky-100: Sensor streams
C_BLUE_BD = "#0284c7"  # Sky-600

C_PURP_BG = "#f3e8ff"  # Purple-100: Vision stream
C_PURP_BD = "#7c3aed"  # Purple-600

C_AMBR_BG = "#fef3c7"  # Amber-100: Distance / Spatial / Blending
C_AMBR_BD = "#d97706"  # Amber-600

C_ROSE_BG = "#ffe4e6"  # Rose-100: Gating / Residual / Severe Crisis
C_ROSE_BD = "#e11d48"  # Rose-600

C_GRN_BG  = "#dcfce7"  # Emerald-100: Outputs / Forecasts
C_GRN_BD  = "#16a34a"  # Emerald-600

C_CARD_BG = "#ffffff"  # Pure white card
C_CARD_BD = "#94a3b8"  # Slate-400

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

def draw_box(ax, xy, width, height, title, subtitle="",
             facecolor=C_CARD_BG, edgecolor=C_DARK, lw=0.9, corner=0.03,
             title_size=7.4, sub_size=6.2, title_color=C_DARK):
    """Draws a clean, single-concept box with guaranteed margins and no text clipping."""
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.008,rounding_size={corner}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw
    )
    ax.add_patch(box)
    
    cx = x + width / 2.0
    if subtitle:
        cy_title = y + height * 0.65
        cy_sub   = y + height * 0.28
        ax.text(cx, cy_title, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color)
        ax.text(cx, cy_sub, subtitle, ha="center", va="center",
                fontsize=sub_size, color=C_SLATE)
    else:
        cy_title = y + height / 2.0
        ax.text(cx, cy_title, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color)
    return box

def draw_arrow(ax, p1, p2, label="", label_pos=0.5, label_offset=(0, 0.06),
               color=C_DARK, lw=0.9, arrowstyle="-|>", fontsize=6.2, ha="center"):
    """Draws a clean directed connector with optional label placed without collision."""
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=lw,
                        shrinkA=1, shrinkB=1, mutation_scale=6)
    )
    if label:
        lx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
        ly = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
        ax.text(lx, ly, label, ha=ha, va="center",
                fontsize=fontsize, fontweight="semibold", color=C_SLATE,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))

def save_fig(fig, base_name):
    for ext in [".png", ".pdf", ".svg"]:
        out_path = OUTPUT_DIR / f"{base_name}{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base_name} (PNG, PDF, SVG)")


# ==============================================================================
# 1. FIG 1: COMPACT MAIN HYBRID ARCHITECTURE (7.16 x 3.30 in)
# ==============================================================================
def generate_fig_arch_overall():
    fig, ax = plt.subplots(figsize=(7.16, 3.30), dpi=300)
    ax.set_xlim(0, 7.16)
    ax.set_ylim(0, 3.30)
    ax.axis("off")

    # Header
    ax.text(3.58, 3.16, "VayuDrishti: Residual Multimodal Hybrid Forecasting Architecture",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_DARK)
    ax.text(3.58, 3.01, "Multimodal Fusion with Quality Gating and Guaranteed Sensor Fallback",
            ha="center", va="center", fontsize=7.0, color=C_MUTED, style="italic")

    # Col 1: Modality Inputs (x = 0.18 to 1.32, w = 1.14)
    draw_box(ax, (0.18, 2.22), 1.14, 0.54, "Sensor Data", "24h × 27 Channels",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD)
    draw_box(ax, (0.18, 1.44), 1.14, 0.54, "Outdoor Image", "224 × 224 × 3 RGB",
             facecolor=C_PURP_BG, edgecolor=C_PURP_BD)
    draw_box(ax, (0.18, 0.66), 1.14, 0.54, "Location Distance", "Station Offset d (km)",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD)

    # Col 2: Feature Encoders (x = 1.60 to 2.85, w = 1.25)
    draw_box(ax, (1.60, 2.22), 1.25, 0.54, "LSTM Sensor Encoder", "2 Layers (64 Units)",
             facecolor=C_CARD_BG, edgecolor=C_BLUE_BD)
    draw_arrow(ax, (1.32, 2.49), (1.60, 2.49))

    draw_box(ax, (1.60, 1.44), 1.25, 0.54, "ResNet-50 Projector", "Backbone → 32D",
             facecolor=C_CARD_BG, edgecolor=C_PURP_BD)
    draw_arrow(ax, (1.32, 1.71), (1.60, 1.71))

    draw_box(ax, (1.60, 0.66), 1.25, 0.54, "Distance Embedding", "Linear + ReLU → 8D",
             facecolor=C_CARD_BG, edgecolor=C_AMBR_BD)
    draw_arrow(ax, (1.32, 0.93), (1.60, 0.93))

    # Col 3: Baseline & Tri-Modal Fusion (x = 3.17 to 4.47, w = 1.30)
    draw_box(ax, (3.17, 2.22), 1.30, 0.54, "Sensor Baseline", "y_sensor = [PM2.5, PM10]",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD)
    draw_arrow(ax, (2.85, 2.49), (3.17, 2.49), label="Baseline", label_pos=0.5, label_offset=(0, 0.07))

    draw_box(ax, (3.17, 0.66), 1.30, 1.32, "Feature Fusion", "[h_s || m·e_v || e_d]\n104D Fused Vector",
             facecolor=C_CARD_BG, edgecolor=C_DARK, lw=1.1)
    draw_arrow(ax, (2.85, 2.30), (3.17, 1.65), label="h_s (64D)", label_pos=0.45, label_offset=(-0.05, 0.07))
    draw_arrow(ax, (2.85, 1.71), (3.17, 1.32), label="e_v (32D)", label_pos=0.45, label_offset=(0, 0.07))
    draw_arrow(ax, (2.85, 0.93), (3.17, 0.98), label="e_d (8D)", label_pos=0.45, label_offset=(0, 0.07))

    # Col 4: Dual Heads (x = 4.77 to 5.92, w = 1.15)
    draw_box(ax, (4.77, 1.44), 1.15, 0.54, "Learned Gate", "α_eff = m · α_raw",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD)
    draw_arrow(ax, (4.47, 1.50), (4.77, 1.71))

    draw_box(ax, (4.77, 0.66), 1.15, 0.54, "Residual Head", "Bounded Δ (±25, ±40)",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD)
    draw_arrow(ax, (4.47, 1.14), (4.77, 0.93))

    # Col 5: Gated Residual & Final Forecast (x = 5.92 to 7.04)
    draw_box(ax, (6.12, 0.85), 0.92, 0.72, "Gated Residual", "δ = α_eff · Δ\n(Zero when m=0)",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD, title_size=7.2, sub_size=5.8)
    draw_arrow(ax, (5.92, 1.71), (6.12, 1.35))
    draw_arrow(ax, (5.92, 0.93), (6.12, 1.05))

    draw_box(ax, (5.92, 2.22), 1.12, 0.54, "PM2.5 / PM10 Forecast", "y_hybrid = max(0, y_s + δ)",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD)
    draw_arrow(ax, (4.47, 2.49), (5.92, 2.49), label="y_sensor", label_pos=0.5, label_offset=(0, 0.07))
    draw_arrow(ax, (6.58, 1.57), (6.58, 2.22), label="+ δ", label_pos=0.5, label_offset=(0.18, 0))

    # Bottom Fallback Banner
    draw_box(ax, (0.18, 0.10), 6.86, 0.40,
             "Guaranteed Sensor Fallback Path (When Images are Missing)",
             "No Image Available (m = 0)   →   Learned Gate α_eff = 0   →   Residual Correction δ = 0   →   y_hybrid ≡ y_sensor identically",
             facecolor="#f8fafc", edgecolor="#64748b", lw=0.8, title_size=7.2, sub_size=6.2)

    save_fig(fig, "fig_arch_overall")


# ==============================================================================
# 2. FIG 2: SENSOR PREPROCESSING PIPELINE (3.5 x 1.70 in)
# ==============================================================================
def generate_fig_arch_preprocessing():
    fig, ax = plt.subplots(figsize=(3.5, 1.70), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 1.70)
    ax.axis("off")

    ax.text(1.75, 1.62, "Sensor Sequence Ingestion & Preprocessing",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color=C_DARK)

    # Step 1: Input Channels (3 split blocks)
    w_b = 0.98
    draw_box(ax, (0.12, 1.22), w_b, 0.30, "Pollutants", "12 Channels",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=6.6, sub_size=5.4)
    draw_box(ax, (1.26, 1.22), w_b, 0.30, "Meteorology", "5 Channels",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=6.6, sub_size=5.4)
    draw_box(ax, (2.40, 1.22), w_b, 0.30, "Spatio-Temporal", "10 Channels",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD, title_size=6.6, sub_size=5.4)

    # Step 2: Quality Validation
    draw_box(ax, (0.25, 0.82), 3.0, 0.28, "Data Quality & Gap Filter", "Reject sequences with gaps > 4h",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=6.8, sub_size=5.5)
    draw_arrow(ax, (1.75, 1.22), (1.75, 1.10))

    # Step 3: Imputation & Standardization
    draw_box(ax, (0.25, 0.44), 3.0, 0.28, "Imputation & Standardization", "Forward-fill + Train Z-score normalization",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=6.8, sub_size=5.5)
    draw_arrow(ax, (1.75, 0.82), (1.75, 0.72))

    # Step 4: Final Sequence Tensor
    draw_box(ax, (0.25, 0.05), 3.0, 0.28, "Sensor Sequence Tensor", "Shape: [Batch, 24 Hours, 27 Features]",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.0, sub_size=5.6)
    draw_arrow(ax, (1.75, 0.44), (1.75, 0.33))

    save_fig(fig, "fig_arch_preprocessing")


# ==============================================================================
# 3. FIG 3: SENSOR-BASED LSTM BASELINE FORECASTER (3.5 x 2.4 in)
# ==============================================================================
def generate_fig_arch_lstm():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    ax.text(1.75, 2.30, "Sensor-Based LSTM Baseline Forecaster",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_DARK)

    # Sequence Input
    draw_box(ax, (0.25, 1.76), 3.0, 0.38, "Sensor Sequence", "24 Hours × 27 Channels",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=7.3, sub_size=5.9)

    # 2-Layer LSTM
    draw_box(ax, (0.25, 1.20), 3.0, 0.38, "LSTM Encoder", "2 Stacked Layers (64 Units each)",
             facecolor=C_CARD_BG, edgecolor=C_BLUE_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 1.76), (1.75, 1.58))

    # Latent State + Regressor
    draw_box(ax, (0.25, 0.64), 3.0, 0.38, "Dense Regressor Head", "Hidden State h_s (64D) → Linear Layers",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 1.20), (1.75, 1.02), label="h_s (64D)", label_pos=0.5)

    # Output Forecast
    draw_box(ax, (0.25, 0.08), 3.0, 0.38, "Sensor Baseline Forecast", "y_sensor = [PM2.5, PM10] (t + 1h)",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 0.64), (1.75, 0.46))

    save_fig(fig, "fig_arch_lstm")


# ==============================================================================
# 4. FIG 4: RESNET-50 VISUAL FEATURE EXTRACTOR (3.5 x 2.4 in)
# ==============================================================================
def generate_fig_arch_resnet():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    ax.text(1.75, 2.30, "ResNet-50 Visual Feature Extractor & Projector",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_DARK)

    # Image Input
    draw_box(ax, (0.25, 1.76), 3.0, 0.38, "Outdoor Surveillance Image", "224 × 224 × 3 RGB Frame",
             facecolor=C_PURP_BG, edgecolor=C_PURP_BD, title_size=7.3, sub_size=5.9)

    # Frozen ResNet
    draw_box(ax, (0.25, 1.20), 3.0, 0.38, "ResNet-50 Backbone", "Pretrained ImageNet Weights (Frozen)",
             facecolor=C_CARD_BG, edgecolor=C_PURP_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 1.76), (1.75, 1.58))

    # Projection Block
    draw_box(ax, (0.25, 0.64), 3.0, 0.38, "Visual Projection Network", "Global Avg Pool (2048D) → Linear + LN → 32D",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 1.20), (1.75, 1.02), label="2048D", label_pos=0.5)

    # Visual Output
    draw_box(ax, (0.25, 0.08), 3.0, 0.38, "Visual Feature Representation", "e_v in R^32 (Feeds Multimodal Fusion)",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 0.64), (1.75, 0.46), label="32D", label_pos=0.5)

    save_fig(fig, "fig_arch_resnet")


# ==============================================================================
# 5. FIG 5: STANDALONE OPTICAL REGRESSOR (3.5 x 2.4 in)
# ==============================================================================
def generate_fig_arch_image_only():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    ax.text(1.75, 2.30, "Standalone Optical Image Regressor (Baseline)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_DARK)

    # Image Input
    draw_box(ax, (0.25, 1.76), 3.0, 0.38, "Outdoor Image", "224 × 224 × 3 RGB Frame",
             facecolor=C_PURP_BG, edgecolor=C_PURP_BD, title_size=7.3, sub_size=5.9)

    # ResNet-50
    draw_box(ax, (0.25, 1.20), 3.0, 0.38, "ResNet-50 Feature Extractor", "Global Average Pooling → 2048D",
             facecolor=C_CARD_BG, edgecolor=C_PURP_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 1.76), (1.75, 1.58))

    # Dense MLP
    draw_box(ax, (0.25, 0.64), 3.0, 0.38, "Dense MLP Regressor Head", "Linear(2048 → 256 → 64 → 1)",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 1.20), (1.75, 1.02))

    # Standalone Estimate
    draw_box(ax, (0.25, 0.08), 3.0, 0.38, "Standalone PM2.5 Estimate", "High Error: MAE = 66.21 ug/m3",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD, title_size=7.4, sub_size=5.9, title_color="#be123c")
    draw_arrow(ax, (1.75, 0.64), (1.75, 0.46))

    save_fig(fig, "fig_arch_image_only")


# ==============================================================================
# 6. FIG 6: TRI-MODAL FEATURE FUSION (3.5 x 2.4 in)
# ==============================================================================
def generate_fig_arch_fusion():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    ax.text(1.75, 2.30, "Tri-Modal Feature Fusion Mechanism",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_DARK)

    # 3 Parallel Branches
    w_b = 0.98
    draw_box(ax, (0.12, 1.62), w_b, 0.48, "Sensor State", "h_s in R^64",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=7.0, sub_size=5.8)
    draw_box(ax, (1.26, 1.62), w_b, 0.48, "Masked Vision", "m·e_v in R^32",
             facecolor=C_PURP_BG, edgecolor=C_PURP_BD, title_size=7.0, sub_size=5.8)
    draw_box(ax, (2.40, 1.62), w_b, 0.48, "Distance", "e_d in R^8",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD, title_size=7.0, sub_size=5.8)

    # Concatenation Layer
    draw_box(ax, (0.25, 0.88), 3.0, 0.42, "Feature Fusion Layer", "Concatenation: [h_s || m·e_v || e_d] → 104D",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.4, sub_size=5.9)

    draw_arrow(ax, (0.61, 1.62), (1.1, 1.30))
    draw_arrow(ax, (1.75, 1.62), (1.75, 1.30))
    draw_arrow(ax, (2.89, 1.62), (2.4, 1.30))

    # Fused Output
    draw_box(ax, (0.25, 0.12), 3.0, 0.42, "Fused Representation (104D)", "Feeds Gating Network & Residual Heads",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.75, 0.88), (1.75, 0.54))

    save_fig(fig, "fig_arch_fusion")


# ==============================================================================
# 7. FIG 7: RESIDUAL MULTIMODAL HYBRID (3.5 x 2.45 in)
# ==============================================================================
def generate_fig_arch_residual_hybrid():
    fig, ax = plt.subplots(figsize=(3.5, 2.45), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.45)
    ax.axis("off")

    ax.text(1.75, 2.34, "Residual Multimodal Hybrid & Quality Gating",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_DARK)

    # Step 1: Fused Feature Representation (Full Width)
    draw_box(ax, (0.15, 1.84), 3.20, 0.36, "Tri-Modal Fused Representation",
             "F = [h_s || m·e_v || e_d] in R^104",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.0, sub_size=5.6)

    # Step 2: Parallel Gating and Residual Heads
    draw_box(ax, (0.15, 1.28), 1.50, 0.38, "Gating Network", "α_eff = m · α_raw in [0, 1]",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD, title_size=6.8, sub_size=5.4)
    draw_arrow(ax, (0.90, 1.84), (0.90, 1.66))

    draw_box(ax, (1.85, 1.28), 1.50, 0.38, "Residual Head", "Bounded Δ in [±25, ±40]",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD, title_size=6.8, sub_size=5.4)
    draw_arrow(ax, (2.60, 1.84), (2.60, 1.66))

    # Step 3: Gated Residual Correction
    draw_box(ax, (0.15, 0.72), 3.20, 0.38, "Bounded Residual Adjustment",
             "Correction: δ = α_eff · Δ (Zero when m = 0)",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.0, sub_size=5.6)
    draw_arrow(ax, (0.90, 1.28), (1.20, 1.10))
    draw_arrow(ax, (2.60, 1.28), (2.30, 1.10))

    # Step 4: Final Non-Negative Fusion
    draw_box(ax, (0.15, 0.12), 3.20, 0.42, "Final Hybrid Forecast",
             "y_hybrid = max(0, y_sensor + α_eff · Δ)\nGuaranteed Sensor Fallback when m = 0",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.2, sub_size=5.7)
    draw_arrow(ax, (1.75, 0.72), (1.75, 0.54), label="y_sensor + δ", label_pos=0.5)

    save_fig(fig, "fig_arch_residual_hybrid")


# ==============================================================================
# 8. FIG 8: SEVERITY-GATED SENSOR ENSEMBLE (3.5 x 2.05 in)
# ==============================================================================
def generate_fig_arch_ensemble():
    fig, ax = plt.subplots(figsize=(3.5, 2.05), dpi=300)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.05)
    ax.axis("off")

    ax.text(1.75, 1.95, "Severity-Gated Sensor Ensemble Forecaster",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_DARK)

    # Input
    draw_box(ax, (0.25, 1.52), 3.0, 0.32, "Sensor Sequence (24h × 27)", "Continuous Numerical Telemetry",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=7.0, sub_size=5.6)

    # Parallel Models
    draw_box(ax, (0.15, 0.98), 1.50, 0.38, "LSTM Forecaster", "Recurrent Temporal Model",
             facecolor=C_CARD_BG, edgecolor=C_BLUE_BD, title_size=6.8, sub_size=5.5)
    draw_arrow(ax, (0.90, 1.52), (0.90, 1.36))

    draw_box(ax, (1.85, 0.98), 1.50, 0.38, "LightGBM (GBDT)", "49 Tabular Lag Features",
             facecolor=C_CARD_BG, edgecolor=C_AMBR_BD, title_size=6.8, sub_size=5.5)
    draw_arrow(ax, (2.60, 1.52), (2.60, 1.36))

    # Severity Threshold Switch
    draw_box(ax, (0.15, 0.48), 3.20, 0.35, "Severity Threshold Decision Switch",
             "Standard: 0.40/0.60 | Crisis (PM2.5 ≥ 250): 1.0 Pure LSTM",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD, title_size=7.0, sub_size=5.5, title_color="#be123c")
    draw_arrow(ax, (0.90, 0.98), (1.10, 0.83))
    draw_arrow(ax, (2.60, 0.98), (2.40, 0.83))

    # Ensemble Output
    draw_box(ax, (0.25, 0.05), 3.00, 0.32, "Operational Ensemble Forecast", "y_ens = w·y_lstm + (1-w)·y_gbdt",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.2, sub_size=5.7)
    draw_arrow(ax, (1.75, 0.48), (1.75, 0.37))

    save_fig(fig, "fig_arch_ensemble")


if __name__ == "__main__":
    print("Generating compact, non-colliding IEEE Architecture Figures...")
    generate_fig_arch_overall()
    generate_fig_arch_preprocessing()
    generate_fig_arch_lstm()
    generate_fig_arch_resnet()
    generate_fig_arch_image_only()
    generate_fig_arch_fusion()
    generate_fig_arch_residual_hybrid()
    generate_fig_arch_ensemble()
    print("All 8 compact architecture figures successfully generated.")
