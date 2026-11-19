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

C_TEAL_BG = "#ccfbf1"  # Teal-100
C_TEAL_BD = "#0d9488"  # Teal-600

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
        try:
            fig.savefig(str(out_path.resolve()), dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Warning saving {base_name}{ext}: {e}")
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

    draw_box(ax, (1.60, 1.44), 1.25, 0.54, "ResNet-50 Projector", "Backbone → e_v (32D)",
             facecolor=C_CARD_BG, edgecolor=C_PURP_BD)
    draw_arrow(ax, (1.32, 1.71), (1.60, 1.71))

    draw_box(ax, (1.60, 0.66), 1.25, 0.54, "Distance Embedding", "Linear + ReLU → e_d (8D)",
             facecolor=C_CARD_BG, edgecolor=C_AMBR_BD)
    draw_arrow(ax, (1.32, 0.93), (1.60, 0.93))

    # Col 3: Baseline & Tri-Modal Fusion (x = 3.17 to 4.47, w = 1.30)
    draw_box(ax, (3.17, 2.22), 1.30, 0.54, "Sensor Baseline", "y_sensor = [PM2.5, PM10]",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD)
    draw_arrow(ax, (2.85, 2.49), (3.17, 2.49))

    draw_box(ax, (3.17, 0.66), 1.30, 1.32, "Feature Fusion", "[h_s || m·e_v || e_d]\n104D Fused Vector",
             facecolor=C_CARD_BG, edgecolor=C_DARK, lw=1.1)
    draw_arrow(ax, (2.85, 2.30), (3.17, 1.65))
    draw_arrow(ax, (2.85, 1.71), (3.17, 1.32))
    draw_arrow(ax, (2.85, 0.93), (3.17, 0.98))

    # Col 4: Dual Heads (x = 4.77 to 5.92, w = 1.15)
    draw_box(ax, (4.77, 1.44), 1.15, 0.54, "Learned Gate", "α_eff = m · α_raw",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD)
    draw_arrow(ax, (4.47, 1.50), (4.77, 1.71))

    draw_box(ax, (4.77, 0.66), 1.15, 0.54, "Residual Head", "Bounded Δ (±25, ±40)",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD)
    draw_arrow(ax, (4.47, 1.14), (4.77, 0.93))

    # Col 5: Gated Residual & Final Forecast (x = 5.92 to 7.04)
    draw_box(ax, (6.15, 0.80), 0.96, 0.76, "Gated Residual", "δ = α_eff · Δ\n(Zero when m=0)",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD, title_size=6.8, sub_size=5.6)
    draw_arrow(ax, (5.92, 1.71), (6.15, 1.35))
    draw_arrow(ax, (5.92, 0.93), (6.15, 0.98))

    draw_box(ax, (5.92, 2.22), 1.12, 0.54, "PM2.5 / PM10 Forecast", "y_hybrid = max(0, y_s + δ)",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD)
    draw_arrow(ax, (4.47, 2.49), (5.92, 2.49), label="y_sensor", label_pos=0.5, label_offset=(0, 0.07))
    draw_arrow(ax, (6.63, 1.56), (6.63, 2.22), label="+ δ", label_pos=0.5, label_offset=(0.18, 0))

    # Bottom Fallback Banner
    draw_box(ax, (0.18, 0.10), 6.86, 0.40,
             "Guaranteed Sensor Fallback Path (When Images are Missing)",
             "No Image Available (m = 0)   →   Learned Gate α_eff = 0   →   Residual Correction δ = 0   →   y_hybrid ≡ y_sensor identically",
             facecolor="#f8fafc", edgecolor="#64748b", lw=0.8, title_size=7.2, sub_size=6.2)

    save_fig(fig, "fig_arch_overall")


# ==============================================================================
# 9. FIG 2.7: GEODESIC IDW SPATIAL INTERPOLATION WORKFLOW (7.16 x 3.20 in)
# ==============================================================================
def generate_fig_arch_idw_spatial():
    fig, ax = plt.subplots(figsize=(7.16, 3.20), dpi=300)
    ax.set_xlim(0, 7.16)
    ax.set_ylim(0, 3.20)
    ax.axis("off")

    ax.text(3.58, 3.06, "Geodesic Inverse Distance Weighting (IDW) Spatial Interpolation Workflow",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=C_DARK)
    ax.text(3.58, 2.91, "Distance Kernel, Physical Co-Location Check, and Sparse-Network Safeguards",
            ha="center", va="center", fontsize=6.8, color=C_MUTED, style="italic")

    # Col 1: Inputs (x = 0.15, w = 1.65)
    draw_box(ax, (0.15, 1.82), 1.65, 0.90, "Target Coordinates", "(lat, lon) Query Point\n1 km Resolution Grid",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.0, sub_size=5.8)
    draw_box(ax, (0.15, 0.65), 1.65, 0.90, "Active Monitor Registry", "Catalogued CAAQMS Stations\nPhysical Ground Truth (z_i)",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=6.7, sub_size=5.8)

    # Col 2: Geodesic Engine (x = 2.30, w = 1.85)
    draw_box(ax, (2.30, 0.95), 1.85, 1.45, "Geodesic Distance Kernel",
             "Haversine Great-Circle Formula\nd_i = 2R · arcsin(√hav(Δφ, Δλ))\nR_earth = 6,371.0 km",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD, title_size=7.4, sub_size=5.9)
    draw_arrow(ax, (1.80, 2.27), (2.30, 1.95))
    draw_arrow(ax, (1.80, 1.10), (2.30, 1.40))

    # Col 3: Provenance Branches (x = 4.80, w = 2.22)
    draw_box(ax, (4.80, 2.15), 2.22, 0.62, "1. Co-Located Observation",
             "Criteria: min(d_i) ≤ 0.05 km\nReturns exact measured ground truth",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.0, sub_size=5.6)
    draw_arrow(ax, (4.15, 1.90), (4.80, 2.35), label="d_min ≤ 50m", label_pos=0.45, label_offset=(0, 0.08), fontsize=5.5)

    draw_box(ax, (4.80, 1.35), 2.22, 0.62, "2. Nearest-Station Fallback",
             "Criteria: N < 3 or min(d_i) > 45 km\nContinuous surface suppressed (1-NN)",
             facecolor=C_ROSE_BG, edgecolor=C_ROSE_BD, title_size=7.0, sub_size=5.6)
    draw_arrow(ax, (4.15, 1.68), (4.80, 1.66), label="N < 3 / d > 45km", label_pos=0.45, label_offset=(0, 0.08), fontsize=5.5)

    draw_box(ax, (4.80, 0.40), 2.22, 0.78, "3. Vectorized IDW Surface",
             "Criteria: N ≥ 3 within 45 km radius\nw_i = (d_i + 0.001)^(-2.0)\nẑ = Σ(w_i·z_i) / Σw_i  (Clamped ≥ 0)",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=7.0, sub_size=5.6)
    draw_arrow(ax, (4.15, 1.45), (4.80, 0.85), label="N ≥ 3 & d ≤ 45km", label_pos=0.45, label_offset=(0, -0.08), fontsize=5.5)

    save_fig(fig, "fig_arch_idw_spatial")


# ==============================================================================
# 10. FIG 2.8: SPATIAL RELIABILITY, TELEMETRY FRESHNESS & GEOMETRY (7.16 x 3.20 in)
# ==============================================================================
def generate_fig_arch_reliability():
    fig, ax = plt.subplots(figsize=(7.16, 3.20), dpi=300)
    ax.set_xlim(0, 7.16)
    ax.set_ylim(0, 3.20)
    ax.axis("off")

    ax.text(3.58, 3.06, "Spatial Reliability, Telemetry Freshness & Geometry Audit Workflow",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=C_DARK)
    ax.text(3.58, 2.91, "5-Tier Latency Classification, Clark-Evans Nearest-Neighbor Ratio, and Urban Tiers",
            ha="center", va="center", fontsize=6.8, color=C_MUTED, style="italic")

    # Col 1: Ingestion (x = 0.15, w = 1.80)
    draw_box(ax, (0.15, 1.50), 1.80, 1.25, "Live Telemetry Ingestion",
             "OpenAQ v3 API / CPCB CAAQMS\nHourly Criteria Pollutants\nTimestamp Verification (t_obs)\nLatency: Δt = t_now - t_obs",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.2, sub_size=5.7)

    # Col 2: 5-Tier Freshness Gate (x = 2.55, w = 1.95)
    draw_box(ax, (2.55, 1.45), 1.95, 1.35, "5-Tier Telemetry Governance",
             "Fresh (Δt ≤ 2h): Active Predictor\nDelayed (2–6h): Cautionary Flag\nStale (6–24h): Excluded from IDW\nOffline (> 24h): Suppressed / Inactive\nUnknown: Missing Timestamp Dropped",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD, title_size=7.0, sub_size=5.5)
    draw_arrow(ax, (1.95, 2.12), (2.55, 2.12), label="Raw Packets", label_pos=0.5, label_offset=(0, 0.12), fontsize=5.5)

    # Col 3: Clark-Evans Geometry Audit (x = 5.10, w = 1.95)
    draw_box(ax, (5.10, 1.45), 1.95, 1.35, "Clark-Evans Geometry Audit",
             "Spatial Clustering Index R = r̄_obs / r̄_exp\nExpected Distance: r̄_exp = 0.5 / √(ρ)\nClustered (R < 0.85): Patna, Kolkata\nRandom (0.85 ≤ R ≤ 1.15): Pune, Lucknow\nDispersed (R > 1.15): Ahmedabad, Agra",
             facecolor=C_TEAL_BG, edgecolor=C_TEAL_BD, title_size=7.0, sub_size=5.4)
    draw_arrow(ax, (4.50, 2.12), (5.10, 2.12), label="Active Stations", label_pos=0.5, label_offset=(0, 0.12), fontsize=5.5)

    # Bottom Banner: 5-Tier Operational Urban Reliability Classification
    draw_box(ax, (0.15, 0.12), 6.90, 1.12, "Operational Urban Reliability Tiers (31 Indian Cities Master Audit)",
             "Tier 1 (Calibrated): Delhi NCR (98 monitors, dense, validated continuous IDW and Conformal coverage)\n"
             "Tier 2 (High Reliability): Mumbai (57), Kolkata (21), Hyderabad (20), Bengaluru (19), Pune (13), Patna (12), Ahmedabad (11)\n"
             "Tier 3 (Moderate Reliability): Chennai (9), Jaipur (9), Lucknow (6), Agra (6), Vijayawada (6), Amaravati (5)\n"
             "Tier 4 (Marginal Reliability): Kochi (3), Coimbatore (3), Kanpur (3) — local nearest-station fallback recommended\n"
             "Tier 5 (Suppressed Interpolation): Visakhapatnam, Tirupati, Bhubaneswar, Surat, Guntur (≤ 2 monitors — IDW barred)",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.1, sub_size=5.4)
    draw_arrow(ax, (6.07, 1.45), (6.07, 1.24))

    save_fig(fig, "fig_arch_reliability")


# ==============================================================================
# 11. FIG 2.9: DISTANCE-SCALED CONFORMAL UNCERTAINTY WORKFLOW (7.16 x 3.20 in)
# ==============================================================================
def generate_fig_arch_conformal():
    fig, ax = plt.subplots(figsize=(7.16, 3.20), dpi=300)
    ax.set_xlim(0, 7.16)
    ax.set_ylim(0, 3.20)
    ax.axis("off")

    ax.text(3.58, 3.06, "Distance-Scaled Conformal Uncertainty Estimation Workflow",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=C_DARK)
    ax.text(3.58, 2.91, "Distribution-Free Prediction Intervals with Distance-Dependent Spatial Expansion",
            ha="center", va="center", fontsize=6.8, color=C_MUTED, style="italic")

    # Step 1: Calibration Set (x = 0.15, w = 1.85)
    draw_box(ax, (0.15, 1.62), 1.85, 1.12, "1. Calibration Dataset",
             "Held-out spatial evaluation points\nGround Truth: y_i (μg/m³)\nPoint Forecast / Estimate: ŷ_i\nNearest Station Distance: d_i (km)",
             facecolor=C_CARD_BG, edgecolor=C_DARK, title_size=7.2, sub_size=5.7)

    # Step 2: Scoring (x = 2.60, w = 1.95)
    draw_box(ax, (2.60, 1.62), 1.95, 1.12, "2. Distance-Scaled Scoring",
             "Non-Conformity Metric:\ns_i = |y_i - ŷ_i| / (1 + γ · d_i)\nScaling Parameter: γ = 0.05 km⁻¹\nNormalizes error far from monitors",
             facecolor=C_AMBR_BG, edgecolor=C_AMBR_BD, title_size=7.2, sub_size=5.6)
    draw_arrow(ax, (2.00, 2.18), (2.60, 2.18), label="Residuals", label_pos=0.5, label_offset=(0, 0.08), fontsize=5.5)

    # Step 3: Quantile (x = 5.10, w = 1.95)
    draw_box(ax, (5.10, 1.62), 1.95, 1.12, "3. Conformal Quantile",
             "Finite-Sample Quantile Calculation:\nq̂_(1-α) = Quantile_p(S)\nIndex p = ⌈(n + 1)(1 - α)⌉ / n\nMarginal Bound: P(Y ∈ C(X)) ≥ 1 - α",
             facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, title_size=7.1, sub_size=5.5)
    draw_arrow(ax, (4.55, 2.18), (5.10, 2.18), label="Score Set S", label_pos=0.5, label_offset=(0, 0.08), fontsize=5.5)

    # Step 4: Operational Prediction Band Banner
    draw_box(ax, (0.15, 0.15), 6.90, 1.25, "4. Operational Conformal Prediction Interval Formulation",
             "Prediction Band: C(x) = [ max(0, ŷ - q̂_(1-α) · (1 + γ · d_min)),   ŷ + q̂_(1-α) · (1 + γ · d_min) ]\n"
             "• Co-Located Calibration (d_min = 0): Interval width is purely 2 · q̂_(1-α) based on physical sensor residuals.\n"
             "• Geographic Blindspots (d_min > 0): Interval expands smoothly by (1 + 0.05 · d_min), honest about unmonitored risk.\n"
             "• Validated Empirical Coverage: 92.8% (80% Tier), 96.5% (90% Tier), 99.37% (95% Tier) on PM2.5.\n"
             "• Honest Governance: Guarantees marginal coverage at the expense of wider bounds during severe winter smog.",
             facecolor=C_GRN_BG, edgecolor=C_GRN_BD, title_size=7.2, sub_size=5.6)
    draw_arrow(ax, (6.07, 1.62), (6.07, 1.40))

    save_fig(fig, "fig_arch_conformal")


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
    print("Generating publication-quality IEEE Architecture and Methodology Figures...")
    generate_fig_arch_overall()
    generate_fig_arch_preprocessing()
    generate_fig_arch_lstm()
    generate_fig_arch_resnet()
    generate_fig_arch_image_only()
    generate_fig_arch_fusion()
    generate_fig_arch_residual_hybrid()
    generate_fig_arch_ensemble()
    generate_fig_arch_idw_spatial()
    generate_fig_arch_reliability()
    generate_fig_arch_conformal()
    print("All 11 architecture figures successfully generated.")
