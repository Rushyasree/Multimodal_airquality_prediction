"""
scripts/generate_missing_report_diagrams.py

Generates publication-quality academic architecture diagrams for:
- Figure 2.7: IDW Spatial Interpolation Workflow
- Figure 2.8: Spatial Reliability and Freshness Assessment Workflow
- Figure 2.9: Conformal Uncertainty Estimation Workflow

Clean white background, high-contrast crisp academic styling, zero overlapping text.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Standardized publication color palette
C_NAVY     = "#0f172a"  # Slate-900 (Main text / titles)
C_SLATE    = "#334155"  # Slate-700 (Body text / equations)
C_MUTED    = "#64748b"  # Slate-500 (Captions / notes)
C_LINE     = "#475569"  # Slate-600 (Connectors)

# Clean academic card tints
C_BLUE_BG  = "#f0f9ff"  # Sky-50
C_BLUE_BD  = "#0284c7"  # Sky-600
C_TEAL_BG  = "#f0fdfa"  # Teal-50
C_TEAL_BD  = "#0d9488"  # Teal-600
C_AMBR_BG  = "#fffbeb"  # Amber-50
C_AMBR_BD  = "#d97706"  # Amber-600
C_ROSE_BG  = "#fff1f2"  # Rose-50
C_ROSE_BD  = "#e11d48"  # Rose-600
C_GRN_BG   = "#f0fdf4"  # Emerald-50
C_GRN_BD   = "#16a34a"  # Emerald-600
C_SLATE_BG = "#f8fafc"  # Slate-50
C_SLATE_BD = "#475569"  # Slate-600

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

def draw_academic_card(ax, xy, w, h, title, subtitle_lines=None,
                       bg=C_SLATE_BG, border=C_SLATE_BD, lw=1.2,
                       title_size=8.8, sub_size=7.2, title_color=None):
    """Draws a crisp academic container with perfectly budgeted text vertical spacing."""
    x, y = xy
    if title_color is None:
        title_color = border

    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=lw,
        edgecolor=border,
        facecolor=bg
    )
    ax.add_patch(box)

    cx = x + w / 2.0
    if not subtitle_lines:
        ax.text(cx, y + h / 2.0, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color)
        return

    # Calculate vertical spacing
    n_sub = len(subtitle_lines)
    y_title = y + h - 0.22
    ax.text(cx, y_title, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=title_color)

    if n_sub == 1:
        ax.text(cx, y + (h - 0.22) / 2.0, subtitle_lines[0], ha="center", va="center",
                fontsize=sub_size, color=C_SLATE)
    else:
        line_spacing = (h - 0.38) / max(1, (n_sub - 1))
        for i, line in enumerate(subtitle_lines):
            y_line = y + h - 0.40 - i * line_spacing
            ax.text(cx, y_line, line, ha="center", va="center",
                    fontsize=sub_size, color=C_SLATE)

def draw_clean_arrow(ax, p1, p2, label="", label_pos=0.5, label_offset=(0, 0.08), color=C_LINE, lw=1.3):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=11)
    )
    if label:
        lx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
        ly = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.2,
                fontweight="semibold", color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.9))

# ==============================================================================
# 1. FIG 2.7: GEODESIC IDW SPATIAL INTERPOLATION WORKFLOW
# ==============================================================================
def generate_idw_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Col 1: Inputs (x = 0.15, w = 1.65)
    draw_academic_card(ax, (0.15, 2.35), 1.65, 1.10,
                       "Target Location",
                       ["Query Point (lat, lon)", "Urban Bounding Box", "Resolution: 1 km Grid"],
                       bg=C_SLATE_BG, border=C_NAVY)

    draw_academic_card(ax, (0.15, 0.65), 1.65, 1.10,
                       "Active Station Registry",
                       ["CPCB / OpenAQ v3 Monitors", "Verified Coordinates (φi, λi)", "Real-time Telemetry (zi)"],
                       bg=C_BLUE_BG, border=C_BLUE_BD)

    # Col 2: Geodesic Distance Engine (x = 2.30, w = 1.95)
    draw_academic_card(ax, (2.30, 1.30), 1.95, 1.45,
                       "Geodesic Distance Engine",
                       ["Haversine Metric (Spherical Earth)",
                        "di = 2R · arcsin(√hav(Δφ, Δλ))",
                        "R_earth = 6,371.0 km",
                        "Computes all monitor distances di"],
                       bg=C_TEAL_BG, border=C_TEAL_BD)

    draw_clean_arrow(ax, (1.80, 2.85), (2.30, 2.25), label="Target (lat, lon)", label_pos=0.5, label_offset=(0, 0.07))
    draw_clean_arrow(ax, (1.80, 1.20), (2.30, 1.75), label="Stations (φi, λi)", label_pos=0.5, label_offset=(0, -0.07))

    # Col 3: Three Provenance Paths (x = 4.75, w = 2.30)
    draw_academic_card(ax, (4.75, 2.65), 2.30, 0.90,
                       "1. Co-Located Observation",
                       ["Criteria: min(di) <= 0.05 km", "Returns Exact Physical Sensor Reading", "Status: Physical Ground Truth"],
                       bg=C_GRN_BG, border=C_GRN_BD)

    draw_academic_card(ax, (4.75, 1.50), 2.30, 0.90,
                       "2. Nearest-Station Fallback",
                       ["Criteria: N < 3 or min(di) > 45 km", "Continuous surface suppressed", "Status: 1-NN Cautionary Fallback"],
                       bg=C_AMBR_BG, border=C_AMBR_BD)

    draw_academic_card(ax, (4.75, 0.25), 2.30, 1.05,
                       "3. Vectorized IDW Surface",
                       ["Criteria: N >= 3 within 45 km",
                        "Weight: wi = (di + 0.001)^(-2.0)",
                        "ẑ = Σ(wi · zi) / Σwi",
                        "Status: Interpolated Continuous Field"],
                       bg=C_BLUE_BG, border=C_BLUE_BD)

    # Arrows from Geodesic Engine to Paths
    draw_clean_arrow(ax, (4.25, 2.20), (4.75, 3.00), label="d_min ≤ 50m", label_pos=0.45, label_offset=(0, 0.06))
    draw_clean_arrow(ax, (4.25, 2.00), (4.75, 1.95), label="N < 3 or d > 45km", label_pos=0.45, label_offset=(0, 0.06))
    draw_clean_arrow(ax, (4.25, 1.80), (4.75, 0.85), label="N ≥ 3 & d ≤ 45km", label_pos=0.45, label_offset=(0, -0.06))

    out_p = OUT_DIR / "fig_arch_idw_spatial.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated clean {out_p.name}")

# ==============================================================================
# 2. FIG 2.8: SPATIAL RELIABILITY, TELEMETRY FRESHNESS & GEOMETRY AUDIT
# ==============================================================================
def generate_reliability_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Stage 1: Ingestion Stream (x = 0.15, w = 1.70)
    draw_academic_card(ax, (0.15, 2.05), 1.70, 1.45,
                       "Live Telemetry Stream",
                       ["OpenAQ v3 API / CPCB CAAQMS",
                        "Hourly Pollutants (PM2.5, PM10)",
                        "Surface Weather Variables",
                        "Observed Timestamp (t_obs)",
                        "Latency: Δt = t_now - t_obs"],
                       bg=C_SLATE_BG, border=C_NAVY)

    # Stage 2: 5-Tier Freshness Filter (x = 2.25, w = 2.10)
    draw_academic_card(ax, (2.25, 1.85), 2.10, 1.75,
                       "5-Tier Telemetry Governance",
                       ["• Fresh (Δt ≤ 2h): Active Model Input",
                        "• Delayed (2 < Δt ≤ 6h): Advisory Flag",
                        "• Stale (6 < Δt ≤ 24h): Excluded IDW",
                        "• Offline (Δt > 24h): Suppressed",
                        "• Unknown: Missing Timestamp Filter"],
                       bg=C_AMBR_BG, border=C_AMBR_BD, sub_size=6.8)

    draw_clean_arrow(ax, (1.85, 2.75), (2.25, 2.75), label="Raw Packets", label_pos=0.5, label_offset=(0, 0.07))

    # Stage 3: Spatial Geometry Audit (x = 4.75, w = 2.30)
    draw_academic_card(ax, (4.75, 1.85), 2.30, 1.75,
                       "Clark-Evans Geometry Audit",
                       ["Spatial Clustering Index:",
                        "R = r̄_obs / r̄_exp",
                        "r̄_exp = 0.5 / √(Density ρ)",
                        "• Clustered: R < 0.85 (e.g., Patna 0.38)",
                        "• Random: 0.85 ≤ R ≤ 1.15",
                        "• Dispersed: R > 1.15 (e.g., Agra 1.35)"],
                       bg=C_TEAL_BG, border=C_TEAL_BD, sub_size=6.8)

    draw_clean_arrow(ax, (4.35, 2.75), (4.75, 2.75), label="Active Monitors", label_pos=0.5, label_offset=(0, 0.07))

    # Bottom Tier Summary Bar (x = 0.15 to 7.05, w = 6.90)
    draw_academic_card(ax, (0.15, 0.20), 6.90, 1.25,
                       "Operational Urban Reliability Classification (31 Indian Cities)",
                       ["Tier 1 (Calibrated): Delhi NCR (98 monitors, dense, validated continuous IDW & Conformal bands)",
                        "Tier 2 (High Reliability): Mumbai, Kolkata, Bengaluru, Hyderabad, Pune, Patna, Ahmedabad (11–57 monitors)",
                        "Tier 3 (Moderate Reliability): Chennai, Jaipur, Lucknow, Agra, Vijayawada, Amaravati (4–9 monitors)",
                        "Tier 4 (Marginal Reliability): Kochi, Coimbatore, Kanpur (3 monitors — local nearest-station fallback recommended)",
                        "Tier 5 (Suppressed Interpolation): Visakhapatnam, Tirupati, Bhubaneswar, Surat, Guntur (≤ 2 monitors — IDW barred)"],
                       bg=C_GRN_BG, border=C_GRN_BD, sub_size=6.6)

    draw_clean_arrow(ax, (5.90, 1.85), (5.90, 1.45), label="Governance Mapping", label_pos=0.5, label_offset=(0, 0.06))

    out_p = OUT_DIR / "fig_arch_reliability.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated clean {out_p.name}")

# ==============================================================================
# 3. FIG 2.9: DISTANCE-SCALED CONFORMAL UNCERTAINTY ESTIMATION WORKFLOW
# ==============================================================================
def generate_conformal_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Step 1: Calibration Set (x = 0.15, w = 1.95)
    draw_academic_card(ax, (0.15, 2.05), 1.95, 1.50,
                       "1. Calibration Dataset",
                       ["Held-out Spatial Samples:",
                        "Ground Truth: yi",
                        "Point Forecast / Estimate: ŷi",
                        "Nearest Monitor Distance: di (km)",
                        "Strict Out-of-Sample Split (n samples)"],
                       bg=C_SLATE_BG, border=C_NAVY)

    # Step 2: Non-Conformity Scoring (x = 2.45, w = 2.20)
    draw_academic_card(ax, (2.45, 2.05), 2.20, 1.50,
                       "2. Distance-Scaled Scoring",
                       ["Non-Conformity Metric:",
                        "si = |yi - ŷi| / (1 + γ · di)",
                        "Scale Factor: γ = 0.05 km⁻¹",
                        "Normalizes error inflation occurring",
                        "in sparse geographic blindspots"],
                       bg=C_TEAL_BG, border=C_TEAL_BD)

    draw_clean_arrow(ax, (2.10, 2.80), (2.45, 2.80), label="Residuals |yi - ŷi|", label_pos=0.5, label_offset=(0, 0.07))

    # Step 3: Empirical Quantile (x = 4.95, w = 2.10)
    draw_academic_card(ax, (4.95, 2.05), 2.10, 1.50,
                       "3. Conformal Quantile",
                       ["Distribution-Free Calibration:",
                        "q̂_(1-α) = Quantile_p(S)",
                        "Index p = ⌈(n + 1)(1 - α)⌉ / n",
                        "Marginal Finite-Sample Guarantee:",
                        "P(Y ∈ C(X)) ≥ 1 - α"],
                       bg=C_BLUE_BG, border=C_BLUE_BD)

    draw_clean_arrow(ax, (4.65, 2.80), (4.95, 2.80), label="Score Set S", label_pos=0.5, label_offset=(0, 0.07))

    # Step 4: Operational Conformal Intervals (x = 0.15 to 7.05, w = 6.90)
    draw_academic_card(ax, (0.15, 0.25), 6.90, 1.35,
                       "4. Operational Conformal Prediction Interval Formulation",
                       ["Prediction Band: C(x) = [ max(0, ŷ - q̂_(1-α) · (1 + γ · d_min)),  ŷ + q̂_(1-α) · (1 + γ · d_min) ]",
                        "• At Reporting Monitor (d_min = 0): Interval width is purely driven by residual quantile 2 · q̂_(1-α)",
                        "• Far From Monitors (d_min > 0): Interval expands proportionally by factor (1 + 0.05 · d_min)",
                        "• Validated Empirical Coverage: 92.8% (80% Tier), 96.5% (90% Tier), 99.37% (95% Tier) on PM2.5",
                        "• Honest Disclosure: Guarantees coverage at the cost of wider bounds during severe episodic smog events"],
                       bg=C_GRN_BG, border=C_GRN_BD, sub_size=6.7)

    draw_clean_arrow(ax, (6.00, 2.05), (6.00, 1.60), label="Calibrated q̂", label_pos=0.5, label_offset=(0, 0.06))

    out_p = OUT_DIR / "fig_arch_conformal.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated clean {out_p.name}")

if __name__ == "__main__":
    generate_idw_workflow()
    generate_reliability_workflow()
    generate_conformal_workflow()
