"""
scripts/generate_missing_report_diagrams.py

Generates publication-quality architecture diagrams for:
- Figure 2.7: IDW Spatial Interpolation Workflow
- Figure 2.8: Spatial Reliability and Freshness Assessment Workflow
- Figure 2.9: Conformal Uncertainty Estimation Workflow
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_BLUE = "#1E40AF"      # Primary Blue
C_LIGHT_BLUE = "#DBEAFE"
C_TEAL = "#0E7490"      # Teal
C_LIGHT_TEAL = "#CFFAFE"
C_NAVY = "#0F172A"      # Slate Navy
C_LIGHT_SLATE = "#F1F5F9"
C_AMBER = "#D97706"     # Amber
C_LIGHT_AMBER = "#FEF3C7"
C_GREEN = "#15803D"     # Emerald Green
C_LIGHT_GREEN = "#DCFCE7"
C_RED = "#B91C1C"       # Crimson Red
C_LIGHT_RED = "#FEE2E2"

def draw_box(ax, xy, w, h, title, subtitle="", bg=C_LIGHT_BLUE, border=C_BLUE, radius=0.03):
    box = patches.FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad={radius},rounding_size=0.02",
        linewidth=1.5,
        edgecolor=border,
        facecolor=bg
    )
    ax.add_patch(box)
    cx = xy[0] + w / 2.0
    cy = xy[1] + h / 2.0
    if subtitle:
        ax.text(cx, cy + 0.08, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=border)
        ax.text(cx, cy - 0.08, subtitle, ha="center", va="center", fontsize=7.5, color="#334155")
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color=border)

def draw_arrow(ax, p1, p2, label="", color="#475569"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color, mutation_scale=12)
    )
    if label:
        mx = (p1[0] + p2[0]) / 2.0
        my = (p1[1] + p2[1]) / 2.0 + 0.04
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.5, color=color, fontweight="semibold")

def generate_idw_workflow():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure 2.7: Geodesic Inverse Distance Weighting (IDW) Spatial Interpolation Workflow",
                 fontsize=11, fontweight="bold", color=C_NAVY, pad=12)

    # Stages
    draw_box(ax, (0.4, 3.4), 2.2, 0.9, "Target Coordinates", "(lat, lon) Query Point\nUrban Bounding Box", bg=C_LIGHT_SLATE, border=C_NAVY)
    draw_box(ax, (0.4, 1.8), 2.2, 0.9, "Active Sensor Registry", "Catalogued CAAQMS Stations\nPhysical Ground Truth", bg=C_LIGHT_BLUE, border=C_BLUE)

    draw_arrow(ax, (2.6, 3.85), (3.4, 3.2), "Haversine Distance")
    draw_arrow(ax, (2.6, 2.25), (3.4, 2.8), "Coordinates (φi, λi)")

    draw_box(ax, (3.4, 2.4), 2.4, 1.2, "Geodesic Distance Kernel", "di = 2R · arcsin(√hav(Δφ, Δλ))\nR_earth = 6,371.0 km", bg=C_LIGHT_TEAL, border=C_TEAL)

    draw_arrow(ax, (5.8, 3.0), (6.6, 3.5), "d_min <= 0.05 km")
    draw_arrow(ax, (5.8, 3.0), (6.6, 2.5), "N < 3 or d > R_max")
    draw_arrow(ax, (5.8, 3.0), (6.6, 1.5), "N >= 3 & d <= 45km")

    draw_box(ax, (6.6, 3.2), 3.0, 0.7, "Co-Located Observation", "Exact sensor reading (Ground truth)", bg=C_LIGHT_GREEN, border=C_GREEN)
    draw_box(ax, (6.6, 2.2), 3.0, 0.7, "Nearest-Station Fallback", "Sparse network safeguard (1-NN)", bg=C_LIGHT_AMBER, border=C_AMBER)
    draw_box(ax, (6.6, 0.9), 3.0, 1.0, "Vectorized IDW Surface", "wi = (di + ε)^(-2.0)\nẑ = Σ(wi·zi) / Σwi (Clamped >= 0)", bg=C_LIGHT_BLUE, border=C_BLUE)

    fig.savefig(OUT_DIR / "fig_arch_idw_spatial.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_arch_idw_spatial.png")

def generate_reliability_workflow():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure 2.8: Spatial Reliability, Telemetry Freshness & Geometry Audit Workflow",
                 fontsize=11, fontweight="bold", color=C_NAVY, pad=12)

    draw_box(ax, (0.4, 2.6), 2.2, 1.1, "Incoming Station Telemetry", "OpenAQ v3 / CPCB Stream\nObserved Timestamp t_obs", bg=C_LIGHT_SLATE, border=C_NAVY)

    draw_arrow(ax, (2.6, 3.15), (3.4, 3.15), "Latency Δt = t_now - t_obs")

    draw_box(ax, (3.4, 2.0), 2.5, 2.3, "5-Tier Freshness Filter",
             "Fresh (<= 2h): Active\nDelayed (2-6h): Warning\nStale (6-24h): Excluded\nOffline (> 24h): Suppressed\nUnknown: Missing stamp",
             bg=C_LIGHT_AMBER, border=C_AMBER)

    draw_arrow(ax, (5.9, 3.15), (6.7, 3.15), "Active Monitors (Fresh/Delayed)")

    draw_box(ax, (6.7, 2.6), 2.9, 1.8, "Spatial Geometry Audit",
             "Clark-Evans Index:\nR = r̄_obs / r̄_exp\nDensity ρ = N / Area\nClustered (R < 0.85)\nRandom (0.85 <= R <= 1.15)\nDispersed (R > 1.15)",
             bg=C_LIGHT_TEAL, border=C_TEAL)

    draw_box(ax, (3.4, 0.4), 6.2, 0.9, "Operational Reliability Tier Assignment",
             "Tier 1: Calibrated (Delhi) | Tier 2: High | Tier 3: Moderate | Tier 4: Marginal | Tier 5: Suppressed",
             bg=C_LIGHT_GREEN, border=C_GREEN)

    draw_arrow(ax, (8.15, 2.6), (8.15, 1.3), "Reliability Mapping")

    fig.savefig(OUT_DIR / "fig_arch_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_arch_reliability.png")

def generate_conformal_workflow():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure 2.9: Distance-Scaled Conformal Uncertainty Estimation Workflow",
                 fontsize=11, fontweight="bold", color=C_NAVY, pad=12)

    draw_box(ax, (0.4, 2.5), 2.3, 1.2, "Calibration Dataset", "Held-out spatial samples\nGround Truth yi & Point Est ŷi\nNearest Station Distance di", bg=C_LIGHT_SLATE, border=C_NAVY)

    draw_arrow(ax, (2.7, 3.1), (3.5, 3.1), "Residuals")

    draw_box(ax, (3.5, 2.3), 2.6, 1.6, "Non-Conformity Scoring", "Normalized Score:\nsi = |yi - ŷi| / (1 + γ · di)\nDistance Scaling γ = 0.05 / km", bg=C_LIGHT_TEAL, border=C_TEAL)

    draw_arrow(ax, (6.1, 3.1), (6.9, 3.1), "Quantile q̂_(1-α)")

    draw_box(ax, (6.9, 2.2), 2.7, 1.8, "Empirical Coverage Bound", "Split Conformal Quantile:\nq̂ = Quantile_((n+1)(1-α)/n)(S)\nGuarantees marginal coverage:\nP(Y ∈ C(X)) >= 1 - α", bg=C_LIGHT_BLUE, border=C_BLUE)

    draw_box(ax, (2.0, 0.5), 6.5, 1.0, "Operational Conformal Interval Output",
             "C(x) = [ max(0, ŷ - q̂(1 + γ·d_min)), ŷ + q̂(1 + γ·d_min) ]\nDistance-dependent interval expansion communicating spatial uncertainty",
             bg=C_LIGHT_GREEN, border=C_GREEN)

    draw_arrow(ax, (8.25, 2.2), (8.25, 1.5))

    fig.savefig(OUT_DIR / "fig_arch_conformal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_arch_conformal.png")

if __name__ == "__main__":
    generate_idw_workflow()
    generate_reliability_workflow()
    generate_conformal_workflow()
