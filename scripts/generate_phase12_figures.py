"""
scripts/generate_phase12_figures.py

Generates publication-quality figures for VayuDrishti Phase 12:
- fig1_conformal_calibration.png: Conformal Prediction Empirical vs Nominal Coverage and Widths.
- fig2_seasonal_model_comparison.png: Seasonal MAE across IMD seasons.
- fig3_multistation_performance.png: Multi-station performance across 15 CAAQMS stations.
- fig4_regime_error_curve.png: Error progression across NAQI Regimes.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PHASE12_DIR = ROOT_DIR / "experiments" / "phase12"
FIGS_DIR = PHASE12_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def generate_fig1_conformal_calibration():
    unc_csv = PHASE12_DIR / "uncertainty_metrics.csv"
    if not unc_csv.exists():
        return
    df = pd.read_csv(unc_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Coverage comparison
    nom_levels = [0.80, 0.90, 0.95]
    pm25_emp = df[df["parameter"] == "PM2.5"]["empirical_coverage"].values
    pm10_emp = df[df["parameter"] == "PM10"]["empirical_coverage"].values

    ax1.plot([0.75, 1.0], [0.75, 1.0], "k--", label="Ideal Calibration ($y=x$)", alpha=0.7)
    ax1.plot(nom_levels, pm25_emp, "o-", color="#1f77b4", linewidth=2, markersize=8, label="PM2.5 Empirical")
    ax1.plot(nom_levels, pm10_emp, "s-", color="#ff7f0e", linewidth=2, markersize=8, label="PM10 Empirical")
    ax1.set_xlim(0.75, 1.0)
    ax1.set_ylim(0.75, 1.0)
    ax1.set_xlabel("Nominal Coverage Level $(1 - \\alpha)$")
    ax1.set_ylabel("Empirical Coverage Frequency")
    ax1.set_title("(a) Split-Conformal Empirical Calibration")
    ax1.legend(loc="upper left")

    # Width comparison
    pm25_w = df[df["parameter"] == "PM2.5"]["mean_interval_width"].values
    pm10_w = df[df["parameter"] == "PM10"]["mean_interval_width"].values
    x = np.arange(len(nom_levels))
    width = 0.35

    ax2.bar(x - width/2, pm25_w, width, label="PM2.5 Width $(\\mu g/m^3)$", color="#1f77b4", alpha=0.85)
    ax2.bar(x + width/2, pm10_w, width, label="PM10 Width $(\\mu g/m^3)$", color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["80%", "90%", "95%"])
    ax2.set_xlabel("Nominal Interval Coverage")
    ax2.set_ylabel("Mean Prediction Interval Width $(\\mu g/m^3)$")
    ax2.set_title("(b) Average Interval Width by Confidence Tier")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = FIGS_DIR / "fig1_conformal_calibration.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path.name}")


def generate_fig2_seasonal_comparison():
    seas_csv = PHASE12_DIR / "seasonal_metrics.csv"
    if not seas_csv.exists():
        return
    df = pd.read_csv(seas_csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.35

    seasons_clean = [s.split("(")[0].strip() for s in df["imd_season"]]

    ax.bar(x - width/2, df["pm25_persistence_mae"], width, label="1-Hour Persistence Baseline", color="#8da4be", alpha=0.8)
    ax.bar(x + width/2, df["pm25_model_mae"], width, label="Adapted Model (Severity-Gated / GBDT)", color="#22c55e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons_clean)
    ax.set_ylabel("PM2.5 MAE $(\\mu g/m^3)$")
    ax.set_title("Forecast Accuracy across Indian Meteorological Department (IMD) Seasons")
    ax.legend()

    # Annotate improvement percentages
    for i in range(len(df)):
        imp = df["pm25_pct_improvement"].iloc[i]
        color = "#15803d" if imp > 0 else "#b91c1c"
        ax.annotate(
            f"{imp:+.1f}%",
            xy=(x[i] + width/2, df["pm25_model_mae"].iloc[i] + 0.15),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    plt.tight_layout()
    out_path = FIGS_DIR / "fig2_seasonal_model_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path.name}")


def generate_fig3_multistation_performance():
    st_csv = PHASE12_DIR / "station_metrics.csv"
    if not st_csv.exists():
        return
    df = pd.read_csv(st_csv).sort_values("pm25_pct_improvement", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#22c55e" if imp > 0 else "#ef4444" for imp in df["pm25_pct_improvement"]]
    bars = ax.barh(df["station_id"] + " (" + df["city"] + ")", df["pm25_pct_improvement"], color=colors, alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Relative PM2.5 MAE Improvement over Persistence (%)")
    ax.set_title("Multi-Station Operational Evaluation across 15 CAAQMS Stations")
    ax.invert_yaxis()

    for bar, imp in zip(bars, df["pm25_pct_improvement"]):
        x_pos = bar.get_width() + (0.3 if imp >= 0 else -0.8)
        ha = "left" if imp >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{imp:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = FIGS_DIR / "fig3_multistation_performance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path.name}")


def generate_fig4_regime_error_curve():
    reg_csv = PHASE12_DIR / "regime_metrics.csv"
    if not reg_csv.exists():
        return
    df = pd.read_csv(reg_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Absolute MAE across regimes
    x = np.arange(len(df))
    ax1.plot(x, df["pm25_persistence_mae"], "o--", color="#8da4be", label="Persistence", linewidth=2)
    ax1.plot(x, df["pm25_gbdt_mae"], "s-", color="#16a34a", label="Adapted GBDT", linewidth=2.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["naqi_regime"], rotation=25, ha="right")
    ax1.set_ylabel("PM2.5 MAE $(\\mu g/m^3)$")
    ax1.set_title("(a) Absolute Error across NAQI Regimes")
    ax1.legend()

    # Percentage improvement
    colors = ["#22c55e" if imp > 0 else "#ef4444" for imp in df["pm25_pct_improvement"]]
    ax2.bar(df["naqi_regime"], df["pm25_pct_improvement"], color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticklabels(df["naqi_regime"], rotation=25, ha="right")
    ax2.set_ylabel("% Error Reduction vs Persistence")
    ax2.set_title("(b) Relative Gain Across Pollution Severity")

    for i, imp in enumerate(df["pm25_pct_improvement"]):
        y_pos = imp + (0.6 if imp >= 0 else -1.8)
        va = "bottom" if imp >= 0 else "top"
        ax2.text(i, y_pos, f"{imp:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = FIGS_DIR / "fig4_regime_error_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path.name}")


if __name__ == "__main__":
    generate_fig1_conformal_calibration()
    generate_fig2_seasonal_comparison()
    generate_fig3_multistation_performance()
    generate_fig4_regime_error_curve()
    print("All figures generated in experiments/phase12/figures/ successfully.")
