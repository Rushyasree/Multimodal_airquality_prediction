"""
scripts/generate_phase13_figures.py

Generates publication-quality figures for Phase 13 Spatial-Temporal Analysis:
1. fig_spatial_cross_correlation.png: CCF curves across lags [-12, +12] hours for Delhi, Mumbai, Kolkata.
2. fig_spatial_forecast_comparison.png: Persistence vs Single-Station GBDT vs Multi-Node Spatial GBDT.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PHASE13_DIR = ROOT_DIR / "experiments" / "phase13"
FIGS_DIR = PHASE13_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

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


def plot_cross_correlation_curves():
    ccf_csv = PHASE13_DIR / "cross_correlation_analysis.csv"
    if not ccf_csv.exists():
        return
    df = pd.read_csv(ccf_csv)

    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        "Delhi NCR Urban Corridor": {"color": "#ef4444", "marker": "o", "label": "Delhi: Anand Vihar <-> DTU (23.0 km)"},
        "Mumbai Coastal Marine Corridor": {"color": "#3b82f6", "marker": "s", "label": "Mumbai: Bandra <-> Colaba (17.2 km)"},
        "Kolkata Gangetic Urban Corridor": {"color": "#10b981", "marker": "^", "label": "Kolkata: Bidhannagar <-> Jadavpur (10.1 km)"},
    }

    for c_name, sub in df[df["parameter"] == "PM2.5"].groupby("cluster_name"):
        sub_sorted = sub.sort_values("lag_hours")
        st = styles.get(c_name, {"color": "black", "marker": "o", "label": c_name})
        ax.plot(
            sub_sorted["lag_hours"],
            sub_sorted["cross_correlation"],
            color=st["color"],
            marker=st["marker"],
            linewidth=2,
            markersize=6,
            label=st["label"],
        )

    ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Synchronous Lag 0")
    ax.set_xlabel("Cross-Correlation Lag $\\tau$ (Hours) [Positive: Station A leads Station B]")
    ax.set_ylabel("Pearson Correlation Coefficient $r(\\tau)$")
    ax.set_title("Intra-Urban Cross-Station PM2.5 Correlation vs. Temporal Lag $\\tau$")
    ax.set_ylim(0.2, 1.0)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = FIGS_DIR / "fig_spatial_cross_correlation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path.name}")


def plot_forecast_comparison():
    fc_csv = PHASE13_DIR / "spatial_forecast_comparison.csv"
    if not fc_csv.exists():
        return
    df = pd.read_csv(fc_csv)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["persistence_mae"], width, label="1-Hour Persistence Baseline", color="#94a3b8")
    ax.bar(x, df["single_station_gbdt_mae"], width, label="Single-Station Autoregressive GBDT", color="#3b82f6")
    ax.bar(x + width, df["multinode_spatial_gbdt_mae"], width, label="Multi-Node Spatial GBDT (with Neighbor Lags)", color="#10b981")

    ax.set_xticks(x)
    labels = [f"{r['city']}\n({r['target_station']})" for _, r in df.iterrows()]
    ax.set_xticklabels(labels)
    ax.set_ylabel("Next-Hour PM2.5 MAE $(\\mu g/m^3)$")
    ax.set_title("Forecast Accuracy: Single-Station vs. Multi-Node Spatial Telemetry")
    ax.legend()

    for i in range(len(df)):
        pers = df["persistence_mae"].iloc[i]
        sg = df["single_station_gbdt_mae"].iloc[i]
        mg = df["multinode_spatial_gbdt_mae"].iloc[i]
        gain_vs_pers = ((pers - sg) / pers) * 100
        ax.annotate(f"{gain_vs_pers:+.1f}% vs pers", xy=(x[i], sg + 0.2), ha="center", fontsize=8.5, fontweight="bold", color="#1d4ed8")

    plt.tight_layout()
    out_path = FIGS_DIR / "fig_spatial_forecast_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path.name}")


if __name__ == "__main__":
    plot_cross_correlation_curves()
    plot_forecast_comparison()
    print("Phase 13 figures generated successfully.")
