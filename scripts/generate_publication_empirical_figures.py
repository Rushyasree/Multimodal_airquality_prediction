"""
scripts/generate_publication_empirical_figures.py

Regenerates empirical and evaluation figures for the VayuDrishti paper
tailored strictly for IEEE Conference Two-Column specifications (3.5" single column).
Eliminates all horizontal squashing, unreadable labels, and text clumping.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHASE12_DIR = Path("experiments/phase12")
PHASE13_DIR = Path("experiments/phase13")
C_DARK = "#0f172a"

# Common IEEE publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


# ------------------------------------------------------------------------------
# 1. FIG 10 & 11: CONFUSION MATRICES (3.5 x 3.5 in)
# ------------------------------------------------------------------------------
def generate_confusion_matrices():
    print("Generating clean single-column Confusion Matrices...")
    categories = ['Good', 'Sat', 'Mod', 'Poor', 'V.Poor', 'Severe']
    
    for pollutant, filename, cmap in [
        ('PM2.5', 'confusion_matrix_pm25', 'Blues'),
        ('PM10', 'confusion_matrix_pm10', 'YlGn')
    ]:
        cm_path = Path(f"reports/confusion_matrix_{pollutant.lower().replace('.', '')}.csv")
        if not cm_path.exists():
            continue
        cm_df = pd.read_csv(cm_path, index_col=0)
        matrix = cm_df.values.astype(int)
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = matrix / np.maximum(row_sums, 1).astype(float)
        
        fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)
        im = ax.imshow(norm_matrix, cmap=cmap, vmin=0, vmax=1.0)
        
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label('Recall Fraction', fontsize=6.5)
        cb.ax.tick_params(labelsize=6.0)
        
        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(range_len := len(categories)))
        ax.set_xticklabels(categories, fontsize=6.2)
        ax.set_yticklabels(categories, fontsize=6.2)
        
        ax.set_xlabel(f'Predicted {pollutant} Category', fontsize=7.0, fontweight='bold', labelpad=2)
        ax.set_ylabel(f'Ground Truth Category', fontsize=7.0, fontweight='bold', labelpad=2)
        
        for i in range(len(categories)):
            for j in range(len(categories)):
                count = matrix[i, j]
                pct = norm_matrix[i, j] * 100
                txt_col = "white" if norm_matrix[i, j] > 0.55 else "#0f172a"
                if count >= 1000:
                    cnt_str = f"{count/1000:.1f}k"
                else:
                    cnt_str = f"{count}"
                ax.text(j, i, f"{cnt_str}\n{pct:.0f}%", ha="center", va="center",
                        color=txt_col, fontsize=5.5, fontweight='bold' if i == j else 'normal')
        
        ax.set_title(f"NAQI {pollutant} Classification Matrix (N=404,431)",
                     fontsize=7.5, fontweight='bold', pad=4)
        
        fig.tight_layout()
        for ext in [".png", ".pdf"]:
            fig.savefig(OUT_DIR / f"{filename}{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")

    # Also generate unified 2-panel stacked confusion matrix (3.5 x 3.3 in)
    cm_path_25 = Path("reports/confusion_matrix_pm25.csv")
    cm_path_10 = Path("reports/confusion_matrix_pm10.csv")
    if cm_path_25.exists() and cm_path_10.exists():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.3), dpi=300)
        
        for ax, p_name, path_c, cmap, title_p in [
            (ax1, 'PM2.5', cm_path_25, 'Blues', '(a) NAQI PM2.5 Confusion Matrix (N=404,431)'),
            (ax2, 'PM10', cm_path_10, 'YlGn', '(b) NAQI PM10 Confusion Matrix (N=404,431)')
        ]:
            df_c = pd.read_csv(path_c, index_col=0)
            mat = df_c.values.astype(int)
            r_sums = mat.sum(axis=1, keepdims=True)
            n_mat = mat / np.maximum(r_sums, 1).astype(float)
            
            im = ax.imshow(n_mat, cmap=cmap, vmin=0, vmax=1.0)
            cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
            cb.set_label('Recall', fontsize=5.8)
            cb.ax.tick_params(labelsize=5.2)
            
            ax.set_xticks(range(len(categories)))
            ax.set_yticks(range(len(categories)))
            ax.set_xticklabels(categories, fontsize=5.6)
            ax.set_yticklabels(categories, fontsize=5.6)
            ax.set_ylabel('True', fontsize=6.2, fontweight='bold', labelpad=1)
            ax.set_xlabel(f'Pred {p_name}', fontsize=6.2, fontweight='bold', labelpad=1)
            ax.set_title(title_p, fontsize=6.8, fontweight='bold', pad=2)
            
            for i in range(len(categories)):
                for j in range(len(categories)):
                    count = mat[i, j]
                    pct = n_mat[i, j] * 100
                    txt_col = "white" if n_mat[i, j] > 0.55 else "#0f172a"
                    cnt_str = f"{count/1000:.1f}k" if count >= 1000 else f"{count}"
                    ax.text(j, i, f"{cnt_str}\n{pct:.0f}%", ha="center", va="center",
                            color=txt_col, fontsize=4.8, fontweight='bold' if i == j else 'normal')
                            
        fig.tight_layout()
        fig.savefig(OUT_DIR / "confusion_matrices_combined.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT_DIR / "confusion_matrices_combined.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved confusion_matrices_combined")


# ------------------------------------------------------------------------------
# 2. FIG 15: CONFORMAL CALIBRATION (SINGLE COLUMN: 3.5 x 3.0 in)
# ------------------------------------------------------------------------------
def generate_conformal_calibration():
    print("Generating single-column Conformal Calibration...")
    unc_csv = PHASE12_DIR / "uncertainty_metrics.csv"
    if not unc_csv.exists():
        return
    df = pd.read_csv(unc_csv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.0), dpi=300)

    # Top: Coverage
    nom_levels = [0.80, 0.90, 0.95]
    pm25_emp = df[df["parameter"] == "PM2.5"]["empirical_coverage"].values
    pm10_emp = df[df["parameter"] == "PM10"]["empirical_coverage"].values

    ax1.plot([0.75, 1.0], [0.75, 1.0], "k--", label="Ideal ($y=x$)", alpha=0.6, lw=1.0)
    ax1.plot(nom_levels, pm25_emp, "o-", color="#0284c7", lw=1.3, ms=4, label="PM2.5 (>=92.8%)")
    ax1.plot(nom_levels, pm10_emp, "s-", color="#d97706", lw=1.3, ms=4, label="PM10 (>=93.6%)")
    ax1.set_xlim(0.75, 1.0)
    ax1.set_ylim(0.75, 1.02)
    ax1.set_xlabel("Nominal Coverage $(1 - \\alpha)$", fontsize=6.8, labelpad=2)
    ax1.set_ylabel("Empirical", fontsize=6.8, labelpad=2)
    ax1.set_title("(a) Split-Conformal Empirical Coverage", fontsize=7.2, fontweight="bold", pad=3)
    ax1.legend(loc="lower right", fontsize=5.8)
    ax1.tick_params(labelsize=6.0, pad=1)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Bottom: Interval Widths
    pm25_w = df[df["parameter"] == "PM2.5"]["mean_interval_width"].values
    pm10_w = df[df["parameter"] == "PM10"]["mean_interval_width"].values
    x = np.arange(len(nom_levels))
    w = 0.35

    b1 = ax2.bar(x - w/2, pm25_w, w, label="PM2.5 Width", color="#0284c7", alpha=0.85)
    b2 = ax2.bar(x + w/2, pm10_w, w, label="PM10 Width", color="#d97706", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["80%", "90%", "95%"], fontsize=6.2)
    ax2.set_xlabel("Nominal Confidence Tier", fontsize=6.8, labelpad=2)
    ax2.set_ylabel("Width (ug/m3)", fontsize=6.8, labelpad=2)
    ax2.set_title("(b) Average Interval Width by Tier", fontsize=7.2, fontweight="bold", pad=3)
    ax2.legend(loc="upper left", fontsize=5.8)
    ax2.tick_params(labelsize=6.0, pad=1)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Annotate bar values
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=5.4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_conformal_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(PHASE12_DIR / "figures/fig1_conformal_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_conformal_calibration")


# ------------------------------------------------------------------------------
# 3. FIG 13: SEASONAL DYNAMICS (SINGLE COLUMN: 3.5 x 2.2 in)
# ------------------------------------------------------------------------------
def generate_seasonal_comparison():
    print("Generating single-column Seasonal Comparison...")
    seas_csv = PHASE12_DIR / "seasonal_metrics.csv"
    if not seas_csv.exists():
        return
    df = pd.read_csv(seas_csv)

    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)
    x = np.arange(len(df))
    w = 0.36
    seasons_clean = ["Winter", "Pre-Mon", "Monsoon", "Post-Mon"]

    ax.bar(x - w/2, df["pm25_persistence_mae"], w, label="Persistence", color="#94a3b8", alpha=0.85)
    bars_m = ax.bar(x + w/2, df["pm25_model_mae"], w, label="Adapted Model", color="#16a34a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(seasons_clean, fontsize=6.5)
    ax.set_ylabel("PM2.5 MAE (ug/m3)", fontsize=6.8, labelpad=2)
    ax.set_title("Forecast Accuracy Across IMD Seasons", fontsize=7.5, fontweight="bold", pad=3)
    ax.legend(fontsize=6.0, loc="upper right")
    ax.tick_params(labelsize=6.2)
    ax.grid(True, alpha=0.3, linestyle="--")

    for i in range(len(df)):
        imp = df["pm25_pct_improvement"].iloc[i]
        col = "#15803d" if imp > 0 else "#b91c1c"
        ax.annotate(f"{imp:+.1f}%", xy=(x[i] + w/2, df["pm25_model_mae"].iloc[i] + 0.15),
                    ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=col)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_seasonal_model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(PHASE12_DIR / "figures/fig2_seasonal_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_seasonal_model_comparison")


# ------------------------------------------------------------------------------
# 4. FIG 11: MULTI-STATION PERFORMANCE (SINGLE COLUMN: 3.5 x 2.45 in)
# ------------------------------------------------------------------------------
def generate_multistation_performance():
    print("Generating single-column Multi-Station Performance...")
    st_csv = PHASE12_DIR / "station_metrics.csv"
    if not st_csv.exists():
        return
    df = pd.read_csv(st_csv).sort_values("pm25_pct_improvement", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.45), dpi=300)
    colors = ["#16a34a" if imp > 0 else "#e11d48" for imp in df["pm25_pct_improvement"]]
    labels = [f"{sid} ({city[:6]})" for sid, city in zip(df["station_id"], df["city"])]
    bars = ax.barh(labels, df["pm25_pct_improvement"], color=colors, alpha=0.85, height=0.62)

    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("PM2.5 MAE Gain vs. Persistence (%)", fontsize=6.5, labelpad=1)
    ax.set_title("15 CAAQMS Multi-Station Evaluation", fontsize=7.2, fontweight="bold", pad=2)
    ax.invert_yaxis()
    ax.tick_params(labelsize=5.3, pad=1)
    ax.grid(True, alpha=0.3, linestyle="--")

    for bar, imp in zip(bars, df["pm25_pct_improvement"]):
        x_pos = bar.get_width() + (0.35 if imp >= 0 else -1.1)
        ha = "left" if imp >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{imp:+.1f}%",
                va="center", ha=ha, fontsize=5.1, fontweight="bold", color=C_DARK)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_multistation_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(PHASE12_DIR / "figures/fig3_multistation_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_multistation_performance")


# ------------------------------------------------------------------------------
# 5. FIG 13: SEVERITY REGIME CURVE (SINGLE COLUMN: 3.5 x 2.45 in)
# ------------------------------------------------------------------------------
def generate_regime_error_curve():
    print("Generating single-column Regime Error Curve...")
    reg_csv = PHASE12_DIR / "regime_metrics.csv"
    if not reg_csv.exists():
        return
    df = pd.read_csv(reg_csv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.45), dpi=300)
    x = np.arange(len(df))

    # Top: Absolute MAE
    ax1.plot(x, df["pm25_persistence_mae"], "o--", color="#94a3b8", label="Persistence", lw=1.2, ms=3.0)
    ax1.plot(x, df["pm25_gbdt_mae"], "s-", color="#16a34a", label="Adapted GBDT", lw=1.4, ms=3.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["naqi_regime"], fontsize=5.5, rotation=10)
    ax1.set_ylabel("MAE (ug/m3)", fontsize=6.5, labelpad=1)
    ax1.set_title("(a) Absolute Error Across NAQI Regimes", fontsize=7.0, fontweight="bold", pad=2)
    ax1.legend(fontsize=5.5)
    ax1.tick_params(labelsize=5.5, pad=1)
    ax1.grid(True, alpha=0.3, linestyle="--")

    colors = ["#16a34a" if imp > 0 else "#e11d48" for imp in df["pm25_pct_improvement"]]
    b = ax2.bar(x, df["pm25_pct_improvement"], color=colors, alpha=0.85, width=0.55)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["naqi_regime"], fontsize=5.5, rotation=10)
    ax2.set_ylabel("% vs Persist", fontsize=6.5, labelpad=1)
    ax2.set_title("(b) Relative Gain Across Severity", fontsize=7.0, fontweight="bold", pad=2)
    ax2.tick_params(labelsize=5.5, pad=1)
    ax2.grid(True, alpha=0.3, linestyle="--")

    for i, imp in enumerate(df["pm25_pct_improvement"]):
        y_pos = imp + (0.5 if imp >= 0 else -1.5)
        va = "bottom" if imp >= 0 else "top"
        ax2.text(i, y_pos, f"{imp:+.1f}%", ha="center", va=va, fontsize=5.1, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_regime_error_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(PHASE12_DIR / "figures/fig4_regime_error_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_regime_error_curve")


# ------------------------------------------------------------------------------
# 6. FIG 16 & 17: SPATIAL ADVECTION PLOTS (SINGLE COLUMN: 3.5 x 1.75 in)
# ------------------------------------------------------------------------------
def generate_spatial_advection():
    print("Generating single-column Spatial Advection figures...")
    ccf_csv = PHASE13_DIR / "cross_correlation_analysis.csv"
    if ccf_csv.exists():
        df = pd.read_csv(ccf_csv)
        fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=300)
        
        styles = {
            "Delhi NCR Urban Corridor": {"color": "#ef4444", "m": "o", "label": "Delhi (23km)"},
            "Mumbai Coastal Marine Corridor": {"color": "#0284c7", "m": "s", "label": "Mumbai (17km)"},
            "Kolkata Gangetic Urban Corridor": {"color": "#10b981", "m": "^", "label": "Kolkata (10km)"},
        }
        for c_name, sub in df[df["parameter"] == "PM2.5"].groupby("cluster_name"):
            sub_sorted = sub.sort_values("lag_hours")
            st = styles.get(c_name, {"color": "black", "m": "o", "label": c_name[:10]})
            ax.plot(sub_sorted["lag_hours"], sub_sorted["cross_correlation"],
                    color=st["color"], marker=st["m"], lw=1.2, ms=2.8, label=st["label"])

        ax.axvline(0, color="gray", linestyle=":", lw=0.9)
        ax.set_xlabel("Lag tau (Hours) [Pos: A leads B]", fontsize=6.5, labelpad=1)
        ax.set_ylabel("Pearson r(tau)", fontsize=6.5, labelpad=1)
        ax.set_title("Cross-Station PM2.5 Correlation vs Lag tau", fontsize=7.2, fontweight="bold", pad=2)
        ax.set_ylim(0.2, 1.0)
        ax.legend(loc="upper right", fontsize=5.5)
        ax.tick_params(labelsize=5.8, pad=1)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_spatial_cross_correlation.png", dpi=300, bbox_inches="tight")
        fig.savefig(PHASE13_DIR / "figures/fig_spatial_cross_correlation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved fig_spatial_cross_correlation")

    fc_csv = PHASE13_DIR / "spatial_forecast_comparison.csv"
    if fc_csv.exists():
        df = pd.read_csv(fc_csv)
        fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=300)
        
        clusters = ["Delhi", "Mumbai", "Kolkata"]
        x = np.arange(len(clusters))
        w = 0.26
        
        p_mae = [df[df["city"] == c]["persistence_mae"].values[0] for c in clusters]
        s_mae = [df[df["city"] == c]["single_station_gbdt_mae"].values[0] for c in clusters]
        m_mae = [df[df["city"] == c]["multinode_spatial_gbdt_mae"].values[0] for c in clusters]

        ax.bar(x - w, p_mae, w, label="Persistence", color="#94a3b8", alpha=0.85)
        ax.bar(x, s_mae, w, label="Single-Station", color="#0284c7", alpha=0.85)
        ax.bar(x + w, m_mae, w, label="Spatial GBDT", color="#16a34a", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(clusters, fontsize=6.2)
        ax.set_ylabel("PM2.5 MAE (ug/m3)", fontsize=6.5, labelpad=1)
        ax.set_title("Single vs. Spatial Multi-Node Forecast", fontsize=7.2, fontweight="bold", pad=2)
        ax.legend(fontsize=5.5, loc="upper right")
        ax.tick_params(labelsize=5.8, pad=1)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_spatial_forecast_comparison.png", dpi=300, bbox_inches="tight")
        fig.savefig(PHASE13_DIR / "figures/fig_spatial_forecast_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved fig_spatial_forecast_comparison")


# ------------------------------------------------------------------------------
# 7. FIG 18: PM25VISION ALIGNMENT DIAGNOSTICS (SINGLE COLUMN: 3.5 x 2.45 in)
# ------------------------------------------------------------------------------
def generate_alignment_diagnostics():
    print("Generating single-column Alignment Diagnostics...")
    manifest_path = Path("data/processed/pm25vision_cnn/pm25vision_clean_manifest.csv")
    aq_path = Path("data/processed/air_quality_weather_merged.csv")
    if not manifest_path.exists() or not aq_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    aq = pd.read_csv(aq_path, low_memory=False)
    aq["Datetime"] = pd.to_datetime(aq["Datetime"], errors="coerce")
    aq["date"] = aq["Datetime"].dt.strftime("%Y-%m-%d")
    
    daily = aq.groupby(["StationId", "date"]).agg({"PM2.5": "mean"}).reset_index()
    within_5km = manifest[manifest["within_threshold"]].copy()
    matched = pd.merge(
        within_5km,
        daily,
        left_on=["nearest_station_id", "capture_date_clean"],
        right_on=["StationId", "date"],
        how="inner"
    ).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.45), dpi=300)

    # Top: Distance histogram
    ax1.hist(manifest["distance_to_station_km"], bins=30, color="#0284c7", edgecolor="white", alpha=0.85)
    ax1.axvline(5.0, color="#e11d48", linestyle="--", lw=1.2, label="5km Thresh (49.4% > 5km)")
    ax1.set_title("(a) Haversine Distance to Station", fontsize=7.0, fontweight="bold", pad=2)
    ax1.set_xlabel("Distance (km)", fontsize=6.5, labelpad=1)
    ax1.set_ylabel("Count", fontsize=6.5, labelpad=1)
    ax1.legend(loc="upper right", fontsize=5.5)
    ax1.tick_params(labelsize=5.8, pad=1)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Bottom: Label Discrepancy Scatter
    ax2.scatter(matched["PM2.5"], matched["pm25_label"], color="#ea580c", alpha=0.55, edgecolors="none", s=16)
    min_p = min(matched["PM2.5"].min(), matched["pm25_label"].min())
    max_p = max(matched["PM2.5"].max(), matched["pm25_label"].max())
    ax2.plot([min_p, max_p], [min_p, max_p], "k--", lw=0.9, label="1:1 Ideal")
    ax2.plot([min_p, max_p], [min_p + 70.74, max_p + 70.74], "r:", lw=0.9, label="+70.7 ug/m3 Bias")
    ax2.set_title("(b) Systematic Positive Image Label Bias", fontsize=7.0, fontweight="bold", pad=2)
    ax2.set_xlabel("Station Mean PM2.5 (ug/m3)", fontsize=6.5, labelpad=1)
    ax2.set_ylabel("Image Label", fontsize=6.5, labelpad=1)
    ax2.legend(loc="lower right", fontsize=5.5)
    ax2.tick_params(labelsize=5.8, pad=1)
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "alignment_discrepancy_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved alignment_discrepancy_distribution")


# ------------------------------------------------------------------------------
# 8. FIG 9: ACTUAL VS PREDICTED (SINGLE COLUMN 2-ROW STACK: 3.5 x 3.3 in)
# ------------------------------------------------------------------------------
def generate_actual_vs_predicted():
    print("Generating single-column stacked Actual vs Predicted...")
    y_test_path = 'data/processed/model_sequences/y_test.npy'
    preds_path = 'models/lstm_baseline/test_predictions.npy'
    if not Path(y_test_path).exists() or not Path(preds_path).exists():
        return
    y_test = np.load(y_test_path)
    preds = np.load(preds_path)

    rng = np.random.RandomState(42)
    indices = rng.choice(len(y_test), size=12000, replace=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.3), dpi=300)

    # Subplot 1: PM2.5
    hb1 = ax1.hexbin(y_test[indices, 0], preds[indices, 0], gridsize=35, cmap='Blues',
                     mincnt=1, bins='log', alpha=0.85, edgecolors='none')
    ax1.plot([0, 350], [0, 350], color='#e11d48', linestyle='--', lw=1.1, label='1:1 Ideal')
    ax1.set_xlim(0, 350)
    ax1.set_ylim(0, 350)
    ax1.set_xlabel('Ground Truth PM2.5 (ug/m3)', fontsize=6.8, labelpad=2)
    ax1.set_ylabel('Predicted PM2.5', fontsize=6.8, labelpad=2)
    ax1.set_title('(a) PM2.5 Forecaster (R2 = 0.9492, MAE = 10.57)', fontsize=7.2, fontweight='bold', pad=3)
    ax1.tick_params(labelsize=6.0, pad=1)
    ax1.legend(loc='lower right', fontsize=5.8)

    # Subplot 2: PM10
    hb2 = ax2.hexbin(y_test[indices, 1], preds[indices, 1], gridsize=35, cmap='YlGn',
                     mincnt=1, bins='log', alpha=0.85, edgecolors='none')
    ax2.plot([0, 600], [0, 600], color='#e11d48', linestyle='--', lw=1.1, label='1:1 Ideal')
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 600)
    ax2.set_xlabel('Ground Truth PM10 (ug/m3)', fontsize=6.8, labelpad=2)
    ax2.set_ylabel('Predicted PM10', fontsize=6.8, labelpad=2)
    ax2.set_title('(b) PM10 Forecaster (R2 = 0.9301, MAE = 18.86)', fontsize=7.2, fontweight='bold', pad=3)
    ax2.tick_params(labelsize=6.0, pad=1)
    ax2.legend(loc='lower right', fontsize=5.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "actual_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved actual_vs_predicted (stacked single-column)")


if __name__ == "__main__":
    generate_confusion_matrices()
    generate_conformal_calibration()
    generate_seasonal_comparison()
    generate_multistation_performance()
    generate_regime_error_curve()
    generate_spatial_advection()
    generate_alignment_diagnostics()
    generate_actual_vs_predicted()
    print("All empirical figures regenerated with native single-column typography.")
