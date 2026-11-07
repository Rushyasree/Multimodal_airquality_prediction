
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# VayuDrishti: Actual vs Predicted Graph
# Uses real saved test targets and LSTM predictions
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

ACTUAL_FILE = ROOT / "data" / "processed" / "model_sequences" / "y_test.npy"
PRED_FILE = ROOT / "models" / "lstm_baseline" / "test_predictions.npy"

OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "actual_vs_predicted.png"
OUTPUT_PDF = OUTPUT_DIR / "actual_vs_predicted.pdf"

POLLUTANTS = [
    ("PM$_{2.5}$", 0),
    ("PM$_{10}$", 1),
]

# Load real test data
if not ACTUAL_FILE.exists():
    raise FileNotFoundError(f"Actual targets not found: {ACTUAL_FILE}")

if not PRED_FILE.exists():
    raise FileNotFoundError(f"Predictions not found: {PRED_FILE}")

y_true = np.load(ACTUAL_FILE)
y_pred = np.load(PRED_FILE)

# Verify dimensions and alignment
if y_true.shape != y_pred.shape:
    raise ValueError(
        f"Shape mismatch: actual={y_true.shape}, "
        f"predicted={y_pred.shape}"
    )

if y_true.ndim != 2 or y_true.shape[1] != 2:
    raise ValueError(
        f"Expected arrays shaped (N, 2), got {y_true.shape}"
    )

if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
    raise ValueError("Arrays contain NaN or infinite values.")

print(f"Actual array shape:    {y_true.shape}")
print(f"Predicted array shape: {y_pred.shape}")
print(f"Test samples:          {len(y_true):,}")

# Scientific publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5.2),
    constrained_layout=True,
)

for ax, (pollutant, column) in zip(axes, POLLUTANTS):
    actual = y_true[:, column].astype(np.float64)
    predicted = y_pred[:, column].astype(np.float64)

    # Metrics use the entire test set, not a sample
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    # Use every test sample in a density-style hexbin plot
    # This avoids hiding points due to overplotting.
    hb = ax.hexbin(
        actual,
        predicted,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )

    # Ideal predictions lie on y = x
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.6,
        color="red",
        label="Ideal prediction ($y=x$)",
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Actual {pollutant} ($\\mu g/m^3$)")
    ax.set_ylabel(f"Predicted {pollutant} ($\\mu g/m^3$)")
    ax.set_title(f"{pollutant}: Actual vs. Predicted")

    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    metrics_text = (
        f"Test samples: {len(actual):,}\n"
        f"MAE = {mae:.4f} $\\mu g/m^3$\n"
        f"RMSE = {rmse:.4f} $\\mu g/m^3$\n"
        f"$R^2$ = {r2:.4f}"
    )

    ax.text(
        0.97,
        0.04,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.92,
        },
    )

    fig.colorbar(
        hb,
        ax=ax,
        label="Log-scaled sample density",
        fraction=0.046,
        pad=0.04,
    )

fig.suptitle(
    "VayuDrishti LSTM: Actual vs. Predicted Concentrations",
    fontsize=14,
    fontweight="bold",
)

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.close(fig)

print("\nGraph generation completed.")
print(f"PNG: {OUTPUT_PNG}")
print(f"PDF: {OUTPUT_PDF}")
print("Metrics are calculated from the full test set.")
