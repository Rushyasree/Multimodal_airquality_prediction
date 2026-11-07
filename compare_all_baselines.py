
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ABLATION_DIR = ROOT / "models" / "ablation"

RESULTS_JSON = ABLATION_DIR / "ablation_comparison_corrected.json"
OUTPUT_DIR = ROOT / "results" / "baseline_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Previously evaluated baseline results.
# These values are from your existing baseline evaluation.
models = {
    "Persistence": {
        "PM2.5": {"MAE": 10.6783, "RMSE": 19.8072, "R2": 0.9423},
        "PM10": {"MAE": 19.7384, "RMSE": 36.2374, "R2": 0.9115},
    },
    "Original LSTM": {
        "PM2.5": {"MAE": 10.5719, "RMSE": 18.5789, "R2": 0.9492},
        "PM10": {"MAE": 18.8643, "RMSE": 32.2027, "R2": 0.9301},
    },
}

if not RESULTS_JSON.exists():
    raise FileNotFoundError(
        f"Corrected ablation results not found: {RESULTS_JSON}"
    )

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    ablation = json.load(f)

models["Ablation - Pollution only"] = ablation["pollution_only"]
models["Ablation - Pollution + weather"] = ablation["pollution_weather"]

# ---------------------------------------------------------
# Save all metrics to CSV
# ---------------------------------------------------------
csv_path = OUTPUT_DIR / "all_model_metrics.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Pollutant",
        "MAE",
        "RMSE",
        "R2",
    ])

    for model_name, targets in models.items():
        for pollutant in ["PM2.5", "PM10"]:
            metrics = targets[pollutant]

            writer.writerow([
                model_name,
                pollutant,
                metrics["MAE"],
                metrics["RMSE"],
                metrics["R2"],
            ])

print("\nALL-MODEL METRICS")
print("-" * 100)
print(
    f"{'Model':<35}{'Pollutant':<12}"
    f"{'MAE':>12}{'RMSE':>12}{'R2':>12}"
)
print("-" * 100)

for model_name, targets in models.items():
    for pollutant in ["PM2.5", "PM10"]:
        m = targets[pollutant]
        print(
            f"{model_name:<35}{pollutant:<12}"
            f"{m['MAE']:>12.4f}"
            f"{m['RMSE']:>12.4f}"
            f"{m['R2']:>12.4f}"
        )

print(f"\nCSV saved to: {csv_path}")

# ---------------------------------------------------------
# Create one chart per pollutant and metric
# ---------------------------------------------------------
model_names = list(models.keys())
metrics_to_plot = ["MAE", "RMSE", "R2"]
pollutants = ["PM2.5", "PM10"]

for pollutant in pollutants:
    for metric in metrics_to_plot:
        values = [
            models[name][pollutant][metric]
            for name in model_names
        ]

        fig, ax = plt.subplots(figsize=(11, 6))

        bars = ax.bar(model_names, values)

        ax.set_title(
            f"{metric} Comparison - {pollutant} Forecasting"
        )
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.4f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()

        filename = (
            f"{pollutant.replace('.', '')}_{metric.lower()}.png"
        )

        figure_path = OUTPUT_DIR / filename
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Chart saved: {figure_path}")

print("\nComparison completed successfully.")
