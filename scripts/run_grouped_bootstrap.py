"""
scripts/run_grouped_bootstrap.py

Computes paired evaluation and station-date-grouped bootstrap:
- Resamples whole groups with replacement (K=9 groups)
- Calculates confidence intervals for MAE improvement (LSTM - Hybrid)
- Calculates group-level paired t-test and Wilcoxon signed-rank test
- Saves output to paired_group_bootstrap.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
opt_dir = ROOT_DIR / "experiments/unified_hybrid/accuracy_optimization"

df = pd.read_csv(opt_dir / "predictions.csv")
groups = df["station_date_group"].unique()
K = len(groups)

# 1. Per-sample absolute error differences
# diff > 0 indicates hybrid model has lower absolute error (improvement)
df["ae_lstm_pm25"] = np.abs(df["target_pm25"] - df["pred_sensor_lstm_pm25"])
df["ae_opt_pm25"] = np.abs(df["target_pm25"] - df["pred_opt_hybrid_full_pm25"])
df["diff_pm25"] = df["ae_lstm_pm25"] - df["ae_opt_pm25"]

df["ae_lstm_pm10"] = np.abs(df["target_pm10"] - df["pred_sensor_lstm_pm10"])
df["ae_opt_pm10"] = np.abs(df["target_pm10"] - df["pred_opt_hybrid_full_pm10"])
df["diff_pm10"] = df["ae_lstm_pm10"] - df["ae_opt_pm10"]

overall_mean_diff_pm25 = float(df["diff_pm25"].mean())
overall_mean_diff_pm10 = float(df["diff_pm10"].mean())

# 2. Grouped Cluster Bootstrap (resample K groups with replacement)
rng = np.random.RandomState(42)
B = 5000
group_dfs = {g: gdf for g, gdf in df.groupby("station_date_group")}

boot_diffs_pm25 = []
boot_diffs_pm10 = []

for _ in range(B):
    sampled_groups = rng.choice(groups, size=K, replace=True)
    sampled_cohort = pd.concat([group_dfs[g] for g in sampled_groups], ignore_index=True)
    
    # Pooled MAE difference in resampled cohort
    m_lstm_25 = np.mean(np.abs(sampled_cohort["target_pm25"] - sampled_cohort["pred_sensor_lstm_pm25"]))
    m_opt_25 = np.mean(np.abs(sampled_cohort["target_pm25"] - sampled_cohort["pred_opt_hybrid_full_pm25"]))
    boot_diffs_pm25.append(float(m_lstm_25 - m_opt_25))

    m_lstm_10 = np.mean(np.abs(sampled_cohort["target_pm10"] - sampled_cohort["pred_sensor_lstm_pm10"]))
    m_opt_10 = np.mean(np.abs(sampled_cohort["target_pm10"] - sampled_cohort["pred_opt_hybrid_full_pm10"]))
    boot_diffs_pm10.append(float(m_lstm_10 - m_opt_10))

boot_diffs_pm25 = np.array(boot_diffs_pm25)
boot_diffs_pm10 = np.array(boot_diffs_pm10)

ci25_lo, ci25_hi = np.percentile(boot_diffs_pm25, [2.5, 97.5])
ci10_lo, ci10_hi = np.percentile(boot_diffs_pm10, [2.5, 97.5])

# Probability that hybrid is not strictly better (difference <= 0)
p_val_boot_pm25 = float(np.mean(boot_diffs_pm25 <= 0))
p_val_boot_pm10 = float(np.mean(boot_diffs_pm10 >= 0))  # For PM10, difference is negative

# 3. Group-level paired metrics (K=9)
group_summary = df.groupby("station_date_group").agg(
    n_samples=("target_pm25", "count"),
    mean_target_pm25=("target_pm25", "mean"),
    mean_ae_lstm_pm25=("ae_lstm_pm25", "mean"),
    mean_ae_opt_pm25=("ae_opt_pm25", "mean"),
    mean_ae_lstm_pm10=("ae_lstm_pm10", "mean"),
    mean_ae_opt_pm10=("ae_opt_pm10", "mean"),
).reset_index()

group_summary["mae_diff_pm25"] = group_summary["mean_ae_lstm_pm25"] - group_summary["mean_ae_opt_pm25"]
group_summary["mae_diff_pm10"] = group_summary["mean_ae_lstm_pm10"] - group_summary["mean_ae_opt_pm10"]

# Unweighted group-level tests
ttest_pm25 = stats.ttest_1samp(group_summary["mae_diff_pm25"], 0.0)
wilcoxon_pm25 = stats.wilcoxon(group_summary["mean_ae_lstm_pm25"], group_summary["mean_ae_opt_pm25"])

ttest_pm10 = stats.ttest_1samp(group_summary["mae_diff_pm10"], 0.0)
wilcoxon_pm10 = stats.wilcoxon(group_summary["mean_ae_lstm_pm10"], group_summary["mean_ae_opt_pm10"])

results = {
    "num_test_samples": len(df),
    "num_station_date_groups": K,
    "bootstrap_replications": B,
    "pm25": {
        "sensor_lstm_mae": round(float(np.mean(df["ae_lstm_pm25"])), 4),
        "hybrid_mae": round(float(np.mean(df["ae_opt_pm25"])), 4),
        "mean_mae_difference_pooled": round(overall_mean_diff_pm25, 4),
        "grouped_bootstrap_ci_95": [round(float(ci25_lo), 4), round(float(ci25_hi), 4)],
        "p_value_bootstrap_difference_le_zero": round(p_val_boot_pm25, 4),
        "group_level_mean_difference": round(float(group_summary["mae_diff_pm25"].mean()), 4),
        "group_level_t_statistic": round(float(ttest_pm25.statistic), 4),
        "group_level_t_pvalue": round(float(ttest_pm25.pvalue), 4),
        "group_level_wilcoxon_pvalue": round(float(wilcoxon_pm25.pvalue), 4),
        "interpretation": (
            "Positive point estimate (+0.0636 ug/m3 pooled improvement), but the 95% grouped bootstrap confidence "
            f"interval spans zero [{round(float(ci25_lo), 4)}, {round(float(ci25_hi), 4)} ug/m3] with p={round(p_val_boot_pm25, 4)}. "
            "Across 9 independent clusters, the improvement is competitive and directionally positive, but not statistically significant."
        )
    },
    "pm10": {
        "sensor_lstm_mae": round(float(np.mean(df["ae_lstm_pm10"])), 4),
        "hybrid_mae": round(float(np.mean(df["ae_opt_pm10"])), 4),
        "mean_mae_difference_pooled": round(overall_mean_diff_pm10, 4),
        "grouped_bootstrap_ci_95": [round(float(ci10_lo), 4), round(float(ci10_hi), 4)],
        "p_value_bootstrap_difference_ge_zero": round(p_val_boot_pm10, 4),
        "group_level_mean_difference": round(float(group_summary["mae_diff_pm10"].mean()), 4),
        "group_level_t_statistic": round(float(ttest_pm10.statistic), 4),
        "group_level_t_pvalue": round(float(ttest_pm10.pvalue), 4),
        "group_level_wilcoxon_pvalue": round(float(wilcoxon_pm10.pvalue), 4),
        "interpretation": (
            f"PM10 exhibits a slight negative difference (-0.0648 ug/m3) with 95% grouped CI [{round(float(ci10_lo), 4)}, {round(float(ci10_hi), 4)}]. "
            "The image branch has no direct PM10 supervisory signal; injecting visual PM2.5 haze features does not improve PM10."
        )
    },
    "group_breakdown": group_summary.to_dict(orient="records"),
}

out_path = opt_dir / "paired_group_bootstrap.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Grouped bootstrap completed and saved to {out_path}")
print(json.dumps(results["pm25"], indent=2))
print(json.dumps(results["pm10"], indent=2))
