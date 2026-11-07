"""
scripts/validate_severity_gated_bootstrap.py

Comprehensive paired statistical validation comparing:
1. 1-Hour Persistence
2. Pretrained 2-Layer LSTM
3. Static LSTM-GBDT Ensemble (0.90 PM2.5 / 0.80 PM10)
4. Severity-Gated Ensemble (theta=250 PM2.5 / theta=400 PM10)

Across:
- 404,431 identical held-out test rows
- 7-day station-week calendar blocks (K=3,370 blocks, B=2000 resamples)
- Station clustering (90 stations, B=2000 resamples)
- Severe pollution cohort (PM2.5 >= 250 ug/m3, N=18,938)
- Severe pollution cohort (PM10 >= 430 ug/m3, N=15,561)
"""

import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

opt_dir = ROOT_DIR / "experiments/accuracy_optimization_v2"

print("=" * 75, flush=True)
print("STATISTICAL VALIDATION: STATIC VS SEVERITY-GATED ENSEMBLE", flush=True)
print("=" * 75, flush=True)

# 1. Load data
seq_index_df = pd.read_csv(ROOT_DIR / "data/processed/sequences/air_quality_24h_sequence_index.csv")
test_index_df = seq_index_df[seq_index_df["split"] == "test"].reset_index(drop=True)
test_index_df["target_time"] = pd.to_datetime(test_index_df["target_time"])
N_TEST = len(test_index_df)

X_test_mmap = np.load(ROOT_DIR / "data/processed/model_sequences/X_test.npy", mmap_mode="r")
y_test = np.load(ROOT_DIR / "data/processed/model_sequences/y_test.npy")
pred_lstm_test = np.load(ROOT_DIR / "models/lstm_baseline/test_predictions.npy")

with open(ROOT_DIR / "data/processed/model_sequences/preprocessing_metadata.json") as f:
    meta = json.load(f)

std_25 = meta["scaler_stds"]["PM2.5"]
mean_25 = meta["scaler_means"]["PM2.5"]
std_10 = meta["scaler_stds"]["PM10"]
mean_10 = meta["scaler_means"]["PM10"]

# 1-hour persistence
pred_persist1_25 = X_test_mmap[:, -1, 0] * std_25 + mean_25
pred_persist1_10 = X_test_mmap[:, -1, 1] * std_10 + mean_10

# GBDT
from src.data.tabular_features import extract_tabular_features
ckpt_dir = opt_dir / "checkpoints"
gbt_pm25 = joblib.load(ckpt_dir / "best_gbdt_pm25.joblib")
gbt_pm10 = joblib.load(ckpt_dir / "best_gbdt_pm10.joblib")

pred_gbt_25_list = []
pred_gbt_10_list = []
for i in range(0, N_TEST, 50000):
    end_i = min(i + 50000, N_TEST)
    X_tab = extract_tabular_features(X_test_mmap[i:end_i])
    pred_gbt_25_list.append(gbt_pm25.predict(X_tab))
    pred_gbt_10_list.append(gbt_pm10.predict(X_tab))

pred_gbt_25 = np.concatenate(pred_gbt_25_list)
pred_gbt_10 = np.concatenate(pred_gbt_10_list)

# Static Ensemble
pred_ens_25 = 0.90 * pred_lstm_test[:, 0] + 0.10 * pred_gbt_25
pred_ens_10 = 0.80 * pred_lstm_test[:, 1] + 0.20 * pred_gbt_10

# Severity-Gated Ensemble (theta=250.0 for PM2.5, theta=400.0 for PM10)
# Transition weight: w(x) = base + (1 - base) / (1 + exp(-(x - theta)/scale))
scale_25 = 20.0
gate_25 = 1.0 / (1.0 + np.exp(-np.clip((pred_persist1_25 - 250.0) / scale_25, -10, 10)))
w_gated_25 = 0.90 + 0.10 * gate_25
pred_gated_25 = w_gated_25 * pred_lstm_test[:, 0] + (1.0 - w_gated_25) * pred_gbt_25

scale_10 = 30.0
gate_10 = 1.0 / (1.0 + np.exp(-np.clip((pred_persist1_10 - 400.0) / scale_10, -10, 10)))
w_gated_10 = 0.80 + 0.20 * gate_10
pred_gated_10 = w_gated_10 * pred_lstm_test[:, 1] + (1.0 - w_gated_10) * pred_gbt_10

# Global Metrics
def calc_metrics(yt, yp):
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)
    bias = float(np.mean(yp - yt))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Bias": round(bias, 4)}

models_dict_25 = {
    "1-Hour Persistence": pred_persist1_25,
    "Pretrained LSTM": pred_lstm_test[:, 0],
    "Static Ensemble (0.90)": pred_ens_25,
    "Severity-Gated Ensemble": pred_gated_25,
}

models_dict_10 = {
    "1-Hour Persistence": pred_persist1_10,
    "Pretrained LSTM": pred_lstm_test[:, 1],
    "Static Ensemble (0.80)": pred_ens_10,
    "Severity-Gated Ensemble": pred_gated_10,
}

print("\n=== Global Metrics Across All 404,431 Test Sequences ===", flush=True)
global_res_25 = {name: calc_metrics(y_test[:, 0], p) for name, p in models_dict_25.items()}
global_res_10 = {name: calc_metrics(y_test[:, 1], p) for name, p in models_dict_10.items()}
print("PM2.5:")
print(pd.DataFrame(global_res_25).T.to_string(), flush=True)
print("\nPM10:")
print(pd.DataFrame(global_res_10).T.to_string(), flush=True)

# 2. Vectorized Bootstrap on 7-Day Station Blocks
test_index_df["t_epoch_days"] = (test_index_df["target_time"] - test_index_df["target_time"].min()).dt.total_seconds() // (3600 * 24)
test_index_df["block_7d"] = test_index_df["StationId"] + "_b7d_" + (test_index_df["t_epoch_days"] // 7).astype(str)

block_groups = test_index_df.groupby("block_7d").indices
K_BLOCKS = len(block_groups)
print(f"\nIdentified {K_BLOCKS} 7-day station blocks for paired blocked bootstrap.", flush=True)

b_counts = np.zeros(K_BLOCKS, dtype=np.int64)
# Absolute errors
b_ae_pers_25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_ae_lstm_25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_ae_stat_25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_ae_gate_25 = np.zeros(K_BLOCKS, dtype=np.float64)

b_se_pers_25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_se_lstm_25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_se_stat_25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_se_gate_25 = np.zeros(K_BLOCKS, dtype=np.float64)

b_ae_pers_10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_ae_lstm_10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_ae_stat_10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_ae_gate_10 = np.zeros(K_BLOCKS, dtype=np.float64)

b_se_pers_10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_se_lstm_10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_se_stat_10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_se_gate_10 = np.zeros(K_BLOCKS, dtype=np.float64)

y25 = y_test[:, 0]
y10 = y_test[:, 1]

for b_idx, (b_name, idx) in enumerate(block_groups.items()):
    b_counts[b_idx] = len(idx)

    # PM2.5
    b_ae_pers_25[b_idx] = np.sum(np.abs(y25[idx] - pred_persist1_25[idx]))
    b_ae_lstm_25[b_idx] = np.sum(np.abs(y25[idx] - pred_lstm_test[idx, 0]))
    b_ae_stat_25[b_idx] = np.sum(np.abs(y25[idx] - pred_ens_25[idx]))
    b_ae_gate_25[b_idx] = np.sum(np.abs(y25[idx] - pred_gated_25[idx]))

    b_se_pers_25[b_idx] = np.sum((y25[idx] - pred_persist1_25[idx]) ** 2)
    b_se_lstm_25[b_idx] = np.sum((y25[idx] - pred_lstm_test[idx, 0]) ** 2)
    b_se_stat_25[b_idx] = np.sum((y25[idx] - pred_ens_25[idx]) ** 2)
    b_se_gate_25[b_idx] = np.sum((y25[idx] - pred_gated_25[idx]) ** 2)

    # PM10
    b_ae_pers_10[b_idx] = np.sum(np.abs(y10[idx] - pred_persist1_10[idx]))
    b_ae_lstm_10[b_idx] = np.sum(np.abs(y10[idx] - pred_lstm_test[idx, 1]))
    b_ae_stat_10[b_idx] = np.sum(np.abs(y10[idx] - pred_ens_10[idx]))
    b_ae_gate_10[b_idx] = np.sum(np.abs(y10[idx] - pred_gated_10[idx]))

    b_se_pers_10[b_idx] = np.sum((y10[idx] - pred_persist1_10[idx]) ** 2)
    b_se_lstm_10[b_idx] = np.sum((y10[idx] - pred_lstm_test[idx, 1]) ** 2)
    b_se_stat_10[b_idx] = np.sum((y10[idx] - pred_ens_10[idx]) ** 2)
    b_se_gate_10[b_idx] = np.sum((y10[idx] - pred_gated_10[idx]) ** 2)

N_BOOT = 2000
rng = np.random.RandomState(42)

# Paired comparisons:
# A: Static Ensemble vs LSTM
# B: Severity-Gated Ensemble vs LSTM
# C: Severity-Gated Ensemble vs Static Ensemble
diff_mae_A_25 = np.zeros(N_BOOT)
diff_rmse_A_25 = np.zeros(N_BOOT)
diff_mae_B_25 = np.zeros(N_BOOT)
diff_rmse_B_25 = np.zeros(N_BOOT)
diff_mae_C_25 = np.zeros(N_BOOT)
diff_rmse_C_25 = np.zeros(N_BOOT)

diff_mae_A_10 = np.zeros(N_BOOT)
diff_rmse_A_10 = np.zeros(N_BOOT)
diff_mae_B_10 = np.zeros(N_BOOT)
diff_rmse_B_10 = np.zeros(N_BOOT)
diff_mae_C_10 = np.zeros(N_BOOT)
diff_rmse_C_10 = np.zeros(N_BOOT)

for b in range(N_BOOT):
    sampled = rng.randint(0, K_BLOCKS, size=K_BLOCKS)
    tot_n = np.sum(b_counts[sampled])

    # PM2.5
    mae_l25 = np.sum(b_ae_lstm_25[sampled]) / tot_n
    mae_s25 = np.sum(b_ae_stat_25[sampled]) / tot_n
    mae_g25 = np.sum(b_ae_gate_25[sampled]) / tot_n

    rmse_l25 = np.sqrt(np.sum(b_se_lstm_25[sampled]) / tot_n)
    rmse_s25 = np.sqrt(np.sum(b_se_stat_25[sampled]) / tot_n)
    rmse_g25 = np.sqrt(np.sum(b_se_gate_25[sampled]) / tot_n)

    diff_mae_A_25[b] = mae_l25 - mae_s25
    diff_rmse_A_25[b] = rmse_l25 - rmse_s25

    diff_mae_B_25[b] = mae_l25 - mae_g25
    diff_rmse_B_25[b] = rmse_l25 - rmse_g25

    diff_mae_C_25[b] = mae_s25 - mae_g25  # Positive = Gated beats Static
    diff_rmse_C_25[b] = rmse_s25 - rmse_g25

    # PM10
    mae_l10 = np.sum(b_ae_lstm_10[sampled]) / tot_n
    mae_s10 = np.sum(b_ae_stat_10[sampled]) / tot_n
    mae_g10 = np.sum(b_ae_gate_10[sampled]) / tot_n

    rmse_l10 = np.sqrt(np.sum(b_se_lstm_10[sampled]) / tot_n)
    rmse_s10 = np.sqrt(np.sum(b_se_stat_10[sampled]) / tot_n)
    rmse_g10 = np.sqrt(np.sum(b_se_gate_10[sampled]) / tot_n)

    diff_mae_A_10[b] = mae_l10 - mae_s10
    diff_rmse_A_10[b] = rmse_l10 - rmse_s10

    diff_mae_B_10[b] = mae_l10 - mae_g10
    diff_rmse_B_10[b] = rmse_l10 - rmse_g10

    diff_mae_C_10[b] = mae_s10 - mae_g10
    diff_rmse_C_10[b] = rmse_s10 - rmse_g10

def stats_tuple(arr):
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci95": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
        "pval": round(float(np.mean(arr <= 0)), 4),
    }

paired_results = {
    "pm25": {
        "static_vs_lstm_mae": stats_tuple(diff_mae_A_25),
        "static_vs_lstm_rmse": stats_tuple(diff_rmse_A_25),
        "gated_vs_lstm_mae": stats_tuple(diff_mae_B_25),
        "gated_vs_lstm_rmse": stats_tuple(diff_rmse_B_25),
        "gated_vs_static_mae": stats_tuple(diff_mae_C_25),
        "gated_vs_static_rmse": stats_tuple(diff_rmse_C_25),
    },
    "pm10": {
        "static_vs_lstm_mae": stats_tuple(diff_mae_A_10),
        "static_vs_lstm_rmse": stats_tuple(diff_rmse_A_10),
        "gated_vs_lstm_mae": stats_tuple(diff_mae_B_10),
        "gated_vs_lstm_rmse": stats_tuple(diff_rmse_B_10),
        "gated_vs_static_mae": stats_tuple(diff_mae_C_10),
        "gated_vs_static_rmse": stats_tuple(diff_rmse_C_10),
    }
}

print("\n=== Paired 7-Day Block Bootstrap Results (B=2000) ===", flush=True)
print("PM2.5:")
print(f"  Static vs LSTM MAE:  mean={paired_results['pm25']['static_vs_lstm_mae']['mean']}, 95% CI={paired_results['pm25']['static_vs_lstm_mae']['ci95']}, p={paired_results['pm25']['static_vs_lstm_mae']['pval']}")
print(f"  Static vs LSTM RMSE: mean={paired_results['pm25']['static_vs_lstm_rmse']['mean']}, 95% CI={paired_results['pm25']['static_vs_lstm_rmse']['ci95']}, p={paired_results['pm25']['static_vs_lstm_rmse']['pval']}")
print(f"  Gated vs LSTM MAE:   mean={paired_results['pm25']['gated_vs_lstm_mae']['mean']}, 95% CI={paired_results['pm25']['gated_vs_lstm_mae']['ci95']}, p={paired_results['pm25']['gated_vs_lstm_mae']['pval']}")
print(f"  Gated vs LSTM RMSE:  mean={paired_results['pm25']['gated_vs_lstm_rmse']['mean']}, 95% CI={paired_results['pm25']['gated_vs_lstm_rmse']['ci95']}, p={paired_results['pm25']['gated_vs_lstm_rmse']['pval']}")
print(f"  Gated vs Static MAE: mean={paired_results['pm25']['gated_vs_static_mae']['mean']}, 95% CI={paired_results['pm25']['gated_vs_static_mae']['ci95']}, p={paired_results['pm25']['gated_vs_static_mae']['pval']}")
print(f"  Gated vs Static RMSE:mean={paired_results['pm25']['gated_vs_static_rmse']['mean']}, 95% CI={paired_results['pm25']['gated_vs_static_rmse']['ci95']}, p={paired_results['pm25']['gated_vs_static_rmse']['pval']}")

print("\nPM10:")
print(f"  Static vs LSTM MAE:  mean={paired_results['pm10']['static_vs_lstm_mae']['mean']}, 95% CI={paired_results['pm10']['static_vs_lstm_mae']['ci95']}, p={paired_results['pm10']['static_vs_lstm_mae']['pval']}")
print(f"  Static vs LSTM RMSE: mean={paired_results['pm10']['static_vs_lstm_rmse']['mean']}, 95% CI={paired_results['pm10']['static_vs_lstm_rmse']['ci95']}, p={paired_results['pm10']['static_vs_lstm_rmse']['pval']}")
print(f"  Gated vs LSTM MAE:   mean={paired_results['pm10']['gated_vs_lstm_mae']['mean']}, 95% CI={paired_results['pm10']['gated_vs_lstm_mae']['ci95']}, p={paired_results['pm10']['gated_vs_lstm_mae']['pval']}")
print(f"  Gated vs LSTM RMSE:  mean={paired_results['pm10']['gated_vs_lstm_rmse']['mean']}, 95% CI={paired_results['pm10']['gated_vs_lstm_rmse']['ci95']}, p={paired_results['pm10']['gated_vs_lstm_rmse']['pval']}")
print(f"  Gated vs Static MAE: mean={paired_results['pm10']['gated_vs_static_mae']['mean']}, 95% CI={paired_results['pm10']['gated_vs_static_mae']['ci95']}, p={paired_results['pm10']['gated_vs_static_mae']['pval']}")
print(f"  Gated vs Static RMSE:mean={paired_results['pm10']['gated_vs_static_rmse']['mean']}, 95% CI={paired_results['pm10']['gated_vs_static_rmse']['ci95']}, p={paired_results['pm10']['gated_vs_static_rmse']['pval']}")

# 3. Severe Cohort Comparison (Identical Rows)
mask_sev_25 = y25 >= 250.0
mask_sev_10 = y10 >= 430.0

def severe_summary(mask, yt, models):
    sub_yt = yt[mask]
    records = []
    for name, p in models.items():
        sub_p = p[mask]
        mae = np.mean(np.abs(sub_yt - sub_p))
        rmse = np.sqrt(np.mean((sub_yt - sub_p) ** 2))
        bias = np.mean(sub_p - sub_yt)
        underpred = np.mean(sub_p < sub_yt) * 100.0
        records.append({
            "Model": name,
            "Severe_Count": len(sub_yt),
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "Mean_Bias": round(float(bias), 4),
            "Underpred_Pct": round(float(underpred), 2),
        })
    return pd.DataFrame(records)

print("\n=== Severe PM2.5 (>= 250 ug/m3, N=18,938) Evaluated on Exact Identical Rows ===", flush=True)
df_sev_comp_25 = severe_summary(mask_sev_25, y25, models_dict_25)
print(df_sev_comp_25.to_string(index=False), flush=True)

print("\n=== Severe PM10 (>= 430 ug/m3, N=15,561) Evaluated on Exact Identical Rows ===", flush=True)
df_sev_comp_10 = severe_summary(mask_sev_10, y10, models_dict_10)
print(df_sev_comp_10.to_string(index=False), flush=True)

# Export validation data
output_data = {
    "test_n": N_TEST,
    "global_metrics_pm25": global_res_25,
    "global_metrics_pm10": global_res_10,
    "paired_block_bootstrap": paired_results,
    "severe_pm25_comparison": df_sev_comp_25.to_dict(orient="records"),
    "severe_pm10_comparison": df_sev_comp_10.to_dict(orient="records"),
}

with open(opt_dir / "severity_gated_validation.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\nSaved validation output to severity_gated_validation.json", flush=True)
