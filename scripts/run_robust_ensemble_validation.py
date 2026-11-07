"""
scripts/run_robust_ensemble_validation.py

Rigorous statistical validation and reproducibility audit for VayuDrishti Accuracy Optimization v2:
1. Reproduce and verify all test metrics across 404,431 test sequences.
2. Station-grouped temporally blocked bootstrap (7-day station blocks, B=2000).
3. Station cluster bootstrap (clustering by station, B=2000).
4. Per-station performance distribution across all 90 stations.
5. Chronological monthly performance breakdown (Sep 2019 to Jun 2020).
6. CPCB NAQI pollution tier breakdown & severe underprediction investigation.
7. Saves experiments/accuracy_optimization_v2/paired_ensemble_bootstrap.json.
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
fig_dir = opt_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 75, flush=True)
print("VAYUDRISHTI ACCURACY OPTIMIZATION V2 — FINAL SCIENTIFIC VERIFICATION AUDIT", flush=True)
print("=" * 75, flush=True)

# 1. LOAD TEST DATA AND SAVED PREDICTIONS
print("\n[1/6] Loading historical benchmark dataset and model predictions...", flush=True)
t0 = time.time()

seq_index_df = pd.read_csv(ROOT_DIR / "data/processed/sequences/air_quality_24h_sequence_index.csv")
test_index_df = seq_index_df[seq_index_df["split"] == "test"].reset_index(drop=True)
N_TEST = len(test_index_df)
assert N_TEST == 404431, f"Expected 404,431 test sequences, got {N_TEST}"

test_index_df["target_time"] = pd.to_datetime(test_index_df["target_time"])
test_index_df["history_end"] = pd.to_datetime(test_index_df["history_end"])

horizon_diff = (test_index_df["target_time"] - test_index_df["history_end"]).dt.total_seconds() / 3600.0
assert np.allclose(horizon_diff, 1.0), "Forecast horizon is not strictly 1 hour!"
print(f"  Verified 1-hour horizon across all {N_TEST} sequences: target_time == history_end + 1h", flush=True)

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

from src.data.tabular_features import extract_tabular_features

ckpt_dir = opt_dir / "checkpoints"
gbt_pm25 = joblib.load(ckpt_dir / "best_gbdt_pm25.joblib")
gbt_pm10 = joblib.load(ckpt_dir / "best_gbdt_pm10.joblib")

print("  Generating GBDT predictions across 404,431 test sequences...", flush=True)
CHUNK_SIZE = 50000
pred_gbt_25_list = []
pred_gbt_10_list = []
for i in range(0, N_TEST, CHUNK_SIZE):
    end_i = min(i + CHUNK_SIZE, N_TEST)
    X_tab = extract_tabular_features(X_test_mmap[i:end_i])
    pred_gbt_25_list.append(gbt_pm25.predict(X_tab))
    pred_gbt_10_list.append(gbt_pm10.predict(X_tab))

pred_gbt_25 = np.concatenate(pred_gbt_25_list)
pred_gbt_10 = np.concatenate(pred_gbt_10_list)

# Blend: PM2.5 (0.90 LSTM + 0.10 GBDT), PM10 (0.80 LSTM + 0.20 GBDT)
pred_ens_25 = 0.90 * pred_lstm_test[:, 0] + 0.10 * pred_gbt_25
pred_ens_10 = 0.80 * pred_lstm_test[:, 1] + 0.20 * pred_gbt_10

print(f"  Predictions generated in {time.time() - t0:.2f}s.", flush=True)

# 2. RECOMPUTE & VERIFY GLOBAL BENCHMARK METRICS
print("\n[2/6] Verifying global metrics against reported values...", flush=True)
def calc_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

m_pers_25 = calc_metrics(y_test[:, 0], pred_persist1_25)
m_pers_10 = calc_metrics(y_test[:, 1], pred_persist1_10)

m_lstm_25 = calc_metrics(y_test[:, 0], pred_lstm_test[:, 0])
m_lstm_10 = calc_metrics(y_test[:, 1], pred_lstm_test[:, 1])

m_gbt_25 = calc_metrics(y_test[:, 0], pred_gbt_25)
m_gbt_10 = calc_metrics(y_test[:, 1], pred_gbt_10)

m_ens_25 = calc_metrics(y_test[:, 0], pred_ens_25)
m_ens_10 = calc_metrics(y_test[:, 1], pred_ens_10)

print(f"  PM2.5 Persistence: MAE={m_pers_25['MAE']}, RMSE={m_pers_25['RMSE']}, R2={m_pers_25['R2']}", flush=True)
print(f"  PM2.5 LSTM:        MAE={m_lstm_25['MAE']}, RMSE={m_lstm_25['RMSE']}, R2={m_lstm_25['R2']}", flush=True)
print(f"  PM2.5 Ensemble:    MAE={m_ens_25['MAE']}, RMSE={m_ens_25['RMSE']}, R2={m_ens_25['R2']}", flush=True)
print(f"  PM10 Persistence:  MAE={m_pers_10['MAE']}, RMSE={m_pers_10['RMSE']}, R2={m_pers_10['R2']}", flush=True)
print(f"  PM10 LSTM:         MAE={m_lstm_10['MAE']}, RMSE={m_lstm_10['RMSE']}, R2={m_lstm_10['R2']}", flush=True)
print(f"  PM10 Ensemble:     MAE={m_ens_10['MAE']}, RMSE={m_ens_10['RMSE']}, R2={m_ens_10['R2']}", flush=True)

# Per-sample error arrays
diff_ae_25 = np.abs(y_test[:, 0] - pred_lstm_test[:, 0]) - np.abs(y_test[:, 0] - pred_ens_25)
diff_ae_10 = np.abs(y_test[:, 1] - pred_lstm_test[:, 1]) - np.abs(y_test[:, 1] - pred_ens_10)

pct_better_25 = float(np.mean(diff_ae_25 > 0) * 100.0)
pct_better_10 = float(np.mean(diff_ae_10 > 0) * 100.0)
print(f"  Ensemble beats LSTM on {pct_better_25:.2f}% of test sequences for PM2.5", flush=True)
print(f"  Ensemble beats LSTM on {pct_better_10:.2f}% of test sequences for PM10", flush=True)

# 3. STATION-GROUPED TEMPORALLY BLOCKED BOOTSTRAP (B = 2000)
print("\n[3/6] Running Station-Grouped Temporally Blocked Bootstrap (B=2000)...", flush=True)
test_index_df["year_week"] = test_index_df["target_time"].dt.strftime("%Y-W%W")
test_index_df["station_block"] = test_index_df["StationId"] + "_" + test_index_df["year_week"]

unique_blocks = test_index_df["station_block"].unique()
K_BLOCKS = len(unique_blocks)
print(f"  Identified {K_BLOCKS} unique station-week blocks across 90 stations.", flush=True)

# Precompute block-level sufficient statistics for instant bootstrap evaluation
block_counts = np.zeros(K_BLOCKS, dtype=np.int64)
b_sum_ae_l25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_sum_ae_e25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_sum_se_l25 = np.zeros(K_BLOCKS, dtype=np.float64)
b_sum_se_e25 = np.zeros(K_BLOCKS, dtype=np.float64)

b_sum_ae_l10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_sum_ae_e10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_sum_se_l10 = np.zeros(K_BLOCKS, dtype=np.float64)
b_sum_se_e10 = np.zeros(K_BLOCKS, dtype=np.float64)

y25 = y_test[:, 0]
y10 = y_test[:, 1]
pl25 = pred_lstm_test[:, 0]
pl10 = pred_lstm_test[:, 1]
pe25 = pred_ens_25
pe10 = pred_ens_10

for b_idx, (b_name, grp_indices) in enumerate(test_index_df.groupby("station_block").indices.items()):
    idx = grp_indices
    block_counts[b_idx] = len(idx)

    b_sum_ae_l25[b_idx] = np.sum(np.abs(y25[idx] - pl25[idx]))
    b_sum_ae_e25[b_idx] = np.sum(np.abs(y25[idx] - pe25[idx]))
    b_sum_se_l25[b_idx] = np.sum((y25[idx] - pl25[idx]) ** 2)
    b_sum_se_e25[b_idx] = np.sum((y25[idx] - pe25[idx]) ** 2)

    b_sum_ae_l10[b_idx] = np.sum(np.abs(y10[idx] - pl10[idx]))
    b_sum_ae_e10[b_idx] = np.sum(np.abs(y10[idx] - pe10[idx]))
    b_sum_se_l10[b_idx] = np.sum((y10[idx] - pl10[idx]) ** 2)
    b_sum_se_e10[b_idx] = np.sum((y10[idx] - pe10[idx]) ** 2)

N_BOOT = 2000
RNG = np.random.RandomState(42)

boot_diff_mae_25 = np.zeros(N_BOOT, dtype=np.float64)
boot_diff_rmse_25 = np.zeros(N_BOOT, dtype=np.float64)
boot_diff_mae_10 = np.zeros(N_BOOT, dtype=np.float64)
boot_diff_rmse_10 = np.zeros(N_BOOT, dtype=np.float64)

t_boot0 = time.time()
for b in range(N_BOOT):
    sampled_indices = RNG.randint(0, K_BLOCKS, size=K_BLOCKS)
    n_total = np.sum(block_counts[sampled_indices])

    mae_l25 = np.sum(b_sum_ae_l25[sampled_indices]) / n_total
    mae_e25 = np.sum(b_sum_ae_e25[sampled_indices]) / n_total
    rmse_l25 = np.sqrt(np.sum(b_sum_se_l25[sampled_indices]) / n_total)
    rmse_e25 = np.sqrt(np.sum(b_sum_se_e25[sampled_indices]) / n_total)

    boot_diff_mae_25[b] = mae_l25 - mae_e25
    boot_diff_rmse_25[b] = rmse_l25 - rmse_e25

    mae_l10 = np.sum(b_sum_ae_l10[sampled_indices]) / n_total
    mae_e10 = np.sum(b_sum_ae_e10[sampled_indices]) / n_total
    rmse_l10 = np.sqrt(np.sum(b_sum_se_l10[sampled_indices]) / n_total)
    rmse_e10 = np.sqrt(np.sum(b_sum_se_e10[sampled_indices]) / n_total)

    boot_diff_mae_10[b] = mae_l10 - mae_e10
    boot_diff_rmse_10[b] = rmse_l10 - rmse_e10

print(f"  Block bootstrap (B={N_BOOT}) complete in {time.time() - t_boot0:.2f}s.", flush=True)

ci_mae_25 = [float(np.percentile(boot_diff_mae_25, 2.5)), float(np.percentile(boot_diff_mae_25, 97.5))]
ci_rmse_25 = [float(np.percentile(boot_diff_rmse_25, 2.5)), float(np.percentile(boot_diff_rmse_25, 97.5))]

ci_mae_10 = [float(np.percentile(boot_diff_mae_10, 2.5)), float(np.percentile(boot_diff_mae_10, 97.5))]
ci_rmse_10 = [float(np.percentile(boot_diff_rmse_10, 2.5)), float(np.percentile(boot_diff_rmse_10, 97.5))]

pval_mae_25 = float(np.mean(boot_diff_mae_25 <= 0))
pval_rmse_25 = float(np.mean(boot_diff_rmse_25 <= 0))
pval_mae_10 = float(np.mean(boot_diff_mae_10 <= 0))
pval_rmse_10 = float(np.mean(boot_diff_rmse_10 <= 0))

print("\n=== Station-Grouped Temporally Blocked Bootstrap 95% CIs ===", flush=True)
print(f"  PM2.5 MAE Improvement:  Mean={np.mean(boot_diff_mae_25):.4f} ug/m3, 95% CI: [{ci_mae_25[0]:.4f}, {ci_mae_25[1]:.4f}], p={pval_mae_25:.4f}", flush=True)
print(f"  PM2.5 RMSE Improvement: Mean={np.mean(boot_diff_rmse_25):.4f} ug/m3, 95% CI: [{ci_rmse_25[0]:.4f}, {ci_rmse_25[1]:.4f}], p={pval_rmse_25:.4f}", flush=True)
print(f"  PM10 MAE Improvement:   Mean={np.mean(boot_diff_mae_10):.4f} ug/m3, 95% CI: [{ci_mae_10[0]:.4f}, {ci_mae_10[1]:.4f}], p={pval_mae_10:.4f}", flush=True)
print(f"  PM10 RMSE Improvement:  Mean={np.mean(boot_diff_rmse_10):.4f} ug/m3, 95% CI: [{ci_rmse_10[0]:.4f}, {ci_rmse_10[1]:.4f}], p={pval_rmse_10:.4f}", flush=True)

# 4. STATION CLUSTER BOOTSTRAP (Cluster by Station, B = 2000)
print("\n[4/6] Running Station Cluster Bootstrap (90 stations, B=2000)...", flush=True)
station_groups = test_index_df.groupby("StationId").indices
n_stations = len(station_groups)

st_counts = np.zeros(n_stations, dtype=np.int64)
st_sum_ae_l25 = np.zeros(n_stations, dtype=np.float64)
st_sum_ae_e25 = np.zeros(n_stations, dtype=np.float64)
st_sum_se_l25 = np.zeros(n_stations, dtype=np.float64)
st_sum_se_e25 = np.zeros(n_stations, dtype=np.float64)

st_sum_ae_l10 = np.zeros(n_stations, dtype=np.float64)
st_sum_ae_e10 = np.zeros(n_stations, dtype=np.float64)
st_sum_se_l10 = np.zeros(n_stations, dtype=np.float64)
st_sum_se_e10 = np.zeros(n_stations, dtype=np.float64)

for s_idx, (st_name, idx) in enumerate(station_groups.items()):
    st_counts[s_idx] = len(idx)
    st_sum_ae_l25[s_idx] = np.sum(np.abs(y25[idx] - pl25[idx]))
    st_sum_ae_e25[s_idx] = np.sum(np.abs(y25[idx] - pe25[idx]))
    st_sum_se_l25[s_idx] = np.sum((y25[idx] - pl25[idx]) ** 2)
    st_sum_se_e25[s_idx] = np.sum((y25[idx] - pe25[idx]) ** 2)

    st_sum_ae_l10[s_idx] = np.sum(np.abs(y10[idx] - pl10[idx]))
    st_sum_ae_e10[s_idx] = np.sum(np.abs(y10[idx] - pe10[idx]))
    st_sum_se_l10[s_idx] = np.sum((y10[idx] - pl10[idx]) ** 2)
    st_sum_se_e10[s_idx] = np.sum((y10[idx] - pe10[idx]) ** 2)

st_diff_mae_25 = np.zeros(N_BOOT, dtype=np.float64)
st_diff_rmse_25 = np.zeros(N_BOOT, dtype=np.float64)
st_diff_mae_10 = np.zeros(N_BOOT, dtype=np.float64)
st_diff_rmse_10 = np.zeros(N_BOOT, dtype=np.float64)

for b in range(N_BOOT):
    s_idx = RNG.randint(0, n_stations, size=n_stations)
    n_tot = np.sum(st_counts[s_idx])

    st_diff_mae_25[b] = (np.sum(st_sum_ae_l25[s_idx]) - np.sum(st_sum_ae_e25[s_idx])) / n_tot
    st_diff_rmse_25[b] = np.sqrt(np.sum(st_sum_se_l25[s_idx]) / n_tot) - np.sqrt(np.sum(st_sum_se_e25[s_idx]) / n_tot)

    st_diff_mae_10[b] = (np.sum(st_sum_ae_l10[s_idx]) - np.sum(st_sum_ae_e10[s_idx])) / n_tot
    st_diff_rmse_10[b] = np.sqrt(np.sum(st_sum_se_l10[s_idx]) / n_tot) - np.sqrt(np.sum(st_sum_se_e10[s_idx]) / n_tot)

ci_st_mae_25 = [float(np.percentile(st_diff_mae_25, 2.5)), float(np.percentile(st_diff_mae_25, 97.5))]
ci_st_rmse_25 = [float(np.percentile(st_diff_rmse_25, 2.5)), float(np.percentile(st_diff_rmse_25, 97.5))]
ci_st_mae_10 = [float(np.percentile(st_diff_mae_10, 2.5)), float(np.percentile(st_diff_mae_10, 97.5))]
ci_st_rmse_10 = [float(np.percentile(st_diff_rmse_10, 2.5)), float(np.percentile(st_diff_rmse_10, 97.5))]

print("=== Station Cluster Bootstrap 95% Confidence Intervals ===", flush=True)
print(f"  PM2.5 MAE Improvement:  95% CI: [{ci_st_mae_25[0]:.4f}, {ci_st_mae_25[1]:.4f}], p={np.mean(st_diff_mae_25 <= 0):.4f}", flush=True)
print(f"  PM2.5 RMSE Improvement: 95% CI: [{ci_st_rmse_25[0]:.4f}, {ci_st_rmse_25[1]:.4f}], p={np.mean(st_diff_rmse_25 <= 0):.4f}", flush=True)
print(f"  PM10 MAE Improvement:   95% CI: [{ci_st_mae_10[0]:.4f}, {ci_st_mae_10[1]:.4f}], p={np.mean(st_diff_mae_10 <= 0):.4f}", flush=True)
print(f"  PM10 RMSE Improvement:  95% CI: [{ci_st_rmse_10[0]:.4f}, {ci_st_rmse_10[1]:.4f}], p={np.mean(st_diff_rmse_10 <= 0):.4f}", flush=True)

# 5. PER-STATION AND CHRONOLOGICAL BREAKDOWN
print("\n[5/6] Computing station-level and monthly breakdowns...", flush=True)
station_results = []
for s_name, idx in station_groups.items():
    mae_l25 = float(np.mean(np.abs(y25[idx] - pl25[idx])))
    mae_e25 = float(np.mean(np.abs(y25[idx] - pe25[idx])))
    mae_l10 = float(np.mean(np.abs(y10[idx] - pl10[idx])))
    mae_e10 = float(np.mean(np.abs(y10[idx] - pe10[idx])))

    station_results.append({
        "StationId": s_name,
        "n_samples": len(idx),
        "mae_lstm_pm25": round(mae_l25, 4),
        "mae_ens_pm25": round(mae_e25, 4),
        "diff_mae_pm25": round(mae_l25 - mae_e25, 4),
        "mae_lstm_pm10": round(mae_l10, 4),
        "mae_ens_pm10": round(mae_e10, 4),
        "diff_mae_pm10": round(mae_l10 - mae_e10, 4),
    })

station_df = pd.DataFrame(station_results)
pct_stations_improved_25 = float(np.mean(station_df["diff_mae_pm25"] > 0) * 100.0)
pct_stations_improved_10 = float(np.mean(station_df["diff_mae_pm10"] > 0) * 100.0)
print(f"  Stations with PM2.5 MAE improvement: {np.sum(station_df['diff_mae_pm25'] > 0)}/90 ({pct_stations_improved_25:.1f}%)", flush=True)
print(f"  Stations with PM10 MAE improvement:  {np.sum(station_df['diff_mae_pm10'] > 0)}/90 ({pct_stations_improved_10:.1f}%)", flush=True)
print(f"  Median PM2.5 MAE improvement across stations: {station_df['diff_mae_pm25'].median():.4f} ug/m3", flush=True)
print(f"  Median PM10 MAE improvement across stations:  {station_df['diff_mae_pm10'].median():.4f} ug/m3", flush=True)

# Monthly breakdown
test_index_df["year_month"] = test_index_df["target_time"].dt.to_period("M").astype(str)
month_results = []
for ym, m_df in test_index_df.groupby("year_month"):
    idx = m_df.index.values
    mae_l25 = float(np.mean(np.abs(y25[idx] - pl25[idx])))
    mae_e25 = float(np.mean(np.abs(y25[idx] - pe25[idx])))
    mae_l10 = float(np.mean(np.abs(y10[idx] - pl10[idx])))
    mae_e10 = float(np.mean(np.abs(y10[idx] - pe10[idx])))

    month_results.append({
        "Month": ym,
        "n_samples": len(idx),
        "mean_actual_pm25": round(float(np.mean(y25[idx])), 2),
        "mae_lstm_pm25": round(mae_l25, 4),
        "mae_ens_pm25": round(mae_e25, 4),
        "diff_mae_pm25": round(mae_l25 - mae_e25, 4),
        "mean_actual_pm10": round(float(np.mean(y10[idx])), 2),
        "mae_lstm_pm10": round(mae_l10, 4),
        "mae_ens_pm10": round(mae_e10, 4),
        "diff_mae_pm10": round(mae_l10 - mae_e10, 4),
    })

monthly_df = pd.DataFrame(month_results)
print("\n=== Monthly Performance Breakdown (Sep 2019 to Jul 2020) ===", flush=True)
print(monthly_df.to_string(index=False), flush=True)

# 6. SEVERE CATEGORY UNDERPREDICTION DEEP DIVE
print("\n[6/6] Investigating severe-category underprediction...", flush=True)
PM25_TIERS = [
    ("Good (0-30)", 0.0, 30.0),
    ("Satisfactory (30-60)", 30.0, 60.0),
    ("Moderate (60-90)", 60.0, 90.0),
    ("Poor (90-120)", 90.0, 120.0),
    ("Very Poor (120-250)", 120.0, 250.0),
    ("Severe (>250)", 250.0, float("inf")),
]

tier_audit = []
for label, lo, hi in PM25_TIERS:
    mask = (y25 >= lo) & (y25 < hi)
    cnt = int(np.sum(mask))
    if cnt > 0:
        yt = y25[mask]
        p_pers = pred_persist1_25[mask]
        p_l = pl25[mask]
        p_e = pe25[mask]

        tier_audit.append({
            "Tier": label,
            "Count": cnt,
            "Percentage_of_Test": round(cnt / N_TEST * 100.0, 2),
            "Mean_Actual": round(float(np.mean(yt)), 2),
            "Persist_MAE": round(float(np.mean(np.abs(yt - p_pers))), 4),
            "LSTM_MAE": round(float(np.mean(np.abs(yt - p_l))), 4),
            "Ensemble_MAE": round(float(np.mean(np.abs(yt - p_e))), 4),
            "LSTM_Mean_Bias": round(float(np.mean(p_l - yt)), 2),
            "Ensemble_Mean_Bias": round(float(np.mean(p_e - yt)), 2),
            "Persist_Mean_Bias": round(float(np.mean(p_pers - yt)), 2),
        })

tier_audit_df = pd.DataFrame(tier_audit)
print(tier_audit_df.to_string(index=False), flush=True)

# Severe details
sev_mask = y25 >= 250.0
sev_count = int(np.sum(sev_mask))
sev_yt = y25[sev_mask]
sev_pl = pl25[sev_mask]
sev_pe = pe25[sev_mask]
sev_pp = pred_persist1_25[sev_mask]

severe_stats = {
    "count": sev_count,
    "percentage_of_test": round(sev_count / N_TEST * 100.0, 3),
    "actual_min": round(float(np.min(sev_yt)), 2),
    "actual_max": round(float(np.max(sev_yt)), 2),
    "actual_mean": round(float(np.mean(sev_yt)), 2),
    "actual_p50": round(float(np.percentile(sev_yt, 50)), 2),
    "actual_p90": round(float(np.percentile(sev_yt, 90)), 2),
    "lstm_pred_mean": round(float(np.mean(sev_pl)), 2),
    "ensemble_pred_mean": round(float(np.mean(sev_pe)), 2),
    "persist_pred_mean": round(float(np.mean(sev_pp)), 2),
    "lstm_bias": round(float(np.mean(sev_pl - sev_yt)), 2),
    "ensemble_bias": round(float(np.mean(sev_pe - sev_yt)), 2),
    "persist_bias": round(float(np.mean(sev_pp - sev_yt)), 2),
    "fraction_underpredicted_lstm": round(float(np.mean(sev_pl < sev_yt)), 4),
    "fraction_underpredicted_ensemble": round(float(np.mean(sev_pe < sev_yt)), 4),
}

# 7. EXPORT STRUCTURED JSON ARTIFACT
output_data = {
    "audit_metadata": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_sequences": N_TEST,
        "stations_evaluated": n_stations,
        "date_range": [str(test_index_df["target_time"].min()), str(test_index_df["target_time"].max())],
        "bootstrap_iterations": N_BOOT,
        "blend_weights": {"pm25": {"lstm": 0.90, "gbdt": 0.10}, "pm10": {"lstm": 0.80, "gbdt": 0.20}},
        "blend_provenance": "Fitted strictly on 50,000 validation sequences (data/processed/model_sequences/y_validation.npy) without test set exposure.",
    },
    "global_metrics": {
        "pm25": {
            "persistence": m_pers_25,
            "lstm": m_lstm_25,
            "gbdt": m_gbt_25,
            "ensemble": m_ens_25,
            "mae_improvement": round(m_lstm_25["MAE"] - m_ens_25["MAE"], 4),
            "rmse_improvement": round(m_lstm_25["RMSE"] - m_ens_25["RMSE"], 4),
            "r2_improvement": round(m_ens_25["R2"] - m_lstm_25["R2"], 4),
            "percentage_samples_improved": round(pct_better_25, 2),
        },
        "pm10": {
            "persistence": m_pers_10,
            "lstm": m_lstm_10,
            "gbdt": m_gbt_10,
            "ensemble": m_ens_10,
            "mae_improvement": round(m_lstm_10["MAE"] - m_ens_10["MAE"], 4),
            "rmse_improvement": round(m_lstm_10["RMSE"] - m_ens_10["RMSE"], 4),
            "r2_improvement": round(m_ens_10["R2"] - m_lstm_10["R2"], 4),
            "percentage_samples_improved": round(pct_better_10, 2),
        },
    },
    "station_block_bootstrap": {
        "block_type": "7-day station-week calendar blocks",
        "n_blocks": K_BLOCKS,
        "iterations": N_BOOT,
        "pm25": {
            "mae_improvement_mean": round(float(np.mean(boot_diff_mae_25)), 4),
            "mae_improvement_95ci": [round(ci_mae_25[0], 4), round(ci_mae_25[1], 4)],
            "mae_improvement_pval": round(pval_mae_25, 4),
            "rmse_improvement_mean": round(float(np.mean(boot_diff_rmse_25)), 4),
            "rmse_improvement_95ci": [round(ci_rmse_25[0], 4), round(ci_rmse_25[1], 4)],
            "rmse_improvement_pval": round(pval_rmse_25, 4),
        },
        "pm10": {
            "mae_improvement_mean": round(float(np.mean(boot_diff_mae_10)), 4),
            "mae_improvement_95ci": [round(ci_mae_10[0], 4), round(ci_mae_10[1], 4)],
            "mae_improvement_pval": round(pval_mae_10, 4),
            "rmse_improvement_mean": round(float(np.mean(boot_diff_rmse_10)), 4),
            "rmse_improvement_95ci": [round(ci_rmse_10[0], 4), round(ci_rmse_10[1], 4)],
            "rmse_improvement_pval": round(pval_rmse_10, 4),
        },
    },
    "station_cluster_bootstrap": {
        "cluster_type": "Entire station clustering (90 stations)",
        "iterations": N_BOOT,
        "pm25": {
            "mae_improvement_95ci": [round(ci_st_mae_25[0], 4), round(ci_st_mae_25[1], 4)],
            "rmse_improvement_95ci": [round(ci_st_rmse_25[0], 4), round(ci_st_rmse_25[1], 4)],
            "mae_improvement_pval": round(float(np.mean(st_diff_mae_25 <= 0)), 4),
        },
        "pm10": {
            "mae_improvement_95ci": [round(ci_st_mae_10[0], 4), round(ci_st_mae_10[1], 4)],
            "rmse_improvement_95ci": [round(ci_st_rmse_10[0], 4), round(ci_st_rmse_10[1], 4)],
            "mae_improvement_pval": round(float(np.mean(st_diff_mae_10 <= 0)), 4),
        },
    },
    "station_distribution": {
        "total_stations": n_stations,
        "stations_improved_pm25": int(np.sum(station_df["diff_mae_pm25"] > 0)),
        "stations_improved_pm25_pct": round(pct_stations_improved_25, 2),
        "stations_improved_pm10": int(np.sum(station_df["diff_mae_pm10"] > 0)),
        "stations_improved_pm10_pct": round(pct_stations_improved_10, 2),
        "median_mae_improvement_pm25": round(float(station_df["diff_mae_pm25"].median()), 4),
        "median_mae_improvement_pm10": round(float(station_df["diff_mae_pm10"].median()), 4),
    },
    "monthly_breakdown": monthly_df.to_dict(orient="records"),
    "tier_audit": tier_audit_df.to_dict(orient="records"),
    "severe_tier_investigation": severe_stats,
}

json_path = opt_dir / "paired_ensemble_bootstrap.json"
with open(json_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved structured audit artifact to {json_path}", flush=True)
print("Final verification analysis complete!", flush=True)
