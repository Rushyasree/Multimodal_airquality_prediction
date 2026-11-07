"""
scripts/analyze_severe_tier.py

Deep investigation of extreme air-pollution performance:
- PM2.5 > 250 ug/m3 (CPCB Severe tier)
- PM10 > 430 ug/m3 (CPCB Severe tier) & PM10 > 250 ug/m3 (CPCB Poor/Very Poor/Severe)
- Metrics: Sample count, Actual mean/std/p50/p90/max, Persist MAE/RMSE/Bias,
  LSTM MAE/RMSE/Bias, GBDT MAE/RMSE/Bias, Ensemble MAE/RMSE/Bias, Underprediction Rate.
- Station-by-station severe pollution analysis.
- Month-by-month severe pollution analysis.
- Exports experiments/accuracy_optimization_v2/severe_tier_analysis.json.
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
print("DEEP SEVERE-POLLUTION PERFORMANCE INVESTIGATION", flush=True)
print("=" * 75, flush=True)

# 1. Load Data
seq_index_df = pd.read_csv(ROOT_DIR / "data/processed/sequences/air_quality_24h_sequence_index.csv")
test_index_df = seq_index_df[seq_index_df["split"] == "test"].reset_index(drop=True)
test_index_df["target_time"] = pd.to_datetime(test_index_df["target_time"])
test_index_df["year_month"] = test_index_df["target_time"].dt.to_period("M").astype(str)

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

N_TEST = len(y_test)
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

pred_ens_25 = 0.90 * pred_lstm_test[:, 0] + 0.10 * pred_gbt_25
pred_ens_10 = 0.80 * pred_lstm_test[:, 1] + 0.20 * pred_gbt_10

y25 = y_test[:, 0]
y10 = y_test[:, 1]
pl25 = pred_lstm_test[:, 0]
pl10 = pred_lstm_test[:, 1]
pe25 = pred_ens_25
pe10 = pred_ens_10
pp25 = pred_persist1_25
pp10 = pred_persist1_10
pg25 = pred_gbt_25
pg10 = pred_gbt_10


def analyze_cohort(name, mask, yt, p_pers, p_lstm, p_gbdt, p_ens):
    cnt = int(np.sum(mask))
    if cnt == 0:
        return {}
    sub_yt = yt[mask]
    sub_pp = p_pers[mask]
    sub_pl = p_lstm[mask]
    sub_pg = p_gbdt[mask]
    sub_pe = p_ens[mask]

    def m_dict(y, p):
        return {
            "MAE": round(float(np.mean(np.abs(y - p))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
            "Mean_Bias": round(float(np.mean(p - y)), 4),
            "Underpred_Rate_Pct": round(float(np.mean(p < y) * 100.0), 2),
            "Mean_Pred": round(float(np.mean(p)), 2),
            "Pred_P50": round(float(np.percentile(p, 50)), 2),
            "Pred_P90": round(float(np.percentile(p, 90)), 2),
        }

    return {
        "cohort_name": name,
        "count": cnt,
        "percentage_of_total_test": round(cnt / len(yt) * 100.0, 3),
        "actual_stats": {
            "mean": round(float(np.mean(sub_yt)), 2),
            "std": round(float(np.std(sub_yt)), 2),
            "min": round(float(np.min(sub_yt)), 2),
            "p50": round(float(np.percentile(sub_yt, 50)), 2),
            "p90": round(float(np.percentile(sub_yt, 90)), 2),
            "max": round(float(np.max(sub_yt)), 2),
        },
        "persistence": m_dict(sub_yt, sub_pp),
        "lstm": m_dict(sub_yt, sub_pl),
        "gbdt": m_dict(sub_yt, sub_pg),
        "ensemble": m_dict(sub_yt, sub_pe),
    }


# Evaluate cohorts
pm25_severe_mask = y25 >= 250.0
pm10_severe_mask = y10 >= 430.0
pm10_elevated_mask = y10 >= 250.0

cohort_pm25_sev = analyze_cohort("PM2.5 Severe (>250 ug/m3)", pm25_severe_mask, y25, pp25, pl25, pg25, pe25)
cohort_pm10_sev = analyze_cohort("PM10 Severe (>430 ug/m3)", pm10_severe_mask, y10, pp10, pl10, pg10, pe10)
cohort_pm10_ele = analyze_cohort("PM10 Elevated (>250 ug/m3)", pm10_elevated_mask, y10, pp10, pl10, pg10, pe10)

print("\n--- Summary of Severe Pollution Performance ---", flush=True)
for c in [cohort_pm25_sev, cohort_pm10_sev, cohort_pm10_ele]:
    print(f"\nCohort: {c['cohort_name']}")
    print(f"  Count: {c['count']} ({c['percentage_of_total_test']}%) | Actual Mean: {c['actual_stats']['mean']} ug/m3 (Max: {c['actual_stats']['max']})")
    print(f"  1h Persist: MAE={c['persistence']['MAE']}, RMSE={c['persistence']['RMSE']}, Bias={c['persistence']['Mean_Bias']}, Underpred={c['persistence']['Underpred_Rate_Pct']}%")
    print(f"  Pretrained: MAE={c['lstm']['MAE']}, RMSE={c['lstm']['RMSE']}, Bias={c['lstm']['Mean_Bias']}, Underpred={c['lstm']['Underpred_Rate_Pct']}%")
    print(f"  GBDT:       MAE={c['gbdt']['MAE']}, RMSE={c['gbdt']['RMSE']}, Bias={c['gbdt']['Mean_Bias']}, Underpred={c['gbdt']['Underpred_Rate_Pct']}%")
    print(f"  Ensemble:   MAE={c['ensemble']['MAE']}, RMSE={c['ensemble']['RMSE']}, Bias={c['ensemble']['Mean_Bias']}, Underpred={c['ensemble']['Underpred_Rate_Pct']}%")

# Station-level breakdown for PM2.5 Severe
station_severe_records = []
for st_id, s_df in test_index_df.groupby("StationId"):
    s_idx = s_df.index.values
    s_mask = pm25_severe_mask[s_idx]
    s_cnt = int(np.sum(s_mask))
    if s_cnt > 0:
        yt_sub = y25[s_idx][s_mask]
        pl_sub = pl25[s_idx][s_mask]
        pe_sub = pe25[s_idx][s_mask]
        pp_sub = pp25[s_idx][s_mask]
        station_severe_records.append({
            "StationId": st_id,
            "severe_count": s_cnt,
            "severe_percentage_of_station": round(s_cnt / len(s_idx) * 100.0, 2),
            "actual_mean": round(float(np.mean(yt_sub)), 2),
            "persist_mae": round(float(np.mean(np.abs(yt_sub - pp_sub))), 2),
            "lstm_mae": round(float(np.mean(np.abs(yt_sub - pl_sub))), 2),
            "ensemble_mae": round(float(np.mean(np.abs(yt_sub - pe_sub))), 2),
            "lstm_bias": round(float(np.mean(pl_sub - yt_sub)), 2),
            "ensemble_bias": round(float(np.mean(pe_sub - yt_sub)), 2),
        })

station_sev_df = pd.DataFrame(station_severe_records).sort_values("severe_count", ascending=False)
print(f"\nIdentified {len(station_sev_df)} stations with severe PM2.5 observations.", flush=True)
print("Top 10 severe-pollution stations:")
print(station_sev_df.head(10).to_string(index=False), flush=True)

# Monthly breakdown for PM2.5 Severe
month_severe_records = []
for ym, m_df in test_index_df.groupby("year_month"):
    m_idx = m_df.index.values
    m_mask = pm25_severe_mask[m_idx]
    m_cnt = int(np.sum(m_mask))
    if m_cnt > 0:
        yt_sub = y25[m_idx][m_mask]
        pl_sub = pl25[m_idx][m_mask]
        pe_sub = pe25[m_idx][m_mask]
        pp_sub = pp25[m_idx][m_mask]
        month_severe_records.append({
            "Month": ym,
            "severe_count": m_cnt,
            "severe_percentage_of_month": round(m_cnt / len(m_idx) * 100.0, 2),
            "actual_mean": round(float(np.mean(yt_sub)), 2),
            "persist_mae": round(float(np.mean(np.abs(yt_sub - pp_sub))), 2),
            "lstm_mae": round(float(np.mean(np.abs(yt_sub - pl_sub))), 2),
            "ensemble_mae": round(float(np.mean(np.abs(yt_sub - pe_sub))), 2),
            "lstm_bias": round(float(np.mean(pl_sub - yt_sub)), 2),
            "ensemble_bias": round(float(np.mean(pe_sub - yt_sub)), 2),
        })

month_sev_df = pd.DataFrame(month_severe_records)
print("\nMonthly severe PM2.5 distribution:")
print(month_sev_df.to_string(index=False), flush=True)

# Station-level breakdown for PM10 Severe (>430 ug/m3)
station_pm10_sev_records = []
for st_id, s_df in test_index_df.groupby("StationId"):
    s_idx = s_df.index.values
    s_mask = pm10_severe_mask[s_idx]
    s_cnt = int(np.sum(s_mask))
    if s_cnt > 0:
        yt_sub = y10[s_idx][s_mask]
        pl_sub = pl10[s_idx][s_mask]
        pe_sub = pe10[s_idx][s_mask]
        pp_sub = pp10[s_idx][s_mask]
        station_pm10_sev_records.append({
            "StationId": st_id,
            "severe_count": s_cnt,
            "actual_mean": round(float(np.mean(yt_sub)), 2),
            "persist_mae": round(float(np.mean(np.abs(yt_sub - pp_sub))), 2),
            "lstm_mae": round(float(np.mean(np.abs(yt_sub - pl_sub))), 2),
            "ensemble_mae": round(float(np.mean(np.abs(yt_sub - pe_sub))), 2),
            "lstm_bias": round(float(np.mean(pl_sub - yt_sub)), 2),
            "ensemble_bias": round(float(np.mean(pe_sub - yt_sub)), 2),
        })

station_pm10_sev_df = pd.DataFrame(station_pm10_sev_records).sort_values("severe_count", ascending=False)

# Monthly breakdown for PM10 Severe (>430 ug/m3)
month_pm10_sev_records = []
for ym, m_df in test_index_df.groupby("year_month"):
    m_idx = m_df.index.values
    m_mask = pm10_severe_mask[m_idx]
    m_cnt = int(np.sum(m_mask))
    if m_cnt > 0:
        yt_sub = y10[m_idx][m_mask]
        pl_sub = pl10[m_idx][m_mask]
        pe_sub = pe10[m_idx][m_mask]
        pp_sub = pp10[m_idx][m_mask]
        month_pm10_sev_records.append({
            "Month": ym,
            "severe_count": m_cnt,
            "actual_mean": round(float(np.mean(yt_sub)), 2),
            "persist_mae": round(float(np.mean(np.abs(yt_sub - pp_sub))), 2),
            "lstm_mae": round(float(np.mean(np.abs(yt_sub - pl_sub))), 2),
            "ensemble_mae": round(float(np.mean(np.abs(yt_sub - pe_sub))), 2),
            "lstm_bias": round(float(np.mean(pl_sub - yt_sub)), 2),
            "ensemble_bias": round(float(np.mean(pe_sub - yt_sub)), 2),
        })

month_pm10_sev_df = pd.DataFrame(month_pm10_sev_records)

severe_analysis_output = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_sequences": N_TEST,
        "criteria": {
            "cpcb_pm25_severe_threshold": 250.0,
            "cpcb_pm10_severe_threshold": 430.0,
            "cpcb_pm10_elevated_threshold": 250.0,
        },
    },
    "cohort_pm25_severe": cohort_pm25_sev,
    "cohort_pm10_severe": cohort_pm10_sev,
    "cohort_pm10_elevated": cohort_pm10_ele,
    "pm25_severe_by_station": station_sev_df.to_dict(orient="records"),
    "pm25_severe_by_month": month_sev_df.to_dict(orient="records"),
    "pm10_severe_by_station": station_pm10_sev_df.to_dict(orient="records"),
    "pm10_severe_by_month": month_pm10_sev_df.to_dict(orient="records"),
}

sev_json_path = opt_dir / "severe_tier_analysis.json"
with open(sev_json_path, "w") as f:
    json.dump(severe_analysis_output, f, indent=2)

print(f"\nSaved severe analysis to {sev_json_path}", flush=True)
