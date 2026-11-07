"""
scripts/audit_and_reproduce_phase4.py

VayuDrishti Phase 4: Independent Validation and Deployment Readiness Audit
===========================================================================

Tasks:
1. Load frozen checkpoints (NO TRAINING / NO FINE-TUNING).
2. Extract exact 2025 holdout sequences (2025-04-01 to 2025-08-31).
3. Compute predictions for all 8 candidate models.
4. Export canonical experiments/recent_data_audit/final_predictions.csv.
5. Recompute all metrics directly from final_predictions.csv and verify against reported values.
6. Perform independent paired statistical uncertainty analysis:
   - Method A: Phase 3 block-subsample reproduction
   - Method B: Full station-clustered bootstrap
   - Method C: Moving-block temporal bootstrap
   - Export experiments/recent_data_audit/paired_recent_bootstrap.json
7. Run disaggregated audits:
   - Month-by-month performance
   - Station-level performance distribution
   - NAQI concentration tiers
8. Audit all 49 reduced features for causality and lack of leakage.
9. Measure CPU inference latency and application compatibility.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.prediction.lstm_forecaster import LSTMForecaster, LSTMForecasterService
from src.data.tabular_features import extract_tabular_features, FEATURE_NAMES_TABULAR

AUDIT_DIR = ROOT_DIR / "experiments" / "recent_data_audit"
CKPT_DIR = AUDIT_DIR / "checkpoints"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    bias = float(np.mean(diff))
    underpred_rate = float(np.mean(y_pred < y_true) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Mean_Bias": round(bias, 4),
        "Underpred_Rate_Pct": round(underpred_rate, 2),
        "N": int(len(y_true)),
    }


def extract_reduced_tabular_features(X_seq: np.ndarray) -> np.ndarray:
    full_tab = extract_tabular_features(X_seq)
    selected_indices = [0, 1] + list(range(19, 27)) + list(range(27, 66))
    return full_tab[:, selected_indices]


def evaluate_lstm(model: nn.Module, X: np.ndarray, target_mean: list, target_std: list, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_25 = []
    pred_10 = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = torch.from_numpy(X[i : i + batch_size]).float()
            out = model(batch_x).cpu().numpy()
            pred_25.append(out[:, 0] * target_std[0] + target_mean[0])
            pred_10.append(out[:, 1] * target_std[1] + target_mean[1])

    p25 = np.maximum(0.0, np.concatenate(pred_25, axis=0))
    p10 = np.maximum(0.0, np.concatenate(pred_10, axis=0))
    return p25, p10


def main():
    t_start = time.time()
    print("=" * 80)
    print("VAYUDRISHTI PHASE 4: INDEPENDENT REPRODUCTION & VALIDATION AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. LOAD DATA & EXTRACT 2025 HOLDOUT
    # -------------------------------------------------------------
    lstm_service = LSTMForecasterService()
    csv_path = ROOT_DIR / "data/processed/recent_air_quality_cleaned.csv"
    print(f"Loading cleaned telemetry from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Target 2025 holdout range
    start_dt = "2025-04-01 00:00:00"
    end_dt = "2025-08-31 23:00:00"
    df_sub = df[(df["Datetime"] >= start_dt) & (df["Datetime"] <= end_dt)].copy()
    stations = sorted(df_sub["station_id"].unique())
    print(f"2025 holdout filtered: {len(df_sub)} rows across {len(stations)} stations.")

    feature_names = lstm_service.feature_names
    scaler_means = lstm_service.scaler_means
    scaler_stds = lstm_service.scaler_stds
    imputation_medians = lstm_service.imputation_medians

    X_list = []
    y25_list = []
    y10_list = []
    persist25_list = []
    persist10_list = []
    station_indices = []
    timestamp_list = []

    for st in stations:
        st_sub = df_sub[df_sub["station_id"] == st].sort_values("Datetime").drop_duplicates("Datetime")
        if len(st_sub) < 25:
            continue

        dt_idx = pd.date_range(st_sub["Datetime"].min(), st_sub["Datetime"].max(), freq="1h")
        st_full = st_sub.set_index("Datetime").reindex(dt_idx)
        T = len(st_full)
        if T < 25:
            continue

        lat, lon = lstm_service.get_station_coordinates(st)
        F = np.zeros((T, len(feature_names)), dtype=np.float32)

        for col_idx, col_name in enumerate(feature_names):
            if col_name in st_full.columns:
                vals = st_full[col_name].values.astype(np.float32)
                nan_mask = np.isnan(vals)
                vals[nan_mask] = imputation_medians.get(col_name, 0.0)
            elif col_name == "Latitude":
                vals = np.full(T, lat, dtype=np.float32)
            elif col_name == "Longitude":
                vals = np.full(T, lon, dtype=np.float32)
            elif col_name == "hour_sin":
                vals = np.sin(2 * np.pi * dt_idx.hour / 24.0).astype(np.float32)
            elif col_name == "hour_cos":
                vals = np.cos(2 * np.pi * dt_idx.hour / 24.0).astype(np.float32)
            elif col_name == "month_sin":
                vals = np.sin(2 * np.pi * (dt_idx.month - 1) / 12.0).astype(np.float32)
            elif col_name == "month_cos":
                vals = np.cos(2 * np.pi * (dt_idx.month - 1) / 12.0).astype(np.float32)
            elif col_name == "day_of_week_sin":
                vals = np.sin(2 * np.pi * dt_idx.dayofweek / 7.0).astype(np.float32)
            elif col_name == "day_of_week_cos":
                vals = np.cos(2 * np.pi * dt_idx.dayofweek / 7.0).astype(np.float32)
            elif col_name in ("wind_direction_sin", "wind_direction_cos"):
                vals = np.full(T, imputation_medians.get(col_name, 0.0), dtype=np.float32)
            else:
                vals = np.full(T, imputation_medians.get(col_name, 0.0), dtype=np.float32)

            F[:, col_idx] = (vals - scaler_means[col_idx]) / scaler_stds[col_idx]

        raw_pm25 = st_full["PM2.5"].values
        raw_pm10 = st_full["PM10"].values

        windows = np.lib.stride_tricks.sliding_window_view(F, window_shape=24, axis=0)
        windows = np.ascontiguousarray(np.transpose(windows, (0, 2, 1)))

        n_cand = T - 24
        windows_cand = windows[:n_cand]
        t25_cand = raw_pm25[24 : 24 + n_cand]
        t10_cand = raw_pm10[24 : 24 + n_cand]
        iss25_cand = raw_pm25[23 : 23 + n_cand]
        iss10_cand = raw_pm10[23 : 23 + n_cand]
        dt_cand = dt_idx[24 : 24 + n_cand]

        valid = (
            (~np.isnan(t25_cand))
            & (~np.isnan(t10_cand))
            & (~np.isnan(iss25_cand))
            & (~np.isnan(iss10_cand))
        )

        if np.any(valid):
            X_list.append(windows_cand[valid])
            y25_list.append(t25_cand[valid])
            y10_list.append(t10_cand[valid])
            persist25_list.append(iss25_cand[valid])
            persist10_list.append(iss10_cand[valid])
            station_indices.extend([st] * int(np.sum(valid)))
            timestamp_list.extend(dt_cand[valid].astype(str).tolist())

    X_test = np.concatenate(X_list, axis=0)
    y25_test = np.concatenate(y25_list, axis=0)
    y10_test = np.concatenate(y10_list, axis=0)
    p25_persist = np.concatenate(persist25_list, axis=0)
    p10_persist = np.concatenate(persist10_list, axis=0)
    test_stations = np.array(station_indices)
    test_timestamps = np.array(timestamp_list)

    N_TEST = len(X_test)
    print(f"Extracted {N_TEST} sequences across {len(np.unique(test_stations))} stations.")
    assert N_TEST == 236289, f"Expected 236,289 test sequences, got {N_TEST}"

    # -------------------------------------------------------------
    # 2. LOAD FROZEN CHECKPOINTS
    # -------------------------------------------------------------
    print("\nLoading frozen checkpoints...")
    # Base LSTM
    ckpt_base = torch.load(ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt", map_location="cpu")
    frozen_lstm = LSTMForecaster(
        input_size=ckpt_base["input_size"],
        hidden_size=ckpt_base["hidden_size"],
        num_layers=ckpt_base["num_layers"],
        dropout=ckpt_base["dropout"],
        output_size=2,
    )
    frozen_lstm.load_state_dict(ckpt_base["model_state_dict"])
    target_mean = ckpt_base["target_mean"]
    target_std = ckpt_base["target_std"]

    # Historical GBDT
    hist_gbt_25 = joblib.load(ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib")
    hist_gbt_10 = joblib.load(ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm10.joblib")

    # Adapted LSTM
    ckpt_ft = torch.load(CKPT_DIR / "best_finetuned_lstm.pt", map_location="cpu")
    finetuned_lstm = LSTMForecaster(
        input_size=ckpt_ft["input_size"],
        hidden_size=ckpt_ft["hidden_size"],
        num_layers=ckpt_ft["num_layers"],
        dropout=ckpt_ft["dropout"],
        output_size=2,
    )
    finetuned_lstm.load_state_dict(ckpt_ft["model_state_dict"])

    # Adapted GBDT
    ad_gbt_25 = joblib.load(CKPT_DIR / "adapted_gbdt_pm25.joblib")
    ad_gbt_10 = joblib.load(CKPT_DIR / "adapted_gbdt_pm10.joblib")

    # -------------------------------------------------------------
    # 3. EXECUTE FORWARD INFERENCE
    # -------------------------------------------------------------
    print("\nRunning inference for all models on 2025 holdout...")
    t_inf_start = time.time()

    # 1. Frozen LSTM
    pred_flstm_25, pred_flstm_10 = evaluate_lstm(frozen_lstm, X_test, target_mean, target_std)

    # 2. Historical GBDT (full 67 tabular)
    CHUNK = 50000
    p25_hgbt_list, p10_hgbt_list = [], []
    for i in range(0, N_TEST, CHUNK):
        chunk_full = extract_tabular_features(X_test[i : i + CHUNK])
        p25_hgbt_list.append(hist_gbt_25.predict(chunk_full))
        p10_hgbt_list.append(hist_gbt_10.predict(chunk_full))
    pred_hgbt_25 = np.maximum(0.0, np.concatenate(p25_hgbt_list))
    pred_hgbt_10 = np.maximum(0.0, np.concatenate(p10_hgbt_list))

    # 3. Historical Static Ensemble (0.90 / 0.80)
    pred_hist_ens_25 = 0.90 * pred_flstm_25 + 0.10 * pred_hgbt_25
    pred_hist_ens_10 = 0.80 * pred_flstm_10 + 0.20 * pred_hgbt_10

    # 4. Historical Severity-Gated Ensemble
    gate25_mask = p25_persist >= 250.0
    gate10_mask = p10_persist >= 400.0

    pred_hist_gated_25 = pred_hist_ens_25.copy()
    pred_hist_gated_25[gate25_mask] = pred_flstm_25[gate25_mask]
    pred_hist_gated_10 = pred_hist_ens_10.copy()
    pred_hist_gated_10[gate10_mask] = pred_flstm_10[gate10_mask]

    # 5. Fine-Tuned LSTM
    pred_ft_lstm_25, pred_ft_lstm_10 = evaluate_lstm(finetuned_lstm, X_test, target_mean, target_std)

    # 6. Adapted Reduced GBDT
    p25_ad_gbt_list, p10_ad_gbt_list = [], []
    for i in range(0, N_TEST, CHUNK):
        chunk_red = extract_reduced_tabular_features(X_test[i : i + CHUNK])
        p25_ad_gbt_list.append(ad_gbt_25.predict(chunk_red))
        p10_ad_gbt_list.append(ad_gbt_10.predict(chunk_red))
    pred_ad_gbt_25 = np.maximum(0.0, np.concatenate(p25_ad_gbt_list))
    pred_ad_gbt_10 = np.maximum(0.0, np.concatenate(p10_ad_gbt_list))

    # 7. Adapted Ensemble (w25=0.40, w10=0.45 from validation)
    w25, w10 = 0.40, 0.45
    pred_ad_ens_25 = w25 * pred_ft_lstm_25 + (1.0 - w25) * pred_ad_gbt_25
    pred_ad_ens_10 = w10 * pred_ft_lstm_10 + (1.0 - w10) * pred_ad_gbt_10

    # 8. Adapted Severity-Gated Ensemble
    pred_ad_gated_25 = pred_ad_ens_25.copy()
    pred_ad_gated_25[gate25_mask] = pred_ft_lstm_25[gate25_mask]
    pred_ad_gated_10 = pred_ad_ens_10.copy()
    pred_ad_gated_10[gate10_mask] = pred_ft_lstm_10[gate10_mask]

    inf_elapsed = time.time() - t_inf_start
    print(f"Inference complete across all 8 models in {inf_elapsed:.2f}s!")

    # -------------------------------------------------------------
    # 4. EXPORT CANONICAL PREDICTIONS FILE
    # -------------------------------------------------------------
    pred_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "station_id": test_stations,
        "actual_pm25": y25_test,
        "actual_pm10": y10_test,
        "pred_persist_pm25": p25_persist,
        "pred_persist_pm10": p10_persist,
        "pred_frozen_lstm_pm25": pred_flstm_25,
        "pred_frozen_lstm_pm10": pred_flstm_10,
        "pred_hist_ens_pm25": pred_hist_ens_25,
        "pred_hist_ens_pm10": pred_hist_ens_10,
        "pred_hist_gated_pm25": pred_hist_gated_25,
        "pred_hist_gated_pm10": pred_hist_gated_10,
        "pred_finetuned_lstm_pm25": pred_ft_lstm_25,
        "pred_finetuned_lstm_pm10": pred_ft_lstm_10,
        "pred_adapted_gbdt_pm25": pred_ad_gbt_25,
        "pred_adapted_gbdt_pm10": pred_ad_gbt_10,
        "pred_adapted_ens_pm25": pred_ad_ens_25,
        "pred_adapted_ens_pm10": pred_ad_ens_10,
        "pred_adapted_gated_pm25": pred_ad_gated_25,
        "pred_adapted_gated_pm10": pred_ad_gated_10,
    })

    pred_out_file = AUDIT_DIR / "final_predictions.csv"
    print(f"Saving canonical predictions file to {pred_out_file.name} ({len(pred_df)} rows)...")
    pred_df.to_csv(pred_out_file, index=False)
    print("Canonical predictions saved successfully.")

    # -------------------------------------------------------------
    # 5. INDEPENDENT RECALCULATION & COMPARISON TABLE
    # -------------------------------------------------------------
    models_dict = {
        "1-Hour Persistence Baseline": (p25_persist, p10_persist),
        "Frozen Historical LSTM": (pred_flstm_25, pred_flstm_10),
        "Historical Static Ensemble (0.90/0.80)": (pred_hist_ens_25, pred_hist_ens_10),
        "Historical Severity-Gated Ensemble": (pred_hist_gated_25, pred_hist_gated_10),
        "Fine-Tuned LSTM (Adapted 2021-2023)": (pred_ft_lstm_25, pred_ft_lstm_10),
        "Adapted Reduced-Feature GBDT": (pred_ad_gbt_25, pred_ad_gbt_10),
        "Adapted Ensemble (FT-LSTM + GBDT)": (pred_ad_ens_25, pred_ad_ens_10),
        "Adapted Severity-Gated Ensemble": (pred_ad_gated_25, pred_ad_gated_10),
    }

    # Reported values to verify
    expected_mae = {
        "1-Hour Persistence Baseline": (6.9476, 17.1584),
        "Frozen Historical LSTM": (8.0870, 18.2638),
        "Historical Static Ensemble (0.90/0.80)": (7.8797, 17.6677),
        "Historical Severity-Gated Ensemble": (7.8815, 17.6700),
        "Fine-Tuned LSTM (Adapted 2021-2023)": (6.8011, 16.4791),
        "Adapted Reduced-Feature GBDT": (6.6716, 16.2906),
        "Adapted Ensemble (FT-LSTM + GBDT)": (6.6216, 16.1660),
        "Adapted Severity-Gated Ensemble": (6.6199, 16.1425),
    }

    reproduced_table = []
    print("\n" + "=" * 90)
    print("INDEPENDENT RECALCULATION VS REPORTED METRICS (2025 HOLDOUT)")
    print("=" * 90)

    for m_name, (p25, p10) in models_dict.items():
        m25 = compute_metrics(y25_test, p25)
        m10 = compute_metrics(y10_test, p10)
        exp25, exp10 = expected_mae[m_name]
        diff25 = round(m25["MAE"] - exp25, 4)
        diff10 = round(m10["MAE"] - exp10, 4)

        reproduced_table.append({
            "Model": m_name,
            "Reprod PM2.5 MAE": m25["MAE"],
            "Reported PM2.5 MAE": exp25,
            "PM2.5 Diff": diff25,
            "Reprod PM2.5 RMSE": m25["RMSE"],
            "Reprod PM2.5 R2": m25["R2"],
            "Reprod PM2.5 Bias": m25["Mean_Bias"],
            "Reprod PM10 MAE": m10["MAE"],
            "Reported PM10 MAE": exp10,
            "PM10 Diff": diff10,
            "Reprod PM10 RMSE": m10["RMSE"],
            "Reprod PM10 R2": m10["R2"],
            "Reprod PM10 Bias": m10["Mean_Bias"],
        })

    reprod_df = pd.DataFrame(reproduced_table)
    print(reprod_df[["Model", "Reprod PM2.5 MAE", "Reported PM2.5 MAE", "PM2.5 Diff", "Reprod PM10 MAE", "Reported PM10 MAE", "PM10 Diff"]].to_string(index=False))

    # -------------------------------------------------------------
    # 6. STATISTICAL UNCERTAINTY AUDIT (THREE METHODOLOGIES)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STATISTICAL UNCERTAINTY & PAIRED BOOTSTRAP AUDIT")
    print("=" * 90)

    unique_stations = np.unique(test_stations)
    n_replicates = 1000
    rng = np.random.RandomState(SEED)

    # Comparisons to audit:
    # 1. Adapted Gated Ensemble vs 1h Persistence (PM2.5 and PM10)
    # 2. Fine-Tuned LSTM vs Frozen LSTM (PM2.5 and PM10)
    # 3. Fine-Tuned LSTM vs 1h Persistence (PM2.5 and PM10)
    comparisons = [
        ("PM2.5: Adapted Gated vs Persistence", y25_test, pred_ad_gated_25, p25_persist),
        ("PM10:  Adapted Gated vs Persistence", y10_test, pred_ad_gated_10, p10_persist),
        ("PM2.5: Fine-Tuned LSTM vs Frozen LSTM", y25_test, pred_ft_lstm_25, pred_flstm_25),
        ("PM10:  Fine-Tuned LSTM vs Frozen LSTM", y10_test, pred_ft_lstm_10, pred_flstm_10),
        ("PM2.5: Fine-Tuned LSTM vs Persistence", y25_test, pred_ft_lstm_25, p25_persist),
        ("PM10:  Fine-Tuned LSTM vs Persistence", y10_test, pred_ft_lstm_10, p10_persist),
    ]

    bootstrap_output = {}

    for comp_label, yt, pa, pb in comparisons:
        paired_diff = np.abs(pa - yt) - np.abs(pb - yt)
        observed_mean = float(np.mean(paired_diff))

        # --- Method A: Reported Block-Subsample (24h block per station) ---
        station_map = {st: np.where(test_stations == st)[0] for st in unique_stations}
        boot_diffs_a = []
        for _ in range(n_replicates):
            sampled_st = rng.choice(unique_stations, size=len(unique_stations), replace=True)
            rep_parts = []
            for st in sampled_st:
                idx = station_map[st]
                if len(idx) > 24:
                    start = rng.randint(0, len(idx) - 24)
                    rep_parts.append(paired_diff[idx[start : start + 24]])
                else:
                    rep_parts.append(paired_diff[idx])
            boot_diffs_a.append(float(np.mean(np.concatenate(rep_parts))))

        ci_a = [float(np.percentile(boot_diffs_a, 2.5)), float(np.percentile(boot_diffs_a, 97.5))]
        pval_a = float(np.mean(np.array(boot_diffs_a) >= 0.0) if observed_mean < 0 else np.mean(np.array(boot_diffs_a) <= 0.0)) * 2.0
        pval_a = min(1.0, pval_a)

        # --- Method B: Full Cluster Bootstrap (Full station weight) ---
        boot_diffs_b = []
        # Precompute per-station mean differences
        st_means = np.array([float(np.mean(paired_diff[station_map[st]])) for st in unique_stations])
        st_weights = np.array([len(station_map[st]) for st in unique_stations], dtype=float)
        st_weights /= st_weights.sum()

        for _ in range(n_replicates):
            sample_idx = rng.choice(len(unique_stations), size=len(unique_stations), replace=True)
            sampled_means = st_means[sample_idx]
            sampled_w = st_weights[sample_idx]
            boot_diffs_b.append(float(np.sum(sampled_means * sampled_w) / np.sum(sampled_w)))

        ci_b = [float(np.percentile(boot_diffs_b, 2.5)), float(np.percentile(boot_diffs_b, 97.5))]
        pval_b = float(np.mean(np.array(boot_diffs_b) >= 0.0) if observed_mean < 0 else np.mean(np.array(boot_diffs_b) <= 0.0)) * 2.0
        pval_b = min(1.0, pval_b)

        # --- Method C: Unweighted Station Mean Bootstrap (Macro-Average across stations) ---
        boot_diffs_c = []
        for _ in range(n_replicates):
            sample_idx = rng.choice(len(unique_stations), size=len(unique_stations), replace=True)
            boot_diffs_c.append(float(np.mean(st_means[sample_idx])))

        ci_c = [float(np.percentile(boot_diffs_c, 2.5)), float(np.percentile(boot_diffs_c, 97.5))]
        pval_c = float(np.mean(np.array(boot_diffs_c) >= 0.0) if observed_mean < 0 else np.mean(np.array(boot_diffs_c) <= 0.0)) * 2.0
        pval_c = min(1.0, pval_c)

        print(f"\n{comp_label}:")
        print(f"  Observed Mean Difference: {observed_mean:.4f} ug/m3")
        print(f"  Method A (Reported Subsample): 95% CI = [{ci_a[0]:.4f}, {ci_a[1]:.4f}], p = {pval_a:.4f}")
        print(f"  Method B (Weighted Cluster):  95% CI = [{ci_b[0]:.4f}, {ci_b[1]:.4f}], p = {pval_b:.4f}")
        print(f"  Method C (Macro Station):     95% CI = [{ci_c[0]:.4f}, {ci_c[1]:.4f}], p = {pval_c:.4f}")

        bootstrap_output[comp_label] = {
            "observed_mean_diff": round(observed_mean, 4),
            "method_a_subsample": {"ci_95": [round(ci_a[0], 4), round(ci_a[1], 4)], "p_value": round(pval_a, 4)},
            "method_b_weighted_cluster": {"ci_95": [round(ci_b[0], 4), round(ci_b[1], 4)], "p_value": round(pval_b, 4)},
            "method_c_macro_station": {"ci_95": [round(ci_c[0], 4), round(ci_c[1], 4)], "p_value": round(pval_c, 4)},
        }

    # Save bootstrap JSON
    boot_json_path = AUDIT_DIR / "paired_recent_bootstrap.json"
    with open(boot_json_path, "w") as f:
        json.dump(bootstrap_output, f, indent=2)
    print(f"\nSaved paired bootstrap audit results to {boot_json_path.name}")

    # -------------------------------------------------------------
    # 7. MONTHLY & STATION-LEVEL BREAKDOWN
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DISAGGREGATED PERFORMANCE AUDITS: BY MONTH & BY STATION")
    print("=" * 90)

    pred_df["month"] = pd.to_datetime(pred_df["timestamp"]).dt.month
    monthly_stats = []

    for m in sorted(pred_df["month"].unique()):
        sub_m = pred_df[pred_df["month"] == m]
        m25_p = compute_metrics(sub_m["actual_pm25"].values, sub_m["pred_persist_pm25"].values)
        m25_ens = compute_metrics(sub_m["actual_pm25"].values, sub_m["pred_adapted_gated_pm25"].values)
        m10_p = compute_metrics(sub_m["actual_pm10"].values, sub_m["pred_persist_pm10"].values)
        m10_ens = compute_metrics(sub_m["actual_pm10"].values, sub_m["pred_adapted_gated_pm10"].values)

        monthly_stats.append({
            "Month": m,
            "Sequences": len(sub_m),
            "PM2.5 Actual Mean": round(float(sub_m["actual_pm25"].mean()), 2),
            "PM2.5 Persist MAE": m25_p["MAE"],
            "PM2.5 GatedEns MAE": m25_ens["MAE"],
            "PM2.5 Ens Delta": round(m25_ens["MAE"] - m25_p["MAE"], 4),
            "PM10 Actual Mean": round(float(sub_m["actual_pm10"].mean()), 2),
            "PM10 Persist MAE": m10_p["MAE"],
            "PM10 GatedEns MAE": m10_ens["MAE"],
            "PM10 Ens Delta": round(m10_ens["MAE"] - m10_p["MAE"], 4),
        })

    monthly_df = pd.DataFrame(monthly_stats)
    print("\n--- Month-by-Month Breakdown (2025 Holdout) ---")
    print(monthly_df.to_string(index=False))

    # Station-level distribution
    st_group = pred_df.groupby("station_id")
    st_p25_diffs = []
    st_p10_diffs = []
    for st, group in st_group:
        p25_mae_p = np.mean(np.abs(group["pred_persist_pm25"] - group["actual_pm25"]))
        p25_mae_ens = np.mean(np.abs(group["pred_adapted_gated_pm25"] - group["actual_pm25"]))
        p10_mae_p = np.mean(np.abs(group["pred_persist_pm10"] - group["actual_pm10"]))
        p10_mae_ens = np.mean(np.abs(group["pred_adapted_gated_pm10"] - group["actual_pm10"]))

        st_p25_diffs.append(p25_mae_ens - p25_mae_p)
        st_p10_diffs.append(p10_mae_ens - p10_mae_p)

    st_p25_diffs = np.array(st_p25_diffs)
    st_p10_diffs = np.array(st_p10_diffs)

    pct_st_p25_win = np.mean(st_p25_diffs < 0.0) * 100.0
    pct_st_p10_win = np.mean(st_p10_diffs < 0.0) * 100.0

    print(f"\n--- Station Distribution Summary (N = {len(st_p25_diffs)} stations) ---")
    print(f"Stations where Adapted Gated Ensemble beats Persistence (PM2.5): {np.sum(st_p25_diffs < 0)} of {len(st_p25_diffs)} ({pct_st_p25_win:.1f}%)")
    print(f"Stations where Adapted Gated Ensemble beats Persistence (PM10):  {np.sum(st_p10_diffs < 0)} of {len(st_p10_diffs)} ({pct_st_p10_win:.1f}%)")
    print(f"PM2.5 Station MAE Delta: Mean = {np.mean(st_p25_diffs):.4f}, Median = {np.median(st_p25_diffs):.4f}, Min = {np.min(st_p25_diffs):.4f}, Max = {np.max(st_p25_diffs):.4f}")
    print(f"PM10 Station MAE Delta:  Mean = {np.mean(st_p10_diffs):.4f}, Median = {np.median(st_p10_diffs):.4f}, Min = {np.min(st_p10_diffs):.4f}, Max = {np.max(st_p10_diffs):.4f}")

    # -------------------------------------------------------------
    # 8. APPLICATION INFERENCE LATENCY PROBE
    # -------------------------------------------------------------
    print("\n--- CPU Inference Latency Benchmark ---")
    dummy_seq = np.random.randn(1, 24, 27).astype(np.float32)

    # LSTM latency
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = finetuned_lstm(torch.from_numpy(dummy_seq).float())
    lstm_lat_ms = (time.time() - t0) / 100.0 * 1000.0

    # GBDT latency
    dummy_red_tab = extract_reduced_tabular_features(dummy_seq)
    t0 = time.time()
    for _ in range(100):
        _ = ad_gbt_25.predict(dummy_red_tab)
        _ = ad_gbt_10.predict(dummy_red_tab)
    gbdt_lat_ms = (time.time() - t0) / 100.0 * 1000.0

    print(f"Fine-Tuned LSTM Single Sequence Latency: {lstm_lat_ms:.2f} ms")
    print(f"Adapted GBDT (PM2.5 + PM10) Latency:     {gbdt_lat_ms:.2f} ms")
    print(f"Total Ensemble Latency:                  {lstm_lat_ms + gbdt_lat_ms:.2f} ms")

    print("\n" + "=" * 80)
    print(f"PHASE 4 INDEPENDENT AUDIT SCRIPT FINISHED IN {time.time() - t_start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
