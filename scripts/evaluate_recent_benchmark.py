"""
scripts/evaluate_recent_benchmark.py

High-performance vectorized evaluation of existing historical models
(Pretrained LSTM, Validated Static Ensemble, Severity-Gated Ensemble)
on post-2020 chronological holdout data with zero retraining.
"""

import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.tabular_features import extract_tabular_features
from src.prediction.lstm_forecaster import LSTMForecasterService

OUT_DIR = ROOT_DIR / "experiments" / "recent_data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, RMSE, R2, Mean Bias, and Underprediction Rate."""
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
        "N": len(y_true),
    }


def construct_sequences_vectorized(
    df: pd.DataFrame,
    lstm_service: LSTMForecasterService,
    min_year: int = 2024,
    max_year: int = 2024,
) -> tuple:
    """Vectorized extraction of continuous 24-hour sequence matrices and targets."""
    print(f"Filtering dataset for target years: {min_year} to {max_year}...", flush=True)
    df_eval = df[(df["Datetime"].dt.year >= min_year) & (df["Datetime"].dt.year <= max_year)].copy()
    stations = df_eval["station_id"].unique()
    print(f"Processing {len(stations)} stations using vectorized sliding windows...", flush=True)

    seq_list = []
    y25_list = []
    y10_list = []
    iss25_list = []
    iss10_list = []
    meta_list = []

    feature_names = lstm_service.feature_names
    scaler_means = lstm_service.scaler_means
    scaler_stds = lstm_service.scaler_stds
    imputation_medians = lstm_service.imputation_medians

    for st in stations:
        st_sub = df_eval[df_eval["station_id"] == st].sort_values("Datetime").drop_duplicates("Datetime")
        if len(st_sub) < 25:
            continue

        # Reindex to full regular hourly frequency
        dt_idx = pd.date_range(st_sub["Datetime"].min(), st_sub["Datetime"].max(), freq="1h")
        st_full = st_sub.set_index("Datetime").reindex(dt_idx)

        # Coordinate features
        lat, lon = lstm_service.get_station_coordinates(st)

        # Build T x 27 standardized feature array
        T = len(st_full)
        F = np.zeros((T, len(feature_names)), dtype=np.float32)

        # Fill available channels
        for col_idx, col_name in enumerate(feature_names):
            if col_name in st_full.columns:
                vals = st_full[col_name].values.astype(np.float32)
                # Fill NaNs with historical median
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

            # Standardize with frozen scaler
            F[:, col_idx] = (vals - scaler_means[col_idx]) / scaler_stds[col_idx]

        # Extract unscaled ground truth PM2.5 and PM10 series
        raw_pm25 = st_full["PM2.5"].values
        raw_pm10 = st_full["PM10"].values

        if T < 25:
            continue

        # Create sliding windows of 24 hours: shape (T - 23, 24, 27)
        windows = np.lib.stride_tricks.sliding_window_view(F, window_shape=24, axis=0)
        # Note: sliding_window_view produces shape (T-23, 27, 24); transpose to (T-23, 24, 27)
        windows = np.ascontiguousarray(np.transpose(windows, (0, 2, 1)))

        # Target at t+1 is index 24..T: length T-24
        # We align:
        # window i has inputs 0..23 (issue time is 23), target is index 24
        # window i has inputs i..i+23 (issue time is i+23), target is index i+24
        n_cand = T - 24
        windows_cand = windows[:n_cand]
        targets_25 = raw_pm25[24 : 24 + n_cand]
        targets_10 = raw_pm10[24 : 24 + n_cand]
        issue_25 = raw_pm25[23 : 23 + n_cand]
        issue_10 = raw_pm10[23 : 23 + n_cand]
        target_dts = dt_idx[24 : 24 + n_cand]
        issue_dts = dt_idx[23 : 23 + n_cand]

        # Raw window observation check: must not have NaNs in the unscaled raw PM2.5 or PM10 across 24h
        raw_pm25_win = np.lib.stride_tricks.sliding_window_view(raw_pm25[: 23 + n_cand], window_shape=24)
        raw_pm10_win = np.lib.stride_tricks.sliding_window_view(raw_pm10[: 23 + n_cand], window_shape=24)

        has_nan_25 = np.isnan(raw_pm25_win).any(axis=1)
        has_nan_10 = np.isnan(raw_pm10_win).any(axis=1)

        valid_mask = (
            (~has_nan_25) & (~has_nan_10)
            & (~np.isnan(targets_25)) & (~np.isnan(targets_10))
            & (targets_25 > 0) & (targets_10 > 0)
            & (~np.isnan(issue_25)) & (~np.isnan(issue_10))
            & (issue_25 > 0) & (issue_10 > 0)
        )

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            continue

        seq_list.append(windows_cand[valid_mask])
        y25_list.append(targets_25[valid_mask])
        y10_list.append(targets_10[valid_mask])
        iss25_list.append(issue_25[valid_mask])
        iss10_list.append(issue_10[valid_mask])

        valid_tg_dts = target_dts[valid_mask]
        valid_is_dts = issue_dts[valid_mask]

        city = st_sub["City"].iloc[0] if "City" in st_sub.columns else ""
        state = st_sub["State"].iloc[0] if "State" in st_sub.columns else ""

        for t_dt, i_dt in zip(valid_tg_dts, valid_is_dts):
            meta_list.append({
                "station_id": st,
                "target_time": str(t_dt),
                "issue_time": str(i_dt),
                "month": t_dt.month,
                "city": city,
                "state": state,
            })

    X_seq = np.concatenate(seq_list, axis=0).astype(np.float32)
    y_25 = np.concatenate(y25_list, axis=0).astype(np.float32)
    y_10 = np.concatenate(y10_list, axis=0).astype(np.float32)
    y_iss_25 = np.concatenate(iss25_list, axis=0).astype(np.float32)
    y_iss_10 = np.concatenate(iss10_list, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_list)

    print(f"Extracted {len(X_seq):,} valid contiguous evaluation sequences across {meta_df['station_id'].nunique()} stations.", flush=True)
    return X_seq, y_25, y_10, y_iss_25, y_iss_10, meta_df


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("VAYUDRISHTI — POST-2020 NEWER-PERIOD BENCHMARK EVALUATION", flush=True)
    print("=" * 80, flush=True)

    clean_csv = ROOT_DIR / "data" / "processed" / "recent_air_quality_cleaned.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {clean_csv}")

    print("Loading cleaned dataset...", flush=True)
    df_clean = pd.read_csv(clean_csv)
    df_clean["Datetime"] = pd.to_datetime(df_clean["Datetime"])
    print(f"Loaded {len(df_clean):,} cleaned rows ({df_clean['Datetime'].min()} to {df_clean['Datetime'].max()})", flush=True)

    # 1. Initialize LSTM Service & Models
    print("\n[1/5] Initializing historical forecasting models & feature extractors...", flush=True)
    lstm_service = LSTMForecasterService()

    ckpt_dir = ROOT_DIR / "experiments" / "accuracy_optimization_v2" / "checkpoints"
    gbdt_pm25 = joblib.load(ckpt_dir / "best_gbdt_pm25.joblib")
    gbdt_pm10 = joblib.load(ckpt_dir / "best_gbdt_pm10.joblib")

    # 2. Extract Valid Sequences for Year 2024
    print("\n[2/5] Extracting continuous 24-hour evaluation sequences for Calendar Year 2024...", flush=True)
    X_seq, y_true_25, y_true_10, y_iss_25, y_iss_10, meta_df = construct_sequences_vectorized(
        df_clean, lstm_service, min_year=2024, max_year=2024
    )

    # 3. Model Inference (Persistence, LSTM, Static Ensemble, Severity-Gated)
    print(f"\n[3/5] Running model predictions on {len(X_seq):,} test sequences...", flush=True)

    # A. 1-Hour Persistence
    pred_pers_25 = y_iss_25
    pred_pers_10 = y_iss_10

    # B. Pretrained 2-Layer LSTM (in batches of 4,096)
    print("  Evaluating Pretrained LSTM on PyTorch batches...", flush=True)
    lstm_preds_list = []
    batch_size = 4096
    with torch.no_grad():
        for b_start in range(0, len(X_seq), batch_size):
            b_end = min(b_start + batch_size, len(X_seq))
            x_b = torch.tensor(X_seq[b_start:b_end], dtype=torch.float32, device=lstm_service.device)
            out_b = lstm_service.model(x_b).cpu().numpy()
            lstm_preds_list.append(out_b)
    raw_lstm_out = np.concatenate(lstm_preds_list, axis=0)
    pred_lstm_25 = raw_lstm_out[:, 0] * lstm_service.target_std[0] + lstm_service.target_mean[0]
    pred_lstm_10 = raw_lstm_out[:, 1] * lstm_service.target_std[1] + lstm_service.target_mean[1]

    # C. Standalone GBDT
    print("  Extracting 67 tabular lag and momentum features...", flush=True)
    tab_feats = extract_tabular_features(X_seq)
    print("  Evaluating HistGradientBoosting trees...", flush=True)
    pred_gbdt_25 = gbdt_pm25.predict(tab_feats)
    pred_gbdt_10 = gbdt_pm10.predict(tab_feats)

    # D. Static Ensemble (0.90 / 0.10 and 0.80 / 0.20)
    pred_static_25 = 0.90 * pred_lstm_25 + 0.10 * pred_gbdt_25
    pred_static_10 = 0.80 * pred_lstm_10 + 0.20 * pred_gbdt_10

    # E. Severity-Gated Ensemble
    gate_25 = 1.0 / (1.0 + np.exp(-(y_iss_25 - 250.0) / 25.0))
    gate_10 = 1.0 / (1.0 + np.exp(-(y_iss_10 - 400.0) / 40.0))
    w25 = 0.90 + 0.10 * gate_25
    w10 = 0.80 + 0.20 * gate_10
    pred_gated_25 = w25 * pred_lstm_25 + (1.0 - w25) * pred_gbdt_25
    pred_gated_10 = w10 * pred_lstm_10 + (1.0 - w10) * pred_gbdt_10

    # 4. Compute Benchmark Performance Metrics
    print("\n[4/5] Computing comprehensive metrics on 2024 test cohort...", flush=True)

    models = {
        "1-Hour Persistence": (pred_pers_25, pred_pers_10),
        "Pretrained LSTM": (pred_lstm_25, pred_lstm_10),
        "Standalone GBDT": (pred_gbdt_25, pred_gbdt_10),
        "Static Ensemble (0.90/0.80)": (pred_static_25, pred_static_10),
        "Severity-Gated Ensemble": (pred_gated_25, pred_gated_10),
    }

    results = {
        "metadata": {
            "evaluation_period": "2024-01-01 to 2024-12-31",
            "test_sequences": len(y_true_25),
            "test_stations": int(meta_df["station_id"].nunique()),
            "actual_pm25_mean": round(float(np.mean(y_true_25)), 2),
            "actual_pm10_mean": round(float(np.mean(y_true_10)), 2),
            "actual_pm25_p50": round(float(np.median(y_true_25)), 2),
            "actual_pm10_p50": round(float(np.median(y_true_10)), 2),
            "actual_pm25_max": round(float(np.max(y_true_25)), 2),
            "actual_pm10_max": round(float(np.max(y_true_10)), 2),
        },
        "overall": {},
        "pm25_tiers": {},
        "pm10_tiers": {},
        "monthly": {},
        "station_summary": {},
    }

    for name, (p25, p10) in models.items():
        results["overall"][name] = {
            "pm25": calculate_metrics(y_true_25, p25),
            "pm10": calculate_metrics(y_true_10, p10),
        }

    # PM2.5 CPCB NAQI Tier Breakdown
    cpcb_tiers_pm25 = [
        ("Good (0-30)", 0.0, 30.0),
        ("Satisfactory (30-60)", 30.0, 60.0),
        ("Moderate (60-90)", 60.0, 90.0),
        ("Poor (90-120)", 90.0, 120.0),
        ("Very Poor (120-250)", 120.0, 250.0),
        ("Severe (>250)", 250.0, 10000.0),
    ]

    for tier_name, low, high in cpcb_tiers_pm25:
        mask = (y_true_25 >= low) & (y_true_25 < high)
        if mask.sum() == 0:
            continue
        results["pm25_tiers"][tier_name] = {"count": int(mask.sum()), "pct": round(float(mask.mean() * 100), 2)}
        for name, (p25, _) in models.items():
            results["pm25_tiers"][tier_name][name] = calculate_metrics(y_true_25[mask], p25[mask])

    # PM10 CPCB NAQI Tier Breakdown
    cpcb_tiers_pm10 = [
        ("Good (0-50)", 0.0, 50.0),
        ("Satisfactory (50-100)", 50.0, 100.0),
        ("Moderate (100-250)", 100.0, 250.0),
        ("Poor (250-350)", 250.0, 350.0),
        ("Very Poor (350-430)", 350.0, 430.0),
        ("Severe (>430)", 430.0, 10000.0),
    ]

    for tier_name, low, high in cpcb_tiers_pm10:
        mask = (y_true_10 >= low) & (y_true_10 < high)
        if mask.sum() == 0:
            continue
        results["pm10_tiers"][tier_name] = {"count": int(mask.sum()), "pct": round(float(mask.mean() * 100), 2)}
        for name, (_, p10) in models.items():
            results["pm10_tiers"][tier_name][name] = calculate_metrics(y_true_10[mask], p10[mask])

    # Monthly Performance Breakdown
    for mo in range(1, 13):
        m_mask = meta_df["month"] == mo
        if m_mask.sum() == 0:
            continue
        results["monthly"][f"2024-{mo:02d}"] = {"count": int(m_mask.sum())}
        for name, (p25, p10) in models.items():
            results["monthly"][f"2024-{mo:02d}"][name] = {
                "pm25_mae": round(float(np.mean(np.abs(p25[m_mask] - y_true_25[m_mask]))), 2),
                "pm10_mae": round(float(np.mean(np.abs(p10[m_mask] - y_true_10[m_mask]))), 2),
            }

    # Save Metrics JSON
    bench_json_path = OUT_DIR / "recent_benchmark_metrics.json"
    with open(bench_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark metrics: {bench_json_path}", flush=True)

    # Display comparison table
    print("\n" + "=" * 90, flush=True)
    print(f"RECENT BENCHMARK RESULTS (CALENDAR YEAR 2024: N = {len(y_true_25):,} SEQUENCES)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Model Architecture':<30} | {'PM2.5 MAE':<10} | {'PM2.5 RMSE':<10} | {'PM2.5 R2':<8} | {'PM10 MAE':<10} | {'PM10 RMSE':<10} | {'PM10 R2':<8}", flush=True)
    print("-" * 90, flush=True)
    for name in models.keys():
        m25 = results["overall"][name]["pm25"]
        m10 = results["overall"][name]["pm10"]
        print(f"{name:<30} | {m25['MAE']:<10.4f} | {m25['RMSE']:<10.4f} | {m25['R2']:<8.4f} | {m10['MAE']:<10.4f} | {m10['RMSE']:<10.4f} | {m10['R2']:<8.4f}", flush=True)
    print("=" * 90, flush=True)

    # 5. Export Summary CSV for Analysis
    pred_summary = meta_df.copy()
    pred_summary["actual_pm25"] = np.round(y_true_25, 2)
    pred_summary["actual_pm10"] = np.round(y_true_10, 2)
    pred_summary["pred_persistence_pm25"] = np.round(pred_pers_25, 2)
    pred_summary["pred_lstm_pm25"] = np.round(pred_lstm_25, 2)
    pred_summary["pred_static_pm25"] = np.round(pred_static_25, 2)
    pred_summary["pred_gated_pm25"] = np.round(pred_gated_25, 2)
    pred_summary["pred_persistence_pm10"] = np.round(pred_pers_10, 2)
    pred_summary["pred_lstm_pm10"] = np.round(pred_lstm_10, 2)
    pred_summary["pred_static_pm10"] = np.round(pred_static_10, 2)
    pred_summary["pred_gated_pm10"] = np.round(pred_gated_10, 2)

    pred_summary_path = OUT_DIR / "recent_predictions_summary.csv"
    pred_summary.to_csv(pred_summary_path, index=False)
    print(f"Saved prediction summary table: {pred_summary_path} (Elapsed: {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
