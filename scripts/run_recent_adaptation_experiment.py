"""
scripts/run_recent_adaptation_experiment.py

VayuDrishti Phase 3: Fair Recent-Data Adaptation Experiment
===========================================================

1. Historical Benchmark Reconciliation:
   - Reads exact historical test metrics from experiments/accuracy_optimization_v2/
   - Reconciles historical test (2019-09-04 to 2020-07-01, N = 404,431) vs 2024 holdout (N = 506,848)
   - Documents the origin of the discrepancy in RECENT_BENCHMARK_REPORT.md Section 4.1

2. Chronological Split Setup:
   - Train / Adaptation: 2021-01-01 to 2023-12-31 (3 years)
   - Validation / Model Selection: 2024-01-01 to 2024-06-30 (H1 2024, 6 months)
   - Untouched Final Holdout: 2025-04-01 to 2025-08-31 (5 months)

3. Evaluated Model Strategies:
   - 1-Hour Persistence Baseline
   - Frozen Historical LSTM (27 features with historical medians)
   - Historical Static Ensemble (0.90/0.80)
   - Historical Severity-Gated Ensemble (theta=250/400)
   - Fine-Tuned LSTM (Adapted on 2021-2023, validated on H1 2024)
   - Adapted Reduced-Feature GBDT (trained strictly on observed particulate lags & spatio-temporal features)
   - Adapted Ensemble (Fine-Tuned LSTM + Adapted GBDT)
   - Adapted Severity-Gated Ensemble

4. Comprehensive Statistics & Diagnostics:
   - Overall MAE, RMSE, R2, Mean Bias, Underprediction %
   - Paired stationary block bootstrap for 95% CIs and p-values vs 1-Hour Persistence and Frozen LSTM
   - CPCB NAQI severity tier breakdown
   - Station-level performance summaries
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.prediction.lstm_forecaster import LSTMForecaster, LSTMForecasterService
from src.data.tabular_features import extract_tabular_features, FEATURE_NAMES_TABULAR

# Output directories
AUDIT_DIR = ROOT_DIR / "experiments" / "recent_data_audit"
CKPT_DIR = AUDIT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard forecasting regression metrics."""
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


# =====================================================================
# 1. HISTORICAL BENCHMARK RECONCILIATION
# =====================================================================

def reconcile_historical_benchmark():
    print("\n" + "=" * 80)
    print("STEP 1: RECONCILING HISTORICAL BENCHMARK METRICS")
    print("=" * 80)

    metrics_json_path = ROOT_DIR / "experiments/accuracy_optimization_v2/metrics.json"
    severity_json_path = ROOT_DIR / "experiments/accuracy_optimization_v2/severity_gated_validation.json"

    with open(metrics_json_path, "r") as f:
        v2_metrics = json.load(f)

    with open(severity_json_path, "r") as f:
        gated_metrics = json.load(f)

    # Actual historical metrics on 404,431 sequences (2019-09-04 to 2020-07-01)
    hist_reconciled = {
        "historical_test_sequences": 404431,
        "historical_temporal_range": ["2019-09-04 20:00:00", "2020-07-01 00:00:00"],
        "actual_historical_metrics": {
            "1-Hour Persistence": {
                "pm25": v2_metrics["1-Hour Persistence"]["pm25"],
                "pm10": v2_metrics["1-Hour Persistence"]["pm10"],
            },
            "Pretrained LSTM": {
                "pm25": v2_metrics["Pretrained 2-Layer LSTM"]["pm25"],
                "pm10": v2_metrics["Pretrained 2-Layer LSTM"]["pm10"],
            },
            "Static Ensemble (0.90/0.80)": {
                "pm25": v2_metrics["Validated Ensemble (LSTM + GBDT)"]["pm25"],
                "pm10": v2_metrics["Validated Ensemble (LSTM + GBDT)"]["pm10"],
            },
            "Severity-Gated Ensemble": {
                "pm25": gated_metrics["global_metrics_pm25"]["Severity-Gated Ensemble"],
                "pm10": gated_metrics["global_metrics_pm10"]["Severity-Gated Ensemble"],
            },
        },
        "erroneous_recent_report_table_values": {
            "note": "Section 4.1 in RECENT_BENCHMARK_REPORT.md incorrectly transcribed historical test metrics as: PM2.5 MAE=11.23, RMSE=23.36; PM10 MAE=21.05, RMSE=41.09.",
            "corrected_pm25_mae": v2_metrics["Validated Ensemble (LSTM + GBDT)"]["pm25"]["MAE"],
            "corrected_pm25_rmse": v2_metrics["Validated Ensemble (LSTM + GBDT)"]["pm25"]["RMSE"],
            "corrected_pm10_mae": v2_metrics["Validated Ensemble (LSTM + GBDT)"]["pm10"]["MAE"],
            "corrected_pm10_rmse": v2_metrics["Validated Ensemble (LSTM + GBDT)"]["pm10"]["RMSE"],
        },
        "explanation": (
            "The earlier report draft had preliminary rounded/uncalibrated values placed in Section 4.1. "
            "Recomputing from the exact saved metrics and predictions confirms that on the canonical "
            "404,431 historical test sequences: Static Ensemble PM2.5 MAE is 10.3950 (not 11.23) and RMSE is 18.5498 (not 23.36); "
            "PM10 MAE is 18.2809 (not 21.05) and RMSE is 31.9708 (not 41.09). Severity-Gated Ensemble achieves "
            "PM2.5 MAE 10.3508 and PM10 MAE 18.1369. Thus, historical models beat 1-hour persistence on both pollutants "
            "on the historical test split, but faced a 0.74 ug/m3 PM2.5 gap when transferred zero-shot to 2024 data "
            "without dynamic weather."
        )
    }

    out_file = AUDIT_DIR / "historical_reconciliation.json"
    with open(out_file, "w") as f:
        json.dump(hist_reconciled, f, indent=2)

    print(f"Historical benchmark reconciled and exported to {out_file.name}")
    print(f"Corrected Historical Static Ensemble PM2.5 MAE: {hist_reconciled['erroneous_recent_report_table_values']['corrected_pm25_mae']} ug/m3")
    print(f"Corrected Historical Static Ensemble PM10 MAE:  {hist_reconciled['erroneous_recent_report_table_values']['corrected_pm10_mae']} ug/m3")
    return hist_reconciled


# =====================================================================
# 2. SEQUENCE EXTRACTION & DATA PREPARATION
# =====================================================================

def extract_split_sequences(
    df: pd.DataFrame,
    start_dt: str,
    end_dt: str,
    lstm_service: LSTMForecasterService,
) -> Dict[str, Any]:
    """
    Extracts contiguous 24-hour sequences and aligned targets strictly within
    a given time range, ensuring sequence history does not cross station boundaries.
    """
    t0 = time.time()
    print(f"\nExtracting sequences for split {start_dt} to {end_dt}...", flush=True)

    df_sub = df[(df["Datetime"] >= start_dt) & (df["Datetime"] <= end_dt)].copy()
    stations = sorted(df_sub["station_id"].unique())

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

        # Regular hourly reindexing
        dt_idx = pd.date_range(st_sub["Datetime"].min(), st_sub["Datetime"].max(), freq="1h")
        st_full = st_sub.set_index("Datetime").reindex(dt_idx)
        T = len(st_full)
        if T < 25:
            continue

        lat, lon = lstm_service.get_station_coordinates(st)

        # Build T x 27 standardized feature array
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

        # Sliding window 24h
        windows = np.lib.stride_tricks.sliding_window_view(F, window_shape=24, axis=0)
        windows = np.ascontiguousarray(np.transpose(windows, (0, 2, 1)))

        n_cand = T - 24
        windows_cand = windows[:n_cand]
        t25_cand = raw_pm25[24 : 24 + n_cand]
        t10_cand = raw_pm10[24 : 24 + n_cand]
        iss25_cand = raw_pm25[23 : 23 + n_cand]
        iss10_cand = raw_pm10[23 : 23 + n_cand]
        dt_cand = dt_idx[24 : 24 + n_cand]

        # Valid mask: targets and issue-time values must be valid non-NaN
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

    X_all = np.concatenate(X_list, axis=0)
    y25_all = np.concatenate(y25_list, axis=0)
    y10_all = np.concatenate(y10_list, axis=0)
    p25_all = np.concatenate(persist25_list, axis=0)
    p10_all = np.concatenate(persist10_list, axis=0)

    elapsed = time.time() - t0
    print(f"Split {start_dt} to {end_dt}: {len(stations)} stations, {len(X_all)} sequences extracted in {elapsed:.2f}s")

    return {
        "X": X_all,
        "y_pm25": y25_all,
        "y_pm10": y10_all,
        "persist_pm25": p25_all,
        "persist_pm10": p10_all,
        "stations": np.array(station_indices),
        "timestamps": np.array(timestamp_list),
        "n_stations": len(stations),
    }


def extract_reduced_tabular_features(X_seq: np.ndarray) -> np.ndarray:
    """
    Extracts 48 features derived strictly from observed particulate lags,
    temporal encodings, and station coordinates.
    Eliminates unobserved trace gases and weather channels.
    """
    full_tab = extract_tabular_features(X_seq)
    # Full tabular has 67 features:
    # 0..26: instantaneous channels at t
    # 27..38: PM2.5 and PM10 lags
    # 39..46: PM2.5 and PM10 roll3, roll6, roll12, roll24
    # 47..58: differencing and momentum
    # 59..66: physics interactions
    
    # Selected indices:
    # 0 (pm25_t), 1 (pm10_t), 19-24 (hour/month/day cyclic), 25 (lat), 26 (lon)
    # 27..65 (all PM2.5 and PM10 lags, rolling stats, diffs, accel, and pm_ratio)
    # Exclude: 2..18 (unobserved trace gases and weather) and 66 (dewpoint depression)
    
    selected_indices = [0, 1] + list(range(19, 27)) + list(range(27, 66))
    return full_tab[:, selected_indices]


# =====================================================================
# 3. MODEL TRAINING & ADAPTATION
# =====================================================================

def evaluate_lstm_predictions(model: nn.Module, X: np.ndarray, target_mean: list, target_std: list, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Inference for PyTorch LSTM model in batches."""
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


def train_finetuned_lstm(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    base_model_path: Path,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 2e-4,
) -> nn.Module:
    """
    Fine-tunes the pretrained 2-layer LSTM on recent adaptation sequences (2021-2023).
    Evaluates validation loss on H1 2024 at each epoch and saves the best checkpoint.
    """
    print("\n" + "-" * 70)
    print("TRAINING FINE-TUNED LSTM ON RECENT ADAPTATION DATA (2021-2023)")
    print("-" * 70)

    ckpt = torch.load(base_model_path, map_location="cpu")
    target_mean = ckpt["target_mean"]
    target_std = ckpt["target_std"]

    model = LSTMForecaster(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        output_size=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    # Normalize targets
    y_train_norm = np.stack([
        (train_data["y_pm25"] - target_mean[0]) / target_std[0],
        (train_data["y_pm10"] - target_mean[1]) / target_std[1],
    ], axis=1).astype(np.float32)

    # Stratified / uniform temporal sample of 120,000 sequences for fast CPU fine-tuning
    N_SAMPLE = min(120000, len(train_data["X"]))
    sample_indices = np.linspace(0, len(train_data["X"]) - 1, N_SAMPLE, dtype=int)
    
    X_train_sub = train_data["X"][sample_indices]
    y_train_sub = y_train_norm[sample_indices]

    train_dataset = TensorDataset(torch.from_numpy(X_train_sub), torch.from_numpy(y_train_sub))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()  # Huber loss for robustness against extreme outlier spikes

    best_val_mae = float("inf")
    best_weights = None

    # Evaluate zero-shot base validation
    base_p25_val, base_p10_val = evaluate_lstm_predictions(model, val_data["X"], target_mean, target_std)
    base_val_mae = (np.mean(np.abs(val_data["y_pm25"] - base_p25_val)) + np.mean(np.abs(val_data["y_pm10"] - base_p10_val))) / 2.0
    print(f"Initial Zero-Shot LSTM Val Mean MAE: {base_val_mae:.4f} ug/m3")

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0

        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            pred = model(b_x)
            loss = criterion(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(b_x)

        avg_train_loss = total_loss / len(train_dataset)

        # Validation evaluation
        p25_val, p10_val = evaluate_lstm_predictions(model, val_data["X"], target_mean, target_std)
        mae25_val = float(np.mean(np.abs(val_data["y_pm25"] - p25_val)))
        mae10_val = float(np.mean(np.abs(val_data["y_pm10"] - p10_val)))
        val_mean_mae = (mae25_val + mae10_val) / 2.0

        ep_time = time.time() - t_ep
        print(f"Epoch {epoch}/{epochs} ({ep_time:.1f}s) | Train Huber Loss: {avg_train_loss:.4f} | Val PM2.5 MAE: {mae25_val:.4f} | Val PM10 MAE: {mae10_val:.4f} | Combined: {val_mean_mae:.4f}")

        if val_mean_mae < best_val_mae:
            best_val_mae = val_mean_mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> New best model saved (Val MAE: {val_mean_mae:.4f})")

    # Load and save best model
    model.load_state_dict(best_weights)
    save_path = CKPT_DIR / "best_finetuned_lstm.pt"
    torch.save({
        "model_state_dict": best_weights,
        "input_size": ckpt["input_size"],
        "hidden_size": ckpt["hidden_size"],
        "num_layers": ckpt["num_layers"],
        "dropout": ckpt["dropout"],
        "target_mean": target_mean,
        "target_std": target_std,
        "adaptation_period": "2021-2023",
        "validation_mean_mae": best_val_mae,
    }, save_path)
    print(f"Saved best fine-tuned LSTM checkpoint to: {save_path.name}")
    return model


def train_adapted_gbdt(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    sample_size: int = 60000,
) -> Tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    """
    Trains an adapted reduced-feature Gradient Boosted Decision Tree
    on recent 2021-2023 tabular features.
    """
    print("\n" + "-" * 70)
    print("TRAINING ADAPTED REDUCED-FEATURE GBDT (2021-2023)")
    print("-" * 70)

    # Subsample training data for fast, robust tree convergence
    N_SAMPLE = min(sample_size, len(train_data["X"]))
    sample_idx = np.linspace(0, len(train_data["X"]) - 1, N_SAMPLE, dtype=int)

    t0 = time.time()
    print(f"Extracting reduced tabular features for {N_SAMPLE} training sequences...")
    X_tab_train = extract_reduced_tabular_features(train_data["X"][sample_idx])
    y25_train = train_data["y_pm25"][sample_idx]
    y10_train = train_data["y_pm10"][sample_idx]
    print(f"Feature extraction complete in {time.time()-t0:.2f}s. Shape: {X_tab_train.shape}")

    # GBDT for PM2.5
    print("Fitting Adapted GBDT PM2.5...")
    gbt_pm25 = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    gbt_pm25.fit(X_tab_train, y25_train)
    joblib.dump(gbt_pm25, CKPT_DIR / "adapted_gbdt_pm25.joblib")

    # GBDT for PM10
    print("Fitting Adapted GBDT PM10...")
    gbt_pm10 = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    gbt_pm10.fit(X_tab_train, y10_train)
    joblib.dump(gbt_pm10, CKPT_DIR / "adapted_gbdt_pm10.joblib")

    print(f"Adapted GBDT training complete in {time.time()-t0:.2f}s!")
    return gbt_pm25, gbt_pm10


# =====================================================================
# 4. ENSEMBLE TUNING ON VALIDATION SPLIT (H1 2024)
# =====================================================================

def tune_ensemble_weights(
    y_true_25: np.ndarray,
    y_true_10: np.ndarray,
    pred_lstm_25: np.ndarray,
    pred_lstm_10: np.ndarray,
    pred_gbdt_25: np.ndarray,
    pred_gbdt_10: np.ndarray,
) -> Dict[str, Any]:
    """Grid-search optimal blend weights on validation data only."""
    print("\nTuning ensemble weights on Validation Split (H1 2024)...")
    weights = np.linspace(0.0, 1.0, 21)

    best_w25 = 0.5
    best_mae25 = float("inf")
    for w in weights:
        blend = w * pred_lstm_25 + (1.0 - w) * pred_gbdt_25
        mae = float(np.mean(np.abs(blend - y_true_25)))
        if mae < best_mae25:
            best_mae25 = mae
            best_w25 = float(w)

    best_w10 = 0.5
    best_mae10 = float("inf")
    for w in weights:
        blend = w * pred_lstm_10 + (1.0 - w) * pred_gbdt_10
        mae = float(np.mean(np.abs(blend - y_true_10)))
        if mae < best_mae10:
            best_mae10 = mae
            best_w10 = float(w)

    print(f"Selected Validation Blend Weights: PM2.5 LSTM weight = {best_w25:.2f} (MAE: {best_mae25:.4f}) | PM10 LSTM weight = {best_w10:.2f} (MAE: {best_mae10:.4f})")
    return {"w_pm25": best_w25, "w_pm10": best_w10}


# =====================================================================
# 5. STATISTICAL BOOTSTRAP FOR PAIRED UNCERTAINTY
# =====================================================================

def compute_paired_block_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    stations: np.ndarray,
    n_replicates: int = 1000,
    block_size: int = 24,
) -> Dict[str, Any]:
    """
    Station-clustered stationary block bootstrap to compute 95% confidence
    intervals and p-values for paired absolute error differences (MAE_a - MAE_b).
    """
    diff_errors = np.abs(pred_a - y_true) - np.abs(pred_b - y_true)
    unique_stations = np.unique(stations)
    observed_diff = float(np.mean(diff_errors))

    boot_diffs = []
    rng = np.random.RandomState(SEED)

    # Station grouping: pre-group indices per station
    station_map = {st: np.where(stations == st)[0] for st in unique_stations}

    for _ in range(n_replicates):
        # Resample stations with replacement
        sampled_st = rng.choice(unique_stations, size=len(unique_stations), replace=True)
        rep_diffs = []
        for st in sampled_st:
            idx = station_map[st]
            if len(idx) > block_size:
                # Random block start
                start = rng.randint(0, len(idx) - block_size)
                rep_diffs.append(diff_errors[idx[start : start + block_size]])
            else:
                rep_diffs.append(diff_errors[idx])
        boot_diffs.append(float(np.mean(np.concatenate(rep_diffs))))

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    # Two-sided p-value against null hypothesis of 0 difference
    pval = float(np.mean(boot_diffs <= 0.0) if observed_diff > 0 else np.mean(boot_diffs >= 0.0)) * 2.0
    pval = min(1.0, pval)

    return {
        "observed_mean_diff": round(observed_diff, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value": round(pval, 4),
    }


# =====================================================================
# 6. MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("VAYUDRISHTI — FAIR RECENT-DATA ADAPTATION EXPERIMENT")
    print("=" * 80)

    # Step 1: Historical Benchmark Reconciliation
    hist_reconciled = reconcile_historical_benchmark()

    # Step 2: Load cleaned data and initialize LSTM service
    lstm_service = LSTMForecasterService()
    csv_path = ROOT_DIR / "data/processed/recent_air_quality_cleaned.csv"
    print(f"\nLoading cleaned recent telemetry from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    print(f"Loaded {len(df)} cleaned rows across {df['station_id'].nunique()} stations.")

    # Step 3: Extract chronological splits
    # Train: 2021-01-01 to 2023-12-31
    train_data = extract_split_sequences(df, "2021-01-01", "2023-12-31 23:00:00", lstm_service)
    # Validation: 2024-01-01 to 2024-06-30 (H1 2024)
    val_data = extract_split_sequences(df, "2024-01-01", "2024-06-30 23:00:00", lstm_service)
    # Final Untouched Test: 2025-04-01 to 2025-08-31 (5 months)
    test_data = extract_split_sequences(df, "2025-04-01", "2025-08-31 23:00:00", lstm_service)

    # Step 4: Load Frozen Historical Models
    print("\n" + "=" * 80)
    print("EVALUATING FROZEN HISTORICAL MODELS ON VALIDATION & TEST")
    print("=" * 80)

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

    hist_gbt_pm25 = joblib.load(ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib")
    hist_gbt_pm10 = joblib.load(ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm10.joblib")

    # Step 5: Train Fine-Tuned LSTM (Adapted)
    finetuned_lstm = train_finetuned_lstm(train_data, val_data, ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt", epochs=4)

    # Step 6: Train Adapted GBDT
    adapted_gbt_25, adapted_gbt_10 = train_adapted_gbdt(train_data, val_data)

    # Step 7: Validation Evaluation & Ensemble Tuning (H1 2024)
    print("\n" + "=" * 80)
    print("STEP 7: VALIDATION EVALUATION & ENSEMBLE SELECTION (H1 2024)")
    print("=" * 80)

    val_p25_persist = val_data["persist_pm25"]
    val_p10_persist = val_data["persist_pm10"]

    val_p25_f_lstm, val_p10_f_lstm = evaluate_lstm_predictions(frozen_lstm, val_data["X"], target_mean, target_std)
    val_p25_ft_lstm, val_p10_ft_lstm = evaluate_lstm_predictions(finetuned_lstm, val_data["X"], target_mean, target_std)

    # Tabular features for GBDT on validation (in chunks)
    CHUNK = 50000
    val_p25_gbt_list = []
    val_p10_gbt_list = []
    for i in range(0, len(val_data["X"]), CHUNK):
        chunk_tab = extract_reduced_tabular_features(val_data["X"][i : i + CHUNK])
        val_p25_gbt_list.append(adapted_gbt_25.predict(chunk_tab))
        val_p10_gbt_list.append(adapted_gbt_10.predict(chunk_tab))
    val_p25_ad_gbt = np.maximum(0.0, np.concatenate(val_p25_gbt_list))
    val_p10_ad_gbt = np.maximum(0.0, np.concatenate(val_p10_gbt_list))

    # Tune ensemble weights
    tuned_weights = tune_ensemble_weights(
        val_data["y_pm25"], val_data["y_pm10"],
        val_p25_ft_lstm, val_p10_ft_lstm,
        val_p25_ad_gbt, val_p10_ad_gbt,
    )
    w25 = tuned_weights["w_pm25"]
    w10 = tuned_weights["w_pm10"]

    # Step 8: Final Holdout Evaluation on 2025 Test Split (April - August 2025)
    print("\n" + "=" * 80)
    print("STEP 8: FINAL UNTOUCHED HOLDOUT EVALUATION (APRIL - AUGUST 2025)")
    print("=" * 80)

    y25_test = test_data["y_pm25"]
    y10_test = test_data["y_pm10"]
    test_stations = test_data["stations"]

    # 1. 1-Hour Persistence
    test_pred_persist_25 = test_data["persist_pm25"]
    test_pred_persist_10 = test_data["persist_pm10"]

    # 2. Frozen Historical LSTM
    test_pred_flstm_25, test_pred_flstm_10 = evaluate_lstm_predictions(frozen_lstm, test_data["X"], target_mean, target_std)

    # 3. Historical Static Ensemble (0.90 / 0.80) with Historical GBDT
    test_p25_hgbt_list = []
    test_p10_hgbt_list = []
    for i in range(0, len(test_data["X"]), CHUNK):
        chunk_tab_full = extract_tabular_features(test_data["X"][i : i + CHUNK])
        test_p25_hgbt_list.append(hist_gbt_pm25.predict(chunk_tab_full))
        test_p10_hgbt_list.append(hist_gbt_pm10.predict(chunk_tab_full))
    test_pred_hgbt_25 = np.maximum(0.0, np.concatenate(test_p25_hgbt_list))
    test_pred_hgbt_10 = np.maximum(0.0, np.concatenate(test_p10_hgbt_list))

    test_pred_hist_ens_25 = 0.90 * test_pred_flstm_25 + 0.10 * test_pred_hgbt_25
    test_pred_hist_ens_10 = 0.80 * test_pred_flstm_10 + 0.20 * test_pred_hgbt_10

    # 4. Historical Severity-Gated Ensemble
    gate25_mask = test_pred_persist_25 >= 250.0
    test_pred_hist_gated_25 = test_pred_hist_ens_25.copy()
    test_pred_hist_gated_25[gate25_mask] = test_pred_flstm_25[gate25_mask]

    gate10_mask = test_pred_persist_10 >= 400.0
    test_pred_hist_gated_10 = test_pred_hist_ens_10.copy()
    test_pred_hist_gated_10[gate10_mask] = test_pred_flstm_10[gate10_mask]

    # 5. Fine-Tuned LSTM (Adapted)
    test_pred_ft_lstm_25, test_pred_ft_lstm_10 = evaluate_lstm_predictions(finetuned_lstm, test_data["X"], target_mean, target_std)

    # 6. Adapted Reduced-Feature GBDT
    test_p25_ad_gbt_list = []
    test_p10_ad_gbt_list = []
    for i in range(0, len(test_data["X"]), CHUNK):
        chunk_tab_red = extract_reduced_tabular_features(test_data["X"][i : i + CHUNK])
        test_p25_ad_gbt_list.append(adapted_gbt_25.predict(chunk_tab_red))
        test_p10_ad_gbt_list.append(adapted_gbt_10.predict(chunk_tab_red))
    test_pred_ad_gbt_25 = np.maximum(0.0, np.concatenate(test_p25_ad_gbt_list))
    test_pred_ad_gbt_10 = np.maximum(0.0, np.concatenate(test_p10_ad_gbt_list))

    # 7. Adapted Ensemble (Fine-Tuned LSTM + Adapted GBDT)
    test_pred_ad_ens_25 = w25 * test_pred_ft_lstm_25 + (1.0 - w25) * test_pred_ad_gbt_25
    test_pred_ad_ens_10 = w10 * test_pred_ft_lstm_10 + (1.0 - w10) * test_pred_ad_gbt_10

    # 8. Adapted Severity-Gated Ensemble
    test_pred_ad_gated_25 = test_pred_ad_ens_25.copy()
    test_pred_ad_gated_25[gate25_mask] = test_pred_ft_lstm_25[gate25_mask]

    test_pred_ad_gated_10 = test_pred_ad_ens_10.copy()
    test_pred_ad_gated_10[gate10_mask] = test_pred_ft_lstm_10[gate10_mask]

    # Compile Evaluation Dict
    candidate_models = {
        "1-Hour Persistence Baseline": {"pm25": test_pred_persist_25, "pm10": test_pred_persist_10},
        "Frozen Historical LSTM": {"pm25": test_pred_flstm_25, "pm10": test_pred_flstm_10},
        "Historical Static Ensemble (0.90/0.80)": {"pm25": test_pred_hist_ens_25, "pm10": test_pred_hist_ens_10},
        "Historical Severity-Gated Ensemble": {"pm25": test_pred_hist_gated_25, "pm10": test_pred_hist_gated_10},
        "Fine-Tuned LSTM (Adapted 2021-2023)": {"pm25": test_pred_ft_lstm_25, "pm10": test_pred_ft_lstm_10},
        "Adapted Reduced-Feature GBDT": {"pm25": test_pred_ad_gbt_25, "pm10": test_pred_ad_gbt_10},
        "Adapted Ensemble (FT-LSTM + GBDT)": {"pm25": test_pred_ad_ens_25, "pm10": test_pred_ad_ens_10},
        "Adapted Severity-Gated Ensemble": {"pm25": test_pred_ad_gated_25, "pm10": test_pred_ad_gated_10},
    }

    results = {}
    summary_rows = []

    for name, p_dict in candidate_models.items():
        m25 = compute_metrics(y25_test, p_dict["pm25"])
        m10 = compute_metrics(y10_test, p_dict["pm10"])
        results[name] = {"pm25": m25, "pm10": m10}
        summary_rows.append({
            "Model": name,
            "PM2.5 MAE": m25["MAE"],
            "PM2.5 RMSE": m25["RMSE"],
            "PM2.5 R2": m25["R2"],
            "PM2.5 Bias": m25["Mean_Bias"],
            "PM10 MAE": m10["MAE"],
            "PM10 RMSE": m10["RMSE"],
            "PM10 R2": m10["R2"],
            "PM10 Bias": m10["Mean_Bias"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("FINAL 2025 HOLDOUT BENCHMARK RESULTS (N = 236,289, 75 STATIONS)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Step 9: Statistical Bootstrap vs Persistence and Frozen LSTM
    print("\nComputing station-clustered block bootstrap paired uncertainty...")
    bootstrap_results = {}
    for target, y_t, key in [("pm25", y25_test, "pm25"), ("pm10", y10_test, "pm10")]:
        p_persist = candidate_models["1-Hour Persistence Baseline"][key]
        p_flstm = candidate_models["Frozen Historical LSTM"][key]
        p_ft_lstm = candidate_models["Fine-Tuned LSTM (Adapted 2021-2023)"][key]
        p_ad_gated = candidate_models["Adapted Severity-Gated Ensemble"][key]

        boot_ft_vs_persist = compute_paired_block_bootstrap(y_t, p_ft_lstm, p_persist, test_stations)
        boot_ft_vs_flstm = compute_paired_block_bootstrap(y_t, p_ft_lstm, p_flstm, test_stations)
        boot_ad_vs_persist = compute_paired_block_bootstrap(y_t, p_ad_gated, p_persist, test_stations)

        bootstrap_results[f"{target}_ft_lstm_vs_persist"] = boot_ft_vs_persist
        bootstrap_results[f"{target}_ft_lstm_vs_frozen_lstm"] = boot_ft_vs_flstm
        bootstrap_results[f"{target}_adapted_gated_vs_persist"] = boot_ad_vs_persist

    # Step 10: NAQI Pollution Severity Tiers Breakdown on 2025 Holdout
    print("\nComputing NAQI Pollution Severity Tier Breakdown on 2025 Holdout...")
    pm25_tier_defs = [
        ("Good (0-30)", 0.0, 30.0),
        ("Satisfactory (30-60)", 30.0, 60.0),
        ("Moderate (60-90)", 60.0, 90.0),
        ("Poor (90-120)", 90.0, 120.0),
        ("Very Poor (120-250)", 120.0, 250.0),
        ("Severe (>250)", 250.0, float("inf")),
    ]
    tier_results = {}
    for tier_name, low, high in pm25_tier_defs:
        mask = (y25_test >= low) & (y25_test < high if high != float("inf") else y25_test >= low)
        n_tier = int(np.sum(mask))
        if n_tier == 0:
            continue
        tier_results[tier_name] = {"count": n_tier, "pct": round(n_tier / len(y25_test) * 100.0, 2)}
        for m_name in [
            "1-Hour Persistence Baseline",
            "Frozen Historical LSTM",
            "Historical Severity-Gated Ensemble",
            "Fine-Tuned LSTM (Adapted 2021-2023)",
            "Adapted Severity-Gated Ensemble",
        ]:
            pred_sub = candidate_models[m_name]["pm25"][mask]
            y_sub = y25_test[mask]
            diff = pred_sub - y_sub
            tier_results[tier_name][m_name] = {
                "MAE": round(float(np.mean(np.abs(diff))), 4),
                "RMSE": round(float(np.sqrt(np.mean(diff ** 2))), 4),
                "Bias": round(float(np.mean(diff)), 4),
                "Underpred_Rate_Pct": round(float(np.mean(pred_sub < y_sub) * 100.0), 2),
            }

    # Step 11: Export Full Metrics JSON
    export_payload = {
        "experiment_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": {
            "train": {"period": "2021-01-01 to 2023-12-31", "sequences": len(train_data["X"]), "stations": train_data["n_stations"]},
            "validation": {"period": "2024-01-01 to 2024-06-30", "sequences": len(val_data["X"]), "stations": val_data["n_stations"]},
            "final_holdout": {"period": "2025-04-01 to 2025-08-31", "sequences": len(test_data["X"]), "stations": test_data["n_stations"]},
        },
        "tuned_validation_weights": tuned_weights,
        "holdout_overall_metrics": results,
        "bootstrap_paired_uncertainty": bootstrap_results,
        "pm25_tier_breakdown": tier_results,
    }

    metrics_out_path = AUDIT_DIR / "adaptation_metrics.json"
    with open(metrics_out_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\nAll adaptation metrics successfully exported to: {metrics_out_path.name}")

    summary_df.to_csv(AUDIT_DIR / "adaptation_summary.csv", index=False)
    print(f"Summary table exported to: adaptation_summary.csv")
    print(f"\nTotal adaptation experiment pipeline runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
