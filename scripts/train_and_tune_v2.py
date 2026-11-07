"""
scripts/train_and_tune_v2.py

Phase 3: Hyperparameter tuning and model training on chronological development data.
- Fits candidate models strictly on training split (X_train.npy, y_train.npy).
- Performs hyperparameter grid search strictly on validation split (X_validation.npy, y_validation.npy).
- Locked test set (X_test.npy, y_test.npy) is NOT accessed.
- Evaluates:
  1. Direct GBDT (HistGradientBoostingRegressor) for PM2.5 and PM10
  2. Residual GBDT (predicting y - y_lstm)
  3. Validated Ensemble Blend (w * y_lstm + (1-w) * y_gbdt)
  4. Weather Ablation (sensor-only without weather vs sensor+weather)
- Saves best checkpoints and validation_results.csv into experiments/accuracy_optimization_v2/
"""

import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.tabular_features import extract_tabular_features
from src.prediction.lstm_forecaster import LSTMForecasterService

opt_dir = ROOT_DIR / "experiments/accuracy_optimization_v2"
ckpt_dir = opt_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Set random seeds for strict reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("PHASE 3: MODEL TRAINING & HYPERPARAMETER TUNING ON DEVELOPMENT DATA")
print("=" * 70)

# 1. Load memory-mapped sequences
X_tr_mmap = np.load(ROOT_DIR / "data/processed/model_sequences/X_train.npy", mmap_mode="r")
y_tr_full = np.load(ROOT_DIR / "data/processed/model_sequences/y_train.npy")

X_va_mmap = np.load(ROOT_DIR / "data/processed/model_sequences/X_validation.npy", mmap_mode="r")
y_va_full = np.load(ROOT_DIR / "data/processed/model_sequences/y_validation.npy")

N_TRAIN_TUNE = 100000
N_VAL_TUNE = 50000

print(f"Subsampling development data for CPU tuning: Train={N_TRAIN_TUNE}, Val={N_VAL_TUNE}")
# Select chronological sub-samples
# For training, take the most recent 100,000 sequences preceding the validation split
X_tr_seq = X_tr_mmap[-N_TRAIN_TUNE:]
y_tr = y_tr_full[-N_TRAIN_TUNE:]

# For validation, evaluate on 50,000 validation sequences
X_va_seq = X_va_mmap[:N_VAL_TUNE]
y_va = y_va_full[:N_VAL_TUNE]

print("Extracting 67 leakage-free tabular features...")
t0 = time.time()
X_tr_tab = extract_tabular_features(X_tr_seq)
X_va_tab = extract_tabular_features(X_va_seq)
print(f"Feature extraction complete in {time.time() - t0:.2f}s. Shape: {X_tr_tab.shape}")

# 2. Get pretrained LSTM predictions on validation subset
lstm_service = LSTMForecasterService()
lstm_service.model.eval()

print("Generating pretrained LSTM predictions on validation cohort...")
with torch.no_grad():
    # Batch inference to conserve RAM
    pred_lstm_va_list = []
    chunk_sz = 10000
    for i in range(0, N_VAL_TUNE, chunk_sz):
        x_chunk = torch.from_numpy(np.array(X_va_seq[i : i + chunk_sz])).float()
        norm_chunk = lstm_service.model(x_chunk).numpy()
        p25 = norm_chunk[:, 0] * 77.4174 + 79.5145
        p10 = norm_chunk[:, 1] * 137.4705 + 172.1068
        pred_lstm_va_list.append(np.column_stack([p25, p10]))
    pred_lstm_va = np.concatenate(pred_lstm_va_list, axis=0)

# LSTM on training subset for residual model
print("Generating pretrained LSTM predictions on training subset...")
with torch.no_grad():
    pred_lstm_tr_list = []
    for i in range(0, N_TRAIN_TUNE, chunk_sz):
        x_chunk = torch.from_numpy(np.array(X_tr_seq[i : i + chunk_sz])).float()
        norm_chunk = lstm_service.model(x_chunk).numpy()
        p25 = norm_chunk[:, 0] * 77.4174 + 79.5145
        p10 = norm_chunk[:, 1] * 137.4705 + 172.1068
        pred_lstm_tr_list.append(np.column_stack([p25, p10]))
    pred_lstm_tr = np.concatenate(pred_lstm_tr_list, axis=0)

# Evaluate LSTM baseline on validation subset
def eval_metrics(yt, yp):
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / ss_tot) if ss_tot > 1e-8 else 0.0
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

m_lstm_va_25 = eval_metrics(y_va[:, 0], pred_lstm_va[:, 0])
m_lstm_va_10 = eval_metrics(y_va[:, 1], pred_lstm_va[:, 1])
print(f"Validation Baseline Pretrained LSTM: PM2.5 MAE={m_lstm_va_25['MAE']}, PM10 MAE={m_lstm_va_10['MAE']}")

# 3. Hyperparameter search for GBDT (HistGradientBoostingRegressor)
param_grid = [
    {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1, "l2_regularization": 1.0, "min_samples_leaf": 20},
    {"max_iter": 150, "max_leaf_nodes": 31, "learning_rate": 0.08, "l2_regularization": 2.0, "min_samples_leaf": 30},
    {"max_iter": 150, "max_leaf_nodes": 63, "learning_rate": 0.08, "l2_regularization": 2.0, "min_samples_leaf": 30},
    {"max_iter": 100, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 5.0, "min_samples_leaf": 50},
]

validation_records = []

best_gbt_pm25 = None
best_mae_pm25 = float("inf")
best_params_pm25 = None

print("\n--- Tuning GBDT PM2.5 on Validation Split ---")
for idx, params in enumerate(param_grid):
    t_start = time.time()
    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        learning_rate=params["learning_rate"],
        l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=SEED,
    )
    model.fit(X_tr_tab, y_tr[:, 0])
    preds_va = model.predict(X_va_tab)
    m = eval_metrics(y_va[:, 0], preds_va)
    elapsed = time.time() - t_start

    record = {
        "Target": "PM2.5",
        "Model_Type": "Direct GBDT",
        "Config_ID": f"cfg_{idx+1}",
        "Params": str(params),
        "Validation_MAE": m["MAE"],
        "Validation_RMSE": m["RMSE"],
        "Validation_R2": m["R2"],
        "Training_Time_s": round(elapsed, 2),
    }
    validation_records.append(record)
    print(f"Config {idx+1}: {params} -> Val MAE: {m['MAE']:.4f}, R2: {m['R2']:.4f} ({elapsed:.1f}s)")

    if m["MAE"] < best_mae_pm25:
        best_mae_pm25 = m["MAE"]
        best_gbt_pm25 = model
        best_params_pm25 = params

best_gbt_pm10 = None
best_mae_pm10 = float("inf")
best_params_pm10 = None

print("\n--- Tuning GBDT PM10 on Validation Split ---")
for idx, params in enumerate(param_grid):
    t_start = time.time()
    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        learning_rate=params["learning_rate"],
        l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=SEED,
    )
    model.fit(X_tr_tab, y_tr[:, 1])
    preds_va = model.predict(X_va_tab)
    m = eval_metrics(y_va[:, 1], preds_va)
    elapsed = time.time() - t_start

    record = {
        "Target": "PM10",
        "Model_Type": "Direct GBDT",
        "Config_ID": f"cfg_{idx+1}",
        "Params": str(params),
        "Validation_MAE": m["MAE"],
        "Validation_RMSE": m["RMSE"],
        "Validation_R2": m["R2"],
        "Training_Time_s": round(elapsed, 2),
    }
    validation_records.append(record)
    print(f"Config {idx+1}: {params} -> Val MAE: {m['MAE']:.4f}, R2: {m['R2']:.4f} ({elapsed:.1f}s)")

    if m["MAE"] < best_mae_pm10:
        best_mae_pm10 = m["MAE"]
        best_gbt_pm10 = model
        best_params_pm10 = params

# 4. Residual Boosting (predicting y - y_lstm)
print("\n--- Training Residual GBDT on Pretrained LSTM Residuals ---")
res_tr_25 = y_tr[:, 0] - pred_lstm_tr[:, 0]
res_tr_10 = y_tr[:, 1] - pred_lstm_tr[:, 1]

res_gbt_25 = HistGradientBoostingRegressor(
    max_iter=100,
    max_leaf_nodes=31,
    learning_rate=0.08,
    l2_regularization=3.0,
    min_samples_leaf=30,
    random_state=SEED,
)
res_gbt_25.fit(X_tr_tab, res_tr_25)
delta_va_25 = res_gbt_25.predict(X_va_tab)
pred_res_va_25 = np.maximum(0.0, pred_lstm_va[:, 0] + delta_va_25)
m_res_25 = eval_metrics(y_va[:, 0], pred_res_va_25)

res_gbt_10 = HistGradientBoostingRegressor(
    max_iter=100,
    max_leaf_nodes=31,
    learning_rate=0.08,
    l2_regularization=3.0,
    min_samples_leaf=30,
    random_state=SEED,
)
res_gbt_10.fit(X_tr_tab, res_tr_10)
delta_va_10 = res_gbt_10.predict(X_va_tab)
pred_res_va_10 = np.maximum(0.0, pred_lstm_va[:, 1] + delta_va_10)
m_res_10 = eval_metrics(y_va[:, 1], pred_res_va_10)

validation_records.append({
    "Target": "PM2.5",
    "Model_Type": "Residual GBDT (LSTM + delta)",
    "Config_ID": "residual_default",
    "Params": "l2=3.0, lr=0.08, leaves=31",
    "Validation_MAE": m_res_25["MAE"],
    "Validation_RMSE": m_res_25["RMSE"],
    "Validation_R2": m_res_25["R2"],
    "Training_Time_s": 0.0,
})
validation_records.append({
    "Target": "PM10",
    "Model_Type": "Residual GBDT (LSTM + delta)",
    "Config_ID": "residual_default",
    "Params": "l2=3.0, lr=0.08, leaves=31",
    "Validation_MAE": m_res_10["MAE"],
    "Validation_RMSE": m_res_10["RMSE"],
    "Validation_R2": m_res_10["R2"],
    "Training_Time_s": 0.0,
})

print(f"Residual GBDT Validation: PM2.5 MAE={m_res_25['MAE']} | PM10 MAE={m_res_10['MAE']}")

# 5. Tune Validated Ensemble Blend Weight strictly on Validation Data
print("\n--- Tuning Ensemble Blend Weights on Validation Split ---")
pred_best_gbt_va_25 = best_gbt_pm25.predict(X_va_tab)
pred_best_gbt_va_10 = best_gbt_pm10.predict(X_va_tab)

best_w_25 = 0.5
best_ensemble_mae_25 = float("inf")
for w in np.linspace(0.0, 1.0, 21):
    blend = w * pred_lstm_va[:, 0] + (1.0 - w) * pred_best_gbt_va_25
    mae = float(np.mean(np.abs(y_va[:, 0] - blend)))
    if mae < best_ensemble_mae_25:
        best_ensemble_mae_25 = mae
        best_w_25 = w

best_w_10 = 0.5
best_ensemble_mae_10 = float("inf")
for w in np.linspace(0.0, 1.0, 21):
    blend = w * pred_lstm_va[:, 1] + (1.0 - w) * pred_best_gbt_va_10
    mae = float(np.mean(np.abs(y_va[:, 1] - blend)))
    if mae < best_ensemble_mae_10:
        best_ensemble_mae_10 = mae
        best_w_10 = w

blend_opt_va_25 = best_w_25 * pred_lstm_va[:, 0] + (1.0 - best_w_25) * pred_best_gbt_va_25
blend_opt_va_10 = best_w_10 * pred_lstm_va[:, 1] + (1.0 - best_w_10) * pred_best_gbt_va_10
m_ens_25 = eval_metrics(y_va[:, 0], blend_opt_va_25)
m_ens_10 = eval_metrics(y_va[:, 1], blend_opt_va_10)

validation_records.append({
    "Target": "PM2.5",
    "Model_Type": "Ensemble (LSTM + GBDT)",
    "Config_ID": f"w_lstm={best_w_25:.2f}",
    "Params": f"weight_lstm={best_w_25:.2f}, weight_gbdt={1-best_w_25:.2f}",
    "Validation_MAE": m_ens_25["MAE"],
    "Validation_RMSE": m_ens_25["RMSE"],
    "Validation_R2": m_ens_25["R2"],
    "Training_Time_s": 0.0,
})
validation_records.append({
    "Target": "PM10",
    "Model_Type": "Ensemble (LSTM + GBDT)",
    "Config_ID": f"w_lstm={best_w_10:.2f}",
    "Params": f"weight_lstm={best_w_10:.2f}, weight_gbdt={1-best_w_10:.2f}",
    "Validation_MAE": m_ens_10["MAE"],
    "Validation_RMSE": m_ens_10["RMSE"],
    "Validation_R2": m_ens_10["R2"],
    "Training_Time_s": 0.0,
})

print(f"Best Ensemble Weights: PM2.5 w_lstm={best_w_25:.2f} (Val MAE={m_ens_25['MAE']}), PM10 w_lstm={best_w_10:.2f} (Val MAE={m_ens_10['MAE']})")

# 6. Weather Ablation (Sensor-Only without weather vs Sensor+Weather)
# Weather features are col 12 to 18 in 27 original features, plus dewpoint depression
# Column indices in 67 tabular features:
# col 12..18 are weather_t, col 66 is dewpoint_depression
weather_indices = [12, 13, 14, 15, 16, 17, 18, 66]
non_weather_indices = [i for i in range(X_tr_tab.shape[1]) if i not in weather_indices]

print("\n--- Running Weather Ablation (without weather features) ---")
m_no_weather = HistGradientBoostingRegressor(
    max_iter=best_params_pm25["max_iter"],
    max_leaf_nodes=best_params_pm25["max_leaf_nodes"],
    learning_rate=best_params_pm25["learning_rate"],
    l2_regularization=best_params_pm25["l2_regularization"],
    min_samples_leaf=best_params_pm25["min_samples_leaf"],
    random_state=SEED,
)
m_no_weather.fit(X_tr_tab[:, non_weather_indices], y_tr[:, 0])
p_no_w_va = m_no_weather.predict(X_va_tab[:, non_weather_indices])
m_no_w_25 = eval_metrics(y_va[:, 0], p_no_w_va)
print(f"PM2.5 without Weather Features: Val MAE={m_no_w_25['MAE']} (vs {best_mae_pm25:.4f} with weather)")

validation_records.append({
    "Target": "PM2.5",
    "Model_Type": "Sensor-Only (Weather Excluded)",
    "Config_ID": "ablation_no_weather",
    "Params": "All 8 weather channels zeroed/excluded",
    "Validation_MAE": m_no_w_25["MAE"],
    "Validation_RMSE": m_no_w_25["RMSE"],
    "Validation_R2": m_no_w_25["R2"],
    "Training_Time_s": 0.0,
})

# 7. Save Checkpoints and Metadata
print("\nSaving winning model checkpoints...")
joblib.dump(best_gbt_pm25, ckpt_dir / "best_gbdt_pm25.joblib")
joblib.dump(best_gbt_pm10, ckpt_dir / "best_gbdt_pm10.joblib")
joblib.dump(res_gbt_25, ckpt_dir / "best_residual_gbdt_pm25.joblib")
joblib.dump(res_gbt_10, ckpt_dir / "best_residual_gbdt_pm10.joblib")

config = {
    "feature_names_tabular": 67,
    "best_params_pm25": best_params_pm25,
    "best_params_pm10": best_params_pm10,
    "best_ensemble_weights": {
        "pm25": {"weight_lstm": round(float(best_w_25), 4), "weight_gbdt": round(float(1.0 - best_w_25), 4)},
        "pm10": {"weight_lstm": round(float(best_w_10), 4), "weight_gbdt": round(float(1.0 - best_w_10), 4)},
    },
    "validation_performance": {
        "pretrained_lstm": {"pm25": m_lstm_va_25, "pm10": m_lstm_va_10},
        "best_gbdt": {"pm25": eval_metrics(y_va[:, 0], pred_best_gbt_va_25), "pm10": eval_metrics(y_va[:, 1], pred_best_gbt_va_10)},
        "residual_gbdt": {"pm25": m_res_25, "pm10": m_res_10},
        "ensemble": {"pm25": m_ens_25, "pm10": m_ens_10},
        "no_weather_ablation_pm25": m_no_w_25,
    },
}

with open(opt_dir / "configs" / "best_model_config.json", "w") as f:
    json.dump(config, f, indent=2)

val_df = pd.DataFrame(validation_records)
val_df.to_csv(opt_dir / "validation_results.csv", index=False)

print(f"Validation results saved to {opt_dir / 'validation_results.csv'}")
print("Phase 3 development tuning complete!")
