"""
scripts/audit_stage4_historical_benchmarks.py

Independent mathematical recomputation of historical benchmark metrics:
- 10,000-sample held-out sensor test set (experiments/accuracy_optimization_v2/predictions.csv)
- 97-sample matched multimodal test set (experiments/unified_hybrid/accuracy_optimization/predictions.csv)

Calculates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R^2)
- Mean Bias Error (MBE)
- Median Absolute Error (MedAE)
- Metric breakdowns across CPCB Air Quality Tiers (Good, Satisfactory/Moderate, Poor, Very Poor, Severe)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    mbe = float(np.mean(diff))
    medae = float(np.median(abs_diff))
    return {
        "count": int(len(y_true)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mbe": round(mbe, 4),
        "medae": round(medae, 4)
    }

def main():
    print("--- Stage 4: Historical Benchmark Independent Audit ---")
    
    # 1. 10,000-sample Sensor Test Cohort
    v2_pred_path = ROOT_DIR / "experiments/accuracy_optimization_v2/predictions.csv"
    df_v2 = pd.read_csv(v2_pred_path)
    print(f"Loaded {len(df_v2)} rows from {v2_pred_path}")
    
    y_true_pm25 = df_v2["actual_pm25"].values
    y_true_pm10 = df_v2["actual_pm10"].values
    
    models_10k = [
        ("Persistence Baseline", "pred_persist1_pm25", "pred_persist1_pm10"),
        ("2-Layer LSTM Forecaster", "pred_lstm_pm25", "pred_lstm_pm10"),
        ("Residual GBDT Model", "pred_gbdt_pm25", "pred_gbdt_pm10"),
        ("Validated Static Ensemble", "pred_ensemble_pm25", "pred_ensemble_pm10"),
    ]
    
    audit_rows = []
    
    print("\n--- 10,000-Sample Test Cohort Results ---")
    for name, col_25, col_10 in models_10k:
        m25 = compute_metrics(y_true_pm25, df_v2[col_25].values)
        m10 = compute_metrics(y_true_pm10, df_v2[col_10].values)
        
        audit_rows.append({
            "cohort": "10k_heldout_sensor",
            "tier": "Overall",
            "model": name,
            "pollutant": "PM2.5",
            **m25
        })
        audit_rows.append({
            "cohort": "10k_heldout_sensor",
            "tier": "Overall",
            "model": name,
            "pollutant": "PM10",
            **m10
        })
        print(f"{name:26s} | PM2.5 MAE: {m25['mae']:6.2f}, RMSE: {m25['rmse']:6.2f}, R2: {m25['r2']:6.4f}, MBE: {m25['mbe']:6.2f} | PM10 MAE: {m10['mae']:6.2f}, RMSE: {m10['rmse']:6.2f}, R2: {m10['r2']:6.4f}")

    # Tier Breakdown for PM2.5 (10,000 samples)
    tiers = [
        ("Good (0-30)", y_true_pm25 <= 30.0),
        ("Moderate (30-60)", (y_true_pm25 > 30.0) & (y_true_pm25 <= 60.0)),
        ("Poor (60-90)", (y_true_pm25 > 60.0) & (y_true_pm25 <= 90.0)),
        ("Very Poor (90-120)", (y_true_pm25 > 90.0) & (y_true_pm25 <= 120.0)),
        ("Severe (>120)", y_true_pm25 > 120.0),
    ]
    
    print("\n--- 10,000-Sample PM2.5 Tier Stratification ---")
    for tier_name, mask in tiers:
        sub_true = y_true_pm25[mask]
        print(f"\n[Tier: {tier_name}] (N = {len(sub_true)})")
        for name, col_25, _ in models_10k:
            sub_pred = df_v2[col_25].values[mask]
            m = compute_metrics(sub_true, sub_pred)
            audit_rows.append({
                "cohort": "10k_heldout_sensor",
                "tier": tier_name,
                "model": name,
                "pollutant": "PM2.5",
                **m
            })
            print(f"  {name:26s} | MAE: {m['mae']:6.2f}, RMSE: {m['rmse']:6.2f}, R2: {m['r2']:6.4f}, MedAE: {m['medae']:6.2f}")

    # 2. 97-sample Matched Multimodal Cohort
    mm_pred_path = ROOT_DIR / "experiments/unified_hybrid/accuracy_optimization/predictions.csv"
    df_mm = pd.read_csv(mm_pred_path)
    print(f"\nLoaded {len(df_mm)} rows from {mm_pred_path}")
    
    mm_true_pm25 = df_mm["target_pm25"].values
    mm_true_pm10 = df_mm["target_pm10"].values
    
    models_mm = [
        ("Persistence Baseline", "pred_persistence_pm25", "pred_persistence_pm10"),
        ("Sensor LSTM Only", "pred_sensor_lstm_pm25", "pred_sensor_lstm_pm10"),
        ("Improved Sensor (LSTM+GBDT)", "pred_improved_sensor_pm25", "pred_improved_sensor_pm10"),
        ("Optical ResNet-50", "pred_image_resnet_pm25", None),
        ("Concat Early Fusion", "pred_concat_fusion_pm25", None),
        ("Optimized Gated Hybrid", "pred_opt_hybrid_full_pm25", "pred_opt_hybrid_full_pm10"),
    ]
    
    print("\n--- 97-Sample Matched Multimodal Cohort Results ---")
    for name, col_25, col_10 in models_mm:
        m25 = compute_metrics(mm_true_pm25, df_mm[col_25].values)
        audit_rows.append({
            "cohort": "97_multimodal_matched",
            "tier": "Overall",
            "model": name,
            "pollutant": "PM2.5",
            **m25
        })
        if col_10 and col_10 in df_mm.columns:
            m10 = compute_metrics(mm_true_pm10, df_mm[col_10].values)
            audit_rows.append({
                "cohort": "97_multimodal_matched",
                "tier": "Overall",
                "model": name,
                "pollutant": "PM10",
                **m10
            })
            print(f"{name:28s} | PM2.5 MAE: {m25['mae']:6.2f}, RMSE: {m25['rmse']:6.2f}, R2: {m25['r2']:6.4f} | PM10 MAE: {m10['mae']:6.2f}, RMSE: {m10['rmse']:6.2f}, R2: {m10['r2']:6.4f}")
        else:
            print(f"{name:28s} | PM2.5 MAE: {m25['mae']:6.2f}, RMSE: {m25['rmse']:6.2f}, R2: {m25['r2']:6.4f} | PM10: N/A")

    df_out = pd.DataFrame(audit_rows)
    out_csv = ROOT_DIR / "HISTORICAL_METRICS_INDEPENDENT_AUDIT.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved independent audit metrics to {out_csv}")

if __name__ == "__main__":
    main()
