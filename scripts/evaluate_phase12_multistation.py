"""
scripts/evaluate_phase12_multistation.py

Comprehensive Multi-Station, Seasonal, Regime, and Uncertainty Evaluation for Phase 12.
Evaluates:
1. Model Comparison: Sensor-only vs Sensor+Meteo vs Persistence vs Ensemble.
2. Station-Wise Evaluation: Across all 15 CAAQMS monitoring stations.
3. NAQI Regime Breakdown: Good, Satisfactory, Moderate, Poor, Very Poor, Severe.
4. IMD Seasonal Breakdown: Winter, Summer, Monsoon, Post-Monsoon.
5. Conformal Prediction Uncertainty Calibration: 80%, 90%, 95% intervals.

Outputs:
- experiments/phase12/model_comparison.csv
- experiments/phase12/station_metrics.csv
- experiments/phase12/regime_metrics.csv
- experiments/phase12/seasonal_metrics.csv
- experiments/phase12/uncertainty_metrics.csv
"""

import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.services.forecast_uncertainty_service import ConformalPredictionCalibrator
from src.services.meteorological_service import MeteorologicalService

PHASE12_DIR = ROOT_DIR / "experiments" / "phase12"
STATION_REGISTRY_CSV = PHASE12_DIR / "station_registry.csv"
FINAL_PREDS_CSV = ROOT_DIR / "experiments" / "recent_data_audit" / "final_predictions.csv"
LIVE_EVAL_CSV = PHASE12_DIR / "live_evaluation.csv"
RECENT_DATA_CLEANED_CSV = ROOT_DIR / "data" / "processed" / "recent_air_quality_cleaned.csv"

# Indian CPCB NAQI Regimes (PM2.5 in ug/m3)
NAQI_PM25_REGIMES = [
    ("Good", 0.0, 30.0),
    ("Satisfactory", 30.01, 60.0),
    ("Moderate", 60.01, 90.0),
    ("Poor", 90.01, 120.0),
    ("Very Poor", 120.01, 250.0),
    ("Severe", 250.01, 1000.0),
]

# IMD Seasons
def get_imd_season(month: int) -> str:
    if month in (1, 2):
        return "Winter (Jan-Feb)"
    elif month in (3, 4, 5):
        return "Summer / Pre-Monsoon (Mar-May)"
    elif month in (6, 7, 8, 9):
        return "Monsoon (Jun-Sep)"
    else:
        return "Post-Monsoon (Oct-Dec)"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan}
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-6 else np.nan
    bias = float(np.mean(yp - yt))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if not np.isnan(r2) else 0.0,
        "bias": round(bias, 4),
    }


def main():
    print("=" * 78)
    print("VAYUDRISHTI PHASE 12: MULTI-STATION & RESEARCH VALIDATION EVALUATION")
    print("=" * 78)

    PHASE12_DIR.mkdir(parents=True, exist_ok=True)
    reg_df = pd.read_csv(STATION_REGISTRY_CSV)
    print(f"Loaded {len(reg_df)} registered monitoring stations.")

    # 1. Load holdout prediction dataset (N=236,289)
    print(f"Loading benchmark prediction dataset from {FINAL_PREDS_CSV.name}...")
    df_preds = pd.read_csv(FINAL_PREDS_CSV)
    df_preds["dt"] = pd.to_datetime(df_preds["timestamp"])
    df_preds["month"] = df_preds["dt"].dt.month
    df_preds["season"] = df_preds["month"].apply(get_imd_season)

    # 2. Add AP001 from live_evaluation if missing in final_predictions
    if LIVE_EVAL_CSV.exists():
        df_live = pd.read_csv(LIVE_EVAL_CSV)
        print(f"Loaded {len(df_live)} live evaluation records from {LIVE_EVAL_CSV.name}.")
    else:
        df_live = pd.DataFrame()

    print(f"Total benchmark sequences available: {len(df_preds):,}")

    # =========================================================================
    # Task 3: Model Comparison (Sensor-only vs Sensor+Meteo vs Persistence vs Ensemble)
    # =========================================================================
    print("\n--- Generating experiments/phase12/model_comparison.csv ---")
    
    # Extract ground truth and candidate predictions
    y_true_25 = df_preds["actual_pm25"].values
    y_true_10 = df_preds["actual_pm10"].values

    p_persist_25 = df_preds["pred_persist_pm25"].values
    p_persist_10 = df_preds["pred_persist_pm10"].values

    p_sensor_25 = df_preds["pred_adapted_gbdt_pm25"].values
    p_sensor_10 = df_preds["pred_adapted_gbdt_pm10"].values

    p_ens_25 = df_preds["pred_adapted_gated_pm25"].values
    p_ens_10 = df_preds["pred_adapted_gated_pm10"].values

    # Sensor + Meteo: Pretrained / Fine-tuned model with meteorology channels
    p_meteo_25 = df_preds["pred_finetuned_lstm_pm25"].values
    p_meteo_10 = df_preds["pred_finetuned_lstm_pm10"].values

    models = [
        ("1-Hour Persistence Baseline", p_persist_25, p_persist_10, "Persistence", "Telemetry at t"),
        ("Sensor-Only Standalone Adapted GBDT", p_sensor_25, p_sensor_10, "Adapted GBDT (49 feats)", "Particulates Only (PM2.5, PM10)"),
        ("Sensor + Meteorology Forecaster", p_meteo_25, p_meteo_10, "Adapted LSTM (27 feats)", "Particulates + ERA5 Weather (T, RH, WS, WD, Precip)"),
        ("Unified Adapted Severity-Gated Ensemble", p_ens_25, p_ens_10, "Gated Hybrid Ensemble", "Particulates + Weather + Particulate Lags"),
    ]

    m_persist_25 = compute_metrics(y_true_25, p_persist_25)
    m_persist_10 = compute_metrics(y_true_10, p_persist_10)

    comp_rows = []
    for name, p25, p10, arch, inputs in models:
        m25 = compute_metrics(y_true_25, p25)
        m10 = compute_metrics(y_true_10, p10)

        imp_25 = ((m_persist_25["mae"] - m25["mae"]) / m_persist_25["mae"]) * 100.0
        imp_10 = ((m_persist_10["mae"] - m10["mae"]) / m_persist_10["mae"]) * 100.0

        comp_rows.append({
            "model_name": name,
            "architecture": arch,
            "input_requirements": inputs,
            "test_instances": len(df_preds),
            "pm25_mae": m25["mae"],
            "pm25_rmse": m25["rmse"],
            "pm25_r2": m25["r2"],
            "pm25_bias": m25["bias"],
            "pm25_pct_improvement_vs_persistence": round(imp_25, 2),
            "pm10_mae": m10["mae"],
            "pm10_rmse": m10["rmse"],
            "pm10_r2": m10["r2"],
            "pm10_bias": m10["bias"],
            "pm10_pct_improvement_vs_persistence": round(imp_10, 2),
        })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(PHASE12_DIR / "model_comparison.csv", index=False)
    print("Saved experiments/phase12/model_comparison.csv")
    print(df_comp[["model_name", "pm25_mae", "pm25_pct_improvement_vs_persistence", "pm10_mae", "pm10_pct_improvement_vs_persistence"]])

    # =========================================================================
    # Task 4: Station-Wise Evaluation Across 15 Stations
    # =========================================================================
    print("\n--- Generating experiments/phase12/station_metrics.csv ---")
    st_rows = []
    for _, s_info in reg_df.iterrows():
        sid = s_info["station_id"]
        cohort = s_info["cohort"]
        name = s_info["station_name"]
        city = s_info["city"]
        state = s_info["state"]

        sub = df_preds[df_preds["station_id"] == sid]
        if sub.empty:
            # Check if station is in live evaluation (e.g. AP001)
            sub_live = df_live[df_live["station_id"] == sid] if not df_live.empty else pd.DataFrame()
            if not sub_live.empty:
                n_samples = len(sub_live)
                m25_g = compute_metrics(sub_live["actual_pm25"].values, sub_live["pred_pm25"].values)
                m25_p = compute_metrics(sub_live["actual_pm25"].values, sub_live["persistence_pm25"].values)
                m10_g = compute_metrics(sub_live["actual_pm10"].values, sub_live["pred_pm10"].values)
                m10_p = compute_metrics(sub_live["actual_pm10"].values, sub_live["persistence_pm10"].values)
                source_tag = "Live Longitudinal Telemetry (2026)"
            else:
                continue
        else:
            n_samples = len(sub)
            m25_g = compute_metrics(sub["actual_pm25"].values, sub["pred_adapted_gbdt_pm25"].values)
            m25_p = compute_metrics(sub["actual_pm25"].values, sub["pred_persist_pm25"].values)
            m10_g = compute_metrics(sub["actual_pm10"].values, sub["pred_adapted_gbdt_pm10"].values)
            m10_p = compute_metrics(sub["actual_pm10"].values, sub["pred_persist_pm10"].values)
            source_tag = "Holdout Benchmark (Apr-Aug 2025)"

        imp25 = ((m25_p["mae"] - m25_g["mae"]) / m25_p["mae"]) * 100.0 if m25_p["mae"] > 0 else 0.0
        imp10 = ((m10_p["mae"] - m10_g["mae"]) / m10_p["mae"]) * 100.0 if m10_p["mae"] > 0 else 0.0

        st_rows.append({
            "station_id": sid,
            "station_name": name,
            "city": city,
            "state": state,
            "cohort": cohort,
            "source": source_tag,
            "sample_count": n_samples,
            "pm25_persistence_mae": m25_p["mae"],
            "pm25_gbdt_mae": m25_g["mae"],
            "pm25_gbdt_rmse": m25_g["rmse"],
            "pm25_gbdt_r2": m25_g["r2"],
            "pm25_gbdt_bias": m25_g["bias"],
            "pm25_pct_improvement": round(imp25, 2),
            "pm10_persistence_mae": m10_p["mae"],
            "pm10_gbdt_mae": m10_g["mae"],
            "pm10_gbdt_rmse": m10_g["rmse"],
            "pm10_gbdt_r2": m10_g["r2"],
            "pm10_gbdt_bias": m10_g["bias"],
            "pm10_pct_improvement": round(imp10, 2),
        })

    df_st = pd.DataFrame(st_rows)
    df_st.to_csv(PHASE12_DIR / "station_metrics.csv", index=False)
    print(f"Saved experiments/phase12/station_metrics.csv ({len(df_st)} stations).")

    # =========================================================================
    # Task 4B: NAQI Regime Breakdown
    # =========================================================================
    print("\n--- Generating experiments/phase12/regime_metrics.csv ---")
    regime_rows = []
    for regime_name, low_b, high_b in NAQI_PM25_REGIMES:
        sub = df_preds[(df_preds["actual_pm25"] >= low_b) & (df_preds["actual_pm25"] <= high_b)]
        if sub.empty:
            continue
        n_sub = len(sub)
        m25_g = compute_metrics(sub["actual_pm25"].values, sub["pred_adapted_gbdt_pm25"].values)
        m25_p = compute_metrics(sub["actual_pm25"].values, sub["pred_persist_pm25"].values)
        m10_g = compute_metrics(sub["actual_pm10"].values, sub["pred_adapted_gbdt_pm10"].values)
        m10_p = compute_metrics(sub["actual_pm10"].values, sub["pred_persist_pm10"].values)

        imp25 = ((m25_p["mae"] - m25_g["mae"]) / m25_p["mae"]) * 100.0 if m25_p["mae"] > 0 else 0.0
        imp10 = ((m10_p["mae"] - m10_g["mae"]) / m10_p["mae"]) * 100.0 if m10_p["mae"] > 0 else 0.0

        regime_rows.append({
            "naqi_regime": regime_name,
            "pm25_range_ug_m3": f"{low_b} - {high_b}",
            "sample_count": n_sub,
            "cohort_share_pct": round((n_sub / len(df_preds)) * 100.0, 2),
            "pm25_persistence_mae": m25_p["mae"],
            "pm25_gbdt_mae": m25_g["mae"],
            "pm25_gbdt_rmse": m25_g["rmse"],
            "pm25_gbdt_r2": m25_g["r2"],
            "pm25_gbdt_bias": m25_g["bias"],
            "pm25_pct_improvement": round(imp25, 2),
            "pm10_persistence_mae": m10_p["mae"],
            "pm10_gbdt_mae": m10_g["mae"],
            "pm10_gbdt_rmse": m10_g["rmse"],
            "pm10_gbdt_r2": m10_g["r2"],
            "pm10_gbdt_bias": m10_g["bias"],
            "pm10_pct_improvement": round(imp10, 2),
        })

    df_regime = pd.DataFrame(regime_rows)
    df_regime.to_csv(PHASE12_DIR / "regime_metrics.csv", index=False)
    print("Saved experiments/phase12/regime_metrics.csv")
    print(df_regime[["naqi_regime", "sample_count", "pm25_gbdt_mae", "pm25_pct_improvement", "pm10_gbdt_mae", "pm10_pct_improvement"]])

    # =========================================================================
    # Task 4C: Seasonal Breakdown (Covering All 4 IMD Seasons from recent_predictions_summary)
    # =========================================================================
    print("\n--- Generating experiments/phase12/seasonal_metrics.csv ---")
    recent_summary_csv = ROOT_DIR / "experiments" / "recent_data_audit" / "recent_predictions_summary.csv"
    if recent_summary_csv.exists():
        print(f"Loading full-year seasonal predictions from {recent_summary_csv.name}...")
        df_season_source = pd.read_csv(recent_summary_csv)
        df_season_source["season"] = df_season_source["month"].apply(get_imd_season)
        pred_col_25 = "pred_gated_pm25"
        pred_col_10 = "pred_gated_pm10"
        pers_col_25 = "pred_persistence_pm25"
        pers_col_10 = "pred_persistence_pm10"
    else:
        df_season_source = df_preds
        pred_col_25 = "pred_adapted_gbdt_pm25"
        pred_col_10 = "pred_adapted_gbdt_pm10"
        pers_col_25 = "pred_persist_pm25"
        pers_col_10 = "pred_persist_pm10"

    seasonal_rows = []
    seasons = [
        "Winter (Jan-Feb)",
        "Summer / Pre-Monsoon (Mar-May)",
        "Monsoon (Jun-Sep)",
        "Post-Monsoon (Oct-Dec)",
    ]
    for season_name in seasons:
        sub = df_season_source[df_season_source["season"] == season_name]
        if sub.empty:
            continue
        n_sub = len(sub)
        m25_g = compute_metrics(sub["actual_pm25"].values, sub[pred_col_25].values)
        m25_p = compute_metrics(sub["actual_pm25"].values, sub[pers_col_25].values)
        m10_g = compute_metrics(sub["actual_pm10"].values, sub[pred_col_10].values)
        m10_p = compute_metrics(sub["actual_pm10"].values, sub[pers_col_10].values)

        imp25 = ((m25_p["mae"] - m25_g["mae"]) / m25_p["mae"]) * 100.0 if m25_p["mae"] > 0 else 0.0
        imp10 = ((m10_p["mae"] - m10_g["mae"]) / m10_p["mae"]) * 100.0 if m10_p["mae"] > 0 else 0.0

        seasonal_rows.append({
            "imd_season": season_name,
            "sample_count": n_sub,
            "cohort_share_pct": round((n_sub / len(df_season_source)) * 100.0, 2),
            "pm25_persistence_mae": m25_p["mae"],
            "pm25_model_mae": m25_g["mae"],
            "pm25_model_rmse": m25_g["rmse"],
            "pm25_model_r2": m25_g["r2"],
            "pm25_model_bias": m25_g["bias"],
            "pm25_pct_improvement": round(imp25, 2),
            "pm10_persistence_mae": m10_p["mae"],
            "pm10_model_mae": m10_g["mae"],
            "pm10_model_rmse": m10_g["rmse"],
            "pm10_model_r2": m10_g["r2"],
            "pm10_model_bias": m10_g["bias"],
            "pm10_pct_improvement": round(imp10, 2),
        })

    df_season = pd.DataFrame(seasonal_rows)
    df_season.to_csv(PHASE12_DIR / "seasonal_metrics.csv", index=False)
    print("Saved experiments/phase12/seasonal_metrics.csv")
    print(df_season[["imd_season", "sample_count", "pm25_model_mae", "pm25_pct_improvement", "pm10_model_mae", "pm10_pct_improvement"]])

    # =========================================================================
    # Task 5: Forecast Uncertainty & Conformal Calibration
    # =========================================================================
    print("\n--- Generating experiments/phase12/uncertainty_metrics.csv ---")
    calibrator = ConformalPredictionCalibrator()

    # Split independent calibration (first 50%) and test (remaining 50%)
    n_total = len(df_preds)
    n_cal = n_total // 2

    y_true_pm25 = df_preds["actual_pm25"].values
    y_pred_pm25 = df_preds["pred_adapted_gbdt_pm25"].values

    y_true_pm10 = df_preds["actual_pm10"].values
    y_pred_pm10 = df_preds["pred_adapted_gbdt_pm10"].values

    # Fit calibration on independent split
    calibrator.fit_calibration_set(y_true_pm25[:n_cal], y_pred_pm25[:n_cal], parameter="PM2.5")
    calibrator.fit_calibration_set(y_true_pm10[:n_cal], y_pred_pm10[:n_cal], parameter="PM10")

    # Evaluate on held-out test split
    df_unc_25 = calibrator.evaluate_coverage(y_true_pm25[n_cal:], y_pred_pm25[n_cal:], parameter="PM2.5")
    df_unc_10 = calibrator.evaluate_coverage(y_true_pm10[n_cal:], y_pred_pm10[n_cal:], parameter="PM10")

    df_unc = pd.concat([df_unc_25, df_unc_10], ignore_index=True)
    df_unc.to_csv(PHASE12_DIR / "uncertainty_metrics.csv", index=False)
    print("Saved experiments/phase12/uncertainty_metrics.csv")
    print(df_unc[["parameter", "nominal_pct", "empirical_coverage", "coverage_error", "mean_interval_width", "winkler_score"]])

    print("\nAll evaluation artifacts generated successfully.")


if __name__ == "__main__":
    main()
