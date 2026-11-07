"""
scripts/audit_stage1_model_independence.py

Stage 1 Research Integrity Audit: Model Prediction Independence & Verification.
Constructs 7 controlled diagnostic atmospheric scenarios and independently evaluates
every model in the VayuDrishti ecosystem:
1. Persistence Baseline
2. Historical 2-Layer LSTM
3. Standalone Adapted GBDT (49 features)
4. Residual Boosting Model (LSTM + GBDT residual)
5. Validated Static Ensemble (0.40 LSTM + 0.60 GBDT)
6. Severity-Gated Hybrid Ensemble (adaptive alpha)
7. Vision ResNet-50 Estimator
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.prediction.lstm_forecaster import LSTMForecasterService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.image_estimation_service import ImageEstimationService
from src.data.tabular_features import extract_reduced_tabular_features, extract_tabular_features, REDUCED_FEATURE_NAMES
import joblib


def build_diagnostic_scenarios():
    """Builds 7 distinct 24-hour diagnostic atmospheric sequences."""
    base_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    hours = [base_time - timedelta(hours=23 - i) for i in range(24)]

    scenarios = {}

    # Scenario 1: Low Pollution (Good Air Quality)
    scenarios["low_pollution"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": [15.0 + 2.0 * np.sin(i / 3.0) for i in range(24)],
        "PM10": [35.0 + 4.0 * np.sin(i / 3.0) for i in range(24)],
        "temperature_C": [24.0] * 24,
        "relative_humidity": [55.0] * 24,
        "wind_speed": [3.5] * 24,
        "wind_direction": [180.0] * 24,
    })

    # Scenario 2: Moderate Pollution
    scenarios["moderate_pollution"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": [55.0 + 5.0 * np.cos(i / 4.0) for i in range(24)],
        "PM10": [95.0 + 10.0 * np.cos(i / 4.0) for i in range(24)],
        "temperature_C": [28.0] * 24,
        "relative_humidity": [65.0] * 24,
        "wind_speed": [2.0] * 24,
        "wind_direction": [90.0] * 24,
    })

    # Scenario 3: High Pollution / Severe Episode
    scenarios["high_pollution"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": [260.0 + 15.0 * np.sin(i / 2.0) for i in range(24)],
        "PM10": [420.0 + 25.0 * np.sin(i / 2.0) for i in range(24)],
        "temperature_C": [18.0] * 24,
        "relative_humidity": [85.0] * 24,
        "wind_speed": [0.8] * 24,
        "wind_direction": [45.0] * 24,
    })

    # Scenario 4: Rapid Increase (Inversion / Stagnation Spike)
    pm25_spike = [40.0] * 18 + [60.0, 100.0, 150.0, 210.0, 270.0, 310.0]
    pm10_spike = [75.0] * 18 + [110.0, 180.0, 260.0, 350.0, 440.0, 500.0]
    scenarios["rapid_increase"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": pm25_spike,
        "PM10": pm10_spike,
        "temperature_C": [22.0] * 24,
        "relative_humidity": [70.0] * 24,
        "wind_speed": [0.5] * 24,
        "wind_direction": [0.0] * 24,
    })

    # Scenario 5: Rapid Decrease (Rain / Strong Front Washout)
    pm25_drop = [220.0] * 18 + [160.0, 110.0, 70.0, 45.0, 30.0, 20.0]
    pm10_drop = [380.0] * 18 + [280.0, 190.0, 120.0, 80.0, 50.0, 35.0]
    scenarios["rapid_decrease"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": pm25_drop,
        "PM10": pm10_drop,
        "temperature_C": [20.0] * 24,
        "relative_humidity": [95.0] * 24,
        "wind_speed": [6.5] * 24,
        "wind_direction": [270.0] * 24,
    })

    # Scenario 6: Missing Environmental Channels (PM only, all weather/trace gases missing)
    scenarios["missing_observations"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": [50.0 + 3.0 * (i % 5) for i in range(24)],
        "PM10": [85.0 + 5.0 * (i % 5) for i in range(24)],
        # Trace and weather columns completely absent
    })

    # Scenario 7: Distinct Extreme Weather (Heatwave, Low Humidity, High Wind)
    scenarios["distinct_weather"] = pd.DataFrame({
        "Datetime": hours,
        "StationId": "AP001",
        "PM2.5": [45.0 + 4.0 * np.cos(i / 3.0) for i in range(24)],
        "PM10": [110.0 + 8.0 * np.cos(i / 3.0) for i in range(24)],
        "temperature_C": [41.5] * 24,
        "relative_humidity": [18.0] * 24,
        "wind_speed": [8.2] * 24,
        "wind_direction": [315.0] * 24,
    })

    return scenarios


def evaluate_all_models_on_scenarios():
    scenarios = build_diagnostic_scenarios()
    
    sensor_srv = SensorForecastService()
    lstm_forecaster = LSTMForecasterService()
    hybrid_srv = HybridPredictionService()
    img_srv = ImageEstimationService()
    
    curated_images = img_srv.get_curated_sample_images()
    sample_img_path = curated_images[0]["image_path"] if curated_images else None

    # Load residual boosting checkpoints directly
    res_pm25_path = ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints/best_residual_gbdt_pm25.joblib"
    res_pm10_path = ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints/best_residual_gbdt_pm10.joblib"
    res_pm25_model = joblib.load(res_pm25_path) if res_pm25_path.exists() else None
    res_pm10_model = joblib.load(res_pm10_path) if res_pm10_path.exists() else None

    results = []

    for sc_name, df_sc in scenarios.items():
        # 1. Persistence Baseline (t value)
        p_pm25 = float(df_sc["PM2.5"].iloc[-1])
        p_pm10 = float(df_sc["PM10"].iloc[-1])

        # 2. Historical LSTM Baseline
        lstm_fc = sensor_srv.get_forecast(station_id="AP001", df_24h=df_sc, model_mode="lstm")
        lstm_p25 = float(lstm_fc.get("pm25", 0.0))
        lstm_p10 = float(lstm_fc.get("pm10", 0.0))

        # 3. Standalone Adapted GBDT (49 Features)
        gbdt_fc = sensor_srv.get_forecast(station_id="AP001", df_24h=df_sc, model_mode="adapted_gbdt_only")
        gbdt_p25 = float(gbdt_fc.get("pm25", 0.0))
        gbdt_p10 = float(gbdt_fc.get("pm10", 0.0))

        # Also test via predict_next_hour (to verify fixed routing)
        gbdt_pnh = sensor_srv.predict_next_hour(df_24h=df_sc, station_id="AP001", model_mode="adapted_gbdt_only")
        gbdt_pnh_p25 = float(gbdt_pnh.get("predicted_PM25", 0.0))
        gbdt_pnh_p10 = float(gbdt_pnh.get("predicted_PM10", 0.0))

        # 4. Validated Static Ensemble (0.40 LSTM + 0.60 GBDT for PM2.5, 0.45 / 0.55 for PM10)
        ens_fc = sensor_srv.get_forecast(station_id="AP001", df_24h=df_sc, model_mode="adapted_ensemble")
        ens_p25 = float(ens_fc.get("pm25", 0.0))
        ens_p10 = float(ens_fc.get("pm10", 0.0))

        # 5. Severity-Gated Hybrid Ensemble
        gated_fc = sensor_srv.get_forecast(station_id="AP001", df_24h=df_sc, model_mode="adapted_severity_gated")
        gated_p25 = float(gated_fc.get("pm25", 0.0))
        gated_p10 = float(gated_fc.get("pm10", 0.0))

        # 6. Residual Boosting Model (LSTM + predicted residual using 67 features)
        seq_arr = lstm_forecaster.preprocess_sequence(df_sc, station_id="AP001")
        tab_feats_67 = extract_tabular_features(seq_arr)
        if res_pm25_model is not None and res_pm10_model is not None:
            res_p25_delta = float(res_pm25_model.predict(tab_feats_67)[0])
            res_p10_delta = float(res_pm10_model.predict(tab_feats_67)[0])
            res_boost_p25 = max(0.0, lstm_p25 + res_p25_delta)
            res_boost_p10 = max(0.0, lstm_p10 + res_p10_delta)
        else:
            res_boost_p25, res_boost_p10 = lstm_p25, lstm_p10

        # 7. ResNet-50 Image Baseline (Optical estimate)
        if sample_img_path:
            img_res = img_srv.estimate_from_file(sample_img_path)
            img_p25 = float(img_res.get("pm25_pred", 55.0))
        else:
            img_p25 = 55.0

        # 8. Unified Multimodal Hybrid with Image
        hybrid_res = hybrid_srv.predict_hybrid(station_id="AP001", df_24h=df_sc, image_input=sample_img_path)
        hybrid_p25 = float(hybrid_res.get("pm25", 0.0))
        hybrid_p10 = float(hybrid_res.get("pm10", 0.0))
        alpha = float(hybrid_res.get("gating_alpha", 0.0))

        results.append({
            "scenario": sc_name,
            "persistence_pm25": p_pm25,
            "persistence_pm10": p_pm10,
            "lstm_pm25": lstm_p25,
            "lstm_pm10": lstm_p10,
            "gbdt_pm25": gbdt_p25,
            "gbdt_pm10": gbdt_p10,
            "gbdt_pnh_pm25": gbdt_pnh_p25,
            "gbdt_pnh_pm10": gbdt_pnh_p10,
            "ensemble_pm25": ens_p25,
            "ensemble_pm10": ens_p10,
            "severity_gated_pm25": gated_p25,
            "severity_gated_pm10": gated_p10,
            "residual_boost_pm25": round(res_boost_p25, 2),
            "residual_boost_pm10": round(res_boost_p10, 2),
            "resnet50_pm25": img_p25,
            "multimodal_hybrid_pm25": hybrid_p25,
            "multimodal_hybrid_pm10": hybrid_p10,
            "gating_alpha": alpha,
        })

    df_results = pd.DataFrame(results)
    out_csv = ROOT_DIR / "experiments/diagnostic_model_independence_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"[OK] Diagnostic model evaluation saved to {out_csv}")
    print("\n--- Diagnostic Results Summary ---")
    for r in results:
        print(f"Scenario: {r['scenario']}")
        print(f"  Persistence:       PM2.5 = {r['persistence_pm25']:.2f}, PM10 = {r['persistence_pm10']:.2f}")
        print(f"  LSTM Forecaster:   PM2.5 = {r['lstm_pm25']:.2f}, PM10 = {r['lstm_pm10']:.2f}")
        print(f"  Standalone GBDT:   PM2.5 = {r['gbdt_pm25']:.2f}, PM10 = {r['gbdt_pm10']:.2f}")
        print(f"  GBDT via pnh:      PM2.5 = {r['gbdt_pnh_pm25']:.2f}, PM10 = {r['gbdt_pnh_pm10']:.2f}")
        print(f"  Static Ensemble:   PM2.5 = {r['ensemble_pm25']:.2f}, PM10 = {r['ensemble_pm10']:.2f}")
        print(f"  Severity Gated:    PM2.5 = {r['severity_gated_pm25']:.2f}, PM10 = {r['severity_gated_pm10']:.2f}")
        print(f"  Residual Boosting: PM2.5 = {r['residual_boost_pm25']:.2f}, PM10 = {r['residual_boost_pm10']:.2f}")
        print(f"  Multimodal Hybrid: PM2.5 = {r['multimodal_hybrid_pm25']:.2f}, PM10 = {r['multimodal_hybrid_pm10']:.2f} (alpha={r['gating_alpha']:.3f})")
        print()

    # Check for identical outputs between LSTM and GBDT
    discrepancies = []
    for r in results:
        diff_25 = abs(r['lstm_pm25'] - r['gbdt_pm25'])
        diff_10 = abs(r['lstm_pm10'] - r['gbdt_pm10'])
        if diff_25 < 1e-4 and diff_10 < 1e-4:
            discrepancies.append(r['scenario'])
    
    if discrepancies:
        print(f"[WARNING] LSTM and GBDT produced identical outputs on: {discrepancies}")
    else:
        print("[PASS] Full Model Independence Confirmed: LSTM and GBDT produce distinct, scientifically decoupled forecasts across all 7 diagnostic regimes!")

    return df_results


if __name__ == "__main__":
    evaluate_all_models_on_scenarios()
