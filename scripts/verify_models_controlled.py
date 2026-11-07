"""
scripts/verify_models_controlled.py

Stage 4: Controlled Model Loading & Forecast Verification
=========================================================
Audits:
1. Historical 2-layer LSTM forecaster (input shape: [1, 24, 27], outputs: [PM2.5, PM10])
2. Standalone Adapted GBDT models (LightGBM/HistGradientBoosting, 49 features)
3. Severity-Gated Hybrid Ensemble (adaptive routing between LSTM and GBDT)
4. Vision ResNet-50 Baseline (2048-d visual feature regression)
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure Windows PyTorch DLLs are accessible
if sys.platform == "win32":
    try:
        import site
        for p in site.getsitepackages():
            torch_lib = os.path.join(p, "torch", "lib")
            if os.path.isdir(torch_lib):
                try:
                    os.add_dll_directory(torch_lib)
                except (AttributeError, OSError):
                    pass
    except Exception:
        pass

from src.prediction.lstm_forecaster import LSTMForecasterService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.image_estimation_service import ImageEstimationService


def audit_lstm_forecaster():
    print("\n--- 1. AUDITING HISTORICAL 2-LAYER LSTM FORECASTER ---")
    ckpt_path = ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt"
    scalers_path = ROOT_DIR / "data/processed/model_sequences/preprocessing_metadata.json"
    
    assert ckpt_path.exists(), f"LSTM checkpoint missing: {ckpt_path}"
    assert scalers_path.exists(), f"Scalers metadata missing: {scalers_path}"
    
    # 1. Direct service verification
    service = LSTMForecasterService(model_path=str(ckpt_path), metadata_path=str(scalers_path))
    assert service.model is not None, "LSTM service failed to load model"
    print("  [PASS] LSTMForecasterService initialized & weights loaded.")
    
    # 2. Production pipeline forecast via SensorForecastService
    sensor_srv = SensorForecastService()
    fc = sensor_srv.predict_next_hour(scenario_id="scenario_01", station_id="AP001", model_mode="lstm")
    
    assert "predicted_PM25" in fc and "predicted_PM10" in fc, "Missing PM2.5 or PM10 in forecast"
    pm25_pred = fc["predicted_PM25"]
    pm10_pred = fc["predicted_PM10"]
    
    assert np.isfinite(pm25_pred), f"PM2.5 prediction is not finite: {pm25_pred}"
    assert np.isfinite(pm10_pred), f"PM10 prediction is not finite: {pm10_pred}"
    assert pm25_pred >= 0, f"PM2.5 prediction must be non-negative: {pm25_pred}"
    assert pm10_pred >= 0, f"PM10 prediction must be non-negative: {pm10_pred}"
    assert "uncertainty_intervals" in fc, "Uncertainty intervals missing"
    
    print(f"  [PASS] Controlled LSTM Inference on Scenario 'scenario_01' (Station AP001):")
    print(f"         Next-Hour PM2.5 = {pm25_pred:.2f} µg/m³ (NAQI: {fc.get('aqi_category_PM25', 'N/A')})")
    print(f"         Next-Hour PM10  = {pm10_pred:.2f} µg/m³ (NAQI: {fc.get('aqi_category_PM10', 'N/A')})")
    u_dict = fc.get("uncertainty_intervals", {})
    p25_low = u_dict.get("PM2.5_lower_80", 0)
    p25_high = u_dict.get("PM2.5_upper_80", 0)
    print(f"         80% Bounds: PM2.5 in [{p25_low:.1f}, {p25_high:.1f}] µg/m³")


def audit_standalone_gbdt():
    print("\n--- 2. AUDITING STANDALONE ADAPTED GBDT (49 FEATURES) ---")
    pm25_ckpt = ROOT_DIR / "experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib"
    pm10_ckpt = ROOT_DIR / "experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib"
    
    assert pm25_ckpt.exists(), f"PM2.5 GBDT missing: {pm25_ckpt}"
    assert pm10_ckpt.exists(), f"PM10 GBDT missing: {pm10_ckpt}"
    from src.data.tabular_features import REDUCED_FEATURE_NAMES
    assert len(REDUCED_FEATURE_NAMES) == 49, f"Expected 49 features, got {len(REDUCED_FEATURE_NAMES)}"
    print(f"  [PASS] Canonical 49-feature manifest verified: {len(REDUCED_FEATURE_NAMES)} features.")
    
    m_pm25 = joblib.load(pm25_ckpt)
    m_pm10 = joblib.load(pm10_ckpt)
    print(f"  [PASS] Checkpoints loaded successfully: {m_pm25.__class__.__name__}")
    
    # Run standalone GBDT forecast through SensorForecastService
    sensor_srv = SensorForecastService()
    fc_gbdt = sensor_srv.predict_next_hour(scenario_id="scenario_01", station_id="AP001", model_mode="adapted_gbdt_only")
    
    p25 = fc_gbdt["predicted_PM25"]
    p10 = fc_gbdt["predicted_PM10"]
    assert np.isfinite(p25), "GBDT PM2.5 prediction not finite"
    assert np.isfinite(p10), "GBDT PM10 prediction not finite"
    print(f"  [PASS] Controlled GBDT Inference Output on Scenario 'scenario_01':")
    print(f"         PM2.5 = {p25:.2f} µg/m³ (finite, physical scale)")
    print(f"         PM10  = {p10:.2f} µg/m³ (finite, physical scale)")


def audit_severity_gated_hybrid():
    print("\n--- 3. AUDITING SEVERITY-GATED HYBRID ENSEMBLE ---")
    sensor_srv = SensorForecastService()
    fc_gated = sensor_srv.predict_next_hour(scenario_id="scenario_01", station_id="AP001", model_mode="adapted_severity_gated")
    
    assert "predicted_PM25" in fc_gated and "predicted_PM10" in fc_gated
    print(f"  [PASS] Severity-Gated Ensemble Inference on Scenario 'scenario_01':")
    print(f"         PM2.5 = {fc_gated['predicted_PM25']:.2f} µg/m³")
    print(f"         PM10  = {fc_gated['predicted_PM10']:.2f} µg/m³")
    
    # Multimodal hybrid service check
    hybrid_srv = HybridPredictionService()
    assert hybrid_srv.is_loaded, "Hybrid service failed to load checkpoints"
    res_fallback = hybrid_srv.predict_hybrid(station_id="AP001", image_input=None)
    assert res_fallback["gating_alpha"] == 0.0, "Telemetry fallback should set alpha=0.0"
    print(f"  [PASS] Multimodal fallback mode verified (alpha = {res_fallback['gating_alpha']}, status = {res_fallback['fallback_status']})")


def audit_vision_baseline():
    print("\n--- 4. AUDITING VISION RESNET-50 BASELINE ---")
    img_srv = ImageEstimationService()
    assert img_srv.underlying_service.model is not None, "Image estimation model is not loaded"
    curated = img_srv.get_curated_sample_images()
    assert len(curated) > 0, "No curated sample images found"
    
    sample_path = curated[0].get("image_path") or curated[0].get("path")
    assert Path(sample_path).exists(), f"Sample image file not found: {sample_path}"
    
    est = img_srv.estimate_from_file(sample_path)
    assert "pm25_pred" in est and "category" in est
    assert np.isfinite(est["pm25_pred"])
    cat_str = est['category'] if isinstance(est['category'], str) else est['category'].get('name', 'N/A')
    print(f"  [PASS] ResNet-50 inference verified on '{Path(sample_path).name}':")
    print(f"         Visual PM2.5 estimate: {est['pm25_pred']:.2f} µg/m³ (Category: {cat_str})")


def main():
    print("=" * 80)
    print("STAGE 4: CONTROLLED MODEL LOADING & FORECAST INFERENCE VERIFICATION")
    print("=" * 80)
    audit_lstm_forecaster()
    audit_standalone_gbdt()
    audit_severity_gated_hybrid()
    audit_vision_baseline()
    print("=" * 80)
    print("ALL MODELS VERIFIED: 100% SUCCESS ACROSS ALL 4 FORECASTING PARADIGMS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
