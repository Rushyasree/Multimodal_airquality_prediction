"""
tests/test_eight_sections_smoke.py

Comprehensive Smoke Test for all 8 Dashboard Sections:
Verifies model loading, service execution, data dependencies, and error handling for:
1. Overview (Dual-Mode Summary)
2. Live Air Quality
3. Individual Forecasting Methods (Mode A)
4. Unified Hybrid Forecast (Mode B)
5. Side-by-Side Comparison
6. Station Map
7. Model Performance
8. Data Sources and System Health
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd
import numpy as np

from src.services.live_air_quality_service import LiveAirQualityService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.image_estimation_service import ImageEstimationService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.model_comparison_service import ModelComparisonService


def test_section_1_overview():
    print("Testing Section 1: Overview Dependencies...")
    live_srv = LiveAirQualityService()
    obs = live_srv.get_live_observation(station_id="AP001", city="Amaravati")
    assert "aqi" in obs or "pm25" in obs, "Overview observation structure missing"
    print("  [PASS] Section 1 Overview data verified.")


def test_section_2_live_air_quality():
    print("Testing Section 2: Live Air Quality Pipeline...")
    live_srv = LiveAirQualityService()
    obs = live_srv.get_live_observation(station_id="AP005", city="Visakhapatnam")
    assert obs.get("station_id") == "AP005", "Station ID resolution failed"
    assert "data_source" in obs, "Data source provenance missing"
    print("  [PASS] Section 2 Live Air Quality verified.")


def test_section_3_individual_methods():
    print("Testing Section 3: Individual Forecasting Methods (Mode A)...")
    sensor_srv = SensorForecastService()
    # Method 1: Sensor Forecast
    fc = sensor_srv.predict_next_hour(scenario_id="scenario_01", station_id="AP001")
    assert "predicted_PM25" in fc and "predicted_PM10" in fc, "Method 1 prediction missing targets"
    assert "uncertainty_intervals" in fc, "Method 1 uncertainty intervals missing"
    print("  [PASS] Method 1 (Sensor-Only LSTM) verified.")

    # Method 3: Image Estimation
    img_srv = ImageEstimationService()
    samples = img_srv.get_curated_sample_images()
    assert len(samples) > 0, "Curated sample images missing"
    sample_path = samples[0].get("image_path") or samples[0].get("path")
    est = img_srv.estimate_from_file(sample_path)
    assert "pm25_pred" in est and "category" in est, "Method 3 image estimate missing fields"
    assert "optical_features" in est, "Method 3 optical features missing"
    print("  [PASS] Method 3 (Image Estimation) verified.")

    # Method 4: Baselines
    comp_file = ROOT_DIR / "models/lstm_baseline/persistence_comparison.json"
    assert comp_file.exists(), "Persistence comparison artifact missing"
    with open(comp_file) as f:
        comp_data = json.load(f)
    assert "persistence" in comp_data and "lstm" in comp_data, "Baseline comparison fields missing"
    print("  [PASS] Method 4 (Baselines & Persistence) verified.")


def test_section_4_unified_hybrid():
    print("Testing Section 4: Unified Hybrid Forecast (Mode B)...")
    hybrid_srv = HybridPredictionService()
    assert hybrid_srv.is_loaded, "Unified Hybrid checkpoint failed to load"
    
    # Run with image
    img_srv = ImageEstimationService()
    samples = img_srv.get_curated_sample_images()
    sample_path = samples[0].get("image_path") or samples[0].get("path")
    res_img = hybrid_srv.predict_hybrid(station_id="AP001", image_input=sample_path)
    assert "pm25" in res_img and "pm10" in res_img, "Hybrid prediction missing targets"
    assert "gating_alpha" in res_img, "Gating alpha missing"
    print(f"  [PASS] Mode B Full Multimodal verified (alpha = {res_img['gating_alpha']}).")

    # Run without image (fallback mode)
    res_fall = hybrid_srv.predict_hybrid(station_id="AP001", image_input=None)
    assert "sensor_telemetry_fallback" in res_fall["fallback_status"], "Fallback not activated"
    assert res_fall["gating_alpha"] == 0.0, "Fallback alpha should be 0.0"
    print("  [PASS] Mode B Missing-Modality Fallback verified (alpha = 0.0).")


def test_section_5_side_by_side():
    print("Testing Section 5: Side-by-Side Comparison...")
    comp_srv = ModelComparisonService()
    res = comp_srv.compare_all(station_id="AP001", city="Amaravati")
    assert "comparison_table" in res, "Comparison table missing"
    df = res["comparison_table"]
    assert len(df) >= 4, "Comparison table missing methods"
    methods = df["Method"].tolist()
    assert "Live Ground Station" in methods, "Live station missing from comparison"
    assert "Sensor-Only LSTM" in methods, "Sensor LSTM missing from comparison"
    assert "Unified Hybrid Model" in methods, "Unified hybrid missing from comparison"
    print("  [PASS] Section 5 Side-by-Side Comparison verified.")


def test_section_6_station_map():
    print("Testing Section 6: Station Map...")
    coords_path = ROOT_DIR / "data/processed/station_coordinates_90.csv"
    assert coords_path.exists(), "Station coordinates file missing"
    df = pd.read_csv(coords_path)
    assert len(df) == 90, f"Expected 90 stations, found {len(df)}"
    assert "Latitude" in df.columns and "Longitude" in df.columns, "Coordinates columns missing"
    print(f"  [PASS] Section 6 Station Map verified ({len(df)} stations loaded).")


def test_section_7_model_performance():
    print("Testing Section 7: Model Performance Artifacts...")
    lstm_m = ROOT_DIR / "models/lstm_baseline/test_metrics.json"
    assert lstm_m.exists(), "LSTM metrics JSON missing"
    
    class_m = ROOT_DIR / "reports/classification_metrics_report.json"
    assert class_m.exists(), "Classification metrics report JSON missing"
    
    v2_m = ROOT_DIR / "experiments/multimodal_fusion/validation_v2/metrics.json"
    assert v2_m.exists(), "Validation v2 metrics missing"

    hybrid_m = ROOT_DIR / "experiments/unified_hybrid/metrics.json"
    assert hybrid_m.exists(), "Unified hybrid metrics missing"

    fig_cm25 = ROOT_DIR / "paper/figures/confusion_matrix_pm25.png"
    fig_cm10 = ROOT_DIR / "paper/figures/confusion_matrix_pm10.png"
    fig_act = ROOT_DIR / "paper/figures/actual_vs_predicted.png"
    fig_ts = ROOT_DIR / "paper/figures/prediction_timeseries.png"
    assert fig_cm25.exists() and fig_cm10.exists(), "Confusion matrix figures missing"
    assert fig_act.exists() and fig_ts.exists(), "Regression/timeseries figures missing"
    print("  [PASS] Section 7 Model Performance evaluation figures and metrics verified.")


def test_section_8_system_health():
    print("Testing Section 8: System Health...")
    # Verify core model checkpoint files exist
    assert (ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt").exists(), "best_lstm_model.pt missing"
    assert (ROOT_DIR / "models/pm25vision_cnn/best_model.pt").exists(), "pm25vision best_model.pt missing"
    assert (ROOT_DIR / "experiments/unified_hybrid/checkpoints/best_unified_hybrid_model.pt").exists(), "best_unified_hybrid_model.pt missing"
    print("  [PASS] Section 8 Model Checkpoints and Pipeline Health verified.")


def main():
    print("=" * 80)
    print("RUNNING END-TO-END SMOKE TEST FOR ALL 8 VAYUDRISHTI DASHBOARD SECTIONS")
    print("=" * 80)
    test_section_1_overview()
    test_section_2_live_air_quality()
    test_section_3_individual_methods()
    test_section_4_unified_hybrid()
    test_section_5_side_by_side()
    test_section_6_station_map()
    test_section_7_model_performance()
    test_section_8_system_health()
    print("=" * 80)
    print("ALL 8 SECTIONS SMOKE TEST PASSED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
