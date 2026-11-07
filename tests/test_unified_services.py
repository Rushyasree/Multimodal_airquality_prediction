"""
tests/test_unified_services.py

Comprehensive tests for the VayuDrishti unified service layer:
- LiveAirQualityService
- SensorForecastService
- ImageEstimationService
- HybridPredictionService
- ModelComparisonService
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image

from src.services.live_air_quality_service import LiveAirQualityService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.image_estimation_service import ImageEstimationService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.model_comparison_service import ModelComparisonService


def test_live_air_quality_service():
    service = LiveAirQualityService()
    res = service.get_live_observation(city="Amaravati", force_refresh=False)

    assert res["result_type"] == "Observation"
    assert "pm25" in res
    assert "observation_time" in res
    assert "data_source" in res
    assert res["unit"] == "ug/m3"
    assert "Live measurements are actual recorded observations" in res["scientific_note"]


def test_sensor_forecast_service():
    service = SensorForecastService()
    res = service.get_forecast(station_id="AP001")

    assert res["result_type"] == "Independent Forecast"
    assert res["pm25"] > 0
    assert res["pm10"] > 0
    assert "t + 1" in res["forecast_horizon"]
    assert "LSTM" in res["model_version"]
    assert res["unit"] == "ug/m3"
    assert "metrics" in res
    assert res["metrics"]["pm25_r2"] > 0.90


def test_image_estimation_service():
    service = ImageEstimationService()
    # Create synthetic test image
    test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    res = service.estimate_from_image(test_img)

    assert res["result_type"] == "Independent Estimate"
    assert res["pm25"] > 0
    assert res["pm10"] is None  # Image only supports PM2.5
    assert "Sub-daily EXIF timestamps absent in PM25Vision" in res["timestamp_precision"]
    assert "optical_features" in res


def test_hybrid_prediction_service_with_image():
    service = HybridPredictionService()
    assert service.is_loaded is True

    test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    res = service.predict_hybrid(station_id="AP001", image_input=test_img)

    assert res["result_type"] == "Unified Hybrid Prediction"
    assert res["pm25"] > 0
    assert res["pm10"] > 0
    assert res["modality_availability"]["visual_environment"] is True
    assert res["fallback_status"] == "active_multimodal_gated"
    assert "gating_alpha" in res
    assert "Hybrid" in res["model_version"]


def test_hybrid_prediction_service_fallback_without_image():
    service = HybridPredictionService()
    res = service.predict_hybrid(station_id="AP001", image_input=None)

    assert res["result_type"] == "Unified Hybrid Prediction"
    assert res["pm25"] > 0
    assert res["pm10"] > 0
    assert res["modality_availability"]["visual_environment"] is False
    assert "sensor_telemetry_fallback" in res["fallback_status"]
    assert res["gating_alpha"] == 0.0  # Masked gate ensures 0 visual contribution


def test_model_comparison_service():
    service = ModelComparisonService()
    res = service.compare_all(station_id="AP001", city="Amaravati", image_input=None)

    table = res["comparison_table"]
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 5

    methods = table["Method"].tolist()
    assert "Live Ground Station" in methods
    assert "Sensor-Only LSTM" in methods
    assert "Image-Only ResNet-50" in methods
    assert "Weather / Context" in methods
    assert "Unified Hybrid Model" in methods

    # Verify result types
    obs_row = table[table["Method"] == "Live Ground Station"].iloc[0]
    assert obs_row["Result Type"] == "Actual Observation"

    hybrid_row = table[table["Method"] == "Unified Hybrid Model"].iloc[0]
    assert hybrid_row["Result Type"] == "Integrated Multimodal Prediction"


if __name__ == "__main__":
    print("Testing live_air_quality_service...")
    test_live_air_quality_service()
    print("Testing sensor_forecast_service...")
    test_sensor_forecast_service()
    print("Testing image_estimation_service...")
    test_image_estimation_service()
    print("Testing hybrid_prediction_service_with_image...")
    test_hybrid_prediction_service_with_image()
    print("Testing hybrid_prediction_service_fallback_without_image...")
    test_hybrid_prediction_service_fallback_without_image()
    print("Testing model_comparison_service...")
    test_model_comparison_service()
    print("\nALL UNIFIED SERVICE TESTS PASSED SUCCESSFULLY!")

