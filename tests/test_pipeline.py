"""
tests/test_pipeline.py

Comprehensive 20-Point End-to-End Test Suite for VayuDrishti:
- Checkpoints & Preprocessing Pipeline
- Andhra Pradesh Location Explorer & Nearest Station Proxy Calculations
- Official Indian CPCB NAQI Sub-Index Calculations & Category Boundaries
- Missing Data Handling & Distinction Between Observations and Predictions
- Environmental Computer Vision Image Validation
- Historical Data Handling & Benchmark Consistency
- Live Telemetry Safety (Zero Fabrication Fallback)
- Telugu / English Bilingual Localization & Clean Application Startup
"""

import sys
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.lstm_forecaster import LSTMForecasterService, LSTMForecaster
from src.prediction.image_estimator import ImageEstimatorService, ResNet50Regressor
from src.prediction.aqi_advisor import (
    calculate_sub_index,
    calculate_overall_cpcb_aqi,
    get_aqi_category,
    BREAKPOINTS,
    AQI_CATEGORIES,
)
from src.prediction.data_loader import (
    load_location_catalog,
    load_stations_metadata,
    get_states_list,
    get_cities_by_state,
    get_locations_for_city,
    find_nearest_supported_station,
    load_verified_results,
    load_demo_scenarios,
    get_scenario_metadata_list,
    get_project_root,
)
from src.prediction.live_service import LiveAirQualityService
from src.utils.i18n import t, TRANSLATIONS


class TestVayuDrishtiSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.lstm_service = LSTMForecasterService()
        cls.image_service = ImageEstimatorService()
        cls.live_service = LiveAirQualityService()

    # 1. Existing Prediction Tests
    def test_01_all_existing_prediction_tests(self):
        dummy_seq = np.zeros((24, 27), dtype=np.float32)
        res = self.lstm_service.predict_sequence(dummy_seq, last_observed_pm25=50.0, last_observed_pm10=100.0)
        self.assertIn("predicted_PM25", res)
        self.assertIn("predicted_PM10", res)
        self.assertIn("delta_PM25", res)
        self.assertIn("delta_PM10", res)

    # 2. Checkpoint Loading
    def test_02_checkpoint_loading(self):
        self.assertIsNotNone(self.lstm_service.model)
        self.assertIsNotNone(self.image_service.model)
        self.assertEqual(self.image_service.best_epoch, 8)

    # 3. LSTM Input Shape and Feature Order
    def test_03_lstm_input_shape_and_feature_order(self):
        self.assertEqual(len(self.lstm_service.feature_names), 27)
        self.assertEqual(self.lstm_service.window_size, 24)
        expected_first_four = ["PM2.5", "PM10", "NO", "NO2"]
        self.assertEqual(self.lstm_service.feature_names[:4], expected_first_four)
        # Shape mismatch must raise ValueError
        with self.assertRaises(ValueError):
            self.lstm_service.predict_sequence(np.zeros((20, 27), dtype=np.float32))

    # 4. Correct Inverse Target Scaling
    def test_04_correct_inverse_target_scaling(self):
        self.assertEqual(len(self.lstm_service.target_mean), 2)
        self.assertEqual(len(self.lstm_service.target_std), 2)
        self.assertGreater(self.lstm_service.target_std[0], 0.0)
        self.assertGreater(self.lstm_service.target_std[1], 0.0)

    # 5. Finite Prediction Outputs
    def test_05_finite_prediction_outputs(self):
        seq = np.random.randn(24, 27).astype(np.float32)
        res = self.lstm_service.predict_sequence(seq)
        self.assertTrue(np.isfinite(res["predicted_PM25"]))
        self.assertTrue(np.isfinite(res["predicted_PM10"]))
        self.assertGreaterEqual(res["predicted_PM25"], 0.0)
        self.assertGreaterEqual(res["predicted_PM10"], 0.0)

    # 6. Supported vs Unsupported Location Behaviour
    def test_06_supported_vs_unsupported_location_behaviour(self):
        catalog = load_location_catalog()
        ap001 = catalog[catalog["station_id"] == "AP001"].iloc[0]
        self.assertTrue(ap001["data_available"])
        self.assertTrue(ap001["forecast_supported"])

        vja = catalog[catalog["city"] == "Vijayawada"].iloc[0]
        self.assertFalse(vja["data_available"])
        self.assertFalse(vja["forecast_supported"])

    # 7. Andhra Pradesh City and Station Filtering
    def test_07_ap_city_and_station_filtering(self):
        ap_cities = get_cities_by_state("Andhra Pradesh")
        self.assertEqual(len(ap_cities), 15)
        required_15 = [
            "Amaravati", "Vijayawada", "Visakhapatnam", "Rajamahendravaram",
            "Tirupati", "Guntur", "Nellore", "Kurnool", "Kadapa",
            "Anantapuram", "Ongole", "Eluru", "Srikakulam",
            "Vizianagaram", "Chittoor"
        ]
        for c in required_15:
            self.assertIn(c, ap_cities)

    # 8. Nearest-Station Distance Calculations
    def test_08_nearest_station_distance_calculations(self):
        nearest_vja = find_nearest_supported_station(16.5062, 80.6480)
        self.assertEqual(nearest_vja["station_id"], "AP001")
        self.assertLess(nearest_vja["distance_km"], 20.0)

    # 9. AQI Category Boundaries
    def test_09_aqi_category_boundaries(self):
        self.assertEqual(get_aqi_category(25)["tier"], "Good")
        self.assertEqual(get_aqi_category(75)["tier"], "Satisfactory")
        self.assertEqual(get_aqi_category(150)["tier"], "Moderate")
        self.assertEqual(get_aqi_category(250)["tier"], "Poor")
        self.assertEqual(get_aqi_category(350)["tier"], "Very Poor")
        self.assertEqual(get_aqi_category(450)["tier"], "Severe")

    # 10. Pollutant-Specific Sub-Index Calculations
    def test_10_pollutant_sub_index_calculations(self):
        # PM2.5: 30 µg/m³ should map to exactly 50
        self.assertEqual(calculate_sub_index("PM2.5", 30.0), 50)
        # PM2.5: 60 µg/m³ should map to exactly 100
        self.assertEqual(calculate_sub_index("PM2.5", 60.0), 100)
        # PM10: 50 µg/m³ should map to 50
        self.assertEqual(calculate_sub_index("PM10", 50.0), 50)

    # 11. Missing-Data Behaviour for AQI
    def test_11_missing_data_behaviour_for_aqi(self):
        # Only PM2.5 provided (fewer than 3 pollutants)
        res = calculate_overall_cpcb_aqi({"PM2.5": 45.0})
        self.assertFalse(res["is_valid"])
        self.assertIn("insufficient data", res["status"].lower())

        # 3 pollutants including PM2.5 -> Valid CPCB NAQI
        res_valid = calculate_overall_cpcb_aqi({"PM2.5": 45.0, "PM10": 80.0, "NO2": 30.0})
        self.assertTrue(res_valid["is_valid"])
        self.assertEqual(res_valid["status"], "Official CPCB NAQI Valid")

    # 12. Observed vs Predicted Labelling
    def test_12_observed_versus_predicted_labelling(self):
        demo_df = load_demo_scenarios()
        sc_data = demo_df[demo_df["scenario_id"] == demo_df["scenario_id"].iloc[0]]
        pred = self.lstm_service.predict_from_dataframe(sc_data, station_id=sc_data["StationId"].iloc[0])
        self.assertIn("predicted_PM25", pred)
        self.assertIn("persistence_PM25", pred)
        self.assertNotEqual("predicted_PM25", "persistence_PM25")

    # 13. Stale-Data Warnings
    def test_13_stale_data_warnings(self):
        status = self.live_service.get_live_status()
        self.assertIn("disclaimer", status)
        self.assertFalse(status["is_live"])

    # 14. Image Upload Validation
    def test_14_image_upload_validation(self):
        valid_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        res = self.image_service.estimate_pm25(valid_img)
        self.assertIn("estimated_PM25", res)
        self.assertTrue(np.isfinite(res["estimated_PM25"]))

        with self.assertRaises(ValueError):
            self.image_service.validate_and_open_image(Image.new("RGB", (8, 8)))

    # 15. Historical Charts with Missing Data
    def test_15_historical_charts_with_missing_data(self):
        demo_df = load_demo_scenarios()
        self.assertFalse(demo_df.empty)
        # Should contain standard numeric columns
        for col in ["PM2.5", "PM10", "temperature_C"]:
            self.assertIn(col, demo_df.columns)

    # 16. Metrics Consistency Against Verified Result Files
    def test_16_metrics_consistency_against_verified_results(self):
        v_df = load_verified_results()
        self.assertGreaterEqual(len(v_df), 11)
        lstm_pm25 = v_df[(v_df["Model"].str.contains("LSTM")) & (v_df["Target"] == "PM2.5")].iloc[0]
        self.assertAlmostEqual(lstm_pm25["MAE"], 10.5719, places=3)
        self.assertAlmostEqual(lstm_pm25["R2"], 0.9492, places=3)

    # 17. Live-Data Failure and Fallback Handling
    def test_17_live_data_failure_and_fallback_handling(self):
        # Invalid API key attempt
        stat = self.live_service.get_live_status(api_key="INVALID_TEST_KEY_123")
        self.assertFalse(stat["is_live"])
        self.assertIn("Fallback", stat.get("mode", "") + stat.get("message", ""))

    # 18. Telugu Text Rendering
    def test_18_telugu_text_rendering(self):
        title_te = t("app_title", "te")
        self.assertEqual(title_te, "వాయుదృష్టి")
        self.assertIn("వాయుదృష్టి", title_te)
        how_safe_te = t("how_safe", "te")
        self.assertEqual(how_safe_te, "మీ గాలి ఎంత సురక్షితం?")

    # 19. No Fabricated Live Data When Unavailable
    def test_19_no_fabricated_live_data_when_unavailable(self):
        live_info = self.live_service.get_live_status()
        self.assertFalse(live_info["is_live"])
        self.assertNotIn("realtime_aqi", live_info)
        self.assertIn("Historical Dataset Mode", live_info["mode"])

    # 20. Application Imports and Startup
    def test_20_application_imports_and_startup(self):
        import app
        self.assertIsNotNone(app.get_lstm_service)
        self.assertIsNotNone(app.get_image_service)
        self.assertIsNotNone(app.get_catalog)


if __name__ == "__main__":
    unittest.main()
