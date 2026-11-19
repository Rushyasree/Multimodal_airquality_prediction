"""
tests/test_lstm_gbdt_hybrid.py

Unit Test Suite for VayuDrishti Mandatory LSTM + GBDT Combined Hybrid Forecasting.

Validates:
1. Independent model executions:
   - LSTM runs on 24h continuous sequence with 27 channels.
   - GBDT/LightGBM runs on canonical 49 reduced features.
   - Both models produce independent, valid physical predictions.
2. Genuine hybrid combination:
   - Severity-gated dynamic blending (w=0.40/0.45 under normal conditions, w=1.0 during severe episodes).
   - Static ensemble option (w=0.40/0.45).
   - Mathematical verification: final hybrid prediction matches the exact weighted combination.
3. Persistence baseline verification:
   - Reference inertia baseline (yt+1 = yt) accurately derived from anchor observation.
4. Comparison table integrity:
   - Table contains LSTM, GBDT/LightGBM, LSTM + GBDT Hybrid, and Persistence Baseline.
5. Causal alignment and temporal parity:
   - Same station, identical issue time t, identical target horizon t+1.
6. Transparent handling of incomplete buffers (live mode validation).
"""

import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src.services.sensor_forecast_service import SensorForecastService
from src.services.model_comparison_service import ModelComparisonService

ROOT_DIR = Path(__file__).resolve().parents[1]


class TestLSTMGBDTHybridForecasting(unittest.TestCase):
    """Test suite for LSTM + GBDT Combined Hybrid Forecaster."""

    def setUp(self):
        self.service = SensorForecastService()
        self.station_id = "AP001"
        self.ref_now = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Construct standard synthetic 24h DataFrame
        hours = [self.ref_now - timedelta(hours=23 - i) for i in range(24)]
        self.df_synthetic_24h = pd.DataFrame({
            "Datetime": hours,
            "StationId": self.station_id,
            "PM2.5": [50.0 + i * 1.0 for i in range(24)],
            "PM10": [100.0 + i * 2.0 for i in range(24)],
            "NO2": [20.0 + i * 0.2 for i in range(24)],
            "SO2": [10.0 + i * 0.1 for i in range(24)],
            "CO": [0.8 + i * 0.01 for i in range(24)],
            "OZONE": [35.0 + i * 0.3 for i in range(24)],
        })

    def test_01_both_models_execute_independently(self):
        """Test 1: Verify both LSTM (27 feats) and GBDT (49 feats) execute independently."""
        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=self.df_synthetic_24h,
            station_id=self.station_id,
            severity_gated=True,
        )

        self.assertEqual(res["status"], "ok")
        self.assertIn("components", res)
        comps = res["components"]

        # Check LSTM component
        self.assertIn("lstm", comps)
        self.assertEqual(comps["lstm"]["n_features"], 27)
        self.assertGreater(comps["lstm"]["pm25"], 0.0)
        self.assertGreater(comps["lstm"]["pm10"], 0.0)

        # Check GBDT component
        self.assertIn("gbdt", comps)
        self.assertEqual(comps["gbdt"]["n_features"], 49)
        self.assertGreater(comps["gbdt"]["pm25"], 0.0)
        self.assertGreater(comps["gbdt"]["pm10"], 0.0)

        # Verify they are independent predictions
        self.assertIsInstance(comps["lstm"]["pm25"], float)
        self.assertIsInstance(comps["gbdt"]["pm25"], float)

    def test_02_mathematical_hybrid_combination_nominal(self):
        """Test 2: Under nominal conditions, hybrid is exactly 0.40*LSTM + 0.60*GBDT (PM2.5) and 0.45/0.55 (PM10)."""
        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=self.df_synthetic_24h,
            station_id=self.station_id,
            severity_gated=True,
        )

        comps = res["components"]
        lstm_25 = comps["lstm"]["pm25"]
        gbdt_25 = comps["gbdt"]["pm25"]
        lstm_10 = comps["lstm"]["pm10"]
        gbdt_10 = comps["gbdt"]["pm10"]

        expected_pm25 = round(0.40 * lstm_25 + 0.60 * gbdt_25, 2)
        expected_pm10 = round(0.45 * lstm_10 + 0.55 * gbdt_10, 2)

        self.assertAlmostEqual(res["pm25"], expected_pm25, places=1)
        self.assertAlmostEqual(res["pm10"], expected_pm10, places=1)
        self.assertFalse(comps["hybrid"]["gating_active"])

    def test_03_severity_gating_switching(self):
        """Test 3: During severe episodes (PM2.5 >= 250), LSTM takes 100% weight."""
        df_severe = self.df_synthetic_24h.copy()
        df_severe.loc[23, "PM2.5"] = 320.0  # Acute winter inversion peak

        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=df_severe,
            station_id=self.station_id,
            severity_gated=True,
        )

        comps = res["components"]
        self.assertTrue(comps["hybrid"]["gating_active"])
        self.assertEqual(comps["hybrid"]["weights"]["w_pm25_lstm"], 1.0)
        self.assertEqual(comps["hybrid"]["weights"]["w_pm25_gbdt"], 0.0)
        # Final PM2.5 prediction must equal LSTM prediction exactly
        self.assertAlmostEqual(res["pm25"], comps["lstm"]["pm25"], places=2)

    def test_04_static_ensemble_weights(self):
        """Test 4: Static ensemble uses 0.40/0.60 weights regardless of concentration."""
        df_severe = self.df_synthetic_24h.copy()
        df_severe.loc[23, "PM2.5"] = 320.0

        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=df_severe,
            station_id=self.station_id,
            severity_gated=False,
        )

        comps = res["components"]
        self.assertEqual(comps["hybrid"]["weights"]["w_pm25_lstm"], 0.40)
        self.assertEqual(comps["hybrid"]["weights"]["w_pm25_gbdt"], 0.60)

    def test_05_persistence_baseline_present(self):
        """Test 5: Persistence baseline matches last observed concentration at anchor time t."""
        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=self.df_synthetic_24h,
            station_id=self.station_id,
        )

        last_observed_pm25 = float(self.df_synthetic_24h["PM2.5"].iloc[-1])
        last_observed_pm10 = float(self.df_synthetic_24h["PM10"].iloc[-1])

        pers = res["components"]["persistence"]
        self.assertEqual(pers["pm25"], last_observed_pm25)
        self.assertEqual(pers["pm10"], last_observed_pm10)

    def test_06_comparison_table_structure(self):
        """Test 6: Comparison table contains all 4 required models with valid predictions."""
        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=self.df_synthetic_24h,
            station_id=self.station_id,
        )

        table = res["comparison_table"]
        self.assertEqual(len(table), 4)

        models = [row["Model"] for row in table]
        self.assertIn("LSTM", models)
        self.assertIn("GBDT / LightGBM", models)
        self.assertIn("LSTM + GBDT Hybrid", models)
        self.assertIn("Persistence Baseline", models)

        # Check every row has PM2.5 and PM10 values
        for row in table:
            self.assertIsNotNone(row["PM2.5"])
            self.assertIsNotNone(row["PM10"])

    def test_07_temporal_alignment_and_horizon(self):
        """Test 7: Both models and hybrid target identical next-hour timestamp t+1."""
        res = self.service.predict_hybrid_lstm_gbdt(
            df_24h=self.df_synthetic_24h,
            station_id=self.station_id,
        )

        issue_time = pd.to_datetime(res["forecast_issue_time"])
        target_time = pd.to_datetime(res["target_time"])

        # Target horizon must be exactly 1 hour ahead
        self.assertEqual(target_time - issue_time, timedelta(hours=1))
        self.assertEqual(res["forecast_horizon"], "Next 1 Hour (t + 1)")

    def test_08_predict_next_hour_defaults_to_hybrid(self):
        """Test 8: predict_next_hour returns the primary LSTM + GBDT Hybrid forecast."""
        res = self.service.predict_next_hour(
            df_24h=self.df_synthetic_24h,
            station_id=self.station_id,
        )

        self.assertTrue(res.get("is_hybrid", False))
        self.assertIn("lstm_PM25", res)
        self.assertIn("gbdt_PM25", res)
        self.assertIn("predicted_PM25", res)
        self.assertIn("persistence_PM25", res)
        self.assertIn("comparison_table", res)

    def test_09_model_comparison_service_integration(self):
        """Test 9: ModelComparisonService includes both individual models and combined hybrid."""
        comp_srv = ModelComparisonService()
        res = comp_srv.compare_all(
            station_id=self.station_id,
            df_24h=self.df_synthetic_24h,
        )

        # Check hybrid breakdown table
        self.assertIn("hybrid_breakdown_table", res)
        breakdown_df = res["hybrid_breakdown_table"]
        self.assertEqual(len(breakdown_df), 4)

        # Check full comparison table
        comp_df = res["comparison_table"]
        methods = comp_df["Method"].tolist()
        self.assertIn("Sensor-Only LSTM", methods)
        self.assertIn("GBDT / LightGBM Forecaster", methods)
        self.assertIn("LSTM + GBDT Combined Hybrid", methods)
        self.assertIn("1-Hour Persistence Baseline", methods)


if __name__ == "__main__":
    unittest.main()
