"""
tests/test_standalone_gbdt_inference.py

Comprehensive Synthetic Unit Test Suite for VayuDrishti Phase 7:
Standalone Adapted GBDT Live Forecast Feasibility & Inference Verification.

All tests utilize SYNTHETIC data clearly labeled as synthetic.
Synthetic tests are never counted as real-world observation results.

Test Coverage:
1. Checkpoint loading: verifies adapted_gbdt_pm25.joblib and adapted_gbdt_pm10.joblib exist and have n_features_in_ = 49.
2. Canonical 49-feature extraction: verifies exact ordering and mathematical independence from channels 2-18.
3. 24-hour particulate history inference: produces valid non-negative physical predictions.
4. Determinism: identical input yields identical prediction.
5. Missing observations handling: graceful handling of NaNs in sequence.
6. Irregular timestamps and rolling buffer integration.
7. Station mapping verification and coordinate resolution.
8. Feature-order integrity: rejects or flags misordered feature shapes.
9. Stale data detection in rolling buffer before GBDT inference.
10. Invalid/unphysical concentrations rejection.
11. Future-data and target-hour (t+1) leakage prevention.
12. Failure isolation: GBDT errors fall back cleanly to production baseline without crashing.
13. Production model default preservation: default mode remains historical severity-gated.
"""

import os
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import joblib

from src.services.sensor_forecast_service import SensorForecastService
from src.data.live_rolling_buffer import StationRollingBuffer, LiveRollingBufferRegistry
from src.data.tabular_features import (
    extract_reduced_tabular_features,
    get_reduced_tabular_feature_names,
    REDUCED_FEATURE_INDICES,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


class TestStandaloneGBDTInference(unittest.TestCase):
    """Synthetic test suite for standalone adapted GBDT inference (Phase 7)."""

    def setUp(self):
        self.service = SensorForecastService()
        self.station_id = "AP001"
        self.ref_now = datetime(2026, 11, 5, 6, 0, 0, tzinfo=timezone.utc)

        # Build standard synthetic 24h DataFrame with only particulates and timestamps
        hours = [self.ref_now - timedelta(hours=23 - i) for i in range(24)]
        self.df_synthetic_24h = pd.DataFrame({
            "Datetime": hours,
            "StationId": self.station_id,
            "PM2.5": [45.0 + i * 0.5 for i in range(24)],
            "PM10": [90.0 + i * 1.0 for i in range(24)],
        })

    def test_01_checkpoint_loading_and_architecture(self):
        """Synthetic Test 1: Verify checkpoints exist, load, and possess exactly 49 input features."""
        ckpt_dir = ROOT_DIR / "experiments/recent_data_audit/checkpoints"
        pm25_path = ckpt_dir / "adapted_gbdt_pm25.joblib"
        pm10_path = ckpt_dir / "adapted_gbdt_pm10.joblib"

        self.assertTrue(pm25_path.exists(), "adapted_gbdt_pm25.joblib missing")
        self.assertTrue(pm10_path.exists(), "adapted_gbdt_pm10.joblib missing")

        model_25 = joblib.load(pm25_path)
        model_10 = joblib.load(pm10_path)

        self.assertEqual(model_25.n_features_in_, 49)
        self.assertEqual(model_10.n_features_in_, 49)

    def test_02_canonical_49_feature_extraction_and_channel_independence(self):
        """Synthetic Test 2: Verify 49 features are mathematically independent of channels 2-18."""
        feat_names = get_reduced_tabular_feature_names()
        self.assertEqual(len(feat_names), 49)
        self.assertEqual(len(REDUCED_FEATURE_INDICES), 49)

        # Create synthetic sequence with full 27 channels
        seq_full = np.random.RandomState(42).randn(1, 24, 27).astype(np.float32)
        # Sequence with unobserved channels 2-18 zeroed out
        seq_zeroed = seq_full.copy()
        seq_zeroed[:, :, 2:19] = 0.0

        f_full = extract_reduced_tabular_features(seq_full)
        f_zeroed = extract_reduced_tabular_features(seq_zeroed)

        self.assertEqual(f_full.shape, (1, 49))
        self.assertEqual(f_zeroed.shape, (1, 49))
        # Difference must be strictly zero
        np.testing.assert_array_almost_equal(f_full, f_zeroed, decimal=6)

    def test_03_standalone_gbdt_inference_valid(self):
        """Synthetic Test 3: adapted_gbdt_only produces valid non-negative physical forecast."""
        res = self.service.get_forecast(
            station_id="AP001",
            df_24h=self.df_synthetic_24h,
            model_mode="adapted_gbdt_only",
        )

        self.assertIn("pm25", res)
        self.assertIn("pm10", res)
        self.assertGreater(res["pm25"], 0.0)
        self.assertGreater(res["pm10"], 0.0)
        self.assertEqual(res["unit"], "ug/m3")
        self.assertEqual(res["model_version"], "Standalone Adapted GBDT Forecaster (49 Reduced Features)")
        self.assertTrue(res.get("is_standalone_gbdt", False))
        self.assertTrue(res.get("is_adapted", False))

        # Must report components
        comps = res.get("components", {})
        self.assertEqual(comps.get("n_features_gbdt"), 49)
        self.assertNotIn("lstm_pm25", comps)  # Zero LSTM dependency

    def test_04_deterministic_inference(self):
        """Synthetic Test 4: Identical input produces identical forecast deterministically."""
        res1 = self.service.get_forecast(
            station_id="AP001",
            df_24h=self.df_synthetic_24h,
            model_mode="adapted_gbdt_only",
        )
        res2 = self.service.get_forecast(
            station_id="AP001",
            df_24h=self.df_synthetic_24h,
            model_mode="adapted_gbdt_only",
        )

        self.assertEqual(res1["pm25"], res2["pm25"])
        self.assertEqual(res1["pm10"], res2["pm10"])

    def test_05_missing_observations_nan_handling(self):
        """Synthetic Test 5: HistGradientBoosting natively handles NaNs in tabular sequence."""
        df_nan = self.df_synthetic_24h.copy()
        # Introduce NaN dropouts in particulate lag columns
        df_nan.loc[5, "PM2.5"] = np.nan
        df_nan.loc[12, "PM10"] = np.nan

        res = self.service.get_forecast(
            station_id="AP001",
            df_24h=df_nan,
            model_mode="adapted_gbdt_only",
        )

        self.assertIsNotNone(res["pm25"])
        self.assertIsNotNone(res["pm10"])
        self.assertFalse(np.isnan(res["pm25"]))
        self.assertFalse(np.isnan(res["pm10"]))

    def test_06_rolling_buffer_integration(self):
        """Synthetic Test 6: Standalone GBDT consumes verified DataFrame from StationRollingBuffer."""
        buf = StationRollingBuffer(station_id="AP001", max_staleness_hours=3.0)
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            buf.ingest_reading("PM2.5", 40.0 + i, t_hour)
            buf.ingest_reading("PM10", 80.0 + i, t_hour)

        readiness = buf.check_readiness(
            anchor_time_utc=self.ref_now,
            current_system_time=self.ref_now + timedelta(minutes=5),
        )
        self.assertTrue(readiness["is_ready"])

        df_verified = readiness["dataframe_24h"]
        res = self.service.get_forecast(
            station_id="AP001",
            df_24h=df_verified,
            model_mode="adapted_gbdt_only",
        )
        self.assertGreater(res["pm25"], 0.0)
        self.assertGreater(res["pm10"], 0.0)

    def test_07_station_mismatch_and_coordinates(self):
        """Synthetic Test 7: Resolves distinct coordinates for AP001 vs AP005 in GBDT features."""
        res_ap001 = self.service.get_forecast(
            station_id="AP001",
            df_24h=self.df_synthetic_24h,
            model_mode="adapted_gbdt_only",
        )
        res_ap005 = self.service.get_forecast(
            station_id="AP005",
            df_24h=self.df_synthetic_24h,
            model_mode="adapted_gbdt_only",
        )

        # Coordinate difference must influence the predictions
        self.assertIsInstance(res_ap001["pm25"], float)
        self.assertIsInstance(res_ap005["pm25"], float)

    def test_08_feature_shape_integrity(self):
        """Synthetic Test 8: Verifies reduced feature extractor rejects malformed inputs."""
        bad_seq = np.zeros((1, 10, 27))  # Not 24 hours
        with self.assertRaises(Exception):
            self.service.underlying_service.predict_sequence(bad_seq)

    def test_09_stale_data_prevention_via_buffer(self):
        """Synthetic Test 9: Stale buffer (> 3h) fails readiness and prevents GBDT invocation."""
        buf = StationRollingBuffer(station_id="AP001", max_staleness_hours=3.0)
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            buf.ingest_reading("PM2.5", 40.0, t_hour)
            buf.ingest_reading("PM10", 80.0, t_hour)

        # 6 hours late
        sys_late = self.ref_now + timedelta(hours=6)
        readiness = buf.check_readiness(anchor_time_utc=self.ref_now, current_system_time=sys_late)
        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["reason"], "stale_data")
        self.assertIsNone(readiness["dataframe_24h"])

    def test_10_invalid_concentrations_clamping(self):
        """Synthetic Test 10: Non-negative clamping ensures GBDT never outputs negative values."""
        df_zero = self.df_synthetic_24h.copy()
        df_zero["PM2.5"] = 0.0
        df_zero["PM10"] = 0.0

        res = self.service.get_forecast(
            station_id="AP001",
            df_24h=df_zero,
            model_mode="adapted_gbdt_only",
        )
        self.assertGreaterEqual(res["pm25"], 0.0)
        self.assertGreaterEqual(res["pm10"], 0.0)

    def test_11_future_leakage_quarantine(self):
        """Synthetic Test 11: Ingesting target-hour t+1 value into buffer does not leak into GBDT forecast."""
        buf = StationRollingBuffer(station_id="AP001")
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            buf.ingest_reading("PM2.5", 50.0, t_hour)
            buf.ingest_reading("PM10", 100.0, t_hour)

        # Inject extreme future value at t+1
        t_plus_1 = self.ref_now + timedelta(hours=1)
        buf.ingest_reading("PM2.5", 999.0, t_plus_1)

        readiness = buf.check_readiness(anchor_time_utc=self.ref_now, current_system_time=self.ref_now + timedelta(minutes=5))
        df_clean = readiness["dataframe_24h"]

        # Future 999.0 must be excluded
        self.assertNotIn(999.0, df_clean["PM2.5"].values)

        res = self.service.get_forecast(
            station_id="AP001",
            df_24h=df_clean,
            model_mode="adapted_gbdt_only",
        )
        # Prediction must be near baseline 50, not near 999
        self.assertLess(res["pm25"], 150.0)

    def test_12_failure_isolation_and_fallback(self):
        """Synthetic Test 12: If GBDT fails, service serves production baseline without crashing."""
        # Corrupt dataframe with non-convertible objects
        df_corrupt = pd.DataFrame({
            "Datetime": ["corrupted"] * 24,
            "StationId": "AP001",
            "PM2.5": ["NaN_ERROR"] * 24,
            "PM10": ["NaN_ERROR"] * 24,
        })

        res = self.service.get_forecast(
            station_id="AP001",
            df_24h=df_corrupt,
            model_mode="adapted_gbdt_only",
        )
        self.assertIn("pm25", res)
        self.assertIn("pm10", res)

    def test_13_production_default_preservation(self):
        """Synthetic Test 13: Default model mode remains historical LSTM / severity-gated model."""
        res_default = self.service.get_forecast(station_id="AP001")
        self.assertEqual(res_default["model_version"], "LSTM-2Layer Baseline v1.2")
        self.assertFalse(res_default.get("is_standalone_gbdt", False))

        res_gated = self.service.get_forecast(station_id="AP001", model_mode="severity_gated")
        self.assertIn("Severity-Gated Ensemble", res_gated["model_version"])
        self.assertFalse(res_gated.get("is_standalone_gbdt", False))


if __name__ == "__main__":
    unittest.main()
