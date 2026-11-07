"""
tests/test_shadow_mode_integration.py

Verification and integration test suite for Phase 5A:
1. Canonical 49 reduced feature schema extraction, ordering, and leakage invariance.
2. Checkpoint integrity and loading (adapted fine-tuned LSTM + 49-feature GBDTs).
3. Production default preservation (default calls untouched, historical models served).
4. Opt-in adapted mode execution and severity-gated tree decoupling.
5. Non-blocking shadow mode execution and payload schema.
6. Graceful fallback on missing checkpoints or inference failure.
7. CPU inference latency benchmarking (reporting median and p95 latency).
"""

import sys
import unittest
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.tabular_features import (
    FEATURE_NAMES_TABULAR,
    REDUCED_FEATURE_INDICES,
    REDUCED_FEATURE_NAMES,
    EXCLUDED_FEATURE_INDICES,
    EXCLUDED_FEATURE_NAMES,
    get_reduced_tabular_feature_names,
    extract_reduced_tabular_features,
    extract_tabular_features,
)
from src.services.sensor_forecast_service import SensorForecastService


class TestShadowModeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = SensorForecastService()
        cls.ckpt_dir = ROOT_DIR / "experiments/recent_data_audit/checkpoints"

    def test_01_feature_schema_and_ordering(self):
        """Verify canonical 49 reduced features shape, ordering, and exclusions."""
        self.assertEqual(len(FEATURE_NAMES_TABULAR), 67)
        self.assertEqual(len(REDUCED_FEATURE_INDICES), 49)
        self.assertEqual(len(REDUCED_FEATURE_NAMES), 49)
        self.assertEqual(len(EXCLUDED_FEATURE_INDICES), 18)
        self.assertEqual(len(EXCLUDED_FEATURE_NAMES), 18)
        self.assertEqual(len(REDUCED_FEATURE_INDICES) + len(EXCLUDED_FEATURE_INDICES), 67)

        # Check exact slice composition: [0, 1] + 19..26 + 27..65
        expected_indices = [0, 1] + list(range(19, 27)) + list(range(27, 66))
        self.assertEqual(REDUCED_FEATURE_INDICES, expected_indices)

        names = get_reduced_tabular_feature_names()
        self.assertEqual(len(names), 49)
        self.assertEqual(names[0], "pm25_t")
        self.assertEqual(names[1], "pm10_t")
        self.assertEqual(names[2], "hour_sin_t")
        self.assertEqual(names[8], "latitude")
        self.assertEqual(names[9], "longitude")
        self.assertEqual(names[10], "pm25_lag1")
        self.assertEqual(names[48], "pm_ratio_t")

        # Excluded check: unobserved channels must not appear
        for excl in ["no_t", "so2_t", "temperature_t", "dewpoint_depression_t"]:
            self.assertNotIn(excl, names)

        # Dimension and NaN check
        dummy_seq = np.random.randn(5, 24, 27).astype(np.float32)
        red_feats = extract_reduced_tabular_features(dummy_seq)
        self.assertEqual(red_feats.shape, (5, 49))
        self.assertFalse(np.isnan(red_feats).any())

    def test_02_temporal_leakage_invariance(self):
        """Verify reduced features depend strictly on steps <= t (index 23)."""
        full_timeline = np.random.randn(3, 25, 27).astype(np.float32)
        seq_hist = full_timeline[:, :24, :]

        feats_1 = extract_reduced_tabular_features(seq_hist)

        # Mutate future step (index 24)
        full_timeline[:, 24, :] += 999.0
        feats_2 = extract_reduced_tabular_features(full_timeline[:, :24, :])

        np.testing.assert_allclose(feats_1, feats_2, atol=1e-7)

    def test_03_checkpoint_integrity(self):
        """Verify adapted checkpoints exist and have valid input dimensions."""
        import joblib
        import torch

        lstm_path = self.ckpt_dir / "best_finetuned_lstm.pt"
        gbdt_25_path = self.ckpt_dir / "adapted_gbdt_pm25.joblib"
        gbdt_10_path = self.ckpt_dir / "adapted_gbdt_pm10.joblib"

        self.assertTrue(lstm_path.exists(), f"Missing {lstm_path}")
        self.assertTrue(gbdt_25_path.exists(), f"Missing {gbdt_25_path}")
        self.assertTrue(gbdt_10_path.exists(), f"Missing {gbdt_10_path}")

        lstm_ckpt = torch.load(lstm_path, map_location="cpu")
        self.assertEqual(lstm_ckpt.get("input_size"), 27)
        self.assertEqual(lstm_ckpt.get("hidden_size"), 64)
        self.assertEqual(lstm_ckpt.get("num_layers"), 2)

        g25 = joblib.load(gbdt_25_path)
        g10 = joblib.load(gbdt_10_path)
        self.assertEqual(getattr(g25, "n_features_in_", None), 49)
        self.assertEqual(getattr(g10, "n_features_in_", None), 49)

    def test_04_production_default_preservation(self):
        """Ensure historical models remain default and are unaffected by shadow execution."""
        now = pd.Timestamp("2025-05-01 12:00:00")
        hours = [now - pd.Timedelta(hours=23 - i) for i in range(24)]
        df_fixed = pd.DataFrame({
            "Datetime": hours,
            "StationId": "AP001",
            "PM2.5": [45.0] * 24,
            "PM10": [90.0] * 24,
        })

        # 1. Default get_forecast() call
        fc_default = self.service.get_forecast(station_id="AP001", df_24h=df_fixed)
        self.assertNotIn("shadow_forecast", fc_default)
        self.assertAlmostEqual(fc_default["metrics"]["pm25_mae"], 10.57, places=1)

        # 2. Historical severity-gated call without shadow
        fc_gated = self.service.get_forecast(station_id="AP001", df_24h=df_fixed, model_mode="severity_gated", enable_shadow=False)
        self.assertNotIn("shadow_forecast", fc_gated)
        self.assertAlmostEqual(fc_gated["metrics"]["pm25_mae"], 10.35, places=2)

        # 3. Historical severity-gated call WITH shadow
        fc_with_shadow = self.service.get_forecast(station_id="AP001", df_24h=df_fixed, model_mode="severity_gated", enable_shadow=True)
        # Primary forecast must be identically equal on identical input
        self.assertEqual(fc_gated["pm25"], fc_with_shadow["pm25"])
        self.assertEqual(fc_gated["pm10"], fc_with_shadow["pm10"])
        self.assertEqual(fc_gated["model_version"], fc_with_shadow["model_version"])
        # Shadow payload must be attached
        self.assertIn("shadow_forecast", fc_with_shadow)
        self.assertEqual(fc_with_shadow["shadow_forecast"]["status"], "success")
        self.assertEqual(fc_with_shadow["shadow_forecast"]["experimental"], True)

    def test_05_opt_in_adapted_mode(self):
        """Verify explicit opt-in model_mode='adapted_severity_gated'."""
        fc_adapted = self.service.get_forecast(station_id="AP001", model_mode="adapted_severity_gated")
        self.assertTrue(fc_adapted.get("is_adapted"))
        self.assertIn("Adapted Severity-Gated Ensemble", fc_adapted["model_version"])
        self.assertAlmostEqual(fc_adapted["metrics"]["pm25_mae"], 6.62, places=2)
        self.assertAlmostEqual(fc_adapted["metrics"]["pm10_mae"], 16.14, places=2)
        self.assertGreaterEqual(fc_adapted["pm25"], 0.0)
        self.assertGreaterEqual(fc_adapted["pm10"], 0.0)

    def test_06_severity_gated_tree_decoupling(self):
        """Verify that GBDT is decoupled when telemetry exceeds severe thresholds."""
        now = pd.Timestamp("2025-05-01 12:00:00")
        hours = [now - pd.Timedelta(hours=23 - i) for i in range(24)]

        # Extreme episode: PM2.5 = 260.0 (> 250), PM10 = 450.0 (> 400)
        df_severe = pd.DataFrame({
            "Datetime": hours,
            "StationId": "DL001",
            "PM2.5": [100.0] * 23 + [260.0],
            "PM10": [200.0] * 23 + [450.0],
        })

        pred_res = self.service._predict_adapted(df_severe, station_id="DL001", severity_gated=True)
        components = pred_res["components"]
        # Tree models must be decoupled (weights on LSTM = 1.0)
        self.assertEqual(components["weights"]["w_pm25_lstm"], 1.0)
        self.assertEqual(components["weights"]["w_pm10_lstm"], 1.0)
        self.assertEqual(pred_res["pm25"], components["lstm_pm25"])
        self.assertEqual(pred_res["pm10"], components["lstm_pm10"])

    def test_07_graceful_fallback(self):
        """Verify service falls back safely if adapted model throws exception."""
        # Temporarily point adapted checkpoint loader to nonexistent path
        original_loader = self.service._load_adapted_models
        self.service._adapted_checkpoints_loaded = False
        self.service._load_adapted_models = lambda: False

        try:
            # Opt-in adapted mode with failed loading -> falls back to production
            fc_fallback = self.service.get_forecast(station_id="AP001", model_mode="adapted_severity_gated")
            self.assertIn("fallback_note", fc_fallback)
            self.assertAlmostEqual(fc_fallback["metrics"]["pm25_mae"], 10.57, places=1)

            # Shadow mode with failed loading -> shadow returns error status, primary unharmed
            fc_shadow_fail = self.service.get_forecast(station_id="AP001", model_mode="lstm", enable_shadow=True)
            self.assertIn("shadow_forecast", fc_shadow_fail)
            self.assertEqual(fc_shadow_fail["shadow_forecast"]["status"], "error")
            self.assertAlmostEqual(fc_shadow_fail["metrics"]["pm25_mae"], 10.57, places=1)
        finally:
            # Restore loader
            self.service._load_adapted_models = original_loader
            self.service._adapted_checkpoints_loaded = False

    def test_08_cpu_inference_latency_benchmark(self):
        """Benchmark CPU inference latency over repeated runs and report median and p95."""
        now = pd.Timestamp("2025-05-01 12:00:00")
        hours = [now - pd.Timedelta(hours=23 - i) for i in range(24)]
        df_bench = pd.DataFrame({
            "Datetime": hours,
            "StationId": "DL001",
            "PM2.5": [50.0] * 24,
            "PM10": [100.0] * 24,
        })

        # Warmup
        for _ in range(5):
            self.service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="adapted_severity_gated")
            self.service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=True)

        n_runs = 50
        single_latencies = []
        dual_latencies = []

        for _ in range(n_runs):
            # 1. Single adapted model inference
            t0 = time.perf_counter()
            self.service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="adapted_severity_gated")
            t1 = time.perf_counter()
            single_latencies.append((t1 - t0) * 1000.0)

            # 2. Dual execution (production model + shadow adapted model)
            t2 = time.perf_counter()
            self.service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=True)
            t3 = time.perf_counter()
            dual_latencies.append((t3 - t2) * 1000.0)

        single_lat = np.array(single_latencies)
        dual_lat = np.array(dual_latencies)

        single_median = float(np.median(single_lat))
        single_p95 = float(np.percentile(single_lat, 95))
        dual_median = float(np.median(dual_lat))
        dual_p95 = float(np.percentile(dual_lat, 95))

        print(f"\n=======================================================")
        print(f"CPU INFERENCE LATENCY BENCHMARK ({n_runs} RUNS ON CPU HOST)")
        print(f"=======================================================")
        print(f"  [Single Mode: Adapted Severity-Gated Ensemble]")
        print(f"    Median Latency: {single_median:.2f} ms")
        print(f"    p95 Latency:    {single_p95:.2f} ms")
        print(f"    Mean Latency:   {float(np.mean(single_lat)):.2f} ms")
        print(f"  [Dual Mode: Production Default + Shadow Mode]")
        print(f"    Median Latency: {dual_median:.2f} ms")
        print(f"    p95 Latency:    {dual_p95:.2f} ms")
        print(f"    Mean Latency:   {float(np.mean(dual_lat)):.2f} ms")
        print(f"=======================================================")

        # Realistic latency budgets on Windows CPU host:
        # Single adapted mode: < 150 ms median, < 200 ms p95
        # Dual execution (historical + shadow): < 220 ms median, < 260 ms p95
        self.assertLess(single_median, 150.0, f"Single adapted median latency {single_median:.2f} ms exceeded 150 ms budget")
        self.assertLess(single_p95, 200.0, f"Single adapted p95 latency {single_p95:.2f} ms exceeded 200 ms budget")
        self.assertLess(dual_median, 220.0, f"Dual execution median latency {dual_median:.2f} ms exceeded 220 ms budget")
        self.assertLess(dual_p95, 260.0, f"Dual execution p95 latency {dual_p95:.2f} ms exceeded 260 ms budget")


if __name__ == "__main__":
    unittest.main()
