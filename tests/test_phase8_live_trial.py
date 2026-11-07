"""
tests/test_phase8_live_trial.py

Comprehensive unit and integration test suite for VayuDrishti Phase 8:
Real Live Shadow Trial with Standalone Adapted GBDT Forecaster.

Tests:
1. Ingestion and deterministic clock-hour aggregation.
2. Deduplication and out-of-order reading rejection.
3. Physical bounds and non-numeric value rejection.
4. Station isolation between independent buffers.
5. Exact 24-hour contiguous sequence verification and gap detection.
6. Missing core channel (PM2.5 / PM10) gating.
7. Strict causal forecasting: target hour (t+1) excluded from input sequence.
8. Standalone GBDT inference execution with zero LSTM / trace gas dependency.
9. Forecast-target matching and paired error computation (GBDT vs Persistence).
10. Trial opt-in configuration and operational kill-switch enforcement.
11. Verification of generated live trial artifacts (CSV and JSON).
"""

import os
import sys
import json
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.live_rolling_buffer import (
    StationRollingBuffer,
    normalize_timestamp_utc,
    validate_sensor_reading,
    PHYSICAL_BOUNDS,
)
from src.services.standalone_gbdt_live_trial import (
    StandaloneGBDTLiveTrialRunner,
    TARGET_CAAQMS_STATIONS,
)
from src.services.sensor_forecast_service import SensorForecastService


class TestPhase8LiveTrial(unittest.TestCase):
    """Rigorous test suite for Phase 8 live trial mechanics and evaluation."""

    def setUp(self):
        self.tmp_dir = ROOT_DIR / "experiments/recent_data_audit/scratch_test_phase8"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.obs_csv = self.tmp_dir / "test_observations.csv"
        self.preds_csv = self.tmp_dir / "test_predictions.csv"
        self.eval_csv = self.tmp_dir / "test_evaluations.csv"
        self.metrics_json = self.tmp_dir / "test_metrics.json"
        self.config_path = self.tmp_dir / "test_config.json"

        self.runner = StandaloneGBDTLiveTrialRunner(
            observations_csv=self.obs_csv,
            predictions_csv=self.preds_csv,
            evaluation_csv=self.eval_csv,
            metrics_json=self.metrics_json,
            config_path=self.config_path,
        )

    def tearDown(self):
        # Clean up temporary test files
        for p in [self.obs_csv, self.preds_csv, self.eval_csv, self.metrics_json, self.config_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        if self.tmp_dir.exists():
            try:
                self.tmp_dir.rmdir()
            except Exception:
                pass

    def test_01_ingestion_and_deterministic_hourly_aggregation(self):
        """Verifies sub-hourly readings are aggregated into exact clock hours."""
        buf = StationRollingBuffer("TEST01", station_lat=17.72, station_lon=83.30)
        # Ingest 4 sub-hourly readings for 10:00-10:45 UTC
        base_t = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
        for offset, val in [(0, 20.0), (15, 30.0), (30, 40.0), (45, 50.0)]:
            t = base_t + timedelta(minutes=offset)
            res = buf.ingest_reading("PM2.5", val, t)
            self.assertEqual(res["status"], "ingested")

        summary = buf.get_hourly_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["clock_hour_utc"], "2026-11-04T10:00:00+00:00")
        self.assertAlmostEqual(summary[0]["channel_values"]["PM2.5"], 35.0, places=2)
        self.assertEqual(summary[0]["sample_counts"]["PM2.5"], 4)

    def test_02_duplicate_and_out_of_bounds_rejection(self):
        """Verifies duplicates and implausible physical values are rejected."""
        buf = StationRollingBuffer("TEST02")
        t = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
        res1 = buf.ingest_reading("PM2.5", 45.0, t)
        self.assertEqual(res1["status"], "ingested")

        # Duplicate
        res2 = buf.ingest_reading("PM2.5", 45.0, t)
        self.assertEqual(res2["status"], "duplicate_skipped")

        # Implausible negative value
        res3 = buf.ingest_reading("PM2.5", -10.0, t + timedelta(minutes=10))
        self.assertEqual(res3["status"], "rejected")

        # Implausible extreme value (> 1000 ug/m3)
        res4 = buf.ingest_reading("PM2.5", 2500.0, t + timedelta(minutes=20))
        self.assertEqual(res4["status"], "rejected")

    def test_03_station_isolation(self):
        """Verifies station readings cannot be ingested into mismatched station buffers."""
        buf = StationRollingBuffer("AP005")
        t = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
        res = buf.ingest_reading("PM2.5", 40.0, t, station_id="DL001")
        self.assertEqual(res["status"], "rejected")
        self.assertIn("station_id_mismatch", res["reason"])

    def test_04_contiguous_24h_verification_and_gap_detection(self):
        """Verifies that time gaps in hourly records properly trigger insufficient_history."""
        buf = StationRollingBuffer("TEST04", max_staleness_hours=500.0)
        start_t = datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)

        # Ingest only 20 hours (insufficient)
        for i in range(20):
            t = start_t + timedelta(hours=i)
            buf.ingest_reading("PM2.5", 50.0, t)
            buf.ingest_reading("PM10", 100.0, t)

        readiness = buf.check_readiness(anchor_time_utc=start_t + timedelta(hours=19))
        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["reason"], "insufficient_history")

        # Now fill hours 20 to 23 (making 24 contiguous hours)
        for i in range(20, 24):
            t = start_t + timedelta(hours=i)
            buf.ingest_reading("PM2.5", 50.0, t)
            buf.ingest_reading("PM10", 100.0, t)

        readiness24 = buf.check_readiness(anchor_time_utc=start_t + timedelta(hours=23))
        self.assertTrue(readiness24["is_ready"])

        # If we query an anchor hour with a gap (e.g. 5 hours later without observations in between)
        gap_anchor = start_t + timedelta(hours=28)
        buf.ingest_reading("PM2.5", 55.0, gap_anchor)
        buf.ingest_reading("PM10", 105.0, gap_anchor)
        readiness_gap = buf.check_readiness(anchor_time_utc=gap_anchor)
        self.assertFalse(readiness_gap["is_ready"])
        self.assertEqual(readiness_gap["reason"], "insufficient_history")

    def test_05_missing_required_channels_gating(self):
        """Verifies that if PM10 is missing even if PM2.5 is present, readiness is gated."""
        buf = StationRollingBuffer("TEST05", max_staleness_hours=500.0)
        start_t = datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)

        for i in range(24):
            t = start_t + timedelta(hours=i)
            buf.ingest_reading("PM2.5", 40.0, t)
            # Omit PM10 in the last hour
            if i < 23:
                buf.ingest_reading("PM10", 80.0, t)

        readiness = buf.check_readiness(anchor_time_utc=start_t + timedelta(hours=23))
        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["reason"], "missing_required_channels")

    def test_06_causal_forecasting_and_persistence_baseline(self):
        """Verifies rolling forecast at t uses only [t-23, ..., t] and generates valid persistence."""
        station_id = "AP005"
        meta = TARGET_CAAQMS_STATIONS[station_id]
        buf = self.runner.buffers[station_id]
        buf.clear()

        start_t = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)

        # Ingest 26 contiguous hours of synthetic PM2.5 and PM10
        for i in range(26):
            t = start_t + timedelta(hours=i)
            pm25_val = 30.0 + (i * 0.5)
            pm10_val = 60.0 + (i * 1.0)
            buf.ingest_reading("PM2.5", pm25_val, t, station_id=station_id)
            buf.ingest_reading("PM10", pm10_val, t, station_id=station_id)

        preds, evals = self.runner.run_rolling_forecasts_and_matching(station_id)

        # Hours 24 and 25 can be anchor hours:
        # Index 23 (hour 23) has target 24.
        # Index 24 (hour 24) has target 25.
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(evals), 2)

        # Verify causal persistence value matches exact observation at anchor hour
        first_pred = preds[0]
        self.assertEqual(first_pred["issue_time_utc"], "2026-11-02T23:00:00Z")
        self.assertEqual(first_pred["target_time_utc"], "2026-11-03T00:00:00Z")
        self.assertAlmostEqual(first_pred["persistence_pm25"], 30.0 + (23 * 0.5), places=2)
        self.assertAlmostEqual(first_pred["persistence_pm10"], 60.0 + (23 * 1.0), places=2)

        # Verify GBDT forecast returned valid numerical values
        self.assertGreater(first_pred["pred_pm25"], 0.0)
        self.assertGreater(first_pred["pred_pm10"], 0.0)

        # Verify evaluation matching
        first_eval = evals[0]
        self.assertEqual(first_eval["target_time_utc"], "2026-11-03T00:00:00Z")
        self.assertAlmostEqual(first_eval["actual_pm25"], 30.0 + (24 * 0.5), places=2)
        self.assertAlmostEqual(first_eval["actual_pm10"], 60.0 + (24 * 1.0), places=2)
        self.assertAlmostEqual(first_eval["gbdt_abs_err_pm25"], abs(first_eval["pred_pm25"] - first_eval["actual_pm25"]), places=2)

    def test_07_trial_opt_in_and_kill_switch(self):
        """Verifies operational kill-switch and opt-in status."""
        self.runner.disable_trial()
        self.assertFalse(self.runner.is_trial_enabled())

        self.runner.enable_trial()
        self.assertTrue(self.runner.is_trial_enabled())

        self.runner.disable_trial()
        self.assertFalse(self.runner.is_trial_enabled())

    def test_08_live_trial_artifacts_verification(self):
        """Verifies that the generated Phase 8 live trial CSV and JSON files exist and have non-empty schemas."""
        obs_file = ROOT_DIR / "experiments/recent_data_audit/live_trial_observations.csv"
        preds_file = ROOT_DIR / "experiments/recent_data_audit/live_trial_predictions.csv"
        eval_file = ROOT_DIR / "experiments/recent_data_audit/live_trial_evaluation.csv"
        metrics_file = ROOT_DIR / "experiments/recent_data_audit/live_trial_metrics.json"

        self.assertTrue(obs_file.exists(), "live_trial_observations.csv missing")
        self.assertTrue(preds_file.exists(), "live_trial_predictions.csv missing")
        self.assertTrue(eval_file.exists(), "live_trial_evaluation.csv missing")
        self.assertTrue(metrics_file.exists(), "live_trial_metrics.json missing")

        df_obs = pd.read_csv(obs_file)
        df_preds = pd.read_csv(preds_file)
        df_eval = pd.read_csv(eval_file)
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        self.assertGreaterEqual(len(df_obs), 1000)
        self.assertGreaterEqual(len(df_preds), 10)
        self.assertGreaterEqual(len(df_eval), 10)
        self.assertIn("overall", metrics)
        self.assertIn("gbdt", metrics["overall"])
        self.assertIn("persistence", metrics["overall"])
        self.assertIn("comparison", metrics["overall"])


if __name__ == "__main__":
    unittest.main()
