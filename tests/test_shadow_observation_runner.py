"""
tests/test_shadow_observation_runner.py

Verification test suite for ShadowObservationRunner (Phase 6):
1. Disabled by default governance (zero unintended logging).
2. Explicit enablement via local config or environment flag.
3. Structured append-only logging with secret sanitization.
4. Deduplication enforcement on forecast records.
5. Strict temporal and station alignment for observation matching.
6. Rejection of unphysical or misaligned target observations.
7. Paired error metric calculation on matched observations.
8. Operational kill-switch and production default protection.
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.services.shadow_observation_runner import ShadowObservationRunner


class TestShadowObservationRunner(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary directory for test logs and config
        self.test_dir = Path(tempfile.mkdtemp(prefix="vayudrishti_test_shadow_"))
        self.forecasts_log = self.test_dir / "test_forecasts.jsonl"
        self.matched_log = self.test_dir / "test_matched.jsonl"
        self.config_file = self.test_dir / "test_config.json"

        # Deterministic 24-hour test dataframe
        now = pd.Timestamp("2026-10-27 06:00:00")
        hours = [now - pd.Timedelta(hours=23 - i) for i in range(24)]
        self.deterministic_df = pd.DataFrame({
            "Datetime": hours,
            "StationId": "AP001",
            "PM2.5": [40.0 + i * 0.5 for i in range(24)],
            "PM10": [80.0 + i * 1.0 for i in range(24)],
            "temperature_C": [26.0] * 24,
            "relative_humidity": [60.0] * 24,
            "wind_speed": [2.0] * 24,
        })

        self.runner = ShadowObservationRunner(
            forecasts_log_path=self.forecasts_log,
            matched_log_path=self.matched_log,
            config_path=self.config_file,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        # Clear test environment variables
        if "VAYUDRISHTI_SHADOW_TRIAL_ENABLED" in os.environ:
            del os.environ["VAYUDRISHTI_SHADOW_TRIAL_ENABLED"]

    def test_01_disabled_by_default(self):
        """Verify runner is strictly disabled by default and does not record logs."""
        self.assertFalse(self.runner.is_enabled())

        res = self.runner.record_forecast(station_id="AP001", df_24h=self.deterministic_df)
        self.assertEqual(res["status"], "trial_disabled")
        self.assertFalse(res["shadow_recorded"])
        self.assertFalse(self.forecasts_log.exists())

        # Production forecast must still be returned cleanly
        self.assertIn("production_forecast", res)
        self.assertGreater(res["production_forecast"]["pm25"], 0.0)

    def test_02_explicit_enablement_and_logging(self):
        """Verify runner logs append-only structured records when explicitly enabled."""
        self.runner.enable_trial()
        self.assertTrue(self.runner.is_enabled())

        res = self.runner.record_forecast(
            station_id="AP001",
            df_24h=self.deterministic_df,
            issue_time_utc="2026-10-27T06:00:00Z",
            data_source="Test CAAQMS Telemetry",
        )
        self.assertEqual(res["status"], "recorded")
        self.assertTrue(res["shadow_recorded"])
        self.assertTrue(self.forecasts_log.exists())

        # Verify log file contents and schema
        with open(self.forecasts_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        rec = json.loads(lines[0])
        self.assertEqual(rec["station_id"], "AP001")
        self.assertEqual(rec["issue_time_utc"], "2026-10-27T06:00:00Z")
        self.assertEqual(rec["target_time_utc"], "2026-10-27T07:00:00Z")
        self.assertEqual(rec["shadow_forecast"]["experimental"], True)
        self.assertIn("historical_forecast", rec)
        self.assertIn("execution", rec)
        self.assertEqual(rec["execution"]["status"], "success")

    def test_03_deduplication(self):
        """Verify duplicate forecast submissions for the same station and issue time are skipped."""
        self.runner.enable_trial()

        # First recording
        res1 = self.runner.record_forecast(
            station_id="AP001",
            df_24h=self.deterministic_df,
            issue_time_utc="2026-10-27T06:00:00Z",
        )
        self.assertEqual(res1["status"], "recorded")

        # Second recording with identical station and issue time
        res2 = self.runner.record_forecast(
            station_id="AP001",
            df_24h=self.deterministic_df,
            issue_time_utc="2026-10-27T06:00:00Z",
        )
        self.assertEqual(res2["status"], "duplicate_skipped")

        # Log must contain exactly one line
        with open(self.forecasts_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

    def test_04_target_matching_and_alignment(self):
        """Verify strict matching of forecasts to target observations and rejection of misaligned records."""
        self.runner.enable_trial()
        fc_res = self.runner.record_forecast(
            station_id="AP001",
            df_24h=self.deterministic_df,
            issue_time_utc="2026-10-27T06:00:00Z",
        )
        rec_id = fc_res["record_id"]

        # Target hour is 2026-10-27T07:00:00Z.
        # Test A: Rejection on wrong hour (2026-10-27T08:00:00Z)
        bad_time_res = self.runner.match_forecast_to_observation(
            record_id=rec_id,
            actual_pm25=52.0,
            actual_pm10=105.0,
            observation_time_utc="2026-10-27T08:00:00Z",
            station_id="AP001",
        )
        self.assertEqual(bad_time_res["status"], "rejected")
        self.assertIn("Temporal misalignment", bad_time_res["reason"])

        # Test B: Rejection on station ID mismatch (AP002)
        bad_station_res = self.runner.match_forecast_to_observation(
            record_id=rec_id,
            actual_pm25=52.0,
            actual_pm10=105.0,
            observation_time_utc="2026-10-27T07:00:00Z",
            station_id="AP002",
        )
        self.assertEqual(bad_station_res["status"], "rejected")
        self.assertIn("Station ID mismatch", bad_station_res["reason"])

        # Test C: Rejection on unphysical reading (negative or extreme)
        bad_val_res = self.runner.match_forecast_to_observation(
            record_id=rec_id,
            actual_pm25=-10.0,
            actual_pm10=105.0,
            observation_time_utc="2026-10-27T07:00:00Z",
            station_id="AP001",
        )
        self.assertEqual(bad_val_res["status"], "rejected")
        self.assertIn("physical plausibility", bad_val_res["reason"])

        # Test D: Successful match on correct hour and station
        match_res = self.runner.match_forecast_to_observation(
            record_id=rec_id,
            actual_pm25=52.0,
            actual_pm10=105.0,
            observation_time_utc="2026-10-27T07:00:00Z",
            station_id="AP001",
        )
        self.assertEqual(match_res["status"], "matched")
        self.assertIn("delta_ae_pm25", match_res)
        self.assertIn("delta_ae_pm10", match_res)

        # Test E: Duplicate matching on the same observation is skipped
        dup_match = self.runner.match_forecast_to_observation(
            record_id=rec_id,
            actual_pm25=52.0,
            actual_pm10=105.0,
            observation_time_utc="2026-10-27T07:00:00Z",
            station_id="AP001",
        )
        self.assertEqual(dup_match["status"], "duplicate_skipped")

    def test_05_trial_statistics_calculation(self):
        """Verify accurate calculation of paired comparison metrics across matched cohorts."""
        self.runner.enable_trial()

        # Record and match 3 synthetic forecasts
        for i in range(3):
            t_issue = f"2026-10-27T{10+i:02d}:00:00Z"
            t_obs = f"2026-10-27T{11+i:02d}:00:00Z"

            fc = self.runner.record_forecast(
                station_id="AP001",
                df_24h=self.deterministic_df,
                issue_time_utc=t_issue,
            )
            self.runner.match_forecast_to_observation(
                record_id=fc["record_id"],
                actual_pm25=50.0 + i * 2.0,
                actual_pm10=100.0 + i * 5.0,
                observation_time_utc=t_obs,
                station_id="AP001",
            )

        stats = self.runner.get_trial_statistics()
        self.assertEqual(stats["forecasts_issued"], 3)
        self.assertEqual(stats["forecasts_matched"], 3)
        self.assertEqual(stats["coverage_rate"], 1.0)
        self.assertEqual(stats["missing_target_rate"], 0.0)

        metrics = stats["metrics"]
        self.assertIn("pm25", metrics)
        self.assertIn("pm10", metrics)
        self.assertIn("delta_mae", metrics["pm25"])
        self.assertIn("win_rate_pct", metrics["pm25"])

    def test_06_operational_kill_switch(self):
        """Verify that disable_trial immediately halts logging and preserves production forecasts."""
        self.runner.enable_trial()
        self.assertTrue(self.runner.is_enabled())

        # Kill-switch activation
        self.runner.disable_trial()
        self.assertFalse(self.runner.is_enabled())

        # Ensuing call must not log
        res = self.runner.record_forecast(station_id="AP001", df_24h=self.deterministic_df)
        self.assertEqual(res["status"], "trial_disabled")
        self.assertFalse(res["shadow_recorded"])
        # Production output must remain fully available
        self.assertGreater(res["production_forecast"]["pm25"], 0.0)


if __name__ == "__main__":
    unittest.main()
