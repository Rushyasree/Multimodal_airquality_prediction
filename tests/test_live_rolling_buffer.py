"""
tests/test_live_rolling_buffer.py

Comprehensive Unit Test Suite for VayuDrishti Phase 6B:
Timestamp-Safe Rolling Telemetry Buffer & Live Readiness Validation.

All tests utilize SYNTHETIC data clearly labeled as synthetic.
Synthetic data is never counted as real-world observation results.

Test Coverage:
1. Exactly 24 contiguous hourly observations.
2. Missing hours handling -> 'insufficient_history'.
3. Duplicate readings deduplication.
4. Sub-hourly irregular timestamps aggregation to clock hour arithmetic mean.
5. Timezone conversions (IST string, ISO UTC, UTC offset).
6. Stale observations detection (> max_staleness_hours) -> 'stale_data'.
7. Missing model-required channels (e.g. missing PM10 or PM2.5) -> 'missing_required_channels'.
8. Unphysical or corrupted sensor readings rejection.
9. Station ID verification and mismatch rejection.
10. Future-data and target-hour (t+1) leakage prevention.
11. Multi-station isolation and buffer reset.
12. End-to-end readiness check and safe shadow runner integration.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src.data.live_rolling_buffer import (
    StationRollingBuffer,
    LiveRollingBufferRegistry,
    normalize_timestamp_utc,
    validate_sensor_reading,
    IST,
)
from src.services.shadow_observation_runner import ShadowObservationRunner
from src.services.sensor_forecast_service import SensorForecastService


class TestLiveRollingBufferSynthetic(unittest.TestCase):
    """Synthetic test suite for StationRollingBuffer and LiveRollingBufferRegistry.
    
    NOTE: All data used here is synthetic and created purely for unit test verification.
    """

    def setUp(self):
        self.station_id = "AP001"
        self.buffer = StationRollingBuffer(
            station_id=self.station_id,
            station_lat=16.515083,
            station_lon=80.518167,
            max_staleness_hours=3.0,
        )

        # Base reference time: 2026-11-05 06:00:00 UTC
        self.ref_now = datetime(2026, 11, 5, 6, 0, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.buffer.clear()

    def test_01_exactly_24_contiguous_hours(self):
        """Synthetic Test 1: Exactly 24 contiguous hours of PM2.5 and PM10 should pass readiness."""
        # Ingest 24 consecutive clock hours [t-23 ... t]
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            res1 = self.buffer.ingest_reading(
                parameter="PM2.5",
                value=40.0 + i * 0.5,
                timestamp=t_hour,
            )
            self.assertEqual(res1["status"], "ingested")

            res2 = self.buffer.ingest_reading(
                parameter="PM10",
                value=80.0 + i * 1.0,
                timestamp=t_hour,
            )
            self.assertEqual(res2["status"], "ingested")

        # Readiness check at anchor t = self.ref_now
        readiness = self.buffer.check_readiness(
            anchor_time_utc=self.ref_now,
            current_system_time=self.ref_now + timedelta(minutes=10),
        )

        self.assertTrue(readiness["is_ready"])
        self.assertEqual(readiness["status"], "ready")
        self.assertIsNone(readiness["reason"])
        self.assertEqual(readiness["available_hours_count"], 24)
        self.assertEqual(readiness["missing_hours_count"], 0)
        self.assertTrue(readiness["adapted_ready"])
        self.assertTrue(readiness["historical_ready"])

        # Check constructed DataFrame
        df_24h = readiness["dataframe_24h"]
        self.assertIsNotNone(df_24h)
        self.assertEqual(len(df_24h), 24)
        self.assertIn("PM2.5", df_24h.columns)
        self.assertIn("PM10", df_24h.columns)
        self.assertIn("Datetime", df_24h.columns)
        self.assertIn("StationId", df_24h.columns)
        self.assertAlmostEqual(df_24h["PM2.5"].iloc[-1], 40.0 + 23 * 0.5)

    def test_02_missing_hours_reported_as_insufficient_history(self):
        """Synthetic Test 2: Incomplete sequence (e.g. 23 hours with 1 missing hour) must fail."""
        # Ingest only 23 hours, skipping hour 12
        for i in range(24):
            if i == 12:
                continue  # Intentional gap in history
            t_hour = self.ref_now - timedelta(hours=23 - i)
            self.buffer.ingest_reading("PM2.5", 45.0, t_hour)
            self.buffer.ingest_reading("PM10", 90.0, t_hour)

        readiness = self.buffer.check_readiness(
            anchor_time_utc=self.ref_now,
            current_system_time=self.ref_now + timedelta(minutes=5),
        )

        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["status"], "not_ready")
        self.assertEqual(readiness["reason"], "insufficient_history")
        self.assertEqual(readiness["available_hours_count"], 23)
        self.assertEqual(readiness["missing_hours_count"], 1)
        self.assertIsNone(readiness["dataframe_24h"])

    def test_03_deduplication_of_readings(self):
        """Synthetic Test 3: Duplicate submissions for identical channel and timestamp are skipped."""
        t_hour = self.ref_now
        res1 = self.buffer.ingest_reading("PM2.5", 50.0, t_hour)
        self.assertEqual(res1["status"], "ingested")

        # Duplicate ingest
        res2 = self.buffer.ingest_reading("PM2.5", 50.0, t_hour)
        self.assertEqual(res2["status"], "duplicate_skipped")

        # Summary count must show only 1 sample
        summary = self.buffer.get_hourly_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["sample_counts"]["PM2.5"], 1)

    def test_04_sub_hourly_irregular_timestamp_aggregation(self):
        """Synthetic Test 4: Sub-hourly readings (e.g., 06:10, 06:25, 06:50) aggregate into 06:00 mean."""
        t_base = datetime(2026, 11, 5, 6, 0, 0, tzinfo=timezone.utc)
        readings = [
            (t_base + timedelta(minutes=10), 30.0),
            (t_base + timedelta(minutes=25), 40.0),
            (t_base + timedelta(minutes=50), 50.0),
        ]

        for ts, val in readings:
            res = self.buffer.ingest_reading("PM2.5", val, ts)
            self.assertEqual(res["status"], "ingested")

        summary = self.buffer.get_hourly_summary()
        self.assertEqual(len(summary), 1)
        # Arithmetic mean of 30, 40, 50 = 40.0
        self.assertAlmostEqual(summary[0]["channel_values"]["PM2.5"], 40.0)
        self.assertEqual(summary[0]["sample_counts"]["PM2.5"], 3)

    def test_05_timezone_conversions(self):
        """Synthetic Test 5: Normalizes timestamps from IST strings, ISO strings, and UTC offsets."""
        # 1. ISO UTC string
        t1 = normalize_timestamp_utc("2026-11-05T06:00:00Z")
        self.assertEqual(t1.hour, 6)
        self.assertEqual(t1.tzinfo, timezone.utc)

        # 2. IST string representation (+05:30)
        # 11:30 AM IST = 06:00 AM UTC
        t2 = normalize_timestamp_utc("2026-11-05T11:30:00+05:30")
        self.assertEqual(t2.hour, 6)
        self.assertEqual(t2.minute, 0)
        self.assertEqual(t2.tzinfo, timezone.utc)

        # 3. Pandas timestamp with localization
        pdt = pd.Timestamp("2026-11-05 06:00:00", tz="UTC")
        t3 = normalize_timestamp_utc(pdt)
        self.assertEqual(t3, t1)

    def test_06_stale_data_rejection(self):
        """Synthetic Test 6: Freshness check rejects observations older than max_staleness_hours."""
        # Ingest 24 hours of data
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            self.buffer.ingest_reading("PM2.5", 40.0, t_hour)
            self.buffer.ingest_reading("PM10", 80.0, t_hour)

        # System time is 5 hours ahead of anchor (threshold is 3.0h)
        sys_time_late = self.ref_now + timedelta(hours=5)
        readiness = self.buffer.check_readiness(
            anchor_time_utc=self.ref_now,
            current_system_time=sys_time_late,
        )

        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["status"], "not_ready")
        self.assertEqual(readiness["reason"], "stale_data")
        self.assertGreater(readiness["staleness_hours"], 3.0)

    def test_07_missing_required_channels(self):
        """Synthetic Test 7: If PM10 is missing across hours, fails with 'missing_required_channels'."""
        # Ingest 24 hours of PM2.5 only (PM10 missing)
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            self.buffer.ingest_reading("PM2.5", 40.0, t_hour)

        readiness = self.buffer.check_readiness(
            anchor_time_utc=self.ref_now,
            current_system_time=self.ref_now + timedelta(minutes=5),
        )

        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["status"], "not_ready")
        self.assertEqual(readiness["reason"], "missing_required_channels")
        self.assertIn("PM10", readiness["detail"])

    def test_08_unphysical_sensor_readings_rejected(self):
        """Synthetic Test 8: Implausible or negative atmospheric values are rejected."""
        # Extreme unphysical PM2.5 (> 1000)
        res_high = self.buffer.ingest_reading("PM2.5", 1500.0, self.ref_now)
        self.assertEqual(res_high["status"], "rejected")
        self.assertIn("physical bounds", res_high["reason"])

        # Negative PM2.5 (< 0)
        res_neg = self.buffer.ingest_reading("PM2.5", -25.0, self.ref_now)
        self.assertEqual(res_neg["status"], "rejected")

        # Non-numeric string
        res_str = self.buffer.ingest_reading("PM2.5", "SENSOR_ERR_TIMEOUT", self.ref_now)
        self.assertEqual(res_str["status"], "rejected")

    def test_09_station_id_mismatch_rejected(self):
        """Synthetic Test 9: Submitting a reading for AP002 to AP001 buffer is rejected."""
        res = self.buffer.ingest_reading(
            parameter="PM2.5",
            value=45.0,
            timestamp=self.ref_now,
            station_id="AP002",  # Mismatched station
        )
        self.assertEqual(res["status"], "rejected")
        self.assertIn("station_id_mismatch", res["reason"])

    def test_10_future_data_leakage_prevention(self):
        """Synthetic Test 10: Target-hour (t+1) and future readings must not enter the 24h forecast window."""
        # Ingest 24 hours up to t = 06:00
        for i in range(24):
            t_hour = self.ref_now - timedelta(hours=23 - i)
            self.buffer.ingest_reading("PM2.5", 40.0 + i, t_hour)
            self.buffer.ingest_reading("PM10", 80.0 + i, t_hour)

        # Ingest target hour t+1 (07:00) with an extreme marker value
        t_plus_1 = self.ref_now + timedelta(hours=1)
        self.buffer.ingest_reading("PM2.5", 999.0, t_plus_1)
        self.buffer.ingest_reading("PM10", 999.0, t_plus_1)

        # Check readiness for forecast at anchor t = 06:00 (target is 07:00)
        readiness = self.buffer.check_readiness(
            anchor_time_utc=self.ref_now,
            current_system_time=self.ref_now + timedelta(minutes=5),
        )

        self.assertTrue(readiness["is_ready"])
        df_24h = readiness["dataframe_24h"]
        # The 24h DataFrame must end strictly at 06:00 with value 40 + 23 = 63.0
        self.assertEqual(df_24h["Datetime"].iloc[-1], self.ref_now)
        self.assertAlmostEqual(df_24h["PM2.5"].iloc[-1], 63.0)
        # 999.0 from t+1 must NOT appear anywhere in the 24h input
        self.assertNotIn(999.0, df_24h["PM2.5"].values)

    def test_11_multi_station_isolation_and_registry(self):
        """Synthetic Test 11: Multi-station registry isolates data between AP001 and AP005."""
        registry = LiveRollingBufferRegistry()

        # AP001 and AP005 must be recognized
        self.assertTrue(registry.is_station_recognized("AP001"))
        self.assertTrue(registry.is_station_recognized("AP005"))
        # Fictional station must NOT be recognized
        self.assertFalse(registry.is_station_recognized("FAKE_STN_999"))

        buf_ap001 = registry.get_or_create_buffer("AP001")
        buf_ap005 = registry.get_or_create_buffer("AP005")

        # Ingest into AP001
        buf_ap001.ingest_reading("PM2.5", 42.0, self.ref_now)
        # AP005 must remain completely empty
        self.assertEqual(len(buf_ap005.get_hourly_summary()), 0)
        self.assertEqual(len(buf_ap001.get_hourly_summary()), 1)

    def test_12_shadow_runner_readiness_integration(self):
        """Synthetic Test 12: Shadow runner check_live_readiness safely validates prerequisites."""
        test_dir = Path(tempfile.mkdtemp(prefix="test_phase6b_runner_"))
        try:
            forecasts_log = test_dir / "shadow_forecasts.jsonl"
            matched_log = test_dir / "matched_evaluations.jsonl"
            config_file = test_dir / "trial_config.json"

            runner = ShadowObservationRunner(
                forecasts_log_path=forecasts_log,
                matched_log_path=matched_log,
                config_path=config_file,
            )

            # Test A: Unrecognized station fails with station_mapping_failed
            bad_stn_check = runner.check_live_readiness("UNKNOWN_STN_999")
            self.assertFalse(bad_stn_check["is_ready"])
            self.assertEqual(bad_stn_check["reason"], "station_mapping_failed")

            # Test B: Buffer with insufficient history fails
            insufficient_check = runner.check_live_readiness("AP001", buffer=self.buffer)
            self.assertFalse(insufficient_check["is_ready"])
            self.assertEqual(insufficient_check["reason"], "insufficient_history")

            # Test C: Ingest complete 24h history and verify readiness + dry-run inference
            for i in range(24):
                t_hour = self.ref_now - timedelta(hours=23 - i)
                self.buffer.ingest_reading("PM2.5", 40.0 + i * 0.5, t_hour)
                self.buffer.ingest_reading("PM10", 80.0 + i * 1.0, t_hour)

            ready_check = runner.check_live_readiness(
                station_id="AP001",
                buffer=self.buffer,
                anchor_time_utc=self.ref_now,
            )
            self.assertTrue(ready_check["is_ready"])
            self.assertEqual(ready_check["status"], "ready")

            # Test D: record_forecast_from_buffer when trial disabled
            # Must return trial_disabled without writing to log
            res = runner.record_forecast_from_buffer("AP001", buffer=self.buffer)
            self.assertEqual(res["status"], "trial_disabled")
            self.assertFalse(forecasts_log.exists())

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
