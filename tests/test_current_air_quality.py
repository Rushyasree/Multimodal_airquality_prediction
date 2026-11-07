"""
tests/test_current_air_quality.py

Comprehensive test suite for the genuine current air quality data service.
Tests mocked OpenAQ API v3 responses, pollutant normalization, physical plausibility checks,
timestamp conversion to IST, freshness badging, CPCB AQI criteria, error resilience,
and prevention of data fabrication.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import pytz
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.current_air_quality_service import (
    CurrentAirQualityService,
    IST,
    PARAM_MAP,
    PHYSICAL_BOUNDS,
)
from src.prediction.lstm_forecaster import LSTMForecasterService


class TestCurrentAirQualityService(unittest.TestCase):
    def setUp(self):
        self.service = CurrentAirQualityService(
            api_key="mock_test_key_abc123",
            timeout_seconds=2.0,
            cache_ttl_seconds=300,
            freshness_live_hours=3.0,
            freshness_recent_hours=24.0,
        )

    # 1. API-Key Handling
    def test_01_api_key_handling(self):
        self.assertTrue(self.service.has_valid_api_key())
        headers = self.service.get_headers()
        self.assertIn("X-API-Key", headers)
        self.assertEqual(headers["X-API-Key"], "mock_test_key_abc123")

        # Test without key
        no_key_srv = CurrentAirQualityService(api_key="")
        self.assertFalse(no_key_srv.has_valid_api_key())
        self.assertNotIn("X-API-Key", no_key_srv.get_headers())

    # 2. Successful Live-Data Retrieval Using Mocked API Response
    def test_02_successful_mocked_openaq_response(self):
        now_utc = datetime.now(timezone.utc)
        recent_iso = (now_utc - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        mock_items = [
            {
                "parameter": {"name": "pm25", "units": "µg/m³"},
                "value": 45.5,
                "datetime": {"utc": recent_iso},
            },
            {
                "parameter": {"name": "pm10", "units": "µg/m³"},
                "value": 85.0,
                "datetime": {"utc": recent_iso},
            },
            {
                "parameter": {"name": "no2", "units": "µg/m³"},
                "value": 28.3,
                "datetime": {"utc": recent_iso},
            },
            {
                "parameter": {"name": "so2", "units": "µg/m³"},
                "value": 12.0,
                "datetime": {"utc": recent_iso},
            },
            {
                "parameter": {"name": "co", "units": "mg/m³"},
                "value": 0.9,
                "datetime": {"utc": recent_iso},
            },
            {
                "parameter": {"name": "o3", "units": "µg/m³"},
                "value": 35.0,
                "datetime": {"utc": recent_iso},
            },
        ]

        payload = self.service.process_raw_measurements(
            mock_items,
            station_name="Visakhapatnam CPCB Monitor",
            city_name="Visakhapatnam",
            distance_km=0.0,
        )

        self.assertTrue(payload["is_live"])
        self.assertEqual(payload["status_badge"], "LIVE DATA")
        self.assertEqual(payload["station_name"], "Visakhapatnam CPCB Monitor")
        self.assertIn("PM2.5", payload["pollutants"])
        self.assertEqual(payload["pollutants"]["PM2.5"], 45.5)
        self.assertEqual(payload["pollutants"]["PM10"], 85.0)
        self.assertIsNotNone(payload["aqi"])
        self.assertGreater(payload["aqi"], 0)

    # 3. Timestamp Conversion to IST
    def test_03_timestamp_conversion_to_ist(self):
        utc_test = "2026-10-15T12:00:00Z"
        dt_obj, formatted_ist, age_hours = self.service.parse_datetime_to_ist(utc_test)
        
        self.assertIsNotNone(dt_obj)
        self.assertIn("IST", formatted_ist)
        # 12:00 UTC is 17:30 IST (+5:30)
        self.assertIn("05:30 PM IST", formatted_ist)

    # 4. Freshness and Stale-Data Labels
    def test_04_freshness_and_stale_data_labels(self):
        # Case A: Live Data (1.5 hours old <= 3.0h)
        badge_a, color_a, _ = self.service.get_freshness_badge(1.5, is_live_source=True)
        self.assertEqual(badge_a, "LIVE DATA")
        self.assertEqual(color_a, "#10b981")

        # Case B: Recent Observation (8.0 hours old > 3.0h and <= 24.0h)
        badge_b, color_b, _ = self.service.get_freshness_badge(8.0, is_live_source=True)
        self.assertEqual(badge_b, "RECENT OBSERVATION")
        self.assertEqual(color_b, "#f59e0b")

        # Case C: Stale Observation (36.0 hours old > 24.0h)
        badge_c, color_c, _ = self.service.get_freshness_badge(36.0, is_live_source=True)
        self.assertEqual(badge_c, "STALE OBSERVATION")
        self.assertEqual(color_c, "#ea580c")

        # Case D: Historical Archive
        badge_d, color_d, _ = self.service.get_freshness_badge(99999.0, is_live_source=False)
        self.assertEqual(badge_d, "HISTORICAL DATA")
        self.assertEqual(color_d, "#38bdf8")

    # 5. Missing and Invalid Pollutant Values
    def test_05_missing_and_invalid_pollutant_values(self):
        # Negative value -> must be rejected
        res_neg = self.service.parse_and_validate_value("pm25", -15.0, "µg/m³")
        self.assertIsNone(res_neg)

        # Impossibly high value (e.g. 5000 µg/m³) -> must be rejected
        res_high = self.service.parse_and_validate_value("pm25", 5000.0, "µg/m³")
        self.assertIsNone(res_high)

        # None value -> must be rejected
        res_none = self.service.parse_and_validate_value("no2", None, "µg/m³")
        self.assertIsNone(res_none)

        # Valid value -> normalized and returned
        res_valid = self.service.parse_and_validate_value("pm25", 55.4, "µg/m³")
        self.assertIsNotNone(res_valid)
        self.assertEqual(res_valid[0], "PM2.5")
        self.assertEqual(res_valid[1], 55.4)

    # 6. CPCB AQI Calculation Requirements (At least 3 pollutants with PM2.5 or PM10)
    def test_06_cpcb_aqi_calculation_requirements(self):
        # Only PM2.5 available (1 pollutant) -> AQI cannot be officially determined
        single_item = [
            {
                "parameter": {"name": "pm25", "units": "µg/m³"},
                "value": 65.0,
                "datetime": {"utc": datetime.now(timezone.utc).isoformat()},
            }
        ]
        res_sparse = self.service.process_raw_measurements(single_item, "AP Sensor", "Amaravati")
        self.assertIsNone(res_sparse["aqi"])
        self.assertIn("Insufficient Pollutants", res_sparse["aqi_tier"])

        # PM2.5, PM10, and NO2 available (3 pollutants) -> AQI is calculated
        three_items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 65.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 110.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 35.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
        ]
        res_full = self.service.process_raw_measurements(three_items, "AP Sensor", "Amaravati")
        self.assertIsNotNone(res_full["aqi"])
        self.assertEqual(res_full["prominent_pollutant"], "PM2.5")

    # 7. Provider Timeouts and API Failures
    @patch("requests.get")
    def test_07_provider_timeout_and_api_failures(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out after 5.0s")
        res = self.service.get_current_observations("Visakhapatnam", force_refresh=True)
        self.assertFalse(res["is_live"])
        self.assertEqual(res["status_badge"], "DATA UNAVAILABLE")
        self.assertIn("timed out", res["badge_desc"])

    # 8. Missing Station Coverage & Unsupported Cities
    @patch.object(CurrentAirQualityService, "fetch_openaq_locations_by_coordinates")
    def test_08_missing_station_coverage(self, mock_loc):
        mock_loc.return_value = []
        # Case A: Valid catalog city with no monitor in radius (e.g. Ongole)
        res_ongole = self.service.get_current_observations("Ongole", force_refresh=True)
        self.assertFalse(res_ongole["is_live"])
        self.assertEqual(res_ongole["status_badge"], "DATA UNAVAILABLE")
        self.assertIn("No active OpenAQ monitoring station found", res_ongole["badge_desc"])
        self.assertIn("nearest_station_option", res_ongole)
        self.assertTrue(res_ongole["nearest_station_option"]["available"])
        self.assertGreater(res_ongole["nearest_station_option"]["distance_km"], 0.0)
        self.assertIn("Explicit user opt-in required", res_ongole["nearest_station_option"]["note"])

        # Case B: Completely uncataloged city (e.g. RemoteAPTown) -> Rejected honestly
        res_unknown = self.service.get_current_observations("RemoteAPTown", force_refresh=True)
        self.assertFalse(res_unknown["is_live"])
        self.assertEqual(res_unknown["status_badge"], "DATA UNAVAILABLE")
        self.assertIn("is not registered in the location catalog", res_unknown["badge_desc"])

    # 9. Response Caching and Force-Refresh
    @patch.object(CurrentAirQualityService, "fetch_openaq_locations_by_coordinates")
    def test_09_response_caching_and_force_refresh(self, mock_loc):
        mock_loc.return_value = []
        # First call hits mock
        res1 = self.service.get_current_observations("Kurnool", force_refresh=True)
        self.assertEqual(mock_loc.call_count, 1)

        # Second call uses in-memory cache without hitting mock again
        res2 = self.service.get_current_observations("Kurnool", force_refresh=False)
        self.assertEqual(mock_loc.call_count, 1)
        self.assertEqual(res1["retrieved_at_ist"], res2["retrieved_at_ist"])

        # Force refresh bypasses cache
        res3 = self.service.get_current_observations("Kurnool", force_refresh=True)
        self.assertEqual(mock_loc.call_count, 2)

    # 10. Preventing Incomplete Live Sequences from Entering the LSTM
    def test_10_preventing_incomplete_live_sequences_entering_lstm(self):
        lstm = LSTMForecasterService()
        
        # Sparse live reading dictionary
        live_sparse_reading = {"PM2.5": 52.0, "PM10": 98.0}
        
        # Verify LSTM requires full 24-step dataframe with 27 engineered features
        # Passing an incomplete sequence of 1 row must raise ValueError
        import pandas as pd
        sparse_df = pd.DataFrame([live_sparse_reading])
        
        with self.assertRaises(ValueError):
            lstm.predict_from_dataframe(sparse_df)

    # 11. No Fabricated Readings Displayed
    def test_11_no_fabricated_readings_displayed(self):
        # If an observation only has PM2.5, other pollutants must NOT be fabricated as 0.0
        partial_items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 48.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}}
        ]
        res = self.service.process_raw_measurements(partial_items, "Station X", "City Y")
        self.assertIn("PM2.5", res["pollutants"])
        self.assertNotIn("SO2", res["pollutants"])
        self.assertNotIn("O3", res["pollutants"])

    # 12. Live Forecast Compatibility Check
    def test_12_live_forecast_compatibility_check(self):
        compat = self.service.check_live_forecast_compatibility("Visakhapatnam")
        self.assertFalse(compat["supported"])
        self.assertEqual(compat["status_badge"], "LIVE FORECAST DISABLED")
        self.assertIn("reasons", compat)
        self.assertGreater(len(compat["reasons"]), 0)
        self.assertEqual(compat["model_requirements"]["sequence_length"], 24)
        self.assertEqual(compat["model_requirements"]["feature_count"], 27)

    # 13. LSTM Sequence Continuity and Feature Validation
    def test_13_lstm_sequence_validation(self):
        import pandas as pd
        lstm = LSTMForecasterService()

        # Case A: Sequence length mismatch
        df_short = pd.DataFrame([{"PM2.5": 40.0, "PM10": 70.0}] * 10)
        ok, msg = lstm.validate_sequence(df_short)
        self.assertFalse(ok)
        self.assertIn("Sequence length mismatch", msg)

        # Case B: Missing both targets
        df_no_target = pd.DataFrame([{"NO2": 25.0}] * 24)
        ok_nt, msg_nt = lstm.validate_sequence(df_no_target)
        self.assertFalse(ok_nt)
        self.assertIn("missing both PM2.5 and PM10", msg_nt)

        # Case C: Physically impossible negative value
        df_neg = pd.DataFrame([{"PM2.5": -10.0, "PM10": 50.0}] * 24)
        ok_neg, msg_neg = lstm.validate_sequence(df_neg)
        self.assertFalse(ok_neg)
        self.assertIn("negative concentration", msg_neg)

        # Case D: Valid 24h sequence
        df_valid = pd.DataFrame([{"PM2.5": 35.0, "PM10": 65.0}] * 24)
        ok_v, msg_v = lstm.validate_sequence(df_valid)
        self.assertTrue(ok_v)

    # 14. AQI Gauge Renders Null as Dashes Not 162
    def test_14_gauge_renders_null_as_dashes_not_162(self):
        from app import render_reference_aqi_gauge
        gauge_html = render_reference_aqi_gauge(None, "Insufficient Pollutants for AQI", "#94a3b8")
        self.assertIn(">--<", gauge_html)
        self.assertNotIn(">162<", gauge_html)
        self.assertNotIn("url(#rainbow-spectrum)", gauge_html)
        self.assertIn("Insufficient Pollutants", gauge_html)

    # 15. Safe Pollutant Formatting Helper
    def test_15_fmt_pollutant_val_missing_handling(self):
        import numpy as np
        from app import fmt_pollutant_val
        self.assertEqual(fmt_pollutant_val(None), "--")
        self.assertEqual(fmt_pollutant_val(np.nan), "--")
        self.assertEqual(fmt_pollutant_val(42.3, 0), "42")
        self.assertEqual(fmt_pollutant_val(0.94, 1), "0.9")
        self.assertEqual(fmt_pollutant_val("invalid"), "--")

    # 16. Insufficient Criteria Pollutants (< 3) Yields None AQI
    def test_16_insufficient_criteria_pollutants_aqi_none(self):
        now_utc = datetime.now(timezone.utc)
        iso_str = now_utc.isoformat().replace("+00:00", "Z")
        # Only PM2.5 and PM10 provided (2 criteria pollutants - CPCB requires >= 3)
        partial_items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 45.0, "datetime": {"utc": iso_str}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 80.0, "datetime": {"utc": iso_str}},
        ]
        res = self.service.process_raw_measurements(partial_items, "Station Alpha", "City Beta")
        self.assertTrue(res["is_live"])
        self.assertIsNone(res["aqi"])
        self.assertEqual(res["aqi_tier"], "Insufficient Pollutants for AQI")
        self.assertEqual(res["pollutants"]["PM2.5"], 45.0)
        self.assertEqual(res["pollutants"]["PM10"], 80.0)
        self.assertNotIn("SO2", res["pollutants"])

    # 17. Stale Observation Badging (> 24h)
    def test_17_stale_observation_badging_over_24h(self):
        badge, color, desc = self.service.get_freshness_badge(age_hours=72.0, is_live_source=True)
        self.assertEqual(badge, "STALE OBSERVATION")
        self.assertNotEqual(badge, "LIVE DATA")
        self.assertEqual(color, "#ea580c")
        self.assertIn("older than 24 hours", desc)

    # 18. Honest Empty Handling for Cities with No Active Monitor
    def test_18_honest_empty_handling_no_active_monitor(self):
        with patch.object(self.service, "fetch_openaq_locations_by_coordinates", return_value=[]):
            res = self.service.get_current_observations("Anantapuram", force_refresh=True)
            self.assertFalse(res["is_live"])
            self.assertEqual(res["status_badge"], "DATA UNAVAILABLE")
            self.assertIsNone(res["aqi"])
            self.assertEqual(res["pollutants"], {})
            self.assertIn("nearest_station_option", res)
            self.assertTrue(res["nearest_station_option"]["available"])


    # 19. Comprehensive Freshness Rules (Live <= 3h, Recent 3-24h, Stale > 24h)
    def test_19_freshness_rules_comprehensive(self):
        now_utc = datetime.now(timezone.utc)
        
        # A: Today / Live (1 hour old)
        ts_1h = (now_utc - timedelta(hours=1)).isoformat()
        items_1h = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 45.0, "datetime": {"utc": ts_1h}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 80.0, "datetime": {"utc": ts_1h}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 25.0, "datetime": {"utc": ts_1h}},
        ]
        res_1h = self.service.process_raw_measurements(items_1h, "Stn Live", "City Live")
        self.assertTrue(res_1h["is_live"])
        self.assertFalse(res_1h["is_stale"])
        self.assertEqual(res_1h["status_badge"], "LIVE DATA")
        self.assertIsNotNone(res_1h["aqi"])
        self.assertEqual(res_1h["stale_warning"], "")

        # B: Recent (12 hours old)
        ts_12h = (now_utc - timedelta(hours=12)).isoformat()
        items_12h = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 50.0, "datetime": {"utc": ts_12h}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 90.0, "datetime": {"utc": ts_12h}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 30.0, "datetime": {"utc": ts_12h}},
        ]
        res_12h = self.service.process_raw_measurements(items_12h, "Stn Recent", "City Recent")
        self.assertTrue(res_12h["is_live"])
        self.assertFalse(res_12h["is_stale"])
        self.assertEqual(res_12h["status_badge"], "RECENT OBSERVATION")
        self.assertIsNotNone(res_12h["aqi"])
        self.assertEqual(res_12h["stale_warning"], "")

        # C: Stale (> 24 hours old, e.g. 72 hours)
        ts_72h = (now_utc - timedelta(hours=72)).isoformat()
        items_72h = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 60.0, "datetime": {"utc": ts_72h}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 110.0, "datetime": {"utc": ts_72h}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 35.0, "datetime": {"utc": ts_72h}},
        ]
        res_72h = self.service.process_raw_measurements(items_72h, "Stn Stale", "City Stale")
        self.assertFalse(res_72h["is_live"])
        self.assertTrue(res_72h["is_stale"])
        self.assertEqual(res_72h["status_badge"], "STALE OBSERVATION")
        self.assertIsNone(res_72h["aqi"])
        self.assertIn("outdated", res_72h["stale_warning"])
        self.assertIn("observed_at_ist", res_72h)
        self.assertIn("retrieved_at_ist", res_72h)

    # 20. Exact UTC to Asia/Kolkata (IST) Conversion
    def test_20_utc_to_ist_conversion(self):
        # 18:00 UTC on 2026-10-13 is exactly 23:30 (11:30 PM) IST on 2026-10-13
        utc_ts = "2026-10-13T18:00:00Z"
        dt_obj, formatted_ist, age_hours = self.service.parse_datetime_to_ist(utc_ts)
        self.assertIsNotNone(dt_obj)
        self.assertEqual(formatted_ist, "2026-10-13 11:30 PM IST")
        self.assertGreater(age_hours, 0)

    # 21. Missing and Invalid Datetime Parsing
    def test_21_invalid_and_missing_timestamps(self):
        dt_none, str_none, age_none = self.service.parse_datetime_to_ist(None)
        self.assertIsNone(dt_none)
        self.assertEqual(str_none, "Timestamp Unavailable")

        dt_empty, str_empty, age_empty = self.service.parse_datetime_to_ist("")
        self.assertIsNone(dt_empty)
        self.assertEqual(str_empty, "Timestamp Unavailable")

        dt_inv, str_inv, age_inv = self.service.parse_datetime_to_ist("not-a-timestamp")
        self.assertIsNone(dt_inv)
        self.assertEqual(str_inv, "not-a-timestamp")
        self.assertEqual(age_inv, 99999.0)

    # 22. Cache Invalidation and Bypass via force_refresh=True
    @patch.object(CurrentAirQualityService, "fetch_openaq_latest_by_location")
    def test_22_cache_invalidation_force_refresh(self, mock_fetch):
        mock_fetch.return_value = {
            "results": [
                {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 40.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
                {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 75.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
                {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 20.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            ]
        }
        stn = {
            "station_id": "AP005",
            "openaq_location_id": 5628,
            "station_name": "GVM Corporation, Visakhapatnam - APPCB",
            "city": "Visakhapatnam",
            "latitude": 17.72,
            "longitude": 83.3,
            "provider": "OpenAQ API v3",
        }
        # Initial request: Cache miss -> triggers fetch
        res1 = self.service.get_current_observations_for_station(stn, force_refresh=False)
        self.assertEqual(mock_fetch.call_count, 1)

        # Second request: Cache hit -> does NOT trigger fetch
        res2 = self.service.get_current_observations_for_station(stn, force_refresh=False)
        self.assertEqual(mock_fetch.call_count, 1)

        # Force refresh: Bypasses cache -> triggers fetch
        res3 = self.service.get_current_observations_for_station(stn, force_refresh=True)
        self.assertEqual(mock_fetch.call_count, 2)


if __name__ == "__main__":
    unittest.main()


