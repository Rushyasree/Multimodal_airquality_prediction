"""
tests/test_live_chart_timestamp_regression.py

Regression test suite verifying correct timestamp handling, timezone conversion to IST,
forecast alignment, graceful failure handling, and strict provenance labeling.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from src.prediction.current_air_quality_service import (
    CurrentAirQualityService,
    fetch_open_meteo_live,
    fetch_open_meteo_hourly_observations,
)
from src.services.sensor_forecast_service import SensorForecastService
from src.prediction.data_loader import (
    load_demo_scenarios,
    get_scenario_metadata_list,
    get_verified_24h_sequence_for_station,
)

IST = ZoneInfo("Asia/Kolkata")


class TestLiveChartTimestampRegression:
    """Regression tests addressing the April 2026 historical timestamp defect."""

    def test_no_hardcoded_april_2026_dates_in_live_pipeline(self):
        """Verify that live hourly observations and live forecasts do not contain April 2026 timestamps."""
        srv = CurrentAirQualityService()
        # Mock API returning realistic past 48h to next 24h telemetry
        base_start = datetime(2026, 11, 12, 0, 0, 0)
        sample_times = [
            (base_start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(72)
        ]
        mock_payload = {
            "hourly": {
                "time": sample_times,
                "pm2_5": [45.0 + (i % 24) * 0.5 for i in range(72)],
                "pm10": [90.0 + (i % 24) * 0.8 for i in range(72)],
                "nitrogen_dioxide": [25.0] * 72,
                "sulphur_dioxide": [8.0] * 72,
                "carbon_monoxide": [400.0] * 72,
                "ozone": [30.0] * 72,
            }
        }
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_payload
            mock_get.return_value = mock_resp

            df_live = srv.get_recent_hourly_observations(station_id="AP001", force_refresh=True)
            assert df_live is not None and not df_live.empty
            assert len(df_live) == 24
            for dt_str in df_live["Datetime"]:
                assert "2026-04" not in str(dt_str), f"Found April 2026 date in live telemetry: {dt_str}"
                assert "2026-11" in str(dt_str)

            # Check SensorForecastService with this live sequence
            forecast_srv = SensorForecastService()
            fc_res = forecast_srv.predict_next_hour(df_24h=df_live, station_id="AP001")
            issue_time = fc_res.get("forecast_issue_time")
            target_time = fc_res.get("target_time")

            assert "2026-04" not in str(issue_time)
            assert "2026-04" not in str(target_time)
            assert "2026-11" in str(issue_time)
            assert fc_res.get("is_live_telemetry") is True

    def test_timestamps_are_timezone_aware_and_ist_converted(self):
        """Verify that parse_datetime_to_ist correctly parses UTC, offsets, and strings to IST."""
        srv = CurrentAirQualityService()
        # 1. UTC timestamp string with Z
        utc_str = "2026-11-14T06:30:00Z"
        parsed_dt, parsed_str, age = srv.parse_datetime_to_ist(utc_str)
        assert parsed_dt is not None
        assert parsed_dt.tzinfo is not None
        # UTC 06:30 + 05:30 = IST 12:00
        assert parsed_dt.hour == 12
        assert parsed_dt.minute == 0
        assert "12:00" in parsed_str

        # 2. String with explicit offset
        offset_str = "2026-11-14T14:00:00+05:30"
        parsed_dt2, parsed_str2, age2 = srv.parse_datetime_to_ist(offset_str)
        assert parsed_dt2.hour == 14
        assert parsed_dt2.minute == 0
        assert "02:00 PM" in parsed_str2

        # 3. Timezone-aware datetime in IST
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        aware_dt = datetime(2026, 11, 14, 15, 45, 0, tzinfo=ist_tz)
        parsed_dt3, parsed_str3, age3 = srv.parse_datetime_to_ist(aware_dt)
        assert parsed_dt3.hour == 15
        assert parsed_dt3.minute == 45
        assert "03:45 PM" in parsed_str3

        # 4. Pandas Timestamp
        ts = pd.Timestamp("2026-11-14 18:20:00", tz="UTC")
        parsed_dt4, parsed_str4, age4 = srv.parse_datetime_to_ist(ts)
        # 18:20 UTC = 23:50 IST
        assert parsed_dt4.hour == 23
        assert parsed_dt4.minute == 50

    def test_actual_observations_use_source_provided_timestamps(self):
        """Verify that fetch_open_meteo_live extracts and preserves the source timestamp rather than fabricating time."""
        mock_api_time = "2026-11-14T11:00"
        mock_payload = {
            "current": {
                "time": mock_api_time,
                "pm2_5": 38.4,
                "pm10": 74.2,
                "nitrogen_dioxide": 18.0,
                "sulphur_dioxide": 6.5,
                "carbon_monoxide": 450.0,
                "ozone": 42.0,
            }
        }
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_payload
            mock_get.return_value = mock_resp

            obs = fetch_open_meteo_live(station_name="Amaravati Ambient", lat=16.515, lon=80.518, city="Amaravati")
            assert obs is not None
            # Must reflect 11:00 source observation time in IST
            assert "11:00 AM IST" in obs["observation_time_ist"]
            assert "14 Nov 2026" in obs["observation_time_ist"]

    def test_prediction_timestamps_aligned_with_observation_time(self):
        """Ensure prediction target timestamp is strictly observation end time + 1 hour."""
        timestamps = [pd.Timestamp("2026-11-14 00:00:00") + pd.Timedelta(hours=i) for i in range(24)]
        df_24h = pd.DataFrame({
            "Datetime": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "StationId": ["AP001"] * 24,
            "PM2.5": [50.0] * 24,
            "PM10": [95.0] * 24,
            "NO": [10.0] * 24,
            "NO2": [25.0] * 24,
            "NOx": [35.0] * 24,
            "NH3": [18.0] * 24,
            "CO": [0.8] * 24,
            "SO2": [8.0] * 24,
            "O3": [30.0] * 24,
            "Benzene": [1.0] * 24,
            "Toluene": [3.0] * 24,
            "Xylene": [0.5] * 24,
            "temperature_C": [28.0] * 24,
            "dewpoint_C": [18.0] * 24,
            "relative_humidity": [60.0] * 24,
            "wind_speed": [2.5] * 24,
            "precipitation_mm": [0.0] * 24,
        })
        forecast_srv = SensorForecastService()
        fc_res = forecast_srv.predict_next_hour(df_24h=df_24h, station_id="AP001")

        expected_issue = "2026-11-14 23:00:00"
        expected_target = "2026-11-15 00:00:00"
        assert fc_res["forecast_issue_time"] == expected_issue
        assert fc_res["target_time"] == expected_target

    def test_missing_timestamps_or_api_failures_handled_gracefully(self):
        """Verify that API failure or missing timestamps do not crash the service and return safe indicators."""
        # Case 1: Missing 'time' in response payload
        mock_payload_no_time = {
            "current": {
                "pm2_5": 30.0,
                "pm10": 60.0,
            }
        }
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_payload_no_time
            mock_get.return_value = mock_resp

            obs = fetch_open_meteo_live(station_name="Amaravati Ambient", lat=16.515, lon=80.518, city="Amaravati")
            assert obs is not None
            assert obs["observation_time_ist"] == "Timestamp Unavailable"

        # Case 2: API 500 error
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_get.return_value = mock_resp

            df_err = fetch_open_meteo_hourly_observations(lat=16.515, lon=80.518)
            assert df_err is None

            srv = CurrentAirQualityService()
            df_fallback = srv.get_recent_hourly_observations(station_id="AP001", force_refresh=True)
            # Service should return None gracefully without raising exception
            assert df_fallback is None

    def test_historical_data_not_incorrectly_labeled_live(self):
        """Verify that historical benchmark scenarios are never mislabeled as live telemetry."""
        demo_df = load_demo_scenarios()
        assert not demo_df.empty

        # Pick scenario 2 (which had April 2026 dates)
        sc2 = demo_df[demo_df["scenario_id"] == "SCENARIO_02_AP001_20260428_15"].sort_values("Datetime").reset_index(drop=True)
        assert not sc2.empty

        forecast_srv = SensorForecastService()
        fc_res = forecast_srv.predict_next_hour(df_24h=sc2, station_id="AP001", scenario_id="SCENARIO_02_AP001_20260428_15")

        assert fc_res.get("is_live_telemetry") is False
        assert fc_res.get("forecast_state") != "LIVE FORECAST — VALIDATED INPUT"

    def test_scenario_metadata_unique_labels_prevent_collision(self):
        """Ensure get_scenario_metadata_list has unique labels so selectbox mappings retain all 24 scenarios."""
        scenarios = get_scenario_metadata_list()
        assert len(scenarios) == 24
        labels = [s["label"] for s in scenarios]
        # All 24 labels must be distinct
        assert len(set(labels)) == len(labels) == 24
