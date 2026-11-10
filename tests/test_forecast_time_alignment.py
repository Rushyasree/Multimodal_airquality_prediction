"""
tests/test_forecast_time_alignment.py

Unit and integration tests for sequence validation, forecast time alignment,
and explicit provenance tracking under Phase 17.1 specifications.
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from src.prediction.sequence_validator import (
    validate_forecast_sequence,
    SequenceValidationResult,
)
from src.prediction.data_loader import (
    load_demo_scenarios,
    get_verified_24h_sequence_for_station,
)
from src.services.sensor_forecast_service import SensorForecastService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.model_comparison_service import ModelComparisonService

IST = timezone(timedelta(hours=5, minutes=30))


def _make_valid_24h_df(
    start_dt: datetime,
    station_id: str = "AP001",
    pm25_val: float = 45.0,
    pm10_val: float = 90.0,
) -> pd.DataFrame:
    """Helper to create a synthetically compliant 24h dataframe."""
    timestamps = [start_dt + timedelta(hours=i) for i in range(24)]
    data = {
        "Datetime": timestamps,
        "StationId": [station_id] * 24,
        "PM2.5": [pm25_val] * 24,
        "PM10": [pm10_val] * 24,
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
    }
    return pd.DataFrame(data)


class TestSequenceValidator:
    """Tests for validate_forecast_sequence()."""

    def test_valid_live_sequence(self):
        """A strictly fresh, 24-hour sequence must pass validation."""
        now = datetime.now()
        start = now - timedelta(hours=23, minutes=30)
        df = _make_valid_24h_df(start, "AP001")

        result = validate_forecast_sequence(
            df,
            station_id="AP001",
            max_age_hours=3.0,
            require_exact_station=True,
            forecast_issue_time=now,
        )

        assert result.valid is True
        assert result.source_type == "LIVE_MEASURED"
        assert result.station_match is True
        assert result.rejection_reason is None
        assert result.target_time == (df["Datetime"].iloc[-1] + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    def test_rejection_of_non_24_rows(self):
        """Sequences with != 24 rows must be rejected."""
        start = datetime(2026, 4, 1, 10, 0, 0)
        df_23 = _make_valid_24h_df(start, "AP001").iloc[:23]
        df_25 = pd.concat([_make_valid_24h_df(start, "AP001"), _make_valid_24h_df(start, "AP001").iloc[:1]])

        res_23 = validate_forecast_sequence(df_23, "AP001")
        assert res_23.valid is False
        assert "Sequence length mismatch" in (res_23.rejection_reason or "")

        res_25 = validate_forecast_sequence(df_25, "AP001")
        assert res_25.valid is False
        assert "Sequence length mismatch" in (res_25.rejection_reason or "")

    def test_rejection_of_duplicate_timestamps(self):
        """Sequences with duplicate timestamps must be rejected."""
        start = datetime(2026, 4, 1, 10, 0, 0)
        df = _make_valid_24h_df(start, "AP001")
        # Introduce a duplicate timestamp
        df.loc[1, "Datetime"] = df.loc[0, "Datetime"]

        res = validate_forecast_sequence(df, "AP001")
        assert res.valid is False
        assert "Duplicate timestamps" in (res.rejection_reason or "")

    def test_rejection_of_non_chronological_timestamps(self):
        """Sequences out of chronological order must be rejected."""
        start = datetime(2026, 4, 1, 10, 0, 0)
        df = _make_valid_24h_df(start, "AP001")
        # Swap rows 5 and 6
        idx = df.index.tolist()
        idx[5], idx[6] = idx[6], idx[5]
        df_swapped = df.reindex(idx)

        res = validate_forecast_sequence(df_swapped, "AP001")
        assert res.valid is False
        assert "not sorted in ascending chronological order" in (res.rejection_reason or "")

    def test_rejection_of_irregular_hourly_cadence_or_gaps(self):
        """Sequences with missing hours (gaps > 1h) must be rejected."""
        start = datetime(2026, 4, 1, 10, 0, 0)
        timestamps = [start + timedelta(hours=i if i < 10 else i + 2) for i in range(24)]
        df = _make_valid_24h_df(start, "AP001")
        df["Datetime"] = timestamps

        res = validate_forecast_sequence(df, "AP001")
        assert res.valid is False
        assert "Discontinuous hourly interval" in (res.rejection_reason or "")

    def test_station_identity_enforcement(self):
        """Regional proxy with station mismatch must be rejected when require_exact_station=True."""
        start = datetime(2026, 4, 1, 10, 0, 0)
        df_ap001 = _make_valid_24h_df(start, "AP001")

        # Expecting AP006 (Nellore proxy request)
        res = validate_forecast_sequence(
            df_ap001,
            station_id="AP006",
            require_exact_station=True,
        )
        assert res.valid is False
        assert res.station_match is False
        assert res.source_type == "REGIONAL_PROXY"
        assert "Station identity mismatch" in (res.rejection_reason or "")

    def test_historical_demo_sequence_never_receives_live_state(self):
        """Historical sequence (> 3h old) must NOT be labeled as LIVE."""
        # 1 year in the past
        past_start = datetime(2025, 4, 1, 10, 0, 0)
        df_past = _make_valid_24h_df(past_start, "AP001")
        now = datetime(2026, 4, 1, 10, 0, 0)

        res = validate_forecast_sequence(
            df_past,
            station_id="AP001",
            max_age_hours=3.0,
            forecast_issue_time=now,
        )
        assert res.valid is False
        assert res.source_type == "HISTORICAL_DEMO"
        assert "exceeds live threshold" in (res.rejection_reason or "")
        assert "Stale" in res.provenance

    def test_target_time_is_strictly_t_plus_one(self):
        """Target time must equal sequence_end_time + 1 hour exactly."""
        start = datetime(2026, 12, 3, 14, 0, 0)
        df = _make_valid_24h_df(start, "AP001")
        end_dt = df["Datetime"].iloc[-1]
        expected_target_str = (end_dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        res = validate_forecast_sequence(df, "AP001")
        assert res.target_time == expected_target_str


class TestForecastServiceTimeAlignment:
    """Tests that SensorForecastService and HybridPredictionService set horizons strictly from sequence end."""

    def test_sensor_service_forecast_horizon_and_state(self):
        """SensorForecastService must base target_time on sequence end time."""
        start = datetime(2026, 10, 15, 0, 0, 0)
        df = _make_valid_24h_df(start, "AP001")
        end_time_str = df["Datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        target_time_str = (df["Datetime"].iloc[-1] + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        srv = SensorForecastService()
        res = srv.get_forecast(station_id="AP001", df_24h=df)

        assert res["forecast_issue_time"] == end_time_str
        assert res["target_time"] == target_time_str
        assert res["forecast_state"] == "HISTORICAL DEMO FORECAST"  # Stale relative to now
        assert "sensor_tensor_digest" in res
        assert len(res["sensor_tensor_digest"]) == 64  # SHA-256

    def test_hybrid_service_forecast_horizon_and_state(self):
        """HybridPredictionService must base target_time on sequence end time."""
        start = datetime(2026, 10, 15, 0, 0, 0)
        df = _make_valid_24h_df(start, "AP001")
        end_time_str = df["Datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        target_time_str = (df["Datetime"].iloc[-1] + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        srv = HybridPredictionService()
        res = srv.predict_hybrid(station_id="AP001", df_24h=df)

        assert res["forecast_issue_time"] == end_time_str
        assert res["target_time"] == target_time_str
        assert res["forecast_state"] == "HISTORICAL DEMO FORECAST"
        assert "sensor_tensor_digest" in res
        assert len(res["sensor_tensor_digest"]) == 64

    def test_demo_sequences_csv_all_aligned(self):
        """Every single scenario in dashboard_demo_sequences.csv must have target_time = end_time + 1h."""
        df_demo = load_demo_scenarios()
        assert not df_demo.empty
        assert "scenario_id" in df_demo.columns

        for scenario_id, grp in df_demo.groupby("scenario_id"):
            assert len(grp) == 24, f"Scenario {scenario_id} must have 24 rows"
            dt_series = pd.to_datetime(grp["Datetime"])
            assert dt_series.is_monotonic_increasing
            end_t = dt_series.iloc[-1]
            exp_target = end_t + timedelta(hours=1)

            val = validate_forecast_sequence(grp, str(grp["StationId"].iloc[0]))
            assert val.target_time == exp_target.strftime("%Y-%m-%d %H:%M:%S")
