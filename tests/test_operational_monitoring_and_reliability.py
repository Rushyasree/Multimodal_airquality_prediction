"""
tests/test_operational_monitoring_and_reliability.py

Phase 17.3: Comprehensive test suite for operational live forecast monitoring,
data reliability, resilient ingestion, forecast audit logging, and observation matching.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

from src.services.operational_health_service import (
    OperationalHealthService,
    OperationalHealthState,
    get_operational_health_service,
)
from src.prediction.station_data_quality import (
    StationDataQualityChecker,
    StationQualityReport,
)
from src.data.resilient_ingestion import (
    ResilientDataIngestionManager,
    get_resilient_ingestion_manager,
    sanitize_text,
)
from src.services.forecast_audit_logger import (
    ForecastAuditLogger,
    get_forecast_audit_logger,
    sanitize_secrets,
)
from src.services.forecast_execution_lifecycle import (
    ForecastExecutionLifecycleManager,
    get_forecast_lifecycle_manager,
    get_aqi_category_and_guidance,
)
from src.services.forecast_observation_matcher import (
    ForecastObservationMatcher,
    get_forecast_observation_matcher,
)


class TestOperationalHealthService:
    """Test Suite 1: Operational health monitoring and state transitions."""

    def setup_method(self):
        self.health_service = get_operational_health_service()
        self.health_service.reset_metrics()

    def test_api_call_metrics_and_latency(self):
        """Service tracks API call counts, latencies, and success rate."""
        self.health_service.record_api_call("OpenAQ_v3", success=True, latency_ms=120.0)
        self.health_service.record_api_call("OpenAQ_v3", success=True, latency_ms=180.0)
        self.health_service.record_api_call("OpenAQ_v3", success=False, latency_ms=500.0, error_msg="Timeout")

        tracker = self.health_service.api_trackers["OpenAQ_v3"]
        assert tracker.total_calls == 3
        assert tracker.successful_calls == 2
        assert tracker.failed_calls == 1
        assert np.isclose(tracker.success_rate_pct, 66.7, atol=0.1)
        assert np.isclose(tracker.avg_latency_ms, 266.67, atol=0.1)
        assert tracker.min_latency_ms == 120.0
        assert tracker.max_latency_ms == 500.0

    def test_station_health_state_transitions(self):
        """Station state transitions strictly: HEALTHY, DEGRADED, STALE, UNAVAILABLE, ERROR."""
        now_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)

        # 1. Fresh physical telemetry (1h old) with complete 24h buffer -> HEALTHY
        fresh_time = now_utc - timedelta(hours=1)
        h1 = self.health_service.evaluate_station_health(
            station_id="AP001",
            observation_timestamp=fresh_time,
            has_pm25=True,
            has_pm10=True,
            buffer_hours_count=24,
            reference_time=now_utc,
        )
        assert h1.state == OperationalHealthState.HEALTHY
        assert h1.forecast_ready is True
        assert h1.rejection_reason is None

        # 2. Incomplete buffer (18h) -> DEGRADED
        h2 = self.health_service.evaluate_station_health(
            station_id="AP002",
            observation_timestamp=fresh_time,
            has_pm25=True,
            has_pm10=True,
            buffer_hours_count=18,
            reference_time=now_utc,
        )
        assert h2.state == OperationalHealthState.DEGRADED
        assert h2.forecast_ready is False
        assert "Incomplete buffer" in (h2.rejection_reason or "")

        # 3. Successful API query returning stale data (5h old) -> STALE (never HEALTHY)
        stale_time = now_utc - timedelta(hours=5)
        h3 = self.health_service.evaluate_station_health(
            station_id="AP003",
            observation_timestamp=stale_time,
            has_pm25=True,
            has_pm10=True,
            buffer_hours_count=24,
            reference_time=now_utc,
        )
        assert h3.state == OperationalHealthState.STALE
        assert h3.forecast_ready is False
        assert "stale" in (h3.rejection_reason or "").lower()

        # 4. Empty buffer -> UNAVAILABLE
        h4 = self.health_service.evaluate_station_health(
            station_id="AP004",
            observation_timestamp=None,
            buffer_hours_count=0,
            reference_time=now_utc,
        )
        assert h4.state == OperationalHealthState.UNAVAILABLE
        assert h4.forecast_ready is False

        # 5. Upstream API error -> ERROR
        h5 = self.health_service.evaluate_station_health(
            station_id="AP005",
            observation_timestamp=fresh_time,
            api_error="502 Bad Gateway",
            reference_time=now_utc,
        )
        assert h5.state == OperationalHealthState.ERROR

    def test_secret_redaction_in_exception_logging(self):
        """Sensitive API keys and tokens are masked in operational error logs."""
        secret_key = "openaq_secret_1234567890abcdef1234567890abcdef"
        self.health_service.record_api_call(
            "OpenAQ_v3",
            success=False,
            latency_ms=100.0,
            error_msg=f"Auth failed with token {secret_key}",
        )
        assert len(self.health_service.exceptions_log) == 1
        err = self.health_service.exceptions_log[0]["error"]
        assert secret_key not in err
        assert "[REDACTED" in err or "Scrubbed" in err


class TestStationDataQualityChecker:
    """Test Suite 2: Station-level data quality validation."""

    def setup_method(self):
        self.checker = StationDataQualityChecker(max_staleness_hours=3.0)

    def test_single_measurement_physical_bounds(self):
        """Unphysical negative or extreme spikes are rejected."""
        now = datetime.now(timezone.utc)
        # Valid PM2.5
        ok, val, _ = self.checker.audit_single_measurement("AP001", "PM2.5", 45.5, now)
        assert ok is True and val == 45.5

        # Negative PM2.5 rejected
        ok_neg, _, issues_neg = self.checker.audit_single_measurement("AP001", "PM2.5", -5.0, now)
        assert ok_neg is False
        assert any(i.check_name == "physical_bounds" for i in issues_neg)

        # Extreme spike PM2.5 > 1000 rejected
        ok_high, _, issues_high = self.checker.audit_single_measurement("AP001", "PM2.5", 1500.0, now)
        assert ok_high is False
        assert any(i.check_name == "physical_bounds" for i in issues_high)

    def test_rejection_of_future_timestamps(self):
        """Observations with timestamps > 5 minutes in the future are rejected."""
        now = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        future_ts = now + timedelta(hours=2)

        ok, _, issues = self.checker.audit_single_measurement(
            "AP001", "PM2.5", 45.0, future_ts, reference_time=now
        )
        assert ok is False
        assert any(i.check_name == "future_timestamp" for i in issues)

    def test_sequence_quality_report_and_rate_of_change(self):
        """Sequence audit detects rate-of-change spikes and computes quality score."""
        anchor = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        rows = []
        for h in range(24):
            dt = anchor - timedelta(hours=23 - h)
            # Normal values except sudden 400 ug/m3 jump at hour 20
            pm25 = 45.0 if h != 20 else 460.0
            rows.append({
                "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "StationId": "AP001",
                "PM2.5": pm25,
                "PM10": 85.0,
            })
        df = pd.DataFrame(rows)

        report = self.checker.audit_sequence(df, "AP001", reference_time=anchor)
        assert isinstance(report, StationQualityReport)
        assert report.station_id == "AP001"
        assert report.total_records == 24
        assert report.spike_anomaly_count >= 1
        assert any(i.check_name == "rate_of_change_spike" for i in report.issues)


class TestResilientDataIngestionManager:
    """Test Suite 3: Resilient ingestion, exponential backoff, rate limiting, and caching."""

    def setup_method(self):
        self.mgr = get_resilient_ingestion_manager()
        self.mgr.clear_cache()

    def test_exponential_backoff_and_recovery(self):
        """Manager retries transient failures with exponential backoff and succeeds on recovery."""
        call_count = 0

        def flaky_api():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary upstream network glitch")
            return {"status": "ok", "pm25": 42.0}

        # Set small base backoff for test speed
        self.mgr.base_backoff_seconds = 0.05
        res = self.mgr.execute_with_retry(flaky_api, service_name="Test_Flaky", max_attempts=4)

        assert call_count == 3
        assert res["status"] == "ok"
        assert res["pm25"] == 42.0

    def test_cache_preservation_during_temporary_outage(self):
        """During upstream outage, preserved valid physical reading is served marked STALE."""
        station_id = "AP001"

        # 1. Successful first retrieval
        first_call = True
        def fetcher(sid):
            nonlocal first_call
            if first_call:
                first_call = False
                return {"station_id": sid, "pm25": 48.0, "pm10": 92.0, "observation_time": "2026-11-12 09:00:00"}
            raise TimeoutError("Upstream API is down")

        obs1 = self.mgr.get_station_observation(station_id, fetcher, force_refresh=True)
        assert obs1["pm25"] == 48.0
        assert obs1["is_cached"] is False

        # 2. Subsequent call encounters upstream outage -> serves preserved reading marked STALE
        obs2 = self.mgr.get_station_observation(station_id, fetcher, force_refresh=True)
        assert obs2["pm25"] == 48.0
        assert obs2["is_outage_preserved"] is True
        assert "STALE" in obs2["freshness_status"]
        assert "outage_notice" in obs2

    def test_zero_fake_reading_generation(self):
        """If no previous valid observation exists, manager returns UNAVAILABLE without fabricating numbers."""
        def failing_fetcher(sid):
            raise ConnectionError("Complete connection failure")

        res = self.mgr.get_station_observation("NEW_STATION", failing_fetcher, force_refresh=True)
        assert res["status"] == "unavailable"
        assert res["pm25"] is None
        assert res["pm10"] is None


class TestForecastExecutionLifecycle:
    """Test Suite 4: 10-step forecast execution lifecycle and error isolation."""

    def setup_method(self):
        self.lifecycle = get_forecast_lifecycle_manager()

    def test_successful_lifecycle_sensor_lstm(self):
        """Executes full 10-step lifecycle for Sensor-Only LSTM on valid sequence."""
        res = self.lifecycle.execute_forecast_lifecycle(
            station_id="AP001",
            model_type="sensor_lstm",
            live_mode=False,
        )
        assert res["status"] == "ok"
        assert res["forecast_status"] == "Forecast Available"
        assert res["pm25"] is not None
        assert res["pm10"] is not None
        assert res["target_time"] != "Unavailable"
        assert res["run_id"].startswith("fc_run_")
        assert res["aqi_category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        assert len(res["health_guidance"]) > 10

    def test_station_isolated_failure_handling(self):
        """Failure on an unphysical input fails safely without crashing dashboard."""
        # DataFrame with unphysical negative values
        bad_df = pd.DataFrame([{"Datetime": f"2026-11-12 {h:02d}:00:00", "StationId": "AP001", "PM2.5": -50.0, "PM10": -100.0} for h in range(24)])

        res = self.lifecycle.execute_forecast_lifecycle(
            station_id="AP001",
            model_type="sensor_lstm",
            df_24h=bad_df,
            live_mode=True,
        )
        assert res["status"] == "unavailable"
        assert res["pm25"] is None
        assert res["run_id"] is not None
        assert "Validation Failed" in res["forecast_status"] or "Unavailable" in res["forecast_status"]

    def test_aqi_category_and_guidance(self):
        """Health guidance is tailored to pollutant concentration category."""
        cat_good, col_good, guide_good = get_aqi_category_and_guidance(25.0)
        assert cat_good == "Good" and col_good == "#22C55E"

        cat_poor, col_poor, guide_poor = get_aqi_category_and_guidance(110.0)
        assert cat_poor == "Poor" and col_poor == "#F97316"

        cat_severe, col_severe, guide_severe = get_aqi_category_and_guidance(350.0)
        assert cat_severe == "Severe" and col_severe == "#7F1D1D"


class TestForecastAuditLogger:
    """Test Suite 5: Audit logging and secret sanitization."""

    def setup_method(self):
        self.audit = get_forecast_audit_logger()
        self.audit.clear_memory()

    def test_audit_record_creation_and_retrieval(self):
        """Creates structured audit record and retrieves recent entries."""
        rec = self.audit.record_run(
            station_id="AP001",
            target_timestamp_utc="2026-11-12 11:00:00",
            input_window_start_utc="2026-11-11 11:00:00",
            input_window_end_utc="2026-11-12 10:00:00",
            observation_provenance="LIVE_MEASURED",
            model_identifier="sensor_lstm",
            model_version="LSTM Baseline v1",
            forecast_output={"pm25": 44.2, "pm10": 88.5},
            execution_duration_ms=45.2,
            status="COMPLETED",
        )
        assert rec.run_id.startswith("fc_run_")
        assert rec.station_id == "AP001"

        recent = self.audit.get_recent_records("AP001")
        assert len(recent) == 1
        assert recent[0].run_id == rec.run_id

    def test_audit_record_secret_sanitization(self):
        """Secrets in error details or metadata are redacted before persistence."""
        secret_token = "auth_token_9876543210fedcba9876543210fedcba"
        rec = self.audit.record_run(
            station_id="AP001",
            target_timestamp_utc="2026-11-12 11:00:00",
            input_window_start_utc="",
            input_window_end_utc="",
            observation_provenance="INVALID",
            model_identifier="sensor_lstm",
            model_version="v1",
            forecast_output={"pm25": None, "pm10": None},
            execution_duration_ms=10.0,
            status="EXECUTION_FAILED",
            error_details=f"Upstream bearer token {secret_token} rejected",
        )
        assert secret_token not in (rec.error_details or "")
        assert "[REDACTED]" in (rec.error_details or "")


class TestForecastObservationMatcher:
    """Test Suite 6: Operational forecast and observation matching and error computation."""

    def setup_method(self):
        self.matcher = get_forecast_observation_matcher()
        self.matcher.clear_memory()
        self.audit = get_forecast_audit_logger()
        self.audit.clear_memory()

    def test_strict_temporal_and_station_matching(self):
        """Matches forecast with corresponding actual measurement by station and hour."""
        target_time = "2026-11-12T11:00:00Z"

        # Record forecast
        rec = self.audit.record_run(
            station_id="AP001",
            target_timestamp_utc=target_time,
            input_window_start_utc="2026-11-11T11:00:00Z",
            input_window_end_utc="2026-11-12T10:00:00Z",
            observation_provenance="LIVE_MEASURED",
            model_identifier="sensor_lstm",
            model_version="v1",
            forecast_output={"pm25": 50.0, "pm10": 100.0},
            baseline_output={"persistence_pm25": 45.0, "persistence_pm10": 90.0},
            execution_duration_ms=25.0,
            status="COMPLETED",
        )

        # Arriving ground observation at 11:00 UTC
        obs = {
            "station_id": "AP001",
            "observation_time": "2026-11-12T11:15:00Z",
            "pm25": 54.0,
            "pm10": 108.0,
        }

        match = self.matcher.match_forecast_with_observation(rec, obs)

        assert match.match_status == "MATCHED"
        assert match.actual_pm25 == 54.0
        assert match.forecast_pm25 == 50.0
        assert match.error_pm25 == -4.0                  # pred - actual = 50 - 54
        assert match.absolute_error_pm25 == 4.0          # |50 - 54|
        assert match.persistence_error_pm25 == 9.0       # |actual - pers| = |54 - 45|

    def test_rejection_of_misaligned_temporal_windows(self):
        """Rejects pairing when observation timestamp belongs to a different clock hour."""
        target_time = "2026-11-12T11:00:00Z"
        rec = self.audit.record_run(
            station_id="AP001",
            target_timestamp_utc=target_time,
            input_window_start_utc="",
            input_window_end_utc="",
            observation_provenance="LIVE_MEASURED",
            model_identifier="sensor_lstm",
            model_version="v1",
            forecast_output={"pm25": 50.0, "pm10": 100.0},
            execution_duration_ms=20.0,
            status="COMPLETED",
        )

        # Observation belongs to 13:00 UTC (+2h discrepancy)
        obs_late = {
            "station_id": "AP001",
            "observation_time": "2026-11-12T13:00:00Z",
            "pm25": 55.0,
            "pm10": 105.0,
        }

        match = self.matcher.match_forecast_with_observation(rec, obs_late)
        assert match.match_status == "MISALIGNED_TEMPORAL_WINDOW"
        assert match.actual_pm25 is None

    def test_rejection_of_station_mismatch(self):
        """Rejects pairing when station identities differ."""
        rec = self.audit.record_run(
            station_id="AP001",
            target_timestamp_utc="2026-11-12T11:00:00Z",
            input_window_start_utc="",
            input_window_end_utc="",
            observation_provenance="LIVE_MEASURED",
            model_identifier="sensor_lstm",
            model_version="v1",
            forecast_output={"pm25": 50.0, "pm10": 100.0},
            execution_duration_ms=20.0,
            status="COMPLETED",
        )

        obs_diff_sid = {
            "station_id": "DL026",
            "observation_time": "2026-11-12T11:00:00Z",
            "pm25": 55.0,
            "pm10": 105.0,
        }

        match = self.matcher.match_forecast_with_observation(rec, obs_diff_sid)
        assert match.match_status == "REJECTED_STATION_MISMATCH"
