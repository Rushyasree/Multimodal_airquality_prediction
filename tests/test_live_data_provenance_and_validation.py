"""
tests/test_live_data_provenance_and_validation.py

Phase 17.2: Comprehensive test suite for authentic live data provenance,
multi-gate live validation, end-to-end forecasting pathways, and model comparison synchronization.
"""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytz

from src.prediction.observation_provenance import (
    ObservationRecord,
    ObservationProvenanceCategory,
    SequenceProvenanceMeta,
    PROVENANCE_BADGE_MAP,
)
from src.data.live_rolling_buffer import (
    StationRollingBuffer,
    LiveRollingBufferRegistry,
    LSTM_27_FEATURES,
)
from src.prediction.live_sequence_builder import LiveSequenceBuilder
from src.prediction.sequence_validator import (
    validate_forecast_sequence,
    validate_live_forecast_sequence,
    LiveSequenceValidationResult,
)
from src.services.sensor_forecast_service import SensorForecastService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.model_comparison_service import ModelComparisonService


def _generate_synthetic_24h_measured_records(
    station_id: str,
    base_anchor_utc: datetime,
    pm25_base: float = 45.0,
    pm10_base: float = 85.0,
    readings_per_hour: int = 2,
):
    """Generates authentic-format sub-hourly observation records for exactly 24 contiguous hours."""
    records = []
    # 24 hours: from (anchor - 23h) up to anchor
    start_utc = base_anchor_utc - timedelta(hours=23)
    
    for h in range(24):
        hour_dt = start_utc + timedelta(hours=h)
        for r in range(readings_per_hour):
            sub_minute = int((60 / readings_per_hour) * r)
            reading_dt = hour_dt.replace(minute=sub_minute, second=0, microsecond=0)
            
            # PM2.5 reading
            records.append(
                ObservationRecord(
                    station_id=station_id,
                    parameter="PM2.5",
                    value=round(pm25_base + np.sin(h / 3.8) * 10 + r * 0.5, 2),
                    measurement_timestamp_utc=reading_dt.isoformat(),
                    source_provider="CPCB Physical CAAQMS Station",
                    is_measured=True,
                    provenance_category=ObservationProvenanceCategory.LIVE_MEASURED,
                )
            )
            # PM10 reading
            records.append(
                ObservationRecord(
                    station_id=station_id,
                    parameter="PM10",
                    value=round(pm10_base + np.cos(h / 3.8) * 15 + r * 1.0, 2),
                    measurement_timestamp_utc=reading_dt.isoformat(),
                    source_provider="CPCB Physical CAAQMS Station",
                    is_measured=True,
                    provenance_category=ObservationProvenanceCategory.LIVE_MEASURED,
                )
            )
            # Environmental readings for complete schema
            for param, val in [("temperature_C", 26.5), ("relative_humidity", 60.0), ("wind_speed", 2.5), ("NO2", 18.0), ("SO2", 12.0), ("CO", 0.8)]:
                records.append(
                    ObservationRecord(
                        station_id=station_id,
                        parameter=param,
                        value=val,
                        measurement_timestamp_utc=reading_dt.isoformat(),
                        source_provider="CPCB Physical CAAQMS Station",
                        is_measured=True,
                        provenance_category=ObservationProvenanceCategory.LIVE_MEASURED,
                    )
                )
    return records


class TestLiveSequenceBuilderAndBuffer:
    """Test Suite 1: Live sequence builder, buffer aggregation, and incomplete buffer rejection."""

    def test_24h_contiguous_sequence_construction(self):
        """Builder constructs valid 24h DataFrame from authentic sub-hourly readings."""
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry, max_staleness_hours=3.0)
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)

        records = _generate_synthetic_24h_measured_records(station_id, anchor_utc)
        for rec in records:
            builder.ingest_observation_record(rec)

        # Build live sequence evaluated at current_system_time = anchor_utc
        df_24h, meta = builder.build_live_sequence(station_id, current_system_time=anchor_utc)

        assert df_24h is not None, f"Expected 24h DataFrame, got None: {meta.rejection_reason}"
        assert len(df_24h) == 24
        assert meta.is_strictly_measured is True
        assert meta.measured_hours_count == 24
        assert meta.completeness_pct == 100.0
        assert meta.provenance_category == ObservationProvenanceCategory.LIVE_MEASURED
        assert "LIVE FORECAST — VALIDATED INPUT" in meta.status_badge

        # Verify hourly aggregation
        diffs = pd.to_datetime(df_24h["Datetime"]).diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all(), "Time delta between contiguous rows must be strictly 1 hour"

    def test_incomplete_buffer_rejection_without_synthetic_substitution(self):
        """When buffer has fewer than 24 hours of measurements, live forecasting is aborted."""
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry, max_staleness_hours=3.0)
        station_id = "AP005"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)

        # Ingest only 15 hours of observations
        start_utc = anchor_utc - timedelta(hours=14)
        for h in range(15):
            hour_dt = start_utc + timedelta(hours=h)
            builder.ingest_raw_measurement(station_id, "PM2.5", 40.0, hour_dt)
            builder.ingest_raw_measurement(station_id, "PM10", 80.0, hour_dt)

        df_24h, meta = builder.build_live_sequence(station_id, current_system_time=anchor_utc)

        assert df_24h is None, "Incomplete buffer must return None DataFrame"
        assert meta.is_strictly_measured is False
        assert meta.measured_hours_count == 15
        assert meta.completeness_pct < 100.0
        assert meta.provenance_category == ObservationProvenanceCategory.LIVE_PARTIAL
        assert "LIVE INPUT UNAVAILABLE" in meta.status_badge
        assert "15/24" in (meta.rejection_reason or "")

    def test_sub_hourly_aggregation_and_deduplication(self):
        """Sub-hourly readings in [H:00, H:59] are aggregated by arithmetic mean."""
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry, max_staleness_hours=3.0)
        station_id = "DL026"
        hour_dt = datetime(2026, 11, 12, 8, 0, 0, tzinfo=timezone.utc)

        # Ingest 3 readings at 08:10 (30.0), 08:25 (40.0), 08:50 (50.0) -> Expected mean = 40.0
        builder.ingest_raw_measurement(station_id, "PM2.5", 30.0, hour_dt.replace(minute=10))
        builder.ingest_raw_measurement(station_id, "PM2.5", 40.0, hour_dt.replace(minute=25))
        builder.ingest_raw_measurement(station_id, "PM2.5", 50.0, hour_dt.replace(minute=50))

        buf = registry.get_or_create_buffer(station_id)
        assert buf.get_sample_count(hour_dt, "PM2.5") == 3
        assert np.isclose(buf.get_hourly_mean(hour_dt, "PM2.5"), 40.0)

    def test_physical_bounds_filtering(self):
        """Unphysical sensor spikes (e.g. negative or >1000 ug/m3 PM2.5) are rejected at ingestion."""
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)
        station_id = "TG006"
        ts = datetime(2026, 11, 12, 9, 0, 0, tzinfo=timezone.utc)

        res_neg = builder.ingest_raw_measurement(station_id, "PM2.5", -5.0, ts)
        assert res_neg["status"] == "rejected"

        res_spike = builder.ingest_raw_measurement(station_id, "PM2.5", 2500.0, ts)
        assert res_spike["status"] == "rejected"


class TestMultiGateLiveValidation:
    """Test Suite 2: Multi-gate validation with independent boolean gates."""

    def test_multi_gate_validation_passes_on_authentic_fresh_sequence(self):
        """Authentic, fresh, 24-hour sequence passes all 7 validation gates."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry, max_staleness_hours=3.0)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        df_24h, _ = builder.build_live_sequence(station_id, current_system_time=anchor_utc)
        assert df_24h is not None

        val = validate_live_forecast_sequence(
            df_24h,
            station_id=station_id,
            reference_time=anchor_utc,
            max_staleness_hours=3.0,
            source_provenance_category="LIVE_MEASURED",
        )

        assert isinstance(val, LiveSequenceValidationResult)
        assert val.sequence_structural_validity is True
        assert val.station_identity_validity is True
        assert val.source_authenticity is True
        assert val.measurement_completeness is True
        assert val.freshness_validity is True
        assert val.scaler_compatibility is True
        assert val.live_forecast_eligible is True
        assert val.rejection_reason is None

    def test_multi_gate_validation_rejects_stale_sequence(self):
        """Sequence with age > max_staleness_hours fails freshness gate and eligibility."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry, max_staleness_hours=10.0)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        df_24h, _ = builder.build_live_sequence(station_id, current_system_time=anchor_utc)
        assert df_24h is not None

        # Reference time 5 hours after sequence end -> stale relative to 3h threshold
        future_ref = anchor_utc + timedelta(hours=5)
        val = validate_live_forecast_sequence(
            df_24h,
            station_id=station_id,
            reference_time=future_ref,
            max_staleness_hours=3.0,
            source_provenance_category="LIVE_MEASURED",
        )

        assert val.freshness_validity is False
        assert val.live_forecast_eligible is False
        assert "stale" in (val.rejection_reason or "").lower()

    def test_multi_gate_validation_rejects_non_authentic_sources(self):
        """Historical demo or synthetic sequences fail source authenticity gate for live forecasting."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        df_24h, _ = builder.build_live_sequence(station_id, current_system_time=anchor_utc)
        assert df_24h is not None

        val = validate_live_forecast_sequence(
            df_24h,
            station_id=station_id,
            reference_time=anchor_utc,
            source_provenance_category="HISTORICAL_DEMO",
        )

        assert val.source_authenticity is False
        assert val.live_forecast_eligible is False
        assert "source authenticity" in (val.rejection_reason or "").lower()


class TestEndToEndLiveForecastingServices:
    """Test Suite 3: End-to-end live forecasting in SensorForecastService and HybridPredictionService."""

    def test_sensor_forecast_service_live_pathway(self):
        """SensorForecastService accepts verified live sequence and marks LIVE FORECAST."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        df_24h, _ = builder.build_live_sequence(station_id, current_system_time=anchor_utc)
        assert df_24h is not None

        service = SensorForecastService()
        res = service.get_forecast(
            station_id=station_id,
            df_24h=df_24h,
            live_mode=True,
            reference_time=anchor_utc,
        )

        assert res["status"] == "ok"
        assert res["pm25"] is not None and res["pm25"] > 0
        assert res["pm10"] is not None and res["pm10"] > 0
        assert "LIVE FORECAST — VALIDATED INPUT" in res["provenance"]
        assert res["sensor_tensor_digest"] is not None
        assert res["forecast_issue_time"] != "Unknown"
        assert res["target_time"] != "Unknown"

    def test_sensor_forecast_service_rejects_stale_in_live_mode(self):
        """SensorForecastService returns unavailable with None concentrations when live sequence is stale."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        df_24h, _ = builder.build_live_sequence(station_id, current_system_time=anchor_utc)
        assert df_24h is not None

        service = SensorForecastService()
        stale_ref = anchor_utc + timedelta(hours=6)
        res = service.get_forecast(
            station_id=station_id,
            df_24h=df_24h,
            live_mode=True,
            reference_time=stale_ref,
        )

        assert res["status"] == "unavailable"
        assert res["pm25"] is None
        assert res["pm10"] is None
        assert "LIVE INPUT UNAVAILABLE" in res["provenance"]
        assert res["rejection_reason"] is not None

    def test_hybrid_prediction_service_live_pathway(self):
        """HybridPredictionService executes multimodal fusion without silent zero fallbacks."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        df_24h, _ = builder.build_live_sequence(station_id, current_system_time=anchor_utc)
        assert df_24h is not None

        service = HybridPredictionService()
        res = service.predict_hybrid(
            station_id=station_id,
            df_24h=df_24h,
            live_mode=True,
            reference_time=anchor_utc,
        )

        assert res["status"] == "ok"
        assert res["pm25"] is not None
        assert res["pm10"] is not None
        assert res["sensor_tensor_digest"] is not None
        assert "LIVE FORECAST — VALIDATED INPUT" in res["forecast_state"]


class TestModelComparisonLiveParityAndRejection:
    """Test Suite 4: Model comparison synchronization, tensor digest parity, and persistence baseline."""

    def test_live_synchronization_and_tensor_digest_parity(self):
        """LSTM and Hybrid ingest identical live tensors and report verified SHA-256 match."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc):
            builder.ingest_observation_record(rec)

        comp_service = ModelComparisonService(live_builder=builder)
        res = comp_service.compare_all(
            station_id=station_id,
            live_mode=True,
            reference_time=anchor_utc,
        )

        assert res["live_input_available"] is True
        assert res["tensor_matched"] is True
        assert res["tensor_digest_lstm"] == res["tensor_digest_hybrid"]
        assert "Strictly Synchronized (Authentic Live Telemetry • Identical Tensor Digest)" in res["synchronization_status"]
        assert "Strictly Synchronized (Identical Tensor Digest)" in res["tensor_synchronization_status"]

        # Check table contents
        df_tbl = res["comparison_table"]
        methods = list(df_tbl["Method"])
        assert "1-Hour Persistence Baseline" in methods
        assert "Sensor-Only LSTM" in methods
        assert "Unified Hybrid Model" in methods

    def test_incomplete_buffer_rejects_live_synchronization(self):
        """When live buffer is incomplete, ModelComparisonService marks both models unavailable and rejects sync."""
        station_id = "DL026"
        registry = LiveRollingBufferRegistry()
        # Empty buffer for DL026
        builder = LiveSequenceBuilder(registry=registry)

        comp_service = ModelComparisonService(live_builder=builder)
        res = comp_service.compare_all(
            station_id=station_id,
            live_mode=True,
            reference_time=datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc),
        )

        assert res["live_input_available"] is False
        assert res["tensor_matched"] is False
        assert "Live Synchronization Rejected" in res["synchronization_status"]
        assert res["sensor_forecast"]["pm25"] is None
        assert res["hybrid_prediction"]["pm25"] is None

        # Check table rows for unavailable status
        df_tbl = res["comparison_table"]
        lstm_row = df_tbl[df_tbl["Method"] == "Sensor-Only LSTM"].iloc[0]
        assert lstm_row["Result Type"] == "LIVE INPUT UNAVAILABLE"
        assert pd.isna(lstm_row["PM2.5 (ug/m3)"])

    def test_persistence_baseline_correctness(self):
        """Persistence baseline reproduces the last antecedent observation yt without modeling."""
        station_id = "AP001"
        anchor_utc = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        registry = LiveRollingBufferRegistry()
        builder = LiveSequenceBuilder(registry=registry)

        for rec in _generate_synthetic_24h_measured_records(station_id, anchor_utc, pm25_base=52.3, pm10_base=94.1):
            builder.ingest_observation_record(rec)

        comp_service = ModelComparisonService(live_builder=builder)
        res = comp_service.compare_all(
            station_id=station_id,
            live_mode=True,
            reference_time=anchor_utc,
        )

        pers = res["persistence_baseline"]
        assert pers["pm25"] is not None
        assert pers["pm10"] is not None

        df_tbl = res["comparison_table"]
        pers_row = df_tbl[df_tbl["Method"] == "1-Hour Persistence Baseline"].iloc[0]
        assert pers_row["Result Type"] == "REFERENCE BASELINE"
        assert pers_row["PM2.5 (ug/m3)"] == pers["pm25"]
        assert pers_row["PM10 (ug/m3)"] == pers["pm10"]
