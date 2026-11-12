"""
src/services/forecast_execution_lifecycle.py

Phase 17.3: Forecast Execution Lifecycle Manager.
Implements the 10-step forecast execution lifecycle:
1. Request forecast: Validate requested parameters & model choice.
2. Identify station: Match station identity against catalog.
3. Retrieve observations: Fetch from LiveSequenceBuilder or supplied dataframe.
4. Validate provenance & freshness: Strict physical inlet & max staleness check.
5. Validate sequence completeness: Exactly 24 continuous hours, zero gaps.
6. Prepare model input: Scaler verification & tensor transformation.
7. Execute forecast: Run parameterized model (Sensor-only LSTM or Unified Hybrid).
8. Validate output: Check physical concentration bounds [0, 1000] µg/m³.
9. Store forecast record: Persist immutable structured audit record via ForecastAuditLogger.
10. Return forecast and status: Clean, station-isolated payload with health guidance.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time
import logging
import pandas as pd
from PIL import Image

from src.prediction.data_loader import load_location_catalog, get_verified_24h_sequence_for_station
from src.prediction.sequence_validator import validate_live_forecast_sequence, validate_forecast_sequence
from src.prediction.live_sequence_builder import get_live_sequence_builder, LiveSequenceBuilder
from src.prediction.station_data_quality import StationDataQualityChecker
from src.services.sensor_forecast_service import SensorForecastService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.forecast_audit_logger import get_forecast_audit_logger, ForecastAuditLogger
from src.services.operational_health_service import get_operational_health_service

logger = logging.getLogger("VayuDrishti.ForecastLifecycle")
ROOT_DIR = Path(__file__).resolve().parents[2]


def get_aqi_category_and_guidance(pm25: Optional[float]) -> Tuple[str, str, str]:
    """Computes Indian National AQI category, color code, and actionable health guidance."""
    if pm25 is None or pm25 < 0:
        return "Unavailable", "#94A3B8", "No forecast data available to compute air quality index."
    elif pm25 <= 30:
        return "Good", "#22C55E", "Air quality is satisfactory; air pollution poses little or no risk."
    elif pm25 <= 60:
        return "Satisfactory", "#84CC16", "Minor breathing discomfort to sensitive people."
    elif pm25 <= 90:
        return "Moderate", "#EAB308", "Breathing discomfort to people with lungs, asthma, and heart diseases."
    elif pm25 <= 120:
        return "Poor", "#F97316", "Breathing discomfort to most people on prolonged exposure."
    elif pm25 <= 250:
        return "Very Poor", "#EF4444", "Respiratory illness on prolonged exposure. Sensitive groups avoid outdoor exertion."
    else:
        return "Severe", "#7F1D1D", "Affects healthy people and seriously impacts those with existing diseases. Stay indoors."


class ForecastExecutionLifecycleManager:
    """Manages robust 10-step forecast execution with station isolation and audit logging."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ForecastExecutionLifecycleManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        live_builder: Optional[LiveSequenceBuilder] = None,
        audit_logger: Optional[ForecastAuditLogger] = None,
    ):
        if getattr(self, "_initialized", False):
            return
        self.catalog = load_location_catalog()
        self.live_builder = live_builder or get_live_sequence_builder()
        self.audit_logger = audit_logger or get_forecast_audit_logger()
        self.health_service = get_operational_health_service()
        self.quality_checker = StationDataQualityChecker()
        self.sensor_service = SensorForecastService()
        self.hybrid_service = HybridPredictionService()
        self._initialized = True

    def execute_forecast_lifecycle(
        self,
        station_id: str,
        model_type: str = "sensor_lstm",     # "sensor_lstm", "hybrid", "persistence"
        df_24h: Optional[pd.DataFrame] = None,
        image_input: Optional[Union[str, Path, bytes, Image.Image]] = None,
        live_mode: bool = False,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Executes the complete 10-step forecast lifecycle safely with station-level isolation."""
        start_wall_time = time.perf_counter()
        clean_sid = str(station_id).strip().upper()
        now_utc = reference_time or datetime.now(timezone.utc)

        # STEP 1: Request forecast & Validate parameters
        supported_models = ["sensor_lstm", "hybrid", "persistence"]
        if model_type not in supported_models:
            return self._fail_safely(
                station_id=clean_sid,
                error_code="INVALID_MODEL_TYPE",
                message=f"Model '{model_type}' is not supported. Must be one of {supported_models}.",
                start_time=start_wall_time,
            )

        # STEP 2: Identify station in catalog
        station_meta = None
        if not self.catalog.empty and "station_id" in self.catalog.columns:
            match = self.catalog[self.catalog["station_id"] == clean_sid]
            if not match.empty:
                station_meta = match.iloc[0].to_dict()

        station_name = station_meta.get("station_name", clean_sid) if station_meta else clean_sid
        city = station_meta.get("city", "Unknown") if station_meta else "Unknown"

        # STEP 3: Retrieve available observations
        df_to_use = None
        seq_provenance = ""
        is_synth = False
        live_meta = None

        if df_24h is not None:
            df_to_use = df_24h
            seq_provenance = f"Supplied 24h Sequence for {clean_sid}"
            is_synth = False
        elif live_mode:
            df_live, live_meta = self.live_builder.build_live_sequence(
                station_id=clean_sid,
                current_system_time=now_utc,
            )
            if df_live is None:
                duration_ms = (time.perf_counter() - start_wall_time) * 1000.0
                rejection = live_meta.rejection_reason if live_meta else "Incomplete live buffer (< 24 continuous hours)."
                # Record in audit log
                audit_rec = self.audit_logger.record_run(
                    station_id=clean_sid,
                    target_timestamp_utc="Unavailable",
                    input_window_start_utc="Unavailable",
                    input_window_end_utc="Unavailable",
                    observation_provenance="LIVE_PARTIAL",
                    model_identifier=model_type,
                    model_version="N/A",
                    forecast_output={"pm25": None, "pm10": None},
                    execution_duration_ms=duration_ms,
                    status="INPUT_UNAVAILABLE",
                    error_details=rejection,
                )
                self.health_service.evaluate_station_health(
                    station_id=clean_sid,
                    buffer_hours_count=live_meta.measured_hours_count if live_meta else 0,
                    reference_time=now_utc,
                )
                return {
                    "status": "unavailable",
                    "forecast_status": "Forecast Unavailable (Incomplete Live Buffer)",
                    "run_id": audit_rec.run_id,
                    "station_id": clean_sid,
                    "station_name": station_name,
                    "city": city,
                    "model_type": model_type,
                    "pm25": None,
                    "pm10": None,
                    "target_time": "Unavailable",
                    "issue_time": "Unavailable",
                    "rejection_reason": rejection,
                    "provenance": "LIVE INPUT UNAVAILABLE",
                    "is_live_forecast": False,
                    "health_guidance": "No live forecast can be issued due to insufficient physical measurements.",
                    "aqi_category": "Unavailable",
                    "aqi_color": "#94A3B8",
                }
            df_to_use = df_live
            seq_provenance = live_meta.source_provider
        else:
            # Historical reproducible demo sequence
            df_to_use, seq_provenance, is_synth = get_verified_24h_sequence_for_station(clean_sid)

        # STEP 4 & 5: Validate provenance, freshness, and sequence completeness
        if live_mode:
            val_res = validate_live_forecast_sequence(
                df_to_use,
                station_id=clean_sid,
                reference_time=now_utc,
                max_staleness_hours=3.0,
                source_provenance_category="LIVE_MEASURED",
            )
            if not val_res.live_forecast_eligible:
                duration_ms = (time.perf_counter() - start_wall_time) * 1000.0
                audit_rec = self.audit_logger.record_run(
                    station_id=clean_sid,
                    target_timestamp_utc=val_res.target_time,
                    input_window_start_utc=val_res.start_time,
                    input_window_end_utc=val_res.end_time,
                    observation_provenance=val_res.provenance_category,
                    model_identifier=model_type,
                    model_version="N/A",
                    forecast_output={"pm25": None, "pm10": None},
                    execution_duration_ms=duration_ms,
                    status="VALIDATION_FAILED",
                    error_details=val_res.rejection_reason,
                    sequence_validation_result=val_res.to_dict(),
                )
                return {
                    "status": "unavailable",
                    "forecast_status": "Input Validation Failed",
                    "run_id": audit_rec.run_id,
                    "station_id": clean_sid,
                    "station_name": station_name,
                    "city": city,
                    "model_type": model_type,
                    "pm25": None,
                    "pm10": None,
                    "target_time": val_res.target_time,
                    "issue_time": val_res.end_time,
                    "rejection_reason": val_res.rejection_reason,
                    "provenance": "LIVE INPUT UNAVAILABLE",
                    "is_live_forecast": False,
                    "health_guidance": "Live forecast aborted due to sequence validation failure.",
                    "aqi_category": "Unavailable",
                    "aqi_color": "#94A3B8",
                }
            issue_time = val_res.end_time
            target_time = val_res.target_time
            provenance_badge = "LIVE FORECAST — VALIDATED INPUT"
            val_dict = val_res.to_dict()
        else:
            val_res = validate_forecast_sequence(df_to_use, station_id=clean_sid)
            issue_time = val_res.end_time or "Unknown"
            target_time = val_res.target_time or "Unknown"
            provenance_badge = "HISTORICAL DEMO FORECAST"
            val_dict = val_res.to_dict()

        # Compute persistence baseline
        pers_pm25 = None
        pers_pm10 = None
        if df_to_use is not None and len(df_to_use) > 0:
            last_row = df_to_use.iloc[-1]
            if "PM2.5" in last_row and pd.notna(last_row["PM2.5"]):
                pers_pm25 = round(float(last_row["PM2.5"]), 2)
            if "PM10" in last_row and pd.notna(last_row["PM10"]):
                pers_pm10 = round(float(last_row["PM10"]), 2)

        # STEP 6 & 7: Prepare model input and Execute forecast
        pm25_pred = None
        pm10_pred = None
        model_version = ""
        tensor_digest = None

        try:
            if model_type == "persistence":
                pm25_pred = pers_pm25
                pm10_pred = pers_pm10
                model_version = "Naïve 1-Hour Persistence Baseline (yt+1 = yt)"
                tensor_digest = None
            elif model_type == "sensor_lstm":
                res = self.sensor_service.get_forecast(
                    station_id=clean_sid,
                    df_24h=df_to_use,
                    live_mode=live_mode,
                    reference_time=now_utc,
                )
                if res.get("status") == "unavailable":
                    return self._fail_safely(
                        station_id=clean_sid,
                        error_code="MODEL_EXECUTION_UNAVAILABLE",
                        message=res.get("scientific_note") or "Sensor forecast service marked input unavailable.",
                        start_time=start_wall_time,
                    )
                pm25_pred = res.get("pm25")
                pm10_pred = res.get("pm10")
                model_version = res.get("model_version", "2-Layer Stacked LSTM Baseline")
                tensor_digest = res.get("sensor_tensor_digest")
            elif model_type == "hybrid":
                res = self.hybrid_service.predict_hybrid(
                    station_id=clean_sid,
                    df_24h=df_to_use,
                    image_input=image_input,
                    live_mode=live_mode,
                    reference_time=now_utc,
                )
                if res.get("status") == "unavailable":
                    return self._fail_safely(
                        station_id=clean_sid,
                        error_code="MODEL_EXECUTION_UNAVAILABLE",
                        message=res.get("scientific_note") or "Hybrid prediction service marked input unavailable.",
                        start_time=start_wall_time,
                    )
                pm25_pred = res.get("pm25")
                pm10_pred = res.get("pm10")
                model_version = res.get("model_version", "Gated Multimodal Fusion v2.0")
                tensor_digest = res.get("sensor_tensor_digest")

        except Exception as e:
            logger.error(f"Execution failure for {clean_sid} on {model_type}: {e}")
            return self._fail_safely(
                station_id=clean_sid,
                error_code="MODEL_EXECUTION_EXCEPTION",
                message=f"Internal model execution error: {str(e)}",
                start_time=start_wall_time,
            )

        # STEP 8: Validate output concentrations
        if pm25_pred is not None and (pm25_pred < 0.0 or pm25_pred > 1000.0):
            return self._fail_safely(
                station_id=clean_sid,
                error_code="UNPHYSICAL_OUTPUT_PM25",
                message=f"Model produced unphysical PM2.5 prediction ({pm25_pred} µg/m³).",
                start_time=start_wall_time,
            )
        if pm10_pred is not None and (pm10_pred < 0.0 or pm10_pred > 1500.0):
            return self._fail_safely(
                station_id=clean_sid,
                error_code="UNPHYSICAL_OUTPUT_PM10",
                message=f"Model produced unphysical PM10 prediction ({pm10_pred} µg/m³).",
                start_time=start_wall_time,
            )

        # STEP 9: Store forecast record in audit logger
        duration_ms = (time.perf_counter() - start_wall_time) * 1000.0
        start_time_iso = str(df_to_use["Datetime"].iloc[0]) if "Datetime" in df_to_use.columns else ""
        end_time_iso = str(df_to_use["Datetime"].iloc[-1]) if "Datetime" in df_to_use.columns else ""

        audit_rec = self.audit_logger.record_run(
            station_id=clean_sid,
            target_timestamp_utc=target_time,
            input_window_start_utc=start_time_iso,
            input_window_end_utc=end_time_iso,
            observation_provenance="LIVE_MEASURED" if live_mode else "HISTORICAL_DEMO",
            model_identifier=model_type,
            model_version=model_version,
            forecast_output={"pm25": pm25_pred, "pm10": pm10_pred},
            baseline_output={"persistence_pm25": pers_pm25, "persistence_pm10": pers_pm10},
            input_tensor_digest=tensor_digest,
            execution_duration_ms=duration_ms,
            status="COMPLETED",
            sequence_validation_result=val_dict,
        )

        # STEP 10: Compute AQI and Return forecast payload
        aqi_cat, aqi_color, guidance = get_aqi_category_and_guidance(pm25_pred)

        return {
            "status": "ok",
            "forecast_status": "Forecast Available",
            "run_id": audit_rec.run_id,
            "station_id": clean_sid,
            "station_name": station_name,
            "city": city,
            "model_type": model_type,
            "model_version": model_version,
            "pm25": pm25_pred,
            "pm10": pm10_pred,
            "target_time": target_time,
            "issue_time": issue_time,
            "provenance": provenance_badge,
            "sequence_provenance": seq_provenance,
            "is_live_forecast": live_mode,
            "is_synthetic_input": is_synth,
            "execution_duration_ms": round(duration_ms, 2),
            "persistence_baseline": {"pm25": pers_pm25, "pm10": pers_pm10},
            "tensor_digest": tensor_digest,
            "aqi_category": aqi_cat,
            "aqi_color": aqi_color,
            "health_guidance": guidance,
        }

    def _fail_safely(
        self,
        station_id: str,
        error_code: str,
        message: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """Returns clean station-isolated failure response without raising exceptions."""
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        audit_rec = self.audit_logger.record_run(
            station_id=station_id,
            target_timestamp_utc="Unavailable",
            input_window_start_utc="Unavailable",
            input_window_end_utc="Unavailable",
            observation_provenance="INVALID",
            model_identifier="unknown",
            model_version="N/A",
            forecast_output={"pm25": None, "pm10": None},
            execution_duration_ms=duration_ms,
            status="EXECUTION_FAILED",
            error_details=message,
        )
        return {
            "status": "error",
            "forecast_status": "Model Execution Failed",
            "run_id": audit_rec.run_id,
            "station_id": station_id,
            "error_code": error_code,
            "message": message,
            "pm25": None,
            "pm10": None,
            "target_time": "Unavailable",
            "issue_time": "Unavailable",
            "provenance": "ERROR",
            "aqi_category": "Unavailable",
            "aqi_color": "#94A3B8",
            "health_guidance": "Forecasting unavailable due to input or processing failure.",
        }


_LIFECYCLE_MANAGER: Optional[ForecastExecutionLifecycleManager] = None


def get_forecast_lifecycle_manager() -> ForecastExecutionLifecycleManager:
    """Returns singleton ForecastExecutionLifecycleManager."""
    global _LIFECYCLE_MANAGER
    if _LIFECYCLE_MANAGER is None:
        _LIFECYCLE_MANAGER = ForecastExecutionLifecycleManager()
    return _LIFECYCLE_MANAGER
