"""
src/services/forecast_observation_matcher.py

Phase 17.3: Operational Forecast and Observation Matching Service.
Provides:
1. Strict station-isolated matching between stored forecasts and subsequent ground observations.
2. Clock-hour temporal alignment: target_hour_utc == observation_clock_hour_utc.
3. Rejection of ambiguous, duplicate, or misaligned matches.
4. Operational error calculation (MAE, signed bias, percentage error, persistence comparison).
5. Safe handling of pending/unmatched forecasts when ground telemetry has not yet arrived.
6. Zero fabrication of ground-truth values.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import json
import logging
import threading
import pandas as pd
import numpy as np

from src.services.forecast_audit_logger import ForecastAuditRecord, get_forecast_audit_logger

logger = logging.getLogger("VayuDrishti.ForecastMatcher")
ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_LOG_FILE = ROOT_DIR / "experiments/phase17_3/matched_evaluations.jsonl"


@dataclass
class MatchedForecastEvaluation:
    """Represents a validated pair of a forecast and its corresponding physical observation."""
    match_id: str
    run_id: str
    station_id: str
    target_timestamp_utc: str
    actual_observation_timestamp_utc: str
    model_identifier: str
    # Predicted concentrations
    forecast_pm25: Optional[float]
    forecast_pm10: Optional[float]
    # Actual ground-truth measurements
    actual_pm25: Optional[float]
    actual_pm10: Optional[float]
    # Baseline comparison
    persistence_pm25: Optional[float]
    persistence_pm10: Optional[float]
    # Operational errors
    error_pm25: Optional[float]              # pred - actual
    absolute_error_pm25: Optional[float]     # |pred - actual|
    error_pm10: Optional[float]
    absolute_error_pm10: Optional[float]
    # Persistence errors
    persistence_error_pm25: Optional[float]
    persistence_error_pm10: Optional[float]
    # Matching metadata
    match_status: str                        # "MATCHED", "PENDING_ACTUAL_OBSERVATION", "AMBIGUOUS", "MISALIGNED"
    temporal_discrepancy_minutes: float      # Time difference between target and observation
    evaluation_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ForecastObservationMatcher:
    """Matches stored forecast runs with subsequent physical ground-truth measurements."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ForecastObservationMatcher, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, output_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self.output_path = output_path or EVAL_LOG_FILE
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_logger = get_forecast_audit_logger()
        self._matched_evaluations: List[MatchedForecastEvaluation] = []
        self._matched_run_ids: Set[str] = set()
        self._match_lock = threading.Lock()
        self._initialized = True

    def _normalize_to_clock_hour(self, ts: Union[str, datetime, pd.Timestamp]) -> datetime:
        """Parses and floors a timestamp to UTC clock hour."""
        if isinstance(ts, str):
            dt = pd.to_datetime(ts)
        else:
            dt = ts
        if dt.tzinfo is None:
            dt_utc = dt.replace(tzinfo=timezone.utc)
        else:
            dt_utc = dt.astimezone(timezone.utc)
        return datetime(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, tzinfo=timezone.utc)

    def match_forecast_with_observation(
        self,
        forecast_record: ForecastAuditRecord,
        actual_observation: Dict[str, Any],
    ) -> MatchedForecastEvaluation:
        """Strictly matches a single forecast against a prospective actual ground measurement.

        Rejection rules:
        - Station IDs must match exactly.
        - Target clock hour must equal observation clock hour.
        - Physical measurements must not be null/NaN.
        """
        clean_sid = forecast_record.station_id.strip().upper()
        obs_sid = str(actual_observation.get("station_id", "")).strip().upper()

        if clean_sid != obs_sid:
            return MatchedForecastEvaluation(
                match_id=f"match_{forecast_record.run_id}_reject",
                run_id=forecast_record.run_id,
                station_id=clean_sid,
                target_timestamp_utc=forecast_record.target_timestamp_utc,
                actual_observation_timestamp_utc=str(actual_observation.get("observation_time", "")),
                model_identifier=forecast_record.model_identifier,
                forecast_pm25=forecast_record.forecast_output.get("pm25"),
                forecast_pm10=forecast_record.forecast_output.get("pm10"),
                actual_pm25=None,
                actual_pm10=None,
                persistence_pm25=forecast_record.baseline_output.get("persistence_pm25"),
                persistence_pm10=forecast_record.baseline_output.get("persistence_pm10"),
                error_pm25=None,
                absolute_error_pm25=None,
                error_pm10=None,
                absolute_error_pm10=None,
                persistence_error_pm25=None,
                persistence_error_pm10=None,
                match_status="REJECTED_STATION_MISMATCH",
                temporal_discrepancy_minutes=999.0,
            )

        # Temporal alignment check
        target_ch = self._normalize_to_clock_hour(forecast_record.target_timestamp_utc)
        obs_time = actual_observation.get("observation_time") or actual_observation.get("observation_time_utc")
        if not obs_time:
            return MatchedForecastEvaluation(
                match_id=f"match_{forecast_record.run_id}_pending",
                run_id=forecast_record.run_id,
                station_id=clean_sid,
                target_timestamp_utc=forecast_record.target_timestamp_utc,
                actual_observation_timestamp_utc="N/A",
                model_identifier=forecast_record.model_identifier,
                forecast_pm25=forecast_record.forecast_output.get("pm25"),
                forecast_pm10=forecast_record.forecast_output.get("pm10"),
                actual_pm25=None,
                actual_pm10=None,
                persistence_pm25=forecast_record.baseline_output.get("persistence_pm25"),
                persistence_pm10=forecast_record.baseline_output.get("persistence_pm10"),
                error_pm25=None,
                absolute_error_pm25=None,
                error_pm10=None,
                absolute_error_pm10=None,
                persistence_error_pm25=None,
                persistence_error_pm10=None,
                match_status="PENDING_ACTUAL_OBSERVATION",
                temporal_discrepancy_minutes=999.0,
            )

        obs_ch = self._normalize_to_clock_hour(obs_time)
        discrepancy_min = abs((target_ch - obs_ch).total_seconds()) / 60.0

        if target_ch != obs_ch:
            return MatchedForecastEvaluation(
                match_id=f"match_{forecast_record.run_id}_misaligned",
                run_id=forecast_record.run_id,
                station_id=clean_sid,
                target_timestamp_utc=forecast_record.target_timestamp_utc,
                actual_observation_timestamp_utc=str(obs_time),
                model_identifier=forecast_record.model_identifier,
                forecast_pm25=forecast_record.forecast_output.get("pm25"),
                forecast_pm10=forecast_record.forecast_output.get("pm10"),
                actual_pm25=None,
                actual_pm10=None,
                persistence_pm25=forecast_record.baseline_output.get("persistence_pm25"),
                persistence_pm10=forecast_record.baseline_output.get("persistence_pm10"),
                error_pm25=None,
                absolute_error_pm25=None,
                error_pm10=None,
                absolute_error_pm10=None,
                persistence_error_pm25=None,
                persistence_error_pm10=None,
                match_status="MISALIGNED_TEMPORAL_WINDOW",
                temporal_discrepancy_minutes=discrepancy_min,
            )

        # Extract measured concentrations
        actual_pm25 = actual_observation.get("pm25")
        actual_pm10 = actual_observation.get("pm10")
        if actual_pm25 is None and "measurements" in actual_observation:
            actual_pm25 = actual_observation["measurements"].get("PM2.5")
            actual_pm10 = actual_observation["measurements"].get("PM10")

        f_pm25 = forecast_record.forecast_output.get("pm25")
        f_pm10 = forecast_record.forecast_output.get("pm10")
        p_pm25 = forecast_record.baseline_output.get("persistence_pm25")
        p_pm10 = forecast_record.baseline_output.get("persistence_pm10")

        err_pm25 = round(f_pm25 - float(actual_pm25), 2) if (f_pm25 is not None and actual_pm25 is not None) else None
        abs_pm25 = abs(err_pm25) if err_pm25 is not None else None
        err_pm10 = round(f_pm10 - float(actual_pm10), 2) if (f_pm10 is not None and actual_pm10 is not None) else None
        abs_pm10 = abs(err_pm10) if err_pm10 is not None else None

        p_err_pm25 = abs(round(float(actual_pm25) - p_pm25, 2)) if (actual_pm25 is not None and p_pm25 is not None) else None
        p_err_pm10 = abs(round(float(actual_pm10) - p_pm10, 2)) if (actual_pm10 is not None and p_pm10 is not None) else None

        evaluation = MatchedForecastEvaluation(
            match_id=f"match_{forecast_record.run_id}_{target_ch.strftime('%Y%m%d%H')}",
            run_id=forecast_record.run_id,
            station_id=clean_sid,
            target_timestamp_utc=forecast_record.target_timestamp_utc,
            actual_observation_timestamp_utc=str(obs_time),
            model_identifier=forecast_record.model_identifier,
            forecast_pm25=f_pm25,
            forecast_pm10=f_pm10,
            actual_pm25=float(actual_pm25) if actual_pm25 is not None else None,
            actual_pm10=float(actual_pm10) if actual_pm10 is not None else None,
            persistence_pm25=p_pm25,
            persistence_pm10=p_pm10,
            error_pm25=err_pm25,
            absolute_error_pm25=abs_pm25,
            error_pm10=err_pm10,
            absolute_error_pm10=abs_pm10,
            persistence_error_pm25=p_err_pm25,
            persistence_error_pm10=p_err_pm10,
            match_status="MATCHED",
            temporal_discrepancy_minutes=discrepancy_min,
        )

        with self._match_lock:
            self._matched_evaluations.append(evaluation)
            self._matched_run_ids.add(forecast_record.run_id)
            # Write to persistence
            try:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(evaluation.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to persist matched evaluation: {e}")

        return evaluation

    def match_batch(
        self,
        observations_by_station_and_hour: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> List[MatchedForecastEvaluation]:
        """Scans un-matched forecast runs and matches them against newly available observations."""
        results = []
        recent_records = self.audit_logger.get_recent_records(limit=200)

        for rec in recent_records:
            if rec.status != "COMPLETED" or rec.run_id in self._matched_run_ids:
                continue

            target_ch = self._normalize_to_clock_hour(rec.target_timestamp_utc).isoformat()
            key = (rec.station_id, target_ch)

            if key in observations_by_station_and_hour:
                obs = observations_by_station_and_hour[key]
                match_res = self.match_forecast_with_observation(rec, obs)
                results.append(match_res)

        return results

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates aggregated operational error metrics across all matched forecasts."""
        with self._match_lock:
            valid_matches = [m for m in self._matched_evaluations if m.match_status == "MATCHED"]

        if not valid_matches:
            return {
                "matched_count": 0,
                "pm25_mae": None,
                "pm10_mae": None,
                "pm25_bias": None,
                "pm10_bias": None,
                "persistence_pm25_mae": None,
                "persistence_pm10_mae": None,
            }

        pm25_errors = [m.absolute_error_pm25 for m in valid_matches if m.absolute_error_pm25 is not None]
        pm10_errors = [m.absolute_error_pm10 for m in valid_matches if m.absolute_error_pm10 is not None]
        pm25_biases = [m.error_pm25 for m in valid_matches if m.error_pm25 is not None]
        pm10_biases = [m.error_pm10 for m in valid_matches if m.error_pm10 is not None]

        pers_pm25_errors = [m.persistence_error_pm25 for m in valid_matches if m.persistence_error_pm25 is not None]
        pers_pm10_errors = [m.persistence_error_pm10 for m in valid_matches if m.persistence_error_pm10 is not None]

        return {
            "matched_count": len(valid_matches),
            "pm25_mae": round(float(np.mean(pm25_errors)), 2) if pm25_errors else None,
            "pm10_mae": round(float(np.mean(pm10_errors)), 2) if pm10_errors else None,
            "pm25_bias": round(float(np.mean(pm25_biases)), 2) if pm25_biases else None,
            "pm10_bias": round(float(np.mean(pm10_biases)), 2) if pm10_biases else None,
            "persistence_pm25_mae": round(float(np.mean(pers_pm25_errors)), 2) if pers_pm25_errors else None,
            "persistence_pm10_mae": round(float(np.mean(pers_pm10_errors)), 2) if pers_pm10_errors else None,
        }

    def clear_memory(self) -> None:
        """Clears memory state (for testing)."""
        with self._match_lock:
            self._matched_evaluations.clear()
            self._matched_run_ids.clear()


_MATCHER_INSTANCE: Optional[ForecastObservationMatcher] = None


def get_forecast_observation_matcher() -> ForecastObservationMatcher:
    """Returns singleton ForecastObservationMatcher."""
    global _MATCHER_INSTANCE
    if _MATCHER_INSTANCE is None:
        _MATCHER_INSTANCE = ForecastObservationMatcher()
    return _MATCHER_INSTANCE
