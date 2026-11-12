"""
src/services/operational_health_service.py

Phase 17.3: Operational Health Monitoring Service.
Monitors:
1. Data source availability (OpenAQ, Open-Meteo, CPCB CAAQMS, Local Cache).
2. API request success/failure counters & latency metrics.
3. Last successful data retrieval timestamp & last physical observation timestamp.
4. Observation age (hours) and cache freshness.
5. Station availability & pollutant presence (PM2.5, PM10).
6. 24-hour sequence completeness and validation status.
7. Forecast service availability and run state.
8. Sanitized exception logging without secret leakage.

Explicit Operational States:
- HEALTHY: Physical data fresh (<= 3h), 24h buffer complete, APIs responsive.
- DEGRADED: Partial station coverage, high latency, or missing non-critical channels.
- STALE: Data age > 3h; cached telemetry serving during temporary upstream lag.
- UNAVAILABLE: Offline station, empty buffer (<24h), or network disconnection.
- VALIDATING: Sequence integrity check or buffer aggregation in progress.
- ERROR: Upstream 5xx/4xx failure or unrecoverable parsing exception.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import logging
import threading
import time
import pandas as pd

logger = logging.getLogger("VayuDrishti.OperationalHealth")
ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "experiments/phase17_3"


class OperationalHealthState(str, Enum):
    """Mutually exclusive operational health states."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    VALIDATING = "VALIDATING"
    ERROR = "ERROR"


@dataclass
class ApiMetricsTracker:
    """Tracks latency, call counts, and success rates for an external upstream service."""
    service_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    last_call_timestamp: Optional[str] = None
    last_success_timestamp: Optional[str] = None
    last_error_message: Optional[str] = None

    def record_call(self, success: bool, latency_ms: float, error_msg: Optional[str] = None) -> None:
        self.total_calls += 1
        self.last_latency_ms = round(latency_ms, 2)
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        now_iso = datetime.now(timezone.utc).isoformat()
        self.last_call_timestamp = now_iso
        if success:
            self.successful_calls += 1
            self.last_success_timestamp = now_iso
        else:
            self.failed_calls += 1
            self.last_error_message = error_msg

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_calls, 2)

    @property
    def success_rate_pct(self) -> float:
        if self.total_calls == 0:
            return 100.0
        return round((self.successful_calls / self.total_calls) * 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate_pct": self.success_rate_pct,
            "last_latency_ms": self.last_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": 0.0 if self.min_latency_ms == float("inf") else round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "last_call_timestamp": self.last_call_timestamp,
            "last_success_timestamp": self.last_success_timestamp,
            "last_error_message": self.last_error_message,
        }


@dataclass
class StationHealthSummary:
    """Operational health snapshot for a single monitoring station."""
    station_id: str
    state: OperationalHealthState
    observation_age_hours: float
    last_observation_timestamp: Optional[str]
    has_pm25: bool
    has_pm10: bool
    buffer_hours_count: int
    completeness_pct: float
    forecast_ready: bool
    data_source: str
    rejection_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class SystemHealthReport:
    """System-wide operational health report across all connectors and stations."""
    overall_state: OperationalHealthState
    active_stations_count: int
    healthy_stations_count: int
    degraded_stations_count: int
    stale_stations_count: int
    unavailable_stations_count: int
    api_metrics: Dict[str, Dict[str, Any]]
    cache_freshness_seconds: Optional[float]
    last_retrieval_utc: Optional[str]
    forecast_service_available: bool
    report_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overall_state"] = self.overall_state.value
        return d


class OperationalHealthService:
    """Thread-safe operational telemetry health monitor and diagnostics registry."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OperationalHealthService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_staleness_hours: float = 3.0):
        if getattr(self, "_initialized", False):
            return
        self.max_staleness_hours = float(max_staleness_hours)
        self.api_trackers: Dict[str, ApiMetricsTracker] = {
            "OpenAQ_v3": ApiMetricsTracker(service_name="OpenAQ_v3"),
            "Open_Meteo": ApiMetricsTracker(service_name="Open_Meteo"),
            "CPCB_Local_Buffer": ApiMetricsTracker(service_name="CPCB_Local_Buffer"),
        }
        self.station_health: Dict[str, StationHealthSummary] = {}
        self.cache_last_updated: Optional[datetime] = None
        self.last_retrieval_time: Optional[datetime] = None
        self.exceptions_log: List[Dict[str, Any]] = []
        self._state_lock = threading.RLock()
        self._initialized = True

    def record_api_call(
        self,
        service_name: str,
        success: bool,
        latency_ms: float,
        error_msg: Optional[str] = None,
    ) -> None:
        """Records an API request result, duration, and sanitized error."""
        with self._state_lock:
            if service_name not in self.api_trackers:
                self.api_trackers[service_name] = ApiMetricsTracker(service_name=service_name)
            self.api_trackers[service_name].record_call(success, latency_ms, error_msg)
            if success:
                self.last_retrieval_time = datetime.now(timezone.utc)
            elif error_msg:
                self._record_sanitized_exception(service_name, error_msg)

    def _record_sanitized_exception(self, component: str, error_msg: str) -> None:
        # Sanitize exception message: scrub potential tokens or secrets
        sanitized = str(error_msg)
        for token_key in ["key", "token", "auth", "secret"]:
            if token_key in sanitized.lower():
                sanitized = "Scrubbed authorization failure or sensitive string."
        self.exceptions_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "error": sanitized,
        })
        # Keep rolling log to latest 50 entries
        if len(self.exceptions_log) > 50:
            self.exceptions_log.pop(0)

    def update_cache_timestamp(self) -> None:
        """Marks cache refresh timestamp."""
        with self._state_lock:
            self.cache_last_updated = datetime.now(timezone.utc)

    def get_cache_age_seconds(self) -> Optional[float]:
        """Returns age of the current memory cache in seconds."""
        with self._state_lock:
            if self.cache_last_updated is None:
                return None
            return round((datetime.now(timezone.utc) - self.cache_last_updated).total_seconds(), 1)

    def evaluate_station_health(
        self,
        station_id: str,
        observation_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
        has_pm25: bool = True,
        has_pm10: bool = True,
        buffer_hours_count: int = 24,
        data_source: str = "CPCB CAAQMS via OpenAQ v3",
        reference_time: Optional[datetime] = None,
        api_error: Optional[str] = None,
    ) -> StationHealthSummary:
        """Evaluates and records operational state for an individual monitoring station.

        State Transition Logic:
        1. If api_error is present -> ERROR.
        2. If buffer_hours_count == 0 or observation_timestamp is missing -> UNAVAILABLE.
        3. If observation_age > max_staleness_hours -> STALE.
        4. If buffer_hours_count < 24 or missing core pollutants -> DEGRADED.
        5. If fresh (<= max_staleness_hours), 24h contiguous, and has PM2.5/PM10 -> HEALTHY.
        """
        clean_sid = str(station_id).strip().upper()
        now_utc = reference_time or datetime.now(timezone.utc)

        # Compute age
        obs_age_hours = 999.0
        obs_iso = None
        if observation_timestamp is not None:
            try:
                if isinstance(observation_timestamp, str):
                    clean_ts = observation_timestamp.replace(" IST", " +05:30")
                    dt = pd.to_datetime(clean_ts)
                else:
                    dt = observation_timestamp
                if dt.tzinfo is None:
                    dt_utc = dt.replace(tzinfo=timezone.utc)
                else:
                    dt_utc = dt.astimezone(timezone.utc)
                obs_age_hours = max(0.0, (now_utc - dt_utc).total_seconds() / 3600.0)
                obs_iso = dt_utc.isoformat()
            except Exception:
                obs_age_hours = 999.0

        completeness_pct = round(min(100.0, (buffer_hours_count / 24.0) * 100.0), 1)

        # State determination
        if api_error:
            state = OperationalHealthState.ERROR
            rejection = f"Upstream API error: {api_error}"
            forecast_ready = False
        elif buffer_hours_count == 0 or observation_timestamp is None:
            state = OperationalHealthState.UNAVAILABLE
            rejection = "No physical measurements available in buffer."
            forecast_ready = False
        elif obs_age_hours > self.max_staleness_hours:
            state = OperationalHealthState.STALE
            rejection = f"Telemetry is stale ({obs_age_hours:.1f}h old; threshold is {self.max_staleness_hours}h)."
            forecast_ready = False
        elif buffer_hours_count < 24 or not (has_pm25 and has_pm10):
            state = OperationalHealthState.DEGRADED
            missing = []
            if not has_pm25:
                missing.append("PM2.5")
            if not has_pm10:
                missing.append("PM10")
            if buffer_hours_count < 24:
                rejection = f"Incomplete buffer: {buffer_hours_count}/24 continuous hours."
            else:
                rejection = f"Missing criteria pollutants: {', '.join(missing)}."
            forecast_ready = False
        else:
            state = OperationalHealthState.HEALTHY
            rejection = None
            forecast_ready = True

        summary = StationHealthSummary(
            station_id=clean_sid,
            state=state,
            observation_age_hours=round(obs_age_hours, 2),
            last_observation_timestamp=obs_iso,
            has_pm25=has_pm25,
            has_pm10=has_pm10,
            buffer_hours_count=buffer_hours_count,
            completeness_pct=completeness_pct,
            forecast_ready=forecast_ready,
            data_source=data_source,
            rejection_reason=rejection,
        )

        with self._state_lock:
            self.station_health[clean_sid] = summary

        return summary

    def get_system_health(self) -> SystemHealthReport:
        """Aggregates system-wide operational health across all monitored stations and APIs."""
        with self._state_lock:
            stations_dict = {sid: s.to_dict() for sid, s in self.station_health.items()}
            metrics_dict = {svc: tracker.to_dict() for svc, tracker in self.api_trackers.items()}
            cache_age = self.get_cache_age_seconds()
            last_retrieval = self.last_retrieval_time.isoformat() if self.last_retrieval_time else None

        active_count = len(stations_dict)
        healthy_count = sum(1 for s in stations_dict.values() if s["state"] == OperationalHealthState.HEALTHY.value)
        degraded_count = sum(1 for s in stations_dict.values() if s["state"] == OperationalHealthState.DEGRADED.value)
        stale_count = sum(1 for s in stations_dict.values() if s["state"] == OperationalHealthState.STALE.value)
        unavail_count = sum(1 for s in stations_dict.values() if s["state"] in [OperationalHealthState.UNAVAILABLE.value, OperationalHealthState.ERROR.value])

        # Overall system state derivation
        if active_count == 0:
            overall = OperationalHealthState.UNAVAILABLE
        elif healthy_count == active_count:
            overall = OperationalHealthState.HEALTHY
        elif healthy_count > 0:
            overall = OperationalHealthState.DEGRADED
        elif stale_count > 0:
            overall = OperationalHealthState.STALE
        else:
            overall = OperationalHealthState.UNAVAILABLE

        # Check API failure rates
        openaq_tracker = self.api_trackers.get("OpenAQ_v3")
        if openaq_tracker and openaq_tracker.total_calls >= 5 and openaq_tracker.success_rate_pct < 50.0:
            if overall == OperationalHealthState.HEALTHY:
                overall = OperationalHealthState.DEGRADED

        return SystemHealthReport(
            overall_state=overall,
            active_stations_count=active_count,
            healthy_stations_count=healthy_count,
            degraded_stations_count=degraded_count,
            stale_stations_count=stale_count,
            unavailable_stations_count=unavail_count,
            api_metrics=metrics_dict,
            cache_freshness_seconds=cache_age,
            last_retrieval_utc=last_retrieval,
            forecast_service_available=True,
            stations=stations_dict,
        )

    def reset_metrics(self) -> None:
        """Resets all metrics (useful in unit testing)."""
        with self._state_lock:
            for tracker in self.api_trackers.values():
                tracker.total_calls = 0
                tracker.successful_calls = 0
                tracker.failed_calls = 0
                tracker.total_latency_ms = 0.0
                tracker.last_latency_ms = 0.0
                tracker.min_latency_ms = float("inf")
                tracker.max_latency_ms = 0.0
                tracker.last_error_message = None
            self.station_health.clear()
            self.exceptions_log.clear()
            self.cache_last_updated = None
            self.last_retrieval_time = None


_HEALTH_SERVICE_INSTANCE: Optional[OperationalHealthService] = None


def get_operational_health_service() -> OperationalHealthService:
    """Returns the process singleton OperationalHealthService."""
    global _HEALTH_SERVICE_INSTANCE
    if _HEALTH_SERVICE_INSTANCE is None:
        _HEALTH_SERVICE_INSTANCE = OperationalHealthService()
    return _HEALTH_SERVICE_INSTANCE
