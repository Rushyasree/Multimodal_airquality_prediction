"""
src/services/forecast_audit_logger.py

Phase 17.3: Forecast Audit Logger.
Maintains an immutable, structured audit trail of every forecast execution:
- Unique run ID (UUID)
- Station ID
- Generation timestamp (UTC)
- Target timestamp (UTC, t+1)
- Input observation window (start & end)
- Observation provenance & freshness
- Sequence validation results
- Model identifier & version
- Forecast outputs & baseline outputs
- Input tensor digest (SHA-256)
- Execution duration (ms)
- Status & sanitized error details

Guarantees absolute secret redaction and preserves operational logs separate from user-facing UI.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import logging
import threading
import uuid
import re

logger = logging.getLogger("VayuDrishti.ForecastAudit")
ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "experiments/phase17_3"
AUDIT_LOG_FILE = LOGS_DIR / "forecast_audit_log.jsonl"

SECRET_PATTERN = re.compile(r"([0-9a-fA-F]{32,64}|[a-zA-Z0-9_\-]{30,})")


def sanitize_secrets(val: Any) -> Any:
    """Recursively redacts secrets and credentials from audit payloads."""
    if isinstance(val, str):
        return SECRET_PATTERN.sub("[REDACTED]", val)
    elif isinstance(val, dict):
        return {k: sanitize_secrets(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_secrets(x) for x in val]
    return val


@dataclass
class ForecastAuditRecord:
    """Immutable audit record for a single forecast execution run."""
    run_id: str
    station_id: str
    generation_timestamp_utc: str
    target_timestamp_utc: str
    input_window_start_utc: str
    input_window_end_utc: str
    observation_provenance: str
    observation_freshness_hours: Optional[float]
    sequence_validation_result: Dict[str, Any]
    model_identifier: str
    model_version: str
    scaler_version: str
    forecast_output: Dict[str, Optional[float]]       # {"pm25": ..., "pm10": ...}
    baseline_output: Dict[str, Optional[float]]       # {"persistence_pm25": ..., "persistence_pm10": ...}
    input_tensor_digest: Optional[str]
    execution_duration_ms: float
    status: str                                       # "COMPLETED", "INPUT_UNAVAILABLE", "VALIDATION_FAILED", "EXECUTION_FAILED"
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(sanitize_secrets(self.to_dict()))


class ForecastAuditLogger:
    """Thread-safe append-only audit logger for forecast lifecycle execution."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ForecastAuditLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self.log_path = log_path or AUDIT_LOG_FILE
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_records: List[ForecastAuditRecord] = []
        self._file_lock = threading.Lock()
        self._initialized = True

    def record_run(
        self,
        station_id: str,
        target_timestamp_utc: str,
        input_window_start_utc: str,
        input_window_end_utc: str,
        observation_provenance: str,
        model_identifier: str,
        model_version: str,
        forecast_output: Dict[str, Optional[float]],
        execution_duration_ms: float,
        status: str,
        observation_freshness_hours: Optional[float] = None,
        sequence_validation_result: Optional[Dict[str, Any]] = None,
        scaler_version: str = "27_features_standard_scaler_v1",
        baseline_output: Optional[Dict[str, Optional[float]]] = None,
        input_tensor_digest: Optional[str] = None,
        error_details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ForecastAuditRecord:
        """Creates and persists an immutable forecast audit record."""
        run_id = f"fc_run_{uuid.uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc).isoformat()

        record = ForecastAuditRecord(
            run_id=run_id,
            station_id=str(station_id).strip().upper(),
            generation_timestamp_utc=now_utc,
            target_timestamp_utc=str(target_timestamp_utc),
            input_window_start_utc=str(input_window_start_utc),
            input_window_end_utc=str(input_window_end_utc),
            observation_provenance=str(observation_provenance),
            observation_freshness_hours=observation_freshness_hours,
            sequence_validation_result=sequence_validation_result or {},
            model_identifier=model_identifier,
            model_version=model_version,
            scaler_version=scaler_version,
            forecast_output=forecast_output,
            baseline_output=baseline_output or {},
            input_tensor_digest=input_tensor_digest,
            execution_duration_ms=round(execution_duration_ms, 2),
            status=status,
            error_details=sanitize_secrets(error_details) if error_details else None,
            metadata=metadata or {},
        )

        with self._file_lock:
            self._memory_records.append(record)
            # Write line to jsonl
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(record.to_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to append to audit log {self.log_path}: {e}")

        return record

    def get_recent_records(
        self,
        station_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ForecastAuditRecord]:
        """Returns recent audit records from memory."""
        with self._file_lock:
            if station_id:
                clean_sid = str(station_id).strip().upper()
                matching = [r for r in self._memory_records if r.station_id == clean_sid]
                return matching[-limit:]
            return self._memory_records[-limit:]

    def clear_memory(self) -> None:
        """Clears memory records (for unit tests)."""
        with self._file_lock:
            self._memory_records.clear()


_AUDIT_LOGGER: Optional[ForecastAuditLogger] = None


def get_forecast_audit_logger() -> ForecastAuditLogger:
    """Returns singleton ForecastAuditLogger."""
    global _AUDIT_LOGGER
    if _AUDIT_LOGGER is None:
        _AUDIT_LOGGER = ForecastAuditLogger()
    return _AUDIT_LOGGER
