"""
src/prediction/live_sequence_builder.py

Strict Live Sequence Builder for VayuDrishti Phase 17.2.
Constructs verifiable, contiguous 24-hour multivariate input sequences from
authentic monitoring station observations with strict provenance tracking.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import pytz
import numpy as np
import pandas as pd

from src.prediction.observation_provenance import (
    ObservationRecord,
    SequenceProvenanceMeta,
    ObservationProvenanceCategory,
    PROVENANCE_BADGE_MAP,
)
from src.data.live_rolling_buffer import (
    StationRollingBuffer,
    LiveRollingBufferRegistry,
    normalize_timestamp_utc,
    validate_sensor_reading,
    LSTM_27_FEATURES,
    CORE_REQUIRED_POLLUTANTS,
)

logger = logging.getLogger("VayuDrishti.LiveSequenceBuilder")
IST = pytz.timezone("Asia/Kolkata")
ROOT_DIR = Path(__file__).resolve().parents[2]


class LiveSequenceBuilder:
    """Builder that constructs verified, contiguous 24-hour measured sequences

    strictly from physical station observations with zero synthetic replacement.
    """

    def __init__(
        self,
        registry: Optional[LiveRollingBufferRegistry] = None,
        max_staleness_hours: float = 3.0,
    ):
        self.registry = registry or LiveRollingBufferRegistry()
        self.max_staleness_hours = float(max_staleness_hours)
        self._raw_observations: Dict[str, List[ObservationRecord]] = {}

    def ingest_observation_record(self, record: ObservationRecord) -> Dict[str, Any]:
        """Ingests an immutable ObservationRecord into the station's live buffer.

        Rejects station mixing, unphysical values, and records provenance.
        """
        sid = str(record.station_id).strip().upper()
        if sid not in self._raw_observations:
            self._raw_observations[sid] = []

        # Store raw observation in immutable audit trail
        self._raw_observations[sid].append(record)

        # Ingest into station buffer
        buf = self.registry.get_or_create_buffer(sid)
        result = buf.ingest_reading(
            parameter=record.parameter,
            value=record.value,
            timestamp=record.measurement_timestamp_utc,
        )
        return result

    def ingest_raw_measurement(
        self,
        station_id: str,
        parameter: str,
        value: Any,
        timestamp: Union[str, datetime, pd.Timestamp],
        source_provider: str = "CPCB CAAQMS Station",
        source_observation_id: Optional[str] = None,
        is_measured: bool = True,
        is_interpolated: bool = False,
        is_imputed: bool = False,
        raw_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validates and ingests a single sensor reading with explicit provenance."""
        sid = str(station_id).strip().upper()
        ts_utc = normalize_timestamp_utc(timestamp)

        is_valid, parsed_val, err_msg = validate_sensor_reading(parameter, value)
        if not is_valid or parsed_val is None:
            return {
                "status": "rejected",
                "reason": err_msg or "invalid_value",
                "station_id": sid,
                "parameter": parameter,
            }

        now_utc = datetime.now(timezone.utc)
        record = ObservationRecord(
            station_id=sid,
            parameter=parameter,
            value=parsed_val,
            unit="µg/m³" if parameter != "CO" else "mg/m³",
            measurement_timestamp_utc=ts_utc,
            source_provider=source_provider,
            retrieval_timestamp_utc=now_utc,
            source_observation_id=source_observation_id,
            is_measured=is_measured,
            is_interpolated=is_interpolated,
            is_imputed=is_imputed,
            raw_value=raw_value if raw_value is not None else parsed_val,
            sample_count=1,
            metadata=metadata or {},
        )
        return self.ingest_observation_record(record)

    def build_live_sequence(
        self,
        station_id: str,
        anchor_time_utc: Optional[Union[str, datetime, pd.Timestamp]] = None,
        current_system_time: Optional[datetime] = None,
        require_all_27_channels: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], SequenceProvenanceMeta]:
        """Constructs an authentic 24-hour antecedent sequence for live forecasting.

        Zero Silent Substitution:
        If fewer than 24 contiguous measured hours exist, live forecast mode is rejected.
        Returns (None, SequenceProvenanceMeta) with LIVE_PARTIAL or INVALID.
        """
        clean_sid = str(station_id).strip().upper()
        sys_time = current_system_time or datetime.now(timezone.utc)
        if sys_time.tzinfo is None:
            sys_time = sys_time.replace(tzinfo=timezone.utc)

        # 1. Check station identity in verified catalog
        if not self.registry.is_station_recognized(clean_sid):
            meta = SequenceProvenanceMeta(
                station_id=clean_sid,
                provenance_category=ObservationProvenanceCategory.INVALID,
                source_provider="Unrecognized Station",
                start_time="",
                end_time="",
                target_time="",
                total_hours_required=24,
                measured_hours_count=0,
                completeness_pct=0.0,
                is_strictly_measured=False,
                rejection_reason=f"Station '{clean_sid}' is not a recognized monitoring station in the location catalog.",
            )
            return None, meta

        # 2. Query buffer readiness
        buf = self.registry.get_or_create_buffer(clean_sid)
        readiness = buf.check_readiness(
            anchor_time_utc=anchor_time_utc,
            model_type="historical" if require_all_27_channels else "both",
            current_system_time=sys_time,
        )

        anchor_str = readiness.get("anchor_hour_utc", "")
        target_str = readiness.get("target_hour_utc", "")
        avail_count = readiness.get("available_hours_count", 0)
        completeness = round((avail_count / 24.0) * 100.0, 1)

        # 3. Check for incomplete or missing history
        if not readiness.get("is_ready", False):
            reason = readiness.get("reason", "insufficient_history")
            detail = readiness.get("detail", "Buffer verification failed")

            if reason == "insufficient_history":
                cat = ObservationProvenanceCategory.LIVE_PARTIAL
                rej_msg = (
                    f"Incomplete measured history: station {clean_sid} has {avail_count}/24 "
                    f"contiguous measured hours ({detail}). Live forecasting rejected."
                )
            elif reason == "missing_required_channels":
                cat = ObservationProvenanceCategory.LIVE_PARTIAL
                rej_msg = (
                    f"Core criteria pollutants missing in station {clean_sid} telemetry: {detail}. "
                    "Live forecasting rejected."
                )
            elif reason == "stale_data":
                cat = ObservationProvenanceCategory.HISTORICAL_DEMO
                rej_msg = (
                    f"Telemetry for station {clean_sid} is stale ({detail}). "
                    "Live forecasting rejected."
                )
            else:
                cat = ObservationProvenanceCategory.INVALID
                rej_msg = f"Station {clean_sid} validation failed: {detail}"

            meta = SequenceProvenanceMeta(
                station_id=clean_sid,
                provenance_category=cat,
                source_provider="CPCB / OpenAQ Real-Time Inlet",
                start_time="",
                end_time=anchor_str,
                target_time=target_str,
                total_hours_required=24,
                measured_hours_count=avail_count,
                completeness_pct=completeness,
                is_strictly_measured=False,
                aggregation_method="Arithmetic mean of sub-hourly measurements [H:00, H:59]",
                retrieval_timestamp=sys_time.isoformat(),
                rejection_reason=rej_msg,
                metadata=readiness,
            )
            return None, meta

        # 4. Verified 24-hour sequence is present and fresh
        df_24h = readiness.get("dataframe_24h")
        if df_24h is None or len(df_24h) != 24:
            meta = SequenceProvenanceMeta(
                station_id=clean_sid,
                provenance_category=ObservationProvenanceCategory.INVALID,
                source_provider="CPCB / OpenAQ Real-Time Inlet",
                start_time="",
                end_time="",
                target_time="",
                total_hours_required=24,
                measured_hours_count=0,
                completeness_pct=0.0,
                is_strictly_measured=False,
                rejection_reason="Failed to extract 24-row DataFrame from validated buffer.",
            )
            return None, meta

        start_time_str = str(df_24h["Datetime"].iloc[0])
        end_time_str = str(df_24h["Datetime"].iloc[-1])
        try:
            end_dt = pd.to_datetime(end_time_str)
            target_time_str = (end_dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            target_time_str = target_str

        # Format Datetime column cleanly for downstream preprocessors
        df_clean = df_24h.copy()
        df_clean["Datetime"] = pd.to_datetime(df_clean["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_clean.attrs["provenance_category"] = ObservationProvenanceCategory.LIVE_MEASURED.value
        df_clean.attrs["is_strictly_measured"] = True

        meta = SequenceProvenanceMeta(
            station_id=clean_sid,
            provenance_category=ObservationProvenanceCategory.LIVE_MEASURED,
            source_provider="CPCB / APPCB Physical Monitor Inlet via Live Telemetry Stream",
            start_time=start_time_str,
            end_time=end_time_str,
            target_time=target_time_str,
            total_hours_required=24,
            measured_hours_count=24,
            completeness_pct=100.0,
            is_strictly_measured=True,
            aggregation_method="Arithmetic mean of sub-hourly physical measurements in [H:00, H:59] UTC",
            retrieval_timestamp=sys_time.isoformat(),
            status_badge="LIVE FORECAST — VALIDATED INPUT",
            rejection_reason=None,
            metadata={
                "staleness_hours": readiness.get("staleness_hours"),
                "observed_channels": readiness.get("observed_channels", []),
                "unobserved_channels_imputed": readiness.get("unobserved_channels_imputed", []),
            },
        )
        return df_clean, meta


_DEFAULT_LIVE_BUILDER: Optional[LiveSequenceBuilder] = None


def get_live_sequence_builder(
    registry: Optional[LiveRollingBufferRegistry] = None,
    max_staleness_hours: float = 3.0,
) -> LiveSequenceBuilder:
    """Returns a module-level singleton LiveSequenceBuilder instance."""
    global _DEFAULT_LIVE_BUILDER
    if _DEFAULT_LIVE_BUILDER is None or registry is not None:
        _DEFAULT_LIVE_BUILDER = LiveSequenceBuilder(registry=registry, max_staleness_hours=max_staleness_hours)
    return _DEFAULT_LIVE_BUILDER

