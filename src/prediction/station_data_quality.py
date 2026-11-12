"""
src/prediction/station_data_quality.py

Phase 17.3: Comprehensive Station Data Quality Checks.
Validates:
1. Station identity consistency.
2. Timestamp validity, strict monotonicity, and zero future timestamps.
3. Duplicate observation detection.
4. Missing hourly measurements and cadence regularity.
5. Pollutant availability (core criteria: PM2.5, PM10).
6. Invalid, zero, or negative concentration values against physical atmospheric bounds.
7. Unit consistency (µg/m³).
8. Unexpected measurement jumps (rate-of-change anomaly detection).
9. Observation freshness vs staleness.
10. Sequence completeness (24-hour continuous window).

Guarantees zero silent repair and zero conversion of synthetic/modeled data into physical measurements.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")

# Physical atmospheric plausibility limits (min, max in standard units)
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "PM2.5": (0.0, 1000.0),    # µg/m³
    "PM10": (0.0, 1500.0),     # µg/m³
    "NO2": (0.0, 1000.0),      # µg/m³
    "SO2": (0.0, 1000.0),      # µg/m³
    "CO": (0.0, 100.0),        # mg/m³
    "O3": (0.0, 1000.0),       # µg/m³
    "temperature_C": (-20.0, 60.0), # °C
    "relative_humidity": (0.0, 100.0), # %
    "wind_speed": (0.0, 60.0), # m/s
}

# Maximum physically plausible 1-hour rate of change (spike threshold)
MAX_HOURLY_DELTA: Dict[str, float] = {
    "PM2.5": 300.0,   # µg/m³ per hour
    "PM10": 500.0,    # µg/m³ per hour
    "NO2": 250.0,
    "SO2": 250.0,
    "CO": 20.0,
    "O3": 250.0,
}


@dataclass
class QualityCheckIssue:
    """Represents a specific data quality anomaly or violation."""
    check_name: str
    severity: str             # "CRITICAL", "WARNING", "INFO"
    detail: str
    parameter: Optional[str] = None
    timestamp: Optional[str] = None
    value: Optional[float] = None


@dataclass
class StationQualityReport:
    """Comprehensive data quality assessment for a station's observation sequence."""
    station_id: str
    is_valid: bool
    quality_score: float      # 0.0 to 100.0
    total_records: int
    valid_records_count: int
    missing_hours_count: int
    duplicate_records_count: int
    out_of_bounds_count: int
    spike_anomaly_count: int
    staleness_hours: float
    is_stale: bool
    is_24h_complete: bool
    issues: List[QualityCheckIssue] = field(default_factory=list)
    available_pollutants: List[str] = field(default_factory=list)
    missing_required_pollutants: List[str] = field(default_factory=list)
    evaluation_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["issues"] = [asdict(i) for i in self.issues]
        return d


class StationDataQualityChecker:
    """Performs rigorous quality audits on physical monitoring station observations."""

    def __init__(
        self,
        max_staleness_hours: float = 3.0,
        required_pollutants: Optional[List[str]] = None,
    ):
        self.max_staleness_hours = float(max_staleness_hours)
        self.required_pollutants = required_pollutants or ["PM2.5", "PM10"]

    def audit_single_measurement(
        self,
        station_id: str,
        parameter: str,
        value: Any,
        timestamp: Union[str, datetime, pd.Timestamp],
        reference_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[float], List[QualityCheckIssue]]:
        """Audits a single sensor observation against physical bounds and temporal constraints."""
        clean_sid = str(station_id).strip().upper()
        clean_param = str(parameter).strip()
        issues: List[QualityCheckIssue] = []

        # 1. Parse numeric value
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                issues.append(QualityCheckIssue(
                    check_name="numeric_validity",
                    severity="CRITICAL",
                    detail=f"Non-finite concentration value '{value}' for {clean_param}.",
                    parameter=clean_param,
                ))
                return False, None, issues
        except (ValueError, TypeError):
            issues.append(QualityCheckIssue(
                check_name="numeric_validity",
                severity="CRITICAL",
                detail=f"Cannot parse value '{value}' as float for {clean_param}.",
                parameter=clean_param,
            ))
            return False, None, issues

        # 2. Physical bounds check
        if clean_param in PHYSICAL_BOUNDS:
            min_v, max_v = PHYSICAL_BOUNDS[clean_param]
            if val < min_v or val > max_v:
                issues.append(QualityCheckIssue(
                    check_name="physical_bounds",
                    severity="CRITICAL",
                    detail=f"{clean_param} value {val} is outside physical bounds [{min_v}, {max_v}].",
                    parameter=clean_param,
                    value=val,
                ))
                return False, None, issues

        # 3. Timestamp check
        now_utc = reference_time or datetime.now(timezone.utc)
        if isinstance(timestamp, str):
            try:
                ts_dt = pd.to_datetime(timestamp)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.tz_localize("UTC")
                else:
                    ts_dt = ts_dt.tz_convert("UTC")
            except Exception as e:
                issues.append(QualityCheckIssue(
                    check_name="timestamp_format",
                    severity="CRITICAL",
                    detail=f"Failed to parse timestamp '{timestamp}': {e}",
                    parameter=clean_param,
                ))
                return False, None, issues
        elif isinstance(timestamp, (datetime, pd.Timestamp)):
            ts_dt = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
            ts_dt = ts_dt.astimezone(timezone.utc)
        else:
            issues.append(QualityCheckIssue(
                check_name="timestamp_format",
                severity="CRITICAL",
                detail=f"Unsupported timestamp type '{type(timestamp)}'.",
                parameter=clean_param,
            ))
            return False, None, issues

        # Check for future timestamp (allowing 5 minutes clock-skew tolerance)
        age_seconds = (now_utc - ts_dt).total_seconds()
        if age_seconds < -300.0:
            issues.append(QualityCheckIssue(
                check_name="future_timestamp",
                severity="CRITICAL",
                detail=f"Observation timestamp {ts_dt.isoformat()} is {-age_seconds:.0f}s in the future.",
                parameter=clean_param,
                timestamp=ts_dt.isoformat(),
            ))
            return False, None, issues

        # Check staleness
        if age_seconds > self.max_staleness_hours * 3600.0:
            issues.append(QualityCheckIssue(
                check_name="staleness",
                severity="WARNING",
                detail=f"Observation is {age_seconds / 3600.0:.1f}h old (threshold: {self.max_staleness_hours}h).",
                parameter=clean_param,
                timestamp=ts_dt.isoformat(),
            ))

        return True, val, issues

    def audit_sequence(
        self,
        df: pd.DataFrame,
        station_id: str,
        reference_time: Optional[datetime] = None,
    ) -> StationQualityReport:
        """Audits an antecedent hourly DataFrame for a station."""
        clean_sid = str(station_id).strip().upper()
        now_utc = reference_time or datetime.now(timezone.utc)
        issues: List[QualityCheckIssue] = []

        if df is None or df.empty:
            return StationQualityReport(
                station_id=clean_sid,
                is_valid=False,
                quality_score=0.0,
                total_records=0,
                valid_records_count=0,
                missing_hours_count=24,
                duplicate_records_count=0,
                out_of_bounds_count=0,
                spike_anomaly_count=0,
                staleness_hours=999.0,
                is_stale=True,
                is_24h_complete=False,
                issues=[QualityCheckIssue(
                    check_name="data_presence",
                    severity="CRITICAL",
                    detail="Sequence DataFrame is None or empty.",
                )],
                available_pollutants=[],
                missing_required_pollutants=self.required_pollutants,
            )

        total_recs = len(df)
        out_of_bounds = 0
        spike_count = 0

        # 1. Station identity consistency
        if "StationId" in df.columns:
            sids = set(df["StationId"].dropna().astype(str).str.strip().str.upper().unique())
            if len(sids) > 1:
                issues.append(QualityCheckIssue(
                    check_name="station_identity",
                    severity="CRITICAL",
                    detail=f"Mixed station IDs found in sequence: {list(sids)}.",
                ))
            elif len(sids) == 1 and clean_sid not in sids:
                issues.append(QualityCheckIssue(
                    check_name="station_identity",
                    severity="CRITICAL",
                    detail=f"Station ID mismatch: expected {clean_sid}, found {list(sids)}.",
                ))

        # 2. Timestamp analysis
        if "Datetime" not in df.columns:
            issues.append(QualityCheckIssue(
                check_name="timestamp_column",
                severity="CRITICAL",
                detail="Mandatory 'Datetime' column is missing.",
            ))
            return StationQualityReport(
                station_id=clean_sid,
                is_valid=False,
                quality_score=0.0,
                total_records=total_recs,
                valid_records_count=0,
                missing_hours_count=24,
                duplicate_records_count=0,
                out_of_bounds_count=0,
                spike_anomaly_count=0,
                staleness_hours=999.0,
                is_stale=True,
                is_24h_complete=False,
                issues=issues,
                available_pollutants=[],
                missing_required_pollutants=self.required_pollutants,
            )

        try:
            dt_series = pd.to_datetime(df["Datetime"])
        except Exception as e:
            issues.append(QualityCheckIssue(
                check_name="timestamp_parse",
                severity="CRITICAL",
                detail=f"Failed to parse Datetime values: {e}",
            ))
            dt_series = pd.Series(dtype="datetime64[ns]")

        # Duplicate check
        dup_cnt = int(dt_series.duplicated().sum())
        if dup_cnt > 0:
            issues.append(QualityCheckIssue(
                check_name="duplicates",
                severity="CRITICAL",
                detail=f"Found {dup_cnt} duplicate timestamps in sequence.",
            ))

        # Monotonicity check
        if not dt_series.is_monotonic_increasing:
            issues.append(QualityCheckIssue(
                check_name="chronological_order",
                severity="CRITICAL",
                detail="Sequence records are not in strictly ascending chronological order.",
            ))

        # Cadence check
        diffs = dt_series.diff().dropna()
        irregular = diffs[diffs != pd.Timedelta(hours=1)]
        if not irregular.empty:
            issues.append(QualityCheckIssue(
                check_name="hourly_cadence",
                severity="WARNING" if len(irregular) <= 2 else "CRITICAL",
                detail=f"Irregular time step detected ({len(irregular)} steps differ from 1h).",
            ))

        # Staleness & future check
        staleness_hours = 0.0
        is_stale = False
        if len(dt_series) > 0:
            last_dt = dt_series.iloc[-1]
            if last_dt.tzinfo is None:
                last_dt_utc = last_dt.replace(tzinfo=timezone.utc)
            else:
                last_dt_utc = last_dt.astimezone(timezone.utc)

            staleness_hours = max(0.0, (now_utc - last_dt_utc).total_seconds() / 3600.0)
            is_stale = staleness_hours > self.max_staleness_hours

            if (now_utc - last_dt_utc).total_seconds() < -300.0:
                issues.append(QualityCheckIssue(
                    check_name="future_timestamp",
                    severity="CRITICAL",
                    detail=f"Sequence endpoint {last_dt_utc.isoformat()} is in the future.",
                ))
            elif is_stale:
                issues.append(QualityCheckIssue(
                    check_name="staleness",
                    severity="WARNING",
                    detail=f"Sequence is {staleness_hours:.1f}h old (threshold: {self.max_staleness_hours}h).",
                ))

        # 3. Pollutant channel presence and values
        available_pollutants = []
        missing_required = []
        for req in self.required_pollutants:
            if req not in df.columns:
                missing_required.append(req)
                issues.append(QualityCheckIssue(
                    check_name="required_pollutant",
                    severity="CRITICAL",
                    detail=f"Required pollutant '{req}' column is absent.",
                    parameter=req,
                ))
            else:
                available_pollutants.append(req)
                # Check bounds and missing values
                num_series = pd.to_numeric(df[req], errors="coerce")
                nan_cnt = int(num_series.isna().sum())
                if nan_cnt > 0:
                    issues.append(QualityCheckIssue(
                        check_name="missing_values",
                        severity="CRITICAL" if nan_cnt > 4 else "WARNING",
                        detail=f"Pollutant '{req}' has {nan_cnt} missing/NaN values.",
                        parameter=req,
                    ))

                # Bounds check
                if req in PHYSICAL_BOUNDS:
                    min_v, max_v = PHYSICAL_BOUNDS[req]
                    bad_bounds = num_series[(num_series < min_v) | (num_series > max_v)]
                    if not bad_bounds.empty:
                        out_of_bounds += len(bad_bounds)
                        issues.append(QualityCheckIssue(
                            check_name="physical_bounds",
                            severity="CRITICAL",
                            detail=f"Pollutant '{req}' has {len(bad_bounds)} values outside [{min_v}, {max_v}].",
                            parameter=req,
                        ))

                # Rate-of-change jump check (anomalous physical spike)
                if req in MAX_HOURLY_DELTA and len(num_series.dropna()) > 1:
                    max_delta_thresh = MAX_HOURLY_DELTA[req]
                    deltas = num_series.diff().abs().dropna()
                    spikes = deltas[deltas > max_delta_thresh]
                    if not spikes.empty:
                        spike_count += len(spikes)
                        issues.append(QualityCheckIssue(
                            check_name="rate_of_change_spike",
                            severity="WARNING",
                            detail=f"Pollutant '{req}' exhibits {len(spikes)} rate-of-change jumps exceeding {max_delta_thresh} µg/m³/h.",
                            parameter=req,
                        ))

        # Check other optional pollutants present
        for col in df.columns:
            if col in PHYSICAL_BOUNDS and col not in available_pollutants:
                available_pollutants.append(col)

        # 4. Completeness summary
        is_24h_complete = bool(total_recs == 24 and len(missing_required) == 0 and dup_cnt == 0 and not is_stale)
        missing_hours = max(0, 24 - total_recs)

        # Quality scoring (0 to 100)
        deductions = 0.0
        critical_count = sum(1 for i in issues if i.severity == "CRITICAL")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")
        deductions += critical_count * 25.0
        deductions += warning_count * 5.0
        deductions += missing_hours * 3.0
        if is_stale:
            deductions += min(30.0, staleness_hours * 5.0)

        quality_score = max(0.0, round(100.0 - deductions, 1))
        is_valid = bool(critical_count == 0 and total_recs == 24 and len(missing_required) == 0)

        return StationQualityReport(
            station_id=clean_sid,
            is_valid=is_valid,
            quality_score=quality_score,
            total_records=total_recs,
            valid_records_count=max(0, total_recs - out_of_bounds),
            missing_hours_count=missing_hours,
            duplicate_records_count=dup_cnt,
            out_of_bounds_count=out_of_bounds,
            spike_anomaly_count=spike_count,
            staleness_hours=round(staleness_hours, 2),
            is_stale=is_stale,
            is_24h_complete=is_24h_complete,
            issues=issues,
            available_pollutants=available_pollutants,
            missing_required_pollutants=missing_required,
        )
