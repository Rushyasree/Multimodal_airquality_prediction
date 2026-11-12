"""
src/prediction/sequence_validator.py

Enforces strict sequence integrity, temporal alignment, and provenance tracking for air quality forecasting.
Provides:
- validate_forecast_sequence(): Backward-compatible validation for general & historical demo forecasting.
- validate_live_forecast_sequence(): Strict multi-gate validation specifically for authentic live measurements.

Exposes independent validation gates:
- sequence_structural_validity
- station_identity_validity
- source_authenticity
- measurement_completeness
- freshness_validity
- scaler_compatibility
- live_forecast_eligibility
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

from src.prediction.observation_provenance import (
    ObservationProvenanceCategory,
    PROVENANCE_BADGE_MAP,
)

IST = timezone(timedelta(hours=5, minutes=30))
CORE_REQUIRED_POLLUTANTS: List[str] = ["PM2.5", "PM10"]

ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT_DIR / "data/processed/model_sequences/preprocessing_metadata.json"


@dataclass
class SequenceValidationResult:
    """General sequence validation result (Phase 17.1 compatible)."""
    valid: bool
    provenance: str
    station_match: bool
    start_time: str
    end_time: str
    target_time: str
    age_hours: Optional[float]
    gaps: List[str]
    source_type: str  # LIVE_MEASURED, HISTORICAL_DEMO, REGIONAL_PROXY, MODELED_REANALYSIS, SYNTHETIC_FALLBACK, INVALID
    rejection_reason: Optional[str] = None
    feature_count: int = 0
    scaler_compatible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveSequenceValidationResult:
    """Strict multi-gate live sequence validation result (Phase 17.2).

    Exposes separate boolean flags for each scientific gate without collapsing into one.
    """
    live_forecast_eligible: bool
    sequence_structural_validity: bool
    station_identity_validity: bool
    source_authenticity: bool
    measurement_completeness: bool
    freshness_validity: bool
    scaler_compatibility: bool
    provenance_category: str
    status_badge: str
    station_id: str
    forecast_issue_time: str
    target_time: str
    start_time: str
    end_time: str
    age_hours: Optional[float]
    measured_hours_count: int
    completeness_pct: float
    rejection_reasons: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    feature_count: int = 0

    @property
    def rejection_reason(self) -> Optional[str]:
        if self.rejection_reasons:
            return "; ".join(self.rejection_reasons)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rejection_reason"] = self.rejection_reason
        return d


def load_model_feature_metadata() -> Dict[str, Any]:
    """Loads preprocessing metadata defining feature names and scalers."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    from src.prediction.lstm_forecaster import FALLBACK_PREPROCESSING_METADATA
    return FALLBACK_PREPROCESSING_METADATA


def validate_live_forecast_sequence(
    df: pd.DataFrame,
    station_id: str,
    max_age_hours: float = 3.0,
    reference_system_time: Optional[datetime] = None,
    source_provenance_category: Optional[Union[str, ObservationProvenanceCategory]] = None,
    require_exact_station: bool = True,
    reference_time: Optional[datetime] = None,
    **kwargs,
) -> LiveSequenceValidationResult:
    """Performs strict, independent multi-gate validation on candidate live forecasting sequences.

    Does NOT collapse checks into a single boolean; explicitly reports individual gate status.
    """
    if reference_system_time is None and reference_time is not None:
        reference_system_time = reference_time
    if "max_staleness_hours" in kwargs:
        max_age_hours = float(kwargs["max_staleness_hours"])

    clean_sid = str(station_id).strip().upper()
    rejection_reasons: List[str] = []
    gaps: List[str] = []

    structural_valid = True
    station_valid = True
    authenticity_valid = True
    completeness_valid = True
    freshness_valid = True
    scaler_valid = False

    start_time_str = ""
    end_time_str = ""
    target_time_str = ""
    age_hours = None
    measured_hours = 0
    completeness_pct = 0.0

    # Gate 1: Sequence Structural Validity
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        structural_valid = False
        rejection_reasons.append("DataFrame is empty or None.")
        return LiveSequenceValidationResult(
            live_forecast_eligible=False,
            sequence_structural_validity=False,
            station_identity_validity=False,
            source_authenticity=False,
            measurement_completeness=False,
            freshness_validity=False,
            scaler_compatibility=False,
            provenance_category=ObservationProvenanceCategory.INVALID.value,
            status_badge=PROVENANCE_BADGE_MAP[ObservationProvenanceCategory.INVALID],
            station_id=clean_sid,
            forecast_issue_time="",
            target_time="",
            start_time="",
            end_time="",
            age_hours=None,
            measured_hours_count=0,
            completeness_pct=0.0,
            rejection_reasons=rejection_reasons,
        )

    if len(df) != 24:
        structural_valid = False
        rejection_reasons.append(f"Window size mismatch: expected exactly 24 hourly rows, found {len(df)}.")

    if "Datetime" not in df.columns:
        structural_valid = False
        rejection_reasons.append("Mandatory 'Datetime' column is absent.")
        dt_series = pd.Series(dtype="datetime64[ns]")
    else:
        try:
            dt_series = pd.to_datetime(df["Datetime"])
            # Duplicate check
            dup_count = int(dt_series.duplicated().sum())
            if dup_count > 0:
                structural_valid = False
                rejection_reasons.append(f"Found {dup_count} duplicate timestamp(s) in sequence.")
            # Chronological monotonicity check
            if not dt_series.is_monotonic_increasing:
                structural_valid = False
                rejection_reasons.append("Timestamps are not in strictly ascending chronological order.")
            # Cadence check
            diffs = dt_series.diff().dropna()
            expected_delta = pd.Timedelta(hours=1)
            irregular = diffs[diffs != expected_delta]
            if not irregular.empty:
                structural_valid = False
                gap_msgs = [f"Step at index {idx}: delta={delta} (expected 1h)" for idx, delta in irregular.items()]
                gaps.extend(gap_msgs)
                rejection_reasons.append(f"Discontinuous hourly interval detected ({len(irregular)} irregular deltas).")

            if len(dt_series) > 0:
                start_time_str = dt_series.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
                end_time_str = dt_series.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
                target_dt = dt_series.iloc[-1] + expected_delta
                target_time_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            structural_valid = False
            rejection_reasons.append(f"Failed to parse Datetime column: {exc}")
            dt_series = pd.Series(dtype="datetime64[ns]")

    # Gate 2: Station Identity Validity
    if "StationId" in df.columns:
        seq_sids = set(df["StationId"].dropna().astype(str).str.strip().str.upper().unique())
        if len(seq_sids) > 1:
            station_valid = False
            rejection_reasons.append(f"Station mixing detected: sequence contains multiple StationIds {list(seq_sids)}.")
        elif len(seq_sids) == 1 and clean_sid not in seq_sids:
            station_valid = False
            rejection_reasons.append(f"Station identity mismatch: expected {clean_sid}, found {list(seq_sids)}.")
    else:
        station_valid = False
        rejection_reasons.append("Mandatory 'StationId' column is absent.")

    # Gate 3: Source Authenticity Check
    prov_cat = str(source_provenance_category or df.attrs.get("provenance_category") or "").upper()
    if prov_cat:
        if prov_cat not in [ObservationProvenanceCategory.LIVE_MEASURED.value, "LIVE_MEASURED"]:
            authenticity_valid = False
            rejection_reasons.append(
                f"Source authenticity rejected for live forecasting: sequence is marked '{prov_cat}', "
                "not authentic physical monitor inlet measurements."
            )
    else:
        # If not explicitly declared, check if caller provided physical measurements
        # Do not automatically assume live from timestamps
        if df.attrs.get("is_strictly_measured") is False:
            authenticity_valid = False
            rejection_reasons.append("Source authenticity not verified: sequence contains unmeasured or proxy rows.")

    # Gate 4: Measurement Completeness Check
    for req_col in CORE_REQUIRED_POLLUTANTS:
        if req_col not in df.columns:
            completeness_valid = False
            rejection_reasons.append(f"Required pollutant channel '{req_col}' is missing from sequence.")
        else:
            vals = pd.to_numeric(df[req_col], errors="coerce")
            nan_count = int(vals.isna().sum())
            if nan_count > 0:
                completeness_valid = False
                rejection_reasons.append(f"Channel '{req_col}' has {nan_count} missing hours (zero missing permitted for LIVE_MEASURED).")

    if completeness_valid and len(df) == 24:
        measured_hours = 24
        completeness_pct = 100.0
    else:
        pm25_valid = pd.to_numeric(df.get("PM2.5", pd.Series([])), errors="coerce").notna()
        pm10_valid = pd.to_numeric(df.get("PM10", pd.Series([])), errors="coerce").notna()
        valid_both = (pm25_valid & pm10_valid).sum()
        measured_hours = int(valid_both)
        completeness_pct = round((measured_hours / 24.0) * 100.0, 1)

    # Gate 5: Freshness Validity Check
    if len(dt_series) > 0:
        has_tz = dt_series.dt.tz is not None
        if reference_system_time is None:
            ref_time = datetime.now(timezone.utc) if has_tz else datetime.now(IST).replace(tzinfo=None)
        else:
            ref_time = reference_system_time

        seq_end_dt = dt_series.iloc[-1]
        if has_tz:
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=timezone.utc)
            age_hours = (ref_time - seq_end_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        else:
            if ref_time.tzinfo is not None:
                ref_utc_naive = ref_time.astimezone(timezone.utc).replace(tzinfo=None)
                ref_ist_naive = ref_time.astimezone(IST).replace(tzinfo=None)
                age_utc = (ref_utc_naive - seq_end_dt).total_seconds() / 3600.0
                age_ist = (ref_ist_naive - seq_end_dt).total_seconds() / 3600.0
                age_hours = age_utc if abs(age_utc) < abs(age_ist) else age_ist
            else:
                age_hours = (ref_time - seq_end_dt).total_seconds() / 3600.0

        if age_hours < -0.05:
            freshness_valid = False
            rejection_reasons.append(f"Future timestamp detected: sequence end {end_time_str} is in the future relative to issue time.")
        elif age_hours > max_age_hours:
            freshness_valid = False
            rejection_reasons.append(f"Stale sequence: age is {age_hours:.1f}h (exceeds live threshold of {max_age_hours:.1f}h).")

    # Gate 6: Scaler & Model Schema Compatibility
    meta = load_model_feature_metadata()
    f_names = meta.get("feature_names", [])
    s_means = meta.get("scaler_means", {})
    s_stds = meta.get("scaler_stds", {})
    scaler_valid = bool(len(f_names) == 27 and len(s_means) == 27 and len(s_stds) == 27)
    if not scaler_valid:
        rejection_reasons.append("Model preprocessing scaler metadata is incompatible or missing 27 channels.")

    # Gate 7: Overall Live Forecast Eligibility
    live_eligible = bool(
        structural_valid
        and station_valid
        and authenticity_valid
        and completeness_valid
        and freshness_valid
        and scaler_valid
    )

    if live_eligible:
        resolved_category = ObservationProvenanceCategory.LIVE_MEASURED.value
        status_badge = "LIVE FORECAST — VALIDATED INPUT"
    elif not station_valid:
        resolved_category = ObservationProvenanceCategory.REGIONAL_PROXY.value
        status_badge = "REGIONAL MODEL ESTIMATE"
    elif not completeness_valid or measured_hours < 24:
        resolved_category = ObservationProvenanceCategory.LIVE_PARTIAL.value
        status_badge = "LIVE INPUT UNAVAILABLE"
    elif not freshness_valid:
        resolved_category = ObservationProvenanceCategory.HISTORICAL_DEMO.value
        status_badge = "HISTORICAL DEMO FORECAST"
    else:
        resolved_category = ObservationProvenanceCategory.INVALID.value
        status_badge = "DATA UNAVAILABLE"

    return LiveSequenceValidationResult(
        live_forecast_eligible=live_eligible,
        sequence_structural_validity=structural_valid,
        station_identity_validity=station_valid,
        source_authenticity=authenticity_valid,
        measurement_completeness=completeness_valid,
        freshness_validity=freshness_valid,
        scaler_compatibility=scaler_valid,
        provenance_category=resolved_category,
        status_badge=status_badge,
        station_id=clean_sid,
        forecast_issue_time=end_time_str,
        target_time=target_time_str,
        start_time=start_time_str,
        end_time=end_time_str,
        age_hours=round(age_hours, 2) if age_hours is not None else None,
        measured_hours_count=measured_hours,
        completeness_pct=completeness_pct,
        rejection_reasons=rejection_reasons,
        gaps=gaps,
        feature_count=len(df.columns),
    )


def validate_forecast_sequence(
    df: pd.DataFrame,
    station_id: str,
    max_age_hours: Optional[float] = None,
    require_exact_station: bool = True,
    forecast_issue_time: Optional[datetime] = None,
) -> SequenceValidationResult:
    """Validates an antecedent 24-hour sequence for general model inference (Phase 17.1 API).

    Retained for complete backward compatibility across all existing test suites.
    """
    clean_sid = str(station_id).strip().upper()
    gaps: List[str] = []

    # 1. Structure and row count check
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return SequenceValidationResult(
            valid=False,
            provenance="Empty or Invalid DataFrame",
            station_match=False,
            start_time="",
            end_time="",
            target_time="",
            age_hours=None,
            gaps=["DataFrame is empty or None"],
            source_type="INVALID",
            rejection_reason="No sequence records supplied.",
        )

    if len(df) != 24:
        return SequenceValidationResult(
            valid=False,
            provenance="Non-24-hour Window",
            station_match=False,
            start_time=str(df.iloc[0].get("Datetime", "")) if len(df) > 0 else "",
            end_time=str(df.iloc[-1].get("Datetime", "")) if len(df) > 0 else "",
            target_time="",
            age_hours=None,
            gaps=[f"Expected exactly 24 hourly rows, found {len(df)}."],
            source_type="INVALID",
            rejection_reason=f"Sequence length mismatch: expected exactly 24 rows, got {len(df)}.",
        )

    # 2. Timestamp analysis
    if "Datetime" not in df.columns:
        return SequenceValidationResult(
            valid=False,
            provenance="Missing Timestamp Column",
            station_match=False,
            start_time="",
            end_time="",
            target_time="",
            age_hours=None,
            gaps=["Missing 'Datetime' column in sequence dataframe."],
            source_type="INVALID",
            rejection_reason="Mandatory 'Datetime' column missing.",
        )

    try:
        dt_series = pd.to_datetime(df["Datetime"])
    except Exception as exc:
        return SequenceValidationResult(
            valid=False,
            provenance="Unparseable Datetime",
            station_match=False,
            start_time="",
            end_time="",
            target_time="",
            age_hours=None,
            gaps=[f"Datetime parse error: {exc}"],
            source_type="INVALID",
            rejection_reason=f"Failed to parse Datetime column: {exc}",
        )

    # 3. Duplicate timestamps check
    dup_count = int(dt_series.duplicated().sum())
    if dup_count > 0:
        dup_timestamps = dt_series[dt_series.duplicated()].tolist()
        return SequenceValidationResult(
            valid=False,
            provenance="Duplicate Timestamps",
            station_match=False,
            start_time=str(dt_series.iloc[0]),
            end_time=str(dt_series.iloc[-1]),
            target_time="",
            age_hours=None,
            gaps=[f"Found {dup_count} duplicate timestamp(s): {dup_timestamps}"],
            source_type="INVALID",
            rejection_reason=f"Duplicate timestamps detected in antecedent sequence ({dup_count} duplicates).",
        )

    # 4. Strictly chronological order check
    if not dt_series.is_monotonic_increasing:
        return SequenceValidationResult(
            valid=False,
            provenance="Non-Chronological Sequence",
            station_match=False,
            start_time=str(dt_series.iloc[0]),
            end_time=str(dt_series.iloc[-1]),
            target_time="",
            age_hours=None,
            gaps=["Timestamps are not strictly in ascending chronological order."],
            source_type="INVALID",
            rejection_reason="Antecedent records are not sorted in ascending chronological order.",
        )

    # 5. Exactly 1-hour intervals check
    diffs = dt_series.diff().dropna()
    expected_delta = pd.Timedelta(hours=1)
    irregular_diffs = diffs[diffs != expected_delta]
    if not irregular_diffs.empty:
        gap_details = [
            f"Step at index {idx}: delta = {delta} (expected 1h)"
            for idx, delta in irregular_diffs.items()
        ]
        return SequenceValidationResult(
            valid=False,
            provenance="Irregular Hourly Frequency / Missing Hours",
            station_match=False,
            start_time=str(dt_series.iloc[0]),
            end_time=str(dt_series.iloc[-1]),
            target_time="",
            age_hours=None,
            gaps=gap_details,
            source_type="INVALID",
            rejection_reason=f"Discontinuous hourly interval detected ({len(irregular_diffs)} gaps).",
        )

    start_time_str = dt_series.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = dt_series.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    target_time_dt = dt_series.iloc[-1] + expected_delta
    target_time_str = target_time_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 6. Station identity verification
    station_match = True
    if "StationId" in df.columns:
        seq_sids = set(df["StationId"].dropna().astype(str).str.strip().str.upper().unique())
        if len(seq_sids) > 1:
            return SequenceValidationResult(
                valid=False,
                provenance="Mixed Stations in Sequence",
                station_match=False,
                start_time=start_time_str,
                end_time=end_time_str,
                target_time=target_time_str,
                age_hours=None,
                gaps=[f"Multiple StationIds detected in single 24h window: {seq_sids}"],
                source_type="INVALID",
                rejection_reason="Sequence mixes records from multiple monitoring stations.",
            )
        if clean_sid not in seq_sids:
            station_match = False
            if require_exact_station:
                return SequenceValidationResult(
                    valid=False,
                    provenance=f"Station Mismatch (Expected {clean_sid}, found {list(seq_sids)})",
                    station_match=False,
                    start_time=start_time_str,
                    end_time=end_time_str,
                    target_time=target_time_str,
                    age_hours=None,
                    gaps=[f"Expected station {clean_sid}, but sequence contains {list(seq_sids)}"],
                    source_type="REGIONAL_PROXY",
                    rejection_reason=f"Station identity mismatch: expected {clean_sid}, got {list(seq_sids)}.",
                )

    # 7. Model feature schema check
    meta = load_model_feature_metadata()
    feature_names = meta.get("feature_names", [])
    scaler_means = meta.get("scaler_means", {})
    scaler_stds = meta.get("scaler_stds", {})

    for req_col in ["PM2.5", "PM10"]:
        if req_col not in df.columns:
            return SequenceValidationResult(
                valid=False,
                provenance="Missing Criteria Pollutants",
                station_match=station_match,
                start_time=start_time_str,
                end_time=end_time_str,
                target_time=target_time_str,
                age_hours=None,
                gaps=[f"Required pollutant column '{req_col}' is absent."],
                source_type="INVALID",
                rejection_reason=f"Required pollutant column '{req_col}' is missing.",
            )
        vals = pd.to_numeric(df[req_col], errors="coerce")
        if vals.isna().sum() > 6:
            return SequenceValidationResult(
                valid=False,
                provenance="Excessive Missing Values",
                station_match=station_match,
                start_time=start_time_str,
                end_time=end_time_str,
                target_time=target_time_str,
                age_hours=None,
                gaps=[f"Excessive null values in '{req_col}' ({vals.isna().sum()} / 24 nulls)."],
                source_type="INVALID",
                rejection_reason=f"Column '{req_col}' has {vals.isna().sum()} missing values (limit: 6).",
            )

    scaler_compat = bool(len(feature_names) == 27 and len(scaler_means) == 27 and len(scaler_stds) == 27)

    # 8. Freshness relative to forecast issue time
    has_tz = dt_series.dt.tz is not None
    if forecast_issue_time is None:
        if has_tz:
            ref_time = datetime.now(timezone.utc)
        else:
            ref_time = datetime.now(IST).replace(tzinfo=None)
    else:
        ref_time = forecast_issue_time

    seq_end_dt = dt_series.iloc[-1]
    if has_tz:
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
        age_hours = (ref_time - seq_end_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
    else:
        if ref_time.tzinfo is not None:
            ref_utc_naive = ref_time.astimezone(timezone.utc).replace(tzinfo=None)
            ref_ist_naive = ref_time.astimezone(IST).replace(tzinfo=None)
            age_utc = (ref_utc_naive - seq_end_dt).total_seconds() / 3600.0
            age_ist = (ref_ist_naive - seq_end_dt).total_seconds() / 3600.0
            age_hours = age_utc if abs(age_utc) < abs(age_ist) else age_ist
        else:
            age_hours = (ref_time - seq_end_dt).total_seconds() / 3600.0

    # Determine classification and live validity
    if max_age_hours is not None:
        if age_hours < -0.05:
            return SequenceValidationResult(
                valid=False,
                provenance="Future Timestamps Detected",
                station_match=station_match,
                start_time=start_time_str,
                end_time=end_time_str,
                target_time=target_time_str,
                age_hours=round(age_hours, 2),
                gaps=[f"Sequence end {end_time_str} is in the future relative to issue time."],
                source_type="INVALID",
                rejection_reason=f"Future timestamp detected in sequence: {end_time_str} is ahead of reference issue time.",
                feature_count=len(df.columns),
                scaler_compatible=scaler_compat,
            )
        if age_hours > max_age_hours:
            return SequenceValidationResult(
                valid=False,
                provenance="Stale Sequence for Live Inference",
                station_match=station_match,
                start_time=start_time_str,
                end_time=end_time_str,
                target_time=target_time_str,
                age_hours=round(age_hours, 2),
                gaps=[f"Sequence age {age_hours:.1f}h exceeds live threshold {max_age_hours:.1f}h"],
                source_type="HISTORICAL_DEMO",
                rejection_reason=f"Sequence is {age_hours:.1f} hours old; exceeds live threshold of {max_age_hours:.1f} hours.",
                feature_count=len(df.columns),
                scaler_compatible=scaler_compat,
            )

    # Valid sequence classification
    if max_age_hours is not None and age_hours <= max_age_hours:
        source_type = "LIVE_MEASURED" if station_match else "REGIONAL_PROXY"
        provenance = (
            f"Live Validated 24h Telemetry (Station {clean_sid}, {start_time_str} to {end_time_str})"
            if station_match
            else f"Live Regional Proxy Telemetry ({start_time_str} to {end_time_str})"
        )
    else:
        source_type = "HISTORICAL_DEMO" if station_match else "REGIONAL_PROXY"
        provenance = (
            f"Curated Historical Benchmark Sequence (Station {clean_sid}, {start_time_str} to {end_time_str})"
            if station_match
            else f"Regional Proxy Historical Benchmark ({start_time_str} to {end_time_str})"
        )

    return SequenceValidationResult(
        valid=True,
        provenance=provenance,
        station_match=station_match,
        start_time=start_time_str,
        end_time=end_time_str,
        target_time=target_time_str,
        age_hours=round(age_hours, 2),
        gaps=[],
        source_type=source_type,
        rejection_reason=None,
        feature_count=len(df.columns),
        scaler_compatible=scaler_compat,
    )
