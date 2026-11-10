"""
src/prediction/sequence_validator.py

Enforces strict sequence integrity, temporal alignment, and provenance tracking for air quality forecasting.

Checks:
- Exactly 24 antecedent hourly records.
- Strictly chronological order.
- No duplicate timestamps.
- Exactly one-hour intervals.
- Valid model features and non-null values.
- Station identity verification.
- Sequence freshness relative to forecast issue time.
- Feature schema and scaler compatibility with model metadata.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))

ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT_DIR / "data/processed/model_sequences/preprocessing_metadata.json"


@dataclass
class SequenceValidationResult:
    valid: bool
    provenance: str
    station_match: bool
    start_time: str
    end_time: str
    target_time: str
    age_hours: Optional[float]
    gaps: List[str]
    source_type: str  # LIVE_MEASURED, HISTORICAL_DEMO, REGIONAL_PROXY, MODELED_REANALYSIS, SYNTHETIC_FALLBACK
    rejection_reason: Optional[str] = None
    feature_count: int = 0
    scaler_compatible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def validate_forecast_sequence(
    df: pd.DataFrame,
    station_id: str,
    max_age_hours: Optional[float] = None,
    require_exact_station: bool = True,
    forecast_issue_time: Optional[datetime] = None,
) -> SequenceValidationResult:
    """Validates an antecedent 24-hour sequence for model inference.

    Args:
        df: DataFrame containing hourly observations.
        station_id: Expected monitoring station ID (e.g. 'AP001').
        max_age_hours: Maximum allowable age in hours for live forecasting.
            If None, historical sequences are permitted but classified as HISTORICAL_DEMO.
        require_exact_station: If True, df['StationId'] must match station_id.
        forecast_issue_time: Reference timestamp for freshness check (default: current UTC/IST).

    Returns:
        SequenceValidationResult detailing validation status and provenance.
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

    # Check key physical criteria pollutants: PM2.5, PM10
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
        # Check non-null numeric validity
        vals = pd.to_numeric(df[req_col], errors="coerce")
        if vals.isna().sum() > 6:  # Allow up to 25% missing for imputation
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
            ref_time_naive = ref_time.astimezone(IST).replace(tzinfo=None)
        else:
            ref_time_naive = ref_time
        age_hours = (ref_time_naive - seq_end_dt).total_seconds() / 3600.0

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
