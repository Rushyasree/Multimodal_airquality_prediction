"""
src/prediction/observation_provenance.py

Explicit Data Provenance Schema for VayuDrishti Phase 17.2.
Provides immutable tracking of observation origin, measurement status,
upstream provider identity, timestamps, and sequence integrity.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union


class ObservationProvenanceCategory(str, Enum):
    """Mutually exclusive provenance classifications for observations and sequences."""
    LIVE_MEASURED = "LIVE_MEASURED"
    LIVE_PARTIAL = "LIVE_PARTIAL"
    HISTORICAL_DEMO = "HISTORICAL_DEMO"
    REGIONAL_PROXY = "REGIONAL_PROXY"
    MODELED_REANALYSIS = "MODELED_REANALYSIS"
    SYNTHETIC_FALLBACK = "SYNTHETIC_FALLBACK"
    INVALID = "INVALID"


# Dashboard display badge mapping
PROVENANCE_BADGE_MAP: Dict[ObservationProvenanceCategory, str] = {
    ObservationProvenanceCategory.LIVE_MEASURED: "LIVE MEASURED OBSERVATION",
    ObservationProvenanceCategory.LIVE_PARTIAL: "LIVE INPUT UNAVAILABLE",
    ObservationProvenanceCategory.HISTORICAL_DEMO: "HISTORICAL DEMO FORECAST",
    ObservationProvenanceCategory.REGIONAL_PROXY: "REGIONAL MODEL ESTIMATE",
    ObservationProvenanceCategory.MODELED_REANALYSIS: "REGIONAL MODEL ESTIMATE",
    ObservationProvenanceCategory.SYNTHETIC_FALLBACK: "SYNTHETIC DEMONSTRATION",
    ObservationProvenanceCategory.INVALID: "DATA UNAVAILABLE",
}


@dataclass
class ObservationRecord:
    """Represents a single verified pollutant observation with complete provenance."""
    station_id: str
    parameter: str                      # e.g., 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'
    value: float
    unit: str = "ug/m3"
    measurement_timestamp_utc: Union[datetime, str] = field(default_factory=lambda: datetime.now(timezone.utc))
    source_provider: str = "CPCB CAAQMS via OpenAQ v3"
    retrieval_timestamp_utc: Optional[Union[datetime, str]] = None
    source_observation_id: Optional[str] = None
    is_measured: bool = True
    is_interpolated: bool = False
    is_imputed: bool = False
    raw_value: Optional[float] = None
    sample_count: int = 1               # Number of sub-hourly samples aggregated into this observation
    provenance_category: ObservationProvenanceCategory = ObservationProvenanceCategory.LIVE_MEASURED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.retrieval_timestamp_utc is None:
            self.retrieval_timestamp_utc = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        meas_ts = self.measurement_timestamp_utc
        d["measurement_timestamp_utc"] = meas_ts.isoformat() if isinstance(meas_ts, datetime) else str(meas_ts)
        ret_ts = self.retrieval_timestamp_utc
        d["retrieval_timestamp_utc"] = ret_ts.isoformat() if isinstance(ret_ts, datetime) else str(ret_ts)
        return d


@dataclass
class SequenceProvenanceMeta:
    """Represents the provenance, completeness, and validity of an antecedent sequence."""
    station_id: str
    provenance_category: ObservationProvenanceCategory
    source_provider: str
    start_time: str
    end_time: str
    target_time: str
    total_hours_required: int = 24
    measured_hours_count: int = 0
    completeness_pct: float = 0.0
    is_strictly_measured: bool = False
    aggregation_method: Optional[str] = None
    retrieval_timestamp: str = ""
    status_badge: str = ""
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.status_badge:
            self.status_badge = PROVENANCE_BADGE_MAP.get(
                self.provenance_category, "DATA UNAVAILABLE"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance_category"] = self.provenance_category.value
        return d
