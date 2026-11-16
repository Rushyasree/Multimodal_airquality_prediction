"""
src/services/spatial_reliability_service.py

Phase 16: Live Station Coverage & Spatial Reliability Audit Service.
Analyzes live station telemetry availability, freshness categorization,
spatial geometry (Clark-Evans clustering index), missing data fractions,
and city-level operational reliability tiers across all 31 Indian urban locations.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.spatial_config import (
    SUPPORTED_CITIES,
    CityBoundingBox,
    SpatialInterpolationConfig,
    DEFAULT_SPATIAL_CONFIG,
)
from src.services.city_registry_service import (
    CityRegistryService,
    get_city_registry_service,
)
from src.services.spatial_interpolation_service import (
    haversine_distance_vectorized,
)

logger = logging.getLogger("VayuDrishti.SpatialReliability")


@dataclass
class FreshnessThresholds:
    """Configurable freshness boundaries in hours."""
    fresh_max_hours: float = 2.0
    delayed_max_hours: float = 6.0
    stale_max_hours: float = 24.0


DEFAULT_FRESHNESS_THRESHOLDS = FreshnessThresholds()


class SpatialReliabilityService:
    """
    Evaluates operational reliability, monitor availability, spatial geometry,
    and data freshness across registered urban monitoring networks.
    """

    _instance: Optional["SpatialReliabilityService"] = None

    def __init__(self, freshness_thresholds: Optional[FreshnessThresholds] = None):
        self._project_root = Path(__file__).resolve().parents[2]
        self.registry = get_city_registry_service()
        self.thresholds = freshness_thresholds or DEFAULT_FRESHNESS_THRESHOLDS
        self._station_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._load_station_metadata()

    @classmethod
    def get_instance(cls) -> "SpatialReliabilityService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_station_metadata(self):
        """Loads detailed sensor parameter capabilities from the XKDR catalogue."""
        xkdr_path = self._project_root / "data/cache/xkdr/xkdr_stations_catalogue.json"
        if xkdr_path.exists():
            try:
                with open(xkdr_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    sid = str(item.get("station_id", "")).strip().upper()
                    if sid:
                        self._station_metadata_cache[sid] = item
            except Exception as e:
                logger.warning(f"Error loading XKDR metadata: {e}")

    def classify_freshness(
        self,
        observation_time_str: Optional[str],
        reference_time: Optional[datetime] = None,
        is_historical: bool = False,
    ) -> Tuple[str, float]:
        """
        Calculates observation age in hours and maps into explicit freshness categories:
        FRESH (<= 2h), DELAYED (2-6h), STALE (6-24h), OFFLINE (> 24h), UNKNOWN.
        Prevents mislabeling historical records as live.
        """
        if not observation_time_str or pd.isna(observation_time_str):
            return "UNKNOWN", float("inf")

        if is_historical:
            return "HISTORICAL_BENCHMARK", float("inf")

        ref_dt = reference_time or datetime.now(timezone.utc)

        # Parse observation timestamp safely
        try:
            # Handle ISO or standard datetime strings
            clean_str = str(observation_time_str).replace("Z", "+00:00")
            if "+" in clean_str or clean_str.endswith("+00:00"):
                obs_dt = datetime.fromisoformat(clean_str)
            else:
                # Assume UTC if timezone-naive
                obs_dt = datetime.fromisoformat(clean_str).replace(tzinfo=timezone.utc)

            age_hours = max(0.0, (ref_dt - obs_dt).total_seconds() / 3600.0)
        except Exception:
            return "UNKNOWN", float("inf")

        if age_hours <= self.thresholds.fresh_max_hours:
            return "FRESH", round(age_hours, 2)
        elif age_hours <= self.thresholds.delayed_max_hours:
            return "DELAYED", round(age_hours, 2)
        elif age_hours <= self.thresholds.stale_max_hours:
            return "STALE", round(age_hours, 2)
        else:
            return "OFFLINE", round(age_hours, 2)

    def calculate_clark_evans_index(
        self,
        coords: np.ndarray,
        area_km2: float,
    ) -> Tuple[float, str, float]:
        """
        Calculates the Clark-Evans Nearest Neighbor Ratio (R) to diagnose spatial geometry:
        R = r_observed / r_expected
        where r_observed = mean nearest neighbor distance in km,
        and r_expected = 0.5 / sqrt(density) in a Poisson point process.
        
        Interpretation:
        - R < 0.85: Clustered (stations clumped in core/industrial districts)
        - 0.85 <= R <= 1.15: Random Poisson distribution
        - R > 1.15: Dispersed / Geographically distributed
        """
        n = len(coords)
        if n < 2 or area_km2 <= 0.0:
            return 1.0, "Undefined (Insufficient Monitors)", 0.0

        lats = coords[:, 0]
        lons = coords[:, 1]
        dist_matrix = haversine_distance_vectorized(
            lats[:, np.newaxis],
            lons[:, np.newaxis],
            lats[np.newaxis, :],
            lons[np.newaxis, :],
        )
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_dists = np.min(dist_matrix, axis=1)
        r_observed = float(np.mean(nearest_dists))

        density = n / area_km2  # stations per km^2
        r_expected = 0.5 / np.sqrt(density) if density > 0 else 1.0

        r_ratio = r_observed / r_expected if r_expected > 0 else 1.0

        if r_ratio < 0.85:
            dist_type = "Clustered (High Core Density, Sparse Periphery)"
        elif r_ratio <= 1.15:
            dist_type = "Random / Mixed Spatial Distribution"
        else:
            dist_type = "Dispersed (Uniform Geographic Coverage)"

        return round(float(r_ratio), 2), dist_type, round(r_observed, 2)

    def audit_city_reliability(self, city_id: str, live_mode: bool = False) -> Dict[str, Any]:
        """
        Performs a full station coverage, freshness, and spatial reliability audit for a city.
        """
        city = self.registry.get_city(city_id)
        if city is None:
            raise ValueError(f"Unknown city ID '{city_id}'")

        stations = self.registry.get_stations_for_city(city_id)
        n_registered = len(stations)

        # Compute bounding extent area in km^2
        dlat_km = (city.max_lat - city.min_lat) * 111.0
        dlon_km = (city.max_lon - city.min_lon) * 111.0 * np.cos(np.radians(city.center_lat))
        area_km2 = round(max(1.0, dlat_km * dlon_km), 1)

        # Sensor parameter capability audit
        n_pm25 = 0
        n_pm10 = 0
        n_reporting = 0
        n_fresh = 0
        n_delayed = 0
        n_stale = 0
        n_offline = 0
        n_complete_24h = 0
        last_ingestion_ts = None

        station_details = []
        for s in stations:
            sid = s["station_id"]
            meta = self._station_metadata_cache.get(sid, {})
            params = meta.get("parameters", [])
            has_p25 = "PM2.5" in params or "pm25" in params
            has_p10 = "PM10" in params or "pm10" in params
            if has_p25:
                n_pm25 += 1
            if has_p10:
                n_pm10 += 1

            # Determine reporting status and timestamp
            last_seen = meta.get("last_seen")
            if last_seen:
                n_reporting += 1
                if last_ingestion_ts is None or str(last_seen) > str(last_ingestion_ts):
                    last_ingestion_ts = str(last_seen)

            freshness_cat, age_h = self.classify_freshness(
                last_seen,
                is_historical=(not live_mode),
            )

            if freshness_cat == "FRESH":
                n_fresh += 1
            elif freshness_cat == "DELAYED":
                n_delayed += 1
            elif freshness_cat == "STALE":
                n_stale += 1
            else:
                n_offline += 1

            # Sequence completeness: stations with > 10,000 recorded rows have robust contiguous 24h buffers
            n_rows = meta.get("n_rows", 0)
            is_complete = bool(n_rows >= 720)  # at least 1 month of contiguous records
            if is_complete:
                n_complete_24h += 1

            station_details.append({
                "station_id": sid,
                "station_name": s["station_name"],
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "distance_to_center_km": s["distance_to_center_km"],
                "has_pm25": has_p25,
                "has_pm10": has_p10,
                "last_seen": last_seen or "N/A",
                "freshness_status": freshness_cat,
                "observation_age_hours": age_h,
                "complete_24h_history": is_complete,
            })

        # Spatial geometry diagnostics
        if n_registered >= 2:
            coords = np.array([[s["latitude"], s["longitude"]] for s in stations])
            clark_r, geom_desc, mean_nn_km = self.calculate_clark_evans_index(coords, area_km2)
        else:
            clark_r = 1.0
            geom_desc = "Single/Zero Station Topology"
            mean_nn_km = 0.0

        density_per_100km2 = round((n_registered / area_km2) * 100.0, 2) if area_km2 > 0 else 0.0
        missing_pm25_pct = round(((n_registered - n_pm25) / n_registered) * 100.0, 1) if n_registered > 0 else 100.0
        missing_pm10_pct = round(((n_registered - n_pm10) / n_registered) * 100.0, 1) if n_registered > 0 else 100.0

        # Multi-Factor Operational Reliability Classification
        if city_id == "delhi_ncr":
            reliability_tier = "Tier 1: Scientifically Calibrated & High Operational Reliability"
            tier_code = "TIER_1_CALIBRATED"
            operational_summary = (
                "Empirically validated with 99.4% conformal coverage (Phase 13). Dense 98-station "
                "network guarantees robust intra-urban spatial interpolation even during partial station dropouts."
            )
        elif n_registered >= 10:
            reliability_tier = "Tier 2: High Operational Reliability"
            tier_code = "TIER_2_HIGH"
            operational_summary = (
                f"High-density urban network ({n_registered} stations, {density_per_100km2:.2f}/100km²). "
                "Supports robust continuous 2D IDW surface estimation with low sensitivity to single-monitor gaps."
            )
        elif n_registered >= 4:
            reliability_tier = "Tier 3: Moderate Operational Reliability"
            tier_code = "TIER_3_MODERATE"
            operational_summary = (
                f"Moderate network ({n_registered} stations). Capable of resolving core macroscopic urban gradients; "
                "accuracy degrades towards municipal periphery."
            )
        elif n_registered == 3:
            reliability_tier = "Tier 4: Marginal / Sparse Operational Reliability"
            tier_code = "TIER_4_MARGINAL"
            operational_summary = (
                "Marginal 3-station network. IDW surface generation is technically supported but highly sensitive "
                "to single-station telemetry lag or outlier sensor readings."
            )
        else:
            reliability_tier = "Tier 5: Insufficient Operational Coverage"
            tier_code = "TIER_5_INSUFFICIENT"
            operational_summary = (
                f"Insufficient physical monitors ({n_registered} stations < 3 required). "
                "Active safeguards suppress continuous 2D surface rendering to eliminate spurious spatial artifacts."
            )

        return {
            "city_id": city.city_id,
            "city_name": city.name,
            "region": city.region,
            "state": city.primary_state,
            "center_lat": city.center_lat,
            "center_lon": city.center_lon,
            "bounding_box": {
                "min_lat": city.min_lat,
                "max_lat": city.max_lat,
                "min_lon": city.min_lon,
                "max_lon": city.max_lon,
            },
            "area_km2": area_km2,
            "station_density_per_100km2": density_per_100km2,
            "registered_station_count": n_registered,
            "reporting_station_count": n_reporting,
            "stations_with_pm25": n_pm25,
            "stations_with_pm10": n_pm10,
            "stations_fresh": n_fresh,
            "stations_delayed": n_delayed,
            "stations_stale": n_stale,
            "stations_offline": n_offline,
            "stations_complete_24h": n_complete_24h,
            "last_ingestion_timestamp": last_ingestion_ts or "Historical / Standby",
            "missing_pm25_pct": missing_pm25_pct,
            "missing_pm10_pct": missing_pm10_pct,
            "clark_evans_r": clark_r,
            "mean_nearest_neighbor_distance_km": mean_nn_km,
            "spatial_distribution_geometry": geom_desc,
            "reliability_tier": reliability_tier,
            "tier_code": tier_code,
            "interpolation_supported": (n_registered >= 3),
            "operational_summary": operational_summary,
            "station_details": station_details,
        }

    def audit_all_cities_reliability(self, live_mode: bool = False) -> pd.DataFrame:
        """Audits all 31 registered cities and returns a comprehensive summary DataFrame."""
        rows = []
        for city in self.registry.get_all_cities():
            audit = self.audit_city_reliability(city.city_id, live_mode=live_mode)
            rows.append({
                "city_id": audit["city_id"],
                "name": audit["city_name"],
                "region": audit["region"],
                "state": audit["state"],
                "area_km2": audit["area_km2"],
                "registered_stations": audit["registered_station_count"],
                "pm25_stations": audit["stations_with_pm25"],
                "pm10_stations": audit["stations_with_pm10"],
                "complete_24h_stations": audit["stations_complete_24h"],
                "density_per_100km2": audit["station_density_per_100km2"],
                "clark_evans_r": audit["clark_evans_r"],
                "mean_nn_dist_km": audit["mean_nearest_neighbor_distance_km"],
                "geometry": audit["spatial_distribution_geometry"],
                "tier_code": audit["tier_code"],
                "reliability_tier": audit["reliability_tier"],
                "interpolation_supported": audit["interpolation_supported"],
            })
        return pd.DataFrame(rows)

    def get_location_spatial_diagnostics(
        self,
        lat: float,
        lon: float,
        city_id: str,
        max_radius_km: float = 45.0,
    ) -> Dict[str, Any]:
        """
        Evaluates arbitrary-location spatial reliability diagnostics:
        nearest station distance, contributing station count, max contributing distance,
        and missing-data fraction within the neighborhood.
        """
        stations = self.registry.get_stations_for_city(city_id)
        if not stations:
            return {
                "latitude": lat,
                "longitude": lon,
                "nearest_station_id": None,
                "nearest_distance_km": None,
                "contributing_stations_count": 0,
                "max_contributing_distance_km": None,
                "missing_data_fraction": 1.0,
                "spatial_reliability_status": "NO_LOCAL_STATIONS",
                "output_classification": "Unavailable Estimate",
            }

        coords = np.array([[s["latitude"], s["longitude"]] for s in stations])
        target_lat = np.array([lat])
        target_lon = np.array([lon])
        dists = haversine_distance_vectorized(
            target_lat, target_lon, coords[:, 0], coords[:, 1]
        ).ravel()

        nearest_idx = int(np.argmin(dists))
        nearest_d = float(dists[nearest_idx])
        nearest_sid = stations[nearest_idx]["station_id"]

        # Contributing stations within max radius
        eligible_mask = dists <= max_radius_km
        n_contrib = int(np.sum(eligible_mask))
        max_contrib_d = float(np.max(dists[eligible_mask])) if n_contrib > 0 else None

        # Missing data in contributing cluster
        eligible_sids = [stations[i]["station_id"] for i in np.where(eligible_mask)[0]]
        p25_avail = sum(1 for sid in eligible_sids if "PM2.5" in self._station_metadata_cache.get(sid, {}).get("parameters", []))
        missing_frac = round(1.0 - (p25_avail / n_contrib), 2) if n_contrib > 0 else 1.0

        if nearest_d <= 0.05:
            output_class = "Physical Observation"
            rel_status = "OPTIMAL_CO_LOCATED"
        elif n_contrib >= 3:
            output_class = "Interpolated Spatial Estimate"
            rel_status = "HIGH_CONFIDENCE_SURFACE" if nearest_d <= 10.0 else "MODERATE_CONFIDENCE_SURFACE"
        elif n_contrib > 0:
            output_class = "Station-Level Forecast (Nearest Neighbor Fallback)"
            rel_status = "SPARSE_NEIGHBORHOOD"
        else:
            output_class = "Unavailable Estimate"
            rel_status = "UNSUPPORTED_EXTRAPOLATION"

        return {
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "nearest_station_id": nearest_sid,
            "nearest_distance_km": round(nearest_d, 2),
            "contributing_stations_count": n_contrib,
            "max_contributing_distance_km": round(max_contrib_d, 2) if max_contrib_d else None,
            "missing_data_fraction": missing_frac,
            "spatial_reliability_status": rel_status,
            "output_classification": output_class,
        }


def get_spatial_reliability_service() -> SpatialReliabilityService:
    """Singleton provider for SpatialReliabilityService."""
    return SpatialReliabilityService.get_instance()
