"""
src/services/spatial_interpolation_service.py

VayuDrishti Spatial Interpolation & Urban Heatmap Service.
Implements modified Inverse Distance Weighting (IDW) with geodesic Haversine distance,
exact station snapping, distance-bounded neighborhood filtering, and strict target timestamp integrity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from src.config.spatial_config import (
    CityBoundingBox,
    SpatialInterpolationConfig,
    SUPPORTED_CITIES,
    DEFAULT_SPATIAL_CONFIG,
)

# Mean Earth Radius in kilometers for WGS84
EARTH_RADIUS_KM = 6371.0088


def haversine_distance_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Computes great-circle Haversine distances in kilometers.
    Accepts 1D or 2D NumPy arrays of latitude and longitude in decimal degrees.
    Broadcasts shapes automatically.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    # Clip to avoid floating point precision exceeding 1.0
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


@dataclass
class StationForecastRecord:
    station_id: str
    latitude: float
    longitude: float
    predicted_pm25: float
    predicted_pm10: float
    target_time: str
    issue_time: Optional[str] = None
    forecast_horizon: str = "Next 1 Hour (t + 1)"
    model_source: str = "VayuDrishti_Ensemble"


@dataclass
class ForecastContext:
    target_time: str
    forecast_horizon: str
    model_source: str
    records: List[StationForecastRecord] = field(default_factory=list)
    issue_time: Optional[str] = None

    @classmethod
    def from_records(
        cls,
        records: List[StationForecastRecord],
        expected_target_time: Optional[str] = None,
        expected_horizon: Optional[str] = None,
    ) -> "ForecastContext":
        if not records:
            raise ValueError("Cannot construct ForecastContext from empty records list.")

        ref_target = expected_target_time or records[0].target_time
        ref_horizon = expected_horizon or records[0].forecast_horizon
        ref_source = records[0].model_source
        ref_issue = records[0].issue_time

        validated: List[StationForecastRecord] = []
        for r in records:
            # Enforce identical forecast horizon and target timestamp across all stations
            if r.target_time != ref_target:
                continue
            if r.forecast_horizon != ref_horizon:
                continue
            validated.append(r)

        return cls(
            target_time=ref_target,
            forecast_horizon=ref_horizon,
            model_source=ref_source,
            records=validated,
            issue_time=ref_issue,
        )


class SpatialInterpolationService:
    """
    Core geostatistical spatial interpolation engine.
    Computes arbitrary-location predictions and continuous urban grid surfaces.
    """

    def __init__(self, config: Optional[SpatialInterpolationConfig] = None):
        self.config = config or DEFAULT_SPATIAL_CONFIG

    def _sanitize_station_inputs(
        self,
        records: List[StationForecastRecord],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Validates, deduplicates, and filters station inputs.
        Returns:
            coords: (N, 2) [latitude, longitude]
            pm25_vals: (N,)
            pm10_vals: (N,)
            station_ids: List[str] of length N
        """
        valid_records: List[StationForecastRecord] = []
        seen_stations = set()

        for r in records:
            # 1. Coordinate validity check
            if not (-90.0 <= r.latitude <= 90.0 and -180.0 <= r.longitude <= 180.0):
                continue
            # 2. Value finite and non-NaN
            if np.isnan(r.predicted_pm25) or np.isnan(r.predicted_pm10):
                continue
            if np.isinf(r.predicted_pm25) or np.isinf(r.predicted_pm10):
                continue

            # 3. Deduplication: keep the first occurrence per station_id
            sid = str(r.station_id).strip().upper()
            if sid in seen_stations:
                continue
            seen_stations.add(sid)
            valid_records.append(r)

        if len(valid_records) < self.config.min_stations:
            raise ValueError(
                f"Insufficient valid stations ({len(valid_records)}). "
                f"Minimum required is {self.config.min_stations}."
            )

        coords = np.array([[r.latitude, r.longitude] for r in valid_records], dtype=np.float64)
        pm25 = np.array([r.predicted_pm25 for r in valid_records], dtype=np.float64)
        pm10 = np.array([r.predicted_pm10 for r in valid_records], dtype=np.float64)
        sids = [r.station_id for r in valid_records]

        return coords, pm25, pm10, sids

    def predict_at_location(
        self,
        latitude: float,
        longitude: float,
        forecast_context: Union[ForecastContext, List[StationForecastRecord]],
    ) -> Dict[str, Any]:
        """
        Predicts PM2.5 and PM10 concentrations at an arbitrary geographic coordinate
        using modified Inverse Distance Weighting with distance-decayed confidence intervals.
        """
        # 1. Coordinate validation
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            return {
                "latitude": latitude,
                "longitude": longitude,
                "predicted_pm25": None,
                "predicted_pm10": None,
                "data_quality_status": "INVALID_COORDINATES",
                "error_message": f"Coordinates out of bounds: lat={latitude}, lon={longitude}.",
            }

        # 2. Extract context
        if isinstance(forecast_context, list):
            context = ForecastContext.from_records(forecast_context)
        else:
            context = forecast_context

        try:
            coords, pm25_vals, pm10_vals, sids = self._sanitize_station_inputs(context.records)
        except ValueError as e:
            return {
                "latitude": latitude,
                "longitude": longitude,
                "predicted_pm25": None,
                "predicted_pm10": None,
                "data_quality_status": "INSUFFICIENT_STATIONS",
                "error_message": str(e),
                "target_time": context.target_time,
                "forecast_horizon": context.forecast_horizon,
            }

        # 3. Vectorized Haversine distance to all stations
        target_lat_arr = np.array([latitude])
        target_lon_arr = np.array([longitude])
        # Shape: (N,)
        distances_km = haversine_distance_vectorized(
            target_lat_arr,
            target_lon_arr,
            coords[:, 0],
            coords[:, 1],
        ).ravel()

        nearest_idx = int(np.argmin(distances_km))
        nearest_dist_km = float(distances_km[nearest_idx])
        nearest_station_id = sids[nearest_idx]

        # 4. Exact station snapping
        if nearest_dist_km <= self.config.exact_match_threshold_km:
            p25_out = float(pm25_vals[nearest_idx])
            p10_out = float(pm10_vals[nearest_idx])
            return {
                "latitude": round(latitude, 5),
                "longitude": round(longitude, 5),
                "predicted_pm25": round(p25_out, 2),
                "predicted_pm10": round(p10_out, 2),
                "nearest_station_id": nearest_station_id,
                "nearest_station_distance_km": round(nearest_dist_km, 3),
                "nearest_station_pm25": round(p25_out, 2),
                "nearest_station_pm10": round(p10_out, 2),
                "num_contributing_stations": 1,
                "contributing_station_ids": [nearest_station_id],
                "data_quality_status": "EXACT_STATION_HIT",
                "interpolation_method": "Exact Station Observation / Prediction",
                "target_time": context.target_time,
                "forecast_issue_time": context.issue_time,
                "forecast_horizon": context.forecast_horizon,
                "model_source": context.model_source,
                "ci_95_pm25": (round(p25_out, 2), round(p25_out, 2)),
                "ci_95_pm10": (round(p10_out, 2), round(p10_out, 2)),
                "ci_heuristic_95_pm25": (round(p25_out, 2), round(p25_out, 2)),
                "ci_heuristic_95_pm10": (round(p10_out, 2), round(p10_out, 2)),
                "ci_conformal_95_pm25": (
                    round(max(0.0, p25_out - 97.24), 2),
                    round(p25_out + 97.24, 2),
                ),
                "ci_conformal_95_pm10": (
                    round(max(0.0, p10_out - 186.83), 2),
                    round(p10_out + 186.83, 2),
                ),
                "is_extrapolation": False,
            }

        # 5. Distance boundary check
        if nearest_dist_km > self.config.max_distance_km:
            return {
                "latitude": round(latitude, 5),
                "longitude": round(longitude, 5),
                "predicted_pm25": None,
                "predicted_pm10": None,
                "nearest_station_id": nearest_station_id,
                "nearest_station_distance_km": round(nearest_dist_km, 2),
                "data_quality_status": "UNSUPPORTED_EXTRAPOLATION",
                "error_message": (
                    f"Query point is {nearest_dist_km:.1f} km from nearest station "
                    f"({nearest_station_id}), exceeding maximum reliable radius ({self.config.max_distance_km:.1f} km)."
                ),
                "target_time": context.target_time,
                "forecast_horizon": context.forecast_horizon,
                "model_source": context.model_source,
                "is_extrapolation": True,
            }

        # 6. Select eligible neighbor stations within max_distance_km
        eligible_mask = distances_km <= self.config.max_distance_km
        eligible_indices = np.where(eligible_mask)[0]

        if len(eligible_indices) < self.config.min_stations:
            # If fewer than min_stations within radius, fall back to nearest min_stations if reasonable
            sorted_indices = np.argsort(distances_km)
            eligible_indices = sorted_indices[: self.config.min_stations]
            status = "SPARSE_INTERPOLATION"
        else:
            # Sort by distance and apply max_neighbors cap if specified
            sorted_by_dist = eligible_indices[np.argsort(distances_km[eligible_indices])]
            if self.config.max_neighbors and len(sorted_by_dist) > self.config.max_neighbors:
                eligible_indices = sorted_by_dist[: self.config.max_neighbors]
            else:
                eligible_indices = sorted_by_dist
            status = "HIGH_CONFIDENCE_INTERPOLATION"

        active_dists = distances_km[eligible_indices]
        active_p25 = pm25_vals[eligible_indices]
        active_p10 = pm10_vals[eligible_indices]
        active_sids = [sids[i] for i in eligible_indices]

        # 7. Compute IDW weights
        weights = 1.0 / ((active_dists + self.config.epsilon_km) ** self.config.power)
        weights_sum = np.sum(weights)

        if weights_sum <= 0.0 or np.isnan(weights_sum):
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights_sum

        pred_pm25 = float(np.sum(weights * active_p25))
        pred_pm10 = float(np.sum(weights * active_p10))

        # Clamp to physical bounds
        pred_pm25 = float(np.clip(pred_pm25, self.config.min_concentration, self.config.max_pm25_concentration))
        pred_pm10 = float(np.clip(pred_pm10, self.config.min_concentration, self.config.max_pm10_concentration))

        # 8. Spatial confidence interval calculation
        # Dispersion increases as distance to nearest physical monitor increases
        ci_half_pm25 = 4.0 + 0.85 * nearest_dist_km
        ci_half_pm10 = 7.5 + 1.40 * nearest_dist_km

        ci_pm25 = (
            round(max(0.0, pred_pm25 - ci_half_pm25), 2),
            round(pred_pm25 + ci_half_pm25, 2),
        )
        ci_pm10 = (
            round(max(0.0, pred_pm10 - ci_half_pm10), 2),
            round(pred_pm10 + ci_half_pm10, 2),
        )

        # 9. Split-conformal calibrated prediction intervals (Empirical 95% nominal guarantee)
        # q_hat fitted on 777 calibration instances, evaluated on 950 unseen instances (Phase 13)
        conformal_half_pm25 = 97.24 * (1.0 + 0.06 * nearest_dist_km)
        conformal_half_pm10 = 186.83 * (1.0 + 0.05 * nearest_dist_km)
        ci_conf_pm25 = (
            round(max(0.0, pred_pm25 - conformal_half_pm25), 2),
            round(pred_pm25 + conformal_half_pm25, 2),
        )
        ci_conf_pm10 = (
            round(max(0.0, pred_pm10 - conformal_half_pm10), 2),
            round(pred_pm10 + conformal_half_pm10, 2),
        )

        return {
            "latitude": round(latitude, 5),
            "longitude": round(longitude, 5),
            "predicted_pm25": round(pred_pm25, 2),
            "predicted_pm10": round(pred_pm10, 2),
            "nearest_station_id": nearest_station_id,
            "nearest_station_distance_km": round(nearest_dist_km, 2),
            "nearest_station_pm25": round(float(pm25_vals[nearest_idx]), 2),
            "nearest_station_pm10": round(float(pm10_vals[nearest_idx]), 2),
            "num_contributing_stations": len(eligible_indices),
            "contributing_station_ids": active_sids,
            "data_quality_status": status,
            "interpolation_method": f"Modified IDW (p={self.config.power:.1f})",
            "target_time": context.target_time,
            "forecast_issue_time": context.issue_time,
            "forecast_horizon": context.forecast_horizon,
            "model_source": context.model_source,
            "ci_95_pm25": ci_pm25,
            "ci_95_pm10": ci_pm10,
            "ci_heuristic_95_pm25": ci_pm25,
            "ci_heuristic_95_pm10": ci_pm10,
            "ci_conformal_95_pm25": ci_conf_pm25,
            "ci_conformal_95_pm10": ci_conf_pm10,
            "is_extrapolation": (status == "SPARSE_INTERPOLATION"),
        }

    def generate_city_grid(
        self,
        city_id: str,
        forecast_context: Union[ForecastContext, List[StationForecastRecord]],
        grid_resolution: int = 50,
    ) -> Dict[str, Any]:
        """
        Generates a continuous 2D spatial estimation grid across a designated urban bounding box.
        Vectorized over M = grid_resolution x grid_resolution points and N stations.
        """
        city_key = city_id.lower().strip()
        if city_key not in SUPPORTED_CITIES:
            raise ValueError(
                f"Unsupported city '{city_id}'. Supported: {list(SUPPORTED_CITIES.keys())}"
            )

        bbox = SUPPORTED_CITIES[city_key]

        # Extract context
        if isinstance(forecast_context, list):
            context = ForecastContext.from_records(forecast_context)
        else:
            context = forecast_context

        try:
            coords, pm25_vals, pm10_vals, sids = self._sanitize_station_inputs(context.records)
        except ValueError as e:
            return {
                "city_id": bbox.city_id,
                "city_name": bbox.name,
                "region": bbox.region,
                "primary_state": bbox.primary_state,
                "is_supported": False,
                "data_quality_status": "INSUFFICIENT_COVERAGE",
                "validation_status": "INSUFFICIENT_COVERAGE",
                "density_category": "Insufficient",
                "error_message": f"Insufficient monitoring coverage ({len(context.records)} monitors). Minimum required for continuous 2D spatial interpolation is {self.config.min_stations}.",
                "stations_used_count": len(context.records),
                "grid_resolution": grid_resolution,
                "total_grid_points": 0,
                "valid_grid_points": 0,
                "coverage_percentage": 0.0,
                "bounding_box": {
                    "min_lat": bbox.min_lat,
                    "max_lat": bbox.max_lat,
                    "min_lon": bbox.min_lon,
                    "max_lon": bbox.max_lon,
                    "center_lat": bbox.center_lat,
                    "center_lon": bbox.center_lon,
                    "default_zoom": bbox.default_zoom,
                },
                "grid_lat": [],
                "grid_lon": [],
                "pm25_grid": [],
                "pm10_grid": [],
                "valid_mask": [],
                "stats": {
                    "pm25_min": 0.0,
                    "pm25_max": 0.0,
                    "pm25_mean": 0.0,
                    "pm10_min": 0.0,
                    "pm10_max": 0.0,
                    "pm10_mean": 0.0,
                },
                "station_ids": [r.station_id for r in context.records],
                "station_coords": [[r.latitude, r.longitude] for r in context.records],
                "station_pm25": [r.predicted_pm25 for r in context.records],
                "station_pm10": [r.predicted_pm10 for r in context.records],
                "target_time": context.target_time,
                "forecast_issue_time": context.issue_time,
                "forecast_horizon": context.forecast_horizon,
                "model_source": context.model_source,
                "interpolation_method": f"Modified IDW (p={self.config.power:.1f})",
            }

        num_stations = len(sids)

        # 1. Construct 2D grid coordinates
        lats = np.linspace(bbox.min_lat, bbox.max_lat, grid_resolution, dtype=np.float64)
        lons = np.linspace(bbox.min_lon, bbox.max_lon, grid_resolution, dtype=np.float64)
        grid_lon, grid_lat = np.meshgrid(lons, lats)

        # Flatten grid points: Shape (M, 2) where M = grid_resolution^2
        M = grid_resolution * grid_resolution
        query_lat = grid_lat.ravel()
        query_lon = grid_lon.ravel()

        # 2. Vectorized distance matrix calculation: Shape (M, N)
        # Broadcasting (M, 1) against (1, N)
        dist_matrix = haversine_distance_vectorized(
            query_lat[:, np.newaxis],
            query_lon[:, np.newaxis],
            coords[np.newaxis, :, 0],
            coords[np.newaxis, :, 1],
        )

        min_distances = np.min(dist_matrix, axis=1)  # Shape: (M,)
        valid_coverage_mask = min_distances <= self.config.max_distance_km

        # 3. Exact snapping for grid points that land directly on a station
        exact_match_mask = dist_matrix <= self.config.exact_match_threshold_km
        has_exact_match = np.any(exact_match_mask, axis=1)

        # 4. Compute IDW weights
        # Apply epsilon to avoid division by zero
        weights = 1.0 / ((dist_matrix + self.config.epsilon_km) ** self.config.power)

        # Mask out stations beyond max_distance_km
        too_far_mask = dist_matrix > self.config.max_distance_km
        weights[too_far_mask] = 0.0

        # For points with exact match, force weight to 1 for matching station
        if np.any(has_exact_match):
            for m_idx in np.where(has_exact_match)[0]:
                match_col = np.where(exact_match_mask[m_idx])[0][0]
                weights[m_idx, :] = 0.0
                weights[m_idx, match_col] = 1.0

        weights_sum = np.sum(weights, axis=1, keepdims=True)
        # Avoid division by zero for unmonitored grid cells
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_weights = np.where(weights_sum > 0.0, weights / weights_sum, 0.0)

        # 5. Interpolate PM2.5 and PM10
        interp_pm25_flat = np.sum(normalized_weights * pm25_vals[np.newaxis, :], axis=1)
        interp_pm10_flat = np.sum(normalized_weights * pm10_vals[np.newaxis, :], axis=1)

        # Set unsupported extrapolation points to NaN
        interp_pm25_flat[~valid_coverage_mask] = np.nan
        interp_pm10_flat[~valid_coverage_mask] = np.nan

        # Reshape to (grid_resolution, grid_resolution)
        pm25_grid = interp_pm25_flat.reshape((grid_resolution, grid_resolution))
        pm10_grid = interp_pm10_flat.reshape((grid_resolution, grid_resolution))
        coverage_grid = valid_coverage_mask.reshape((grid_resolution, grid_resolution))

        valid_pm25 = interp_pm25_flat[valid_coverage_mask]
        valid_pm10 = interp_pm10_flat[valid_coverage_mask]

        coverage_pct = float(np.mean(valid_coverage_mask) * 100.0)

        return {
            "city_id": bbox.city_id,
            "city_name": bbox.name,
            "region": bbox.region,
            "primary_state": bbox.primary_state,
            "is_supported": True,
            "data_quality_status": "VALIDATED_CONFORMAL" if bbox.city_id == "delhi_ncr" else "EXPLORATORY_SUFFICIENT",
            "validation_status": bbox.validation_status,
            "density_category": bbox.density_category,
            "grid_resolution": grid_resolution,
            "total_grid_points": M,
            "valid_grid_points": int(np.sum(valid_coverage_mask)),
            "coverage_percentage": round(coverage_pct, 2),
            "bounding_box": {
                "min_lat": bbox.min_lat,
                "max_lat": bbox.max_lat,
                "min_lon": bbox.min_lon,
                "max_lon": bbox.max_lon,
                "center_lat": bbox.center_lat,
                "center_lon": bbox.center_lon,
                "default_zoom": bbox.default_zoom,
            },
            "grid_lat": grid_lat.tolist(),
            "grid_lon": grid_lon.tolist(),
            "pm25_grid": pm25_grid.tolist(),
            "pm10_grid": pm10_grid.tolist(),
            "valid_mask": coverage_grid.tolist(),
            "stats": {
                "pm25_min": round(float(np.nanmin(valid_pm25)), 2) if len(valid_pm25) > 0 else 0.0,
                "pm25_max": round(float(np.nanmax(valid_pm25)), 2) if len(valid_pm25) > 0 else 0.0,
                "pm25_mean": round(float(np.nanmean(valid_pm25)), 2) if len(valid_pm25) > 0 else 0.0,
                "pm10_min": round(float(np.nanmin(valid_pm10)), 2) if len(valid_pm10) > 0 else 0.0,
                "pm10_max": round(float(np.nanmax(valid_pm10)), 2) if len(valid_pm10) > 0 else 0.0,
                "pm10_mean": round(float(np.nanmean(valid_pm10)), 2) if len(valid_pm10) > 0 else 0.0,
            },
            "stations_used_count": num_stations,
            "station_ids": sids,
            "station_coords": coords.tolist(),
            "station_pm25": pm25_vals.tolist(),
            "station_pm10": pm10_vals.tolist(),
            "target_time": context.target_time,
            "forecast_issue_time": context.issue_time,
            "forecast_horizon": context.forecast_horizon,
            "model_source": context.model_source,
            "interpolation_method": f"Modified IDW (p={self.config.power:.1f})",
        }
