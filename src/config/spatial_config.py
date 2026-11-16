"""
src/config/spatial_config.py

Spatial bounding boxes, grid resolutions, and interpolation parameters
for the VayuDrishti Spatial Prediction & Urban Heatmap Layer.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

@dataclass(frozen=True)
class CityBoundingBox:
    city_id: str
    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    center_lat: float
    center_lon: float
    default_zoom: int
    primary_state: str

# Defined bounding boxes covering urban agglomerations and surrounding CAAQMS stations
SUPPORTED_CITIES: Dict[str, CityBoundingBox] = {
    "delhi_ncr": CityBoundingBox(
        city_id="delhi_ncr",
        name="Delhi NCR",
        min_lat=28.35,
        max_lat=28.95,
        min_lon=76.85,
        max_lon=77.45,
        center_lat=28.65,
        center_lon=77.15,
        default_zoom=10,
        primary_state="Delhi",
    ),
    "mumbai_mmr": CityBoundingBox(
        city_id="mumbai_mmr",
        name="Mumbai MMR",
        min_lat=18.85,
        max_lat=19.35,
        min_lon=72.75,
        max_lon=73.20,
        center_lat=19.07,
        center_lon=72.88,
        default_zoom=10,
        primary_state="Maharashtra",
    ),
    "kolkata": CityBoundingBox(
        city_id="kolkata",
        name="Kolkata Metropolitan",
        min_lat=22.35,
        max_lat=22.75,
        min_lon=88.20,
        max_lon=88.55,
        center_lat=22.57,
        center_lon=88.36,
        default_zoom=11,
        primary_state="West Bengal",
    ),
    "bengaluru": CityBoundingBox(
        city_id="bengaluru",
        name="Bengaluru Urban",
        min_lat=12.75,
        max_lat=13.20,
        min_lon=77.40,
        max_lon=77.80,
        center_lat=12.97,
        center_lon=77.59,
        default_zoom=11,
        primary_state="Karnataka",
    ),
}

@dataclass
class SpatialInterpolationConfig:
    # IDW exponent: p=2.0 is standard Inverse Distance Squared
    power: float = 2.0
    # Minimum physical stations with valid predictions required to compute spatial surface
    min_stations: int = 3
    # Maximum radius (in km) beyond which a station is not used or extrapolation is rejected
    max_distance_km: float = 40.0
    # Maximum number of nearest neighbor stations to consider (None = all within max_distance_km)
    max_neighbors: Optional[int] = 12
    # Threshold in km below which query point is treated as an exact station hit
    exact_match_threshold_km: float = 0.05
    # Small epsilon in km to prevent division by zero in weights
    epsilon_km: float = 0.001
    # Grid resolutions available
    available_grid_resolutions: Tuple[int, ...] = (25, 50, 75)
    default_grid_resolution: int = 50
    # Concentration bounds (ug/m3) for physical plausibility clamping
    min_concentration: float = 0.0
    max_pm25_concentration: float = 1200.0
    max_pm10_concentration: float = 2000.0

DEFAULT_SPATIAL_CONFIG = SpatialInterpolationConfig()
