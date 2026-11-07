"""
src/services/spatial_advection_service.py

Spatial-Temporal Cross-Station Advection & Multi-Node Modeling Service for VayuDrishti Phase 13.
Investigates spatial correlation, atmospheric advection, and causal predictive value of
neighboring CAAQMS observations across urban clusters (Delhi, Mumbai, Kolkata).

Guarantees strict temporal causality: all neighbor observations are indexed <= t.
Zero future lookahead is permitted.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATION_REGISTRY_CSV = ROOT_DIR / "experiments" / "phase12" / "station_registry.csv"
WEATHER_DIR = ROOT_DIR / "data" / "raw" / "era5_land" / "station_weather"
RECENT_AQ_CSV = ROOT_DIR / "data" / "processed" / "recent_air_quality_cleaned.csv"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in kilometers between two coordinates."""
    r_earth = 6371.0  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes initial great-circle bearing (azimuth) from point 1 to point 2 in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    theta = math.degrees(math.atan2(y, x))
    return (theta + 360.0) % 360.0


def compute_cross_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = 12,
) -> Dict[int, float]:
    """
    Computes normalized Pearson cross-correlation function (CCF) at discrete lags tau in [-max_lag, +max_lag].
    Convention: Positive lag tau > 0 means series_a at (t - tau) correlates with series_b at t (A leads B).
    Negative lag tau < 0 means series_b leads series_a.
    """
    s_a = np.asarray(series_a, dtype=float)
    s_b = np.asarray(series_b, dtype=float)

    mask = np.isfinite(s_a) & np.isfinite(s_b)
    s_a = s_a[mask]
    s_b = s_b[mask]

    n = len(s_a)
    if n <= max_lag + 10:
        return {lag: 0.0 for lag in range(-max_lag, max_lag + 1)}

    # Standardize
    a_norm = (s_a - np.mean(s_a)) / (np.std(s_a) + 1e-9)
    b_norm = (s_b - np.mean(s_b)) / (np.std(s_b) + 1e-9)

    ccf = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # a[t - lag] vs b[t] -> a leads b
            r = np.mean(a_norm[:-lag] * b_norm[lag:])
        elif lag < 0:
            # a[t + |lag|] vs b[t] -> b leads a
            k = abs(lag)
            r = np.mean(a_norm[k:] * b_norm[:-k])
        else:
            r = np.mean(a_norm * b_norm)
        ccf[lag] = round(float(r), 4)

    return ccf


def compute_along_transect_wind(
    wind_speed: np.ndarray,
    wind_direction_deg: np.ndarray,
    transect_bearing_deg: float,
) -> np.ndarray:
    """
    Computes along-transect wind velocity component (v_parallel):
    v_parallel = wind_speed * cos(wind_direction - transect_bearing)

    Convention:
    - Positive v_parallel (> 0) indicates wind blowing in direction of transect (from Station A to Station B).
    - Negative v_parallel (< 0) indicates opposing wind.
    """
    ws = np.asarray(wind_speed, dtype=float)
    wd = np.asarray(wind_direction_deg, dtype=float)

    # Angle difference in radians
    angle_diff = np.radians(wd - transect_bearing_deg)
    v_parallel = ws * np.cos(angle_diff)
    return v_parallel


class SpatialAdvectionService:
    """Core analytical service for multi-node cross-station air quality advection."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = Path(registry_path or STATION_REGISTRY_CSV)
        self.registry = pd.read_csv(self.registry_path) if self.registry_path.exists() else pd.DataFrame()

        # Predefined urban clusters (< 25 km distance)
        self.clusters = [
            {
                "cluster_name": "Delhi NCR Urban Corridor",
                "city": "Delhi",
                "station_a": "DL002",
                "station_a_name": "Anand Vihar, Delhi - DPCC",
                "station_b": "DL008",
                "station_b_name": "DTU, Delhi - CPCB",
            },
            {
                "cluster_name": "Mumbai Coastal Marine Corridor",
                "city": "Mumbai",
                "station_a": "MH005",
                "station_a_name": "Bandra, Mumbai - MPCB",
                "station_b": "MH008",
                "station_b_name": "Colaba, Mumbai - MPCB",
            },
            {
                "cluster_name": "Kolkata Gangetic Urban Corridor",
                "city": "Kolkata",
                "station_a": "WB008",
                "station_a_name": "Bidhannagar, Kolkata - WBPCB",
                "station_b": "WB010",
                "station_b_name": "Jadavpur, Kolkata - WBPCB",
            },
        ]

        self._populate_cluster_geodesics()

    def _populate_cluster_geodesics(self) -> None:
        """Computes distance, bearing, and reverse bearing for each cluster pair."""
        if self.registry.empty:
            return

        reg_map = self.registry.set_index("station_id")
        for cl in self.clusters:
            sa = cl["station_a"]
            sb = cl["station_b"]
            if sa in reg_map.index and sb in reg_map.index:
                lat1, lon1 = float(reg_map.loc[sa, "latitude"]), float(reg_map.loc[sa, "longitude"])
                lat2, lon2 = float(reg_map.loc[sb, "latitude"]), float(reg_map.loc[sb, "longitude"])
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                b_ab = calculate_bearing(lat1, lon1, lat2, lon2)
                b_ba = calculate_bearing(lat2, lon2, lat1, lon1)

                cl["distance_km"] = round(dist, 2)
                cl["bearing_a_to_b"] = round(b_ab, 1)
                cl["bearing_b_to_a"] = round(b_ba, 1)

    def analyze_cross_correlation(
        self,
        df_paired: pd.DataFrame,
        param: str = "PM2.5",
        max_lag: int = 12,
    ) -> Dict[str, Any]:
        """
        Analyzes CCF between Station A and Station B for parameter PM2.5 or PM10.
        Finds optimal empirical lag tau* and maximum correlation.
        """
        col_a = f"{param}_A"
        col_b = f"{param}_B"
        if col_a not in df_paired.columns or col_b not in df_paired.columns:
            return {}

        ccf_dict = compute_cross_correlation(df_paired[col_a].values, df_paired[col_b].values, max_lag=max_lag)

        # Identify peak lag
        best_lag = max(ccf_dict, key=lambda k: abs(ccf_dict[k]))
        peak_corr = ccf_dict[best_lag]
        lag_zero_corr = ccf_dict[0]

        return {
            "parameter": param,
            "max_lag_evaluated": max_lag,
            "lag_zero_correlation": lag_zero_corr,
            "peak_correlation_lag_hours": best_lag,
            "peak_correlation_value": peak_corr,
            "ccf_curve": ccf_dict,
        }

    def evaluate_wind_advection_alignment(
        self,
        df_paired_weather: pd.DataFrame,
        distance_km: float,
        transect_bearing: float,
        param: str = "PM2.5",
    ) -> Dict[str, Any]:
        """
        Evaluates physical wind-driven transport along the transect:
        1. Filters hours where wind is oriented along the transect (v_parallel >= 1.5 m/s).
        2. Computes theoretical transit time tau = d / (v_parallel * 3.6).
        3. Tests whether downwind episodes exhibit stronger lagged correlation than opposing wind.
        """
        if "wind_speed" not in df_paired_weather.columns or "wind_direction" not in df_paired_weather.columns:
            return {}

        ws = df_paired_weather["wind_speed"].values
        wd = df_paired_weather["wind_direction"].values

        v_par = compute_along_transect_wind(ws, wd, transect_bearing)
        df_eval = df_paired_weather.copy()
        df_eval["v_parallel"] = v_par

        # Favorable wind subset (blowing A -> B with speed >= 1.5 m/s)
        favorable_mask = df_eval["v_parallel"] >= 1.5
        opposing_mask = df_eval["v_parallel"] <= -1.5

        df_fav = df_eval[favorable_mask]
        df_opp = df_eval[opposing_mask]

        mean_v_fav = float(df_fav["v_parallel"].mean()) if not df_fav.empty else 2.5
        # Physical transit time in hours: d_km / (v_ms * 3.6 km/h per m/s)
        physical_transit_hours = distance_km / (mean_v_fav * 3.6) if mean_v_fav > 0 else np.nan

        # CCF under favorable wind
        ccf_fav = compute_cross_correlation(df_fav[f"{param}_A"].values, df_fav[f"{param}_B"].values, max_lag=12) if len(df_fav) > 50 else {}
        # CCF under opposing wind
        ccf_opp = compute_cross_correlation(df_opp[f"{param}_A"].values, df_opp[f"{param}_B"].values, max_lag=12) if len(df_opp) > 50 else {}

        peak_lag_fav = max(ccf_fav, key=lambda k: abs(ccf_fav[k])) if ccf_fav else 0
        peak_corr_fav = ccf_fav.get(peak_lag_fav, 0.0)

        return {
            "distance_km": round(distance_km, 2),
            "transect_bearing_deg": round(transect_bearing, 1),
            "favorable_wind_hours": int(favorable_mask.sum()),
            "opposing_wind_hours": int(opposing_mask.sum()),
            "mean_favorable_velocity_ms": round(mean_v_fav, 2),
            "theoretical_advection_hours": round(physical_transit_hours, 2),
            "empirical_peak_lag_under_wind": peak_lag_fav,
            "empirical_peak_corr_under_wind": peak_corr_fav,
            "ccf_favorable_wind": ccf_fav,
            "ccf_opposing_wind": ccf_opp,
        }
