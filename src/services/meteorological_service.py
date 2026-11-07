"""
src/services/meteorological_service.py

Meteorological Feature Integration Service for VayuDrishti Phase 12.
Provides genuine, timestamped meteorological telemetry (temperature, relative humidity,
wind speed, wind direction, dewpoint, precipitation) from ERA5-Land station files and
Open-Meteo archive/live feeds.

Guarantees strict zero-lookahead temporal causality (all meteorological features are <= t).
"""

from __future__ import annotations

import os
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("VayuDrishti.MeteorologicalService")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ERA5_WEATHER_DIR = ROOT_DIR / "data" / "raw" / "era5_land" / "station_weather"
STATION_REGISTRY_CSV = ROOT_DIR / "experiments" / "phase12" / "station_registry.csv"
WEATHER_CACHE_DIR = ROOT_DIR / "experiments" / "phase12" / "weather_cache"


class MeteorologicalService:
    """Provides validated, leakage-free meteorological features aligned to air quality stations."""

    CANONICAL_WEATHER_COLUMNS = [
        "temperature_C",
        "dewpoint_C",
        "relative_humidity",
        "wind_speed",
        "wind_direction",
        "precipitation_mm",
    ]

    def __init__(
        self,
        weather_dir: Optional[Path] = None,
        registry_path: Optional[Path] = None,
        enable_api_fallback: bool = True,
    ):
        self.weather_dir = Path(weather_dir or ERA5_WEATHER_DIR)
        self.registry_path = Path(registry_path or STATION_REGISTRY_CSV)
        self.enable_api_fallback = enable_api_fallback
        WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_dir = WEATHER_CACHE_DIR

        # Load station registry for coordinate lookups
        self.station_coords: Dict[str, Tuple[float, float]] = {}
        if self.registry_path.exists():
            df_reg = pd.read_csv(self.registry_path)
            for _, r in df_reg.iterrows():
                self.station_coords[str(r["station_id"])] = (float(r["latitude"]), float(r["longitude"]))

        # In-memory station weather dataframes
        self._local_weather_cache: Dict[str, pd.DataFrame] = {}

    def _load_local_era5_data(self, station_id: str) -> Optional[pd.DataFrame]:
        """Loads and indexes local ERA5-Land CSV for station_id if available."""
        if station_id in self._local_weather_cache:
            return self._local_weather_cache[station_id]

        f1 = self.weather_dir / f"{station_id}_weather.csv"
        f2 = self.weather_dir / f"{station_id}.csv"

        target_file = f1 if f1.exists() else (f2 if f2.exists() else None)
        if not target_file:
            return None

        try:
            df = pd.read_csv(target_file)
            # Parse timestamps to UTC
            time_col = "timestamp" if "timestamp" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
            if not time_col:
                return None

            df["dt_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.dropna(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)
            self._local_weather_cache[station_id] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to read local ERA5 file {target_file}: {e}")
            return None

    def fetch_open_meteo_telemetry(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches genuine historical or current weather from Open-Meteo API with local disk caching.
        Returns hourly temperature, relative humidity, wind speed, wind direction, and precipitation.
        """
        if not self.enable_api_fallback:
            return None

        lat_lon = self.station_coords.get(station_id)
        if not lat_lon:
            return None

        lat, lon = lat_lon
        cache_file = self.cache_dir / f"{station_id}_{start_date}_{end_date}.csv"
        if cache_file.exists():
            try:
                df_c = pd.read_csv(cache_file)
                df_c["dt_utc"] = pd.to_datetime(df_c["timestamp"], utc=True)
                return df_c
            except Exception:
                pass

        try:
            import urllib.request
            import json

            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
                f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,wind_direction_10m,precipitation"
                f"&timezone=UTC"
            )

            req = urllib.request.Request(url, headers={"User-Agent": "VayuDrishti-Research-Platform/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))

            hourly = payload.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                return None

            df_out = pd.DataFrame({
                "timestamp": [f"{t}:00Z" for t in times],
                "temperature_C": hourly.get("temperature_2m", [np.nan] * len(times)),
                "dewpoint_C": hourly.get("dew_point_2m", [np.nan] * len(times)),
                "relative_humidity": hourly.get("relative_humidity_2m", [np.nan] * len(times)),
                "wind_speed": hourly.get("wind_speed_10m", [np.nan] * len(times)),
                "wind_direction": hourly.get("wind_direction_10m", [np.nan] * len(times)),
                "precipitation_mm": hourly.get("precipitation", [0.0] * len(times)),
            })
            df_out.to_csv(cache_file, index=False)
            df_out["dt_utc"] = pd.to_datetime(df_out["timestamp"], utc=True)
            return df_out
        except Exception as e:
            logger.info(f"Open-Meteo API query skipped or unavailable ({e}); using offline ERA5 archive.")
            return None

    def get_aligned_weather_features(
        self,
        station_id: str,
        issue_time_utc: Union[str, datetime],
        window_hours: int = 24,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves a contiguous hourly sequence of weather features strictly <= issue_time_utc.
        Zero future lookahead: observations at t+1 or beyond are completely excluded.

        Args:
            station_id: Station ID.
            issue_time_utc: Anchor forecast issue timestamp t (UTC).
            window_hours: Number of trailing hours [t - window_hours + 1, t]. Default 24.

        Returns:
            DataFrame of shape [window_hours, 6] with canonical weather columns, or None if unavailable.
        """
        dt_anchor = pd.to_datetime(issue_time_utc, utc=True).floor("h")

        # Required hourly window: [t - 23h, t]
        target_hours = [dt_anchor - timedelta(hours=window_hours - 1 - i) for i in range(window_hours)]

        # Check local ERA5 data
        df_local = self._load_local_era5_data(station_id)
        if df_local is not None and not df_local.empty:
            min_local = df_local["dt_utc"].min()
            max_local = df_local["dt_utc"].max()
            if target_hours[0] >= min_local and target_hours[-1] <= max_local:
                # Local ERA5 covers the window
                sub = df_local[(df_local["dt_utc"] >= target_hours[0]) & (df_local["dt_utc"] <= target_hours[-1])]
                if len(sub) == window_hours:
                    cols = [c for c in self.CANONICAL_WEATHER_COLUMNS if c in sub.columns]
                    res = sub[cols].copy().reset_index(drop=True)
                    return res

        # Try Open-Meteo cache / query for recent dates
        start_str = target_hours[0].strftime("%Y-%m-%d")
        end_str = (target_hours[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
        df_om = self.fetch_open_meteo_telemetry(station_id, start_str, end_str)
        if df_om is not None and not df_om.empty:
            sub = df_om[(df_om["dt_utc"] >= target_hours[0]) & (df_om["dt_utc"] <= target_hours[-1])]
            if len(sub) == window_hours:
                cols = [c for c in self.CANONICAL_WEATHER_COLUMNS if c in sub.columns]
                return sub[cols].copy().reset_index(drop=True)

        return None

    def enrich_sensor_sequence_with_meteo(
        self,
        station_id: str,
        df_sensor_24h: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges genuine meteorological observations with 24-hour particulate telemetry.
        If weather is unavailable for a timestamp, preserves original values without fabrication.
        """
        df_out = df_sensor_24h.copy()
        if "Datetime" not in df_out.columns or df_out.empty:
            return df_out

        anchor_time = pd.to_datetime(df_out["Datetime"].iloc[-1], utc=True)
        weather_df = self.get_aligned_weather_features(station_id, anchor_time, window_hours=len(df_out))
        if weather_df is not None and len(weather_df) == len(df_out):
            for col in self.CANONICAL_WEATHER_COLUMNS:
                if col in weather_df.columns:
                    df_out[col] = weather_df[col].values
            df_out["meteo_source"] = "genuine_aligned_era5_or_open_meteo"
        else:
            df_out["meteo_source"] = "unobserved_sensor_only"

        return df_out
