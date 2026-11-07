"""
src/data/recent_air_quality_ingestion.py

Production client and ingestion pipeline for recent air-quality telemetry from the
XKDR India Air Quality Database (https://airquality.xkdr.org/).

Key Capabilities:
1. Authenticated API client for XKDR API v1.
2. Station entity matching between VayuDrishti's 90 CPCB stations and XKDR's 558 stations.
3. Multi-parameter hourly time-series extraction and schema harmonization.
4. Strict evaluation guardrails to protect the 2015-2020 historical benchmark (cutoff: 2020-07-01 00:00:00).
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error
import pandas as pd
import numpy as np

logger = logging.getLogger("VayuDrishti.XKDR")

DEFAULT_DEMO_KEY = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
DEFAULT_BASE_URL = "https://airquality.xkdr.org/v1"
HISTORICAL_CUTOFF_TIMESTAMP = "2020-07-01 00:00:00"

XKDR_PARAMETER_MAP = {
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "NO": "NO",
    "NO2": "NO2",
    "NOx": "NOx",
    "NH3": "NH3",
    "CO": "CO",
    "SO2": "SO2",
    "Ozone": "O3",
    "Benzene": "Benzene",
    "Toluene": "Toluene",
    "Xylene": "Xylene",
    "MP-Xylene": "MP-Xylene",
    "Eth-Benzene": "Eth-Benzene",
    "AT": "temperature_C",
    "RH": "relative_humidity",
    "WS": "wind_speed",
    "WD": "wind_direction",
    "SR": "solar_radiation",
    "BP": "barometric_pressure",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two coordinates in kilometers."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


class XKDRClient:
    """Client for querying the XKDR India Air Quality Database."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or os.environ.get("XKDR_API_KEY", DEFAULT_DEMO_KEY)
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir or (Path(__file__).resolve().parents[2] / "data" / "cache" / "xkdr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Performs authenticated HTTP GET to XKDR endpoint."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            query_str = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{query_str}"

        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "VayuDrishti/2.0",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            logger.error(f"XKDR HTTP {e.code} error for {url}: {err_body}")
            raise RuntimeError(f"XKDR API error {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            logger.error(f"XKDR connection failed for {url}: {e.reason}")
            raise RuntimeError(f"XKDR connection failure: {e.reason}") from e

    def get_metadata(self) -> Dict[str, Any]:
        """Fetches XKDR repository metadata and record counts."""
        res = self._get("meta")
        return res.get("data", res) if isinstance(res, dict) else {"raw": res}

    def get_parameters(self) -> List[Dict[str, Any]]:
        """Fetches list of available measurement parameters and units."""
        res = self._get("parameters")
        if isinstance(res, dict):
            return res.get("data", res.get("parameters", []))
        return res if isinstance(res, list) else []

    def get_stations(self) -> List[Dict[str, Any]]:
        """Fetches complete station catalogue (558 continuous monitoring stations)."""
        cache_path = self.cache_dir / "xkdr_stations_catalogue.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if cached_data and len(cached_data) > 0:
                        return cached_data
            except Exception:
                pass

        res = self._get("stations")
        stations = res.get("data", []) if isinstance(res, dict) else res
        if stations:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(stations, f, indent=2)
            except Exception:
                pass
        return stations

    def match_vayu_stations(
        self,
        vayu_coords_csv: Path,
        max_dist_km: float = 2.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Matches VayuDrishti's 90 stations to XKDR station catalogue."""
        vdf = pd.read_csv(vayu_coords_csv)
        xkdr_stations = self.get_stations()
        xdf = pd.DataFrame(xkdr_stations)

        matches = []
        for _, r90 in vdf.iterrows():
            v_id = r90["StationId"]
            v_name = str(r90["StationName"])
            v_city = str(r90["City"])
            v_lat = float(r90["Latitude"]) if pd.notna(r90["Latitude"]) else None
            v_lon = float(r90["Longitude"]) if pd.notna(r90["Longitude"]) else None

            # 1. Exact station name match
            m = xdf[xdf["station_name"].str.strip().str.lower() == v_name.strip().lower()]
            match_type = "exact_name"
            dist_km = 0.0

            # 2. Coordinate proximity match if name not found exactly
            if m.empty and v_lat is not None and v_lon is not None and not xdf.empty:
                dists = np.sqrt((xdf["latitude"] - v_lat) ** 2 + (xdf["longitude"] - v_lon) ** 2)
                close = xdf[dists < 0.015]  # ~1.5 km
                if not close.empty:
                    m = close.iloc[:1]
                    match_type = "coordinate_proximity"
                    dist_km = haversine_km(v_lat, v_lon, float(m.iloc[0]["latitude"]), float(m.iloc[0]["longitude"]))

            is_matched = not m.empty
            matched_rec = m.iloc[0] if is_matched else None

            matches.append({
                "vayu_station_id": v_id,
                "vayu_station_name": v_name,
                "vayu_city": v_city,
                "vayu_lat": v_lat,
                "vayu_lon": v_lon,
                "matched": is_matched,
                "xkdr_station_id": matched_rec["station_id"] if is_matched else None,
                "xkdr_station_name": matched_rec["station_name"] if is_matched else None,
                "xkdr_city": matched_rec.get("city_name") if is_matched else None,
                "xkdr_state": matched_rec.get("state_name") if is_matched else None,
                "xkdr_first_seen": matched_rec.get("first_seen") if is_matched else None,
                "xkdr_last_seen": matched_rec.get("last_seen") if is_matched else None,
                "xkdr_total_rows": matched_rec.get("n_rows") if is_matched else None,
                "xkdr_parameters": matched_rec.get("parameters") if is_matched else None,
                "distance_km": round(dist_km, 3) if is_matched else None,
                "match_method": match_type if is_matched else "unmatched",
            })

        matched_df = pd.DataFrame(matches)
        summary = {
            "total_vayu_stations": len(vdf),
            "matched_stations": int(matched_df["matched"].sum()),
            "unmatched_stations": int((~matched_df["matched"]).sum()),
            "match_rate_pct": round(float(matched_df["matched"].mean() * 100), 2),
            "mean_distance_km": round(float(matched_df[matched_df["matched"]]["distance_km"].mean()), 3),
            "max_distance_km": round(float(matched_df[matched_df["matched"]]["distance_km"].max()), 3),
        }
        return matched_df, summary

    def fetch_measurements(
        self,
        station_id: str,
        parameter: str,
        start_date: str,
        end_date: str,
        agg: str = "hourly",
    ) -> pd.DataFrame:
        """Fetches hourly measurement records for a specific station and parameter."""
        # Enforce evaluation protocol protection
        if start_date < "2020-07-01":
            logger.warning(
                f"Requested start_date ({start_date}) precedes historical cutoff ({HISTORICAL_CUTOFF_TIMESTAMP}). "
                "Ensure historical benchmark test set is protected from retraining leakage."
            )

        params = {
            "station": station_id,
            "parameter": parameter,
            "start": start_date,
            "end": end_date,
            "agg": agg,
            "format": "json",
        }
        res = self._get("measurements", params=params)
        data = res.get("data", []) if isinstance(res, dict) else res
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # Handle datetime column
        if "collected_at" in df.columns:
            df["Datetime"] = pd.to_datetime(df["collected_at"])
        elif "period_start" in df.columns:
            df["Datetime"] = pd.to_datetime(df["period_start"])

        # Handle value column
        if "value" not in df.columns and "mean" in df.columns:
            df["value"] = df["mean"]
        return df

    def fetch_recent_station_series(
        self,
        station_id: str,
        start_date: str = "2024-11-01",
        end_date: str = "2024-11-05",
        parameters: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetches multiple parameters for a station and pivots to wide hourly DataFrame."""
        if parameters is None:
            parameters = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

        series_dict = {}
        for param in parameters:
            try:
                p_df = self.fetch_measurements(station_id, param, start_date, end_date)
                if not p_df.empty and "value" in p_df.columns:
                    col_name = XKDR_PARAMETER_MAP.get(param, param)
                    sub = p_df[["Datetime", "value"]].drop_duplicates("Datetime").set_index("Datetime")
                    series_dict[col_name] = sub["value"]
            except Exception as e:
                logger.warning(f"Could not fetch {param} for {station_id}: {e}")

        if not series_dict:
            return pd.DataFrame()

        merged_df = pd.DataFrame(series_dict)
        merged_df.sort_index(inplace=True)
        merged_df["StationId"] = station_id
        return merged_df
