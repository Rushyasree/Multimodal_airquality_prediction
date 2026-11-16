"""
src/services/city_registry_service.py

Centralized Urban Location and Station Coverage Registry for Phase 15.
Manages 31 Indian urban locations across South, West, North, East, and Central India.
Integrates verified CPCB, APPCB, and OpenAQ monitoring station metadata,
deduplicating station identifiers and enforcing sparse-coverage safeguards.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.config.spatial_config import (
    CityBoundingBox,
    SUPPORTED_CITIES,
)

logger = logging.getLogger("VayuDrishti.CityRegistryService")


class CityRegistryService:
    """
    Centralized registry for Indian urban locations and verified physical monitoring stations.
    Prevents city hardcoding across dashboard pages and guarantees data-driven station matching.
    """

    _instance: Optional["CityRegistryService"] = None

    def __init__(self):
        self._project_root = Path(__file__).resolve().parents[2]
        self._cities: Dict[str, CityBoundingBox] = dict(SUPPORTED_CITIES)
        self._stations_cache: Optional[pd.DataFrame] = None
        self._city_stations_map: Dict[str, List[Dict[str, Any]]] = {}
        self._load_and_index_stations()

    @classmethod
    def get_instance(cls) -> "CityRegistryService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_and_index_stations(self):
        """Loads and deduplicates all verified CAAQMS monitoring stations from available datasets."""
        records: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1. Load XKDR official CPCB CAAQMS catalogue (558 stations)
        xkdr_path = self._project_root / "data/cache/xkdr/xkdr_stations_catalogue.json"
        if xkdr_path.exists():
            try:
                with open(xkdr_path, "r", encoding="utf-8") as f:
                    xkdr_data = json.load(f)
                for item in xkdr_data:
                    sid = str(item.get("station_id", "")).strip().upper()
                    raw_lat = item.get("latitude")
                    raw_lon = item.get("longitude")
                    if raw_lat is None or raw_lon is None:
                        continue
                    try:
                        lat = float(raw_lat)
                        lon = float(raw_lon)
                    except (ValueError, TypeError):
                        continue

                    if sid and sid not in seen_ids:
                        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                            seen_ids.add(sid)
                            records.append({
                                "station_id": sid,
                                "station_name": str(item.get("station_name", sid)),
                                "city": str(item.get("city_name", "")),
                                "state": str(item.get("state_name", "")),
                                "latitude": round(lat, 5),
                                "longitude": round(lon, 5),
                                "source": "CPCB_CAAQMS_XKDR",
                            })
            except Exception as e:
                logger.warning(f"Error loading xkdr catalogue: {e}")

        # 2. Supplement from location_catalog.csv (local ground-truth & AP stations)
        loc_path = self._project_root / "data/processed/location_catalog.csv"
        if loc_path.exists():
            try:
                df_loc = pd.read_csv(loc_path)
                for _, r in df_loc.iterrows():
                    sid = str(r.get("station_id", "")).strip().upper()
                    raw_lat = r.get("latitude")
                    raw_lon = r.get("longitude")
                    if raw_lat is None or raw_lon is None or pd.isna(raw_lat) or pd.isna(raw_lon):
                        continue
                    try:
                        lat = float(raw_lat)
                        lon = float(raw_lon)
                    except (ValueError, TypeError):
                        continue

                    if sid and sid != "NAN" and sid not in seen_ids:
                        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                            seen_ids.add(sid)
                            records.append({
                                "station_id": sid,
                                "station_name": str(r.get("station_name", sid)),
                                "city": str(r.get("city", "")),
                                "state": str(r.get("state", "")),
                                "latitude": round(lat, 5),
                                "longitude": round(lon, 5),
                                "source": "VAYUDRISHTI_LOCATION_CATALOG",
                            })
            except Exception as e:
                logger.warning(f"Error loading location catalog: {e}")

        self._stations_cache = pd.DataFrame(records)

        # 3. Pre-index stations for each registered city
        for city_id, bbox in self._cities.items():
            matched = self._match_stations_for_bbox(bbox)
            self._city_stations_map[city_id] = matched

    def _match_stations_for_bbox(self, bbox: CityBoundingBox) -> List[Dict[str, Any]]:
        """Matches physical stations strictly located within or directly adjacent to city bounds."""
        if self._stations_cache is None or self._stations_cache.empty:
            return []

        pad = 0.04  # ~4.4 km buffer for boundary monitoring facilities
        in_bbox = self._stations_cache[
            (self._stations_cache["latitude"] >= bbox.min_lat - pad)
            & (self._stations_cache["latitude"] <= bbox.max_lat + pad)
            & (self._stations_cache["longitude"] >= bbox.min_lon - pad)
            & (self._stations_cache["longitude"] <= bbox.max_lon + pad)
        ].copy()

        # Filter or match by city / municipal name if appropriate
        name_root = bbox.name.split()[0].strip()
        by_name = self._stations_cache[
            self._stations_cache["city"].str.contains(name_root, case=False, na=False)
            & self._stations_cache["state"].str.contains(bbox.primary_state.split()[0], case=False, na=False)
        ].copy()

        combined = pd.concat([in_bbox, by_name]).drop_duplicates(subset=["station_id"])
        
        # Ensure coordinates are within reasonable regional distance to city center (< 50 km)
        valid_stations = []
        for _, row in combined.iterrows():
            d_km = np.sqrt(
                ((row["latitude"] - bbox.center_lat) * 111.0) ** 2
                + ((row["longitude"] - bbox.center_lon) * 111.0 * np.cos(np.radians(bbox.center_lat))) ** 2
            )
            if d_km <= 50.0:
                valid_stations.append({
                    "station_id": row["station_id"],
                    "station_name": row["station_name"],
                    "city": row["city"],
                    "state": row["state"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "source": row["source"],
                    "distance_to_center_km": round(float(d_km), 2),
                })

        return sorted(valid_stations, key=lambda s: s["distance_to_center_km"])

    def get_city(self, city_id: str) -> Optional[CityBoundingBox]:
        """Retrieves configuration for a specific city ID."""
        return self._cities.get(city_id.lower().strip())

    def get_all_cities(self) -> List[CityBoundingBox]:
        """Returns all registered urban locations."""
        return list(self._cities.values())

    def get_all_city_ids(self) -> List[str]:
        """Returns all registered city IDs."""
        return list(self._cities.keys())

    def get_cities_by_region(self) -> Dict[str, List[CityBoundingBox]]:
        """Groups all registered cities by geographical macro-region."""
        grouped: Dict[str, List[CityBoundingBox]] = {
            "South": [],
            "West": [],
            "North": [],
            "East": [],
            "Central": [],
        }
        for city in self._cities.values():
            reg = city.region
            if reg not in grouped:
                grouped[reg] = []
            grouped[reg].append(city)
        return grouped

    def get_stations_for_city(self, city_id: str) -> List[Dict[str, Any]]:
        """Returns verified physical monitoring stations for the given city."""
        clean_id = city_id.lower().strip()
        return self._city_stations_map.get(clean_id, [])

    def get_station_count(self, city_id: str) -> int:
        """Returns the number of verified physical monitoring stations in the city."""
        return len(self.get_stations_for_city(city_id))

    def is_interpolation_supported(self, city_id: str) -> bool:
        """
        Determines whether 2D spatial interpolation is scientifically supported.
        Enforces sparse coverage safeguards: cities with < 3 stations cannot produce
        a continuous spatial interpolation surface.
        """
        city = self.get_city(city_id)
        if not city:
            return False
        st_count = self.get_station_count(city_id)
        return st_count >= city.min_stations_required

    def get_validation_status(self, city_id: str) -> str:
        """Returns the scientific validation status of spatial estimates for the city."""
        city = self.get_city(city_id)
        if not city:
            return "UNKNOWN"
        st_count = self.get_station_count(city_id)
        if st_count < city.min_stations_required:
            return "INSUFFICIENT_COVERAGE"
        if city.city_id == "delhi_ncr":
            return "VALIDATED_CONFORMAL"
        return "EXPLORATORY_SUFFICIENT"

    def get_monitoring_density(self, city_id: str) -> str:
        """Categorizes monitoring density based on active physical stations."""
        st_count = self.get_station_count(city_id)
        if st_count >= 10:
            return "Dense"
        elif st_count >= 4:
            return "Moderate"
        elif st_count >= 1:
            return "Sparse"
        else:
            return "Insufficient"

    def get_coverage_summary(self) -> pd.DataFrame:
        """Generates an audit dataframe of all registered cities with station counts and status."""
        rows = []
        for city in self.get_all_cities():
            st_count = self.get_station_count(city.city_id)
            val_status = self.get_validation_status(city.city_id)
            density = self.get_monitoring_density(city.city_id)
            is_supp = self.is_interpolation_supported(city.city_id)
            rows.append({
                "city_id": city.city_id,
                "name": city.name,
                "region": city.region,
                "state": city.primary_state,
                "center_lat": city.center_lat,
                "center_lon": city.center_lon,
                "min_lat": city.min_lat,
                "max_lat": city.max_lat,
                "min_lon": city.min_lon,
                "max_lon": city.max_lon,
                "station_count": st_count,
                "interpolation_supported": is_supp,
                "validation_status": val_status,
                "density_category": density,
            })
        return pd.DataFrame(rows)


def get_city_registry_service() -> CityRegistryService:
    """Singleton provider for CityRegistryService."""
    return CityRegistryService.get_instance()
