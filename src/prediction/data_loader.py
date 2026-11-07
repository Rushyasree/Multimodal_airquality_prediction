"""
src/prediction/data_loader.py

Data utilities for dashboard visualization, Andhra Pradesh location exploration,
cascading geographical filters, and metadata management.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_location_catalog() -> pd.DataFrame:
    """Loads the unified location catalog containing 90 CAAQMS monitoring stations

    and 13 verified Andhra Pradesh municipal city-center locations.
    """
    root = get_project_root()
    catalog_path = root / "data/processed/location_catalog.csv"

    if not catalog_path.exists():
        # Fallback to generating it dynamically if not found
        from scratch.generate_location_catalog import output_path  # type: ignore
        catalog_path = output_path

    df = pd.read_csv(catalog_path)
    df["station_id"] = df["station_id"].fillna("").astype(str).str.strip().str.upper()
    df["data_available"] = df["data_available"].astype(bool)
    df["forecast_supported"] = df["forecast_supported"].astype(bool)
    return df


def load_stations_metadata() -> pd.DataFrame:
    """Loads station metadata and coordinates for all 90 CAAQMS monitoring stations."""
    catalog = load_location_catalog()
    stations = catalog[catalog["location_type"] == "cpcb_station"].copy()
    stations = stations.rename(columns={
        "station_id": "StationId",
        "station_name": "StationName",
        "city": "City",
        "state": "State",
        "latitude": "Latitude",
        "longitude": "Longitude",
    })
    return stations


def get_states_list() -> List[str]:
    """Returns sorted list of states with 'Andhra Pradesh' featured at the top."""
    catalog = load_location_catalog()
    states = sorted(catalog["state"].unique().tolist())
    if "Andhra Pradesh" in states:
        states.remove("Andhra Pradesh")
        states.insert(0, "Andhra Pradesh")
    return states


def get_cities_by_state(state: str) -> List[str]:
    """Returns sorted list of cities for the selected state."""
    catalog = load_location_catalog()
    cities = sorted(catalog[catalog["state"] == state]["city"].unique().tolist())
    return cities


def get_locations_for_city(state: str, city: str) -> pd.DataFrame:
    """Returns catalog rows for the specified state and city."""
    catalog = load_location_catalog()
    return catalog[(catalog["state"] == state) & (catalog["city"] == city)].copy()


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance in kilometers between two coordinate pairs."""
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def find_nearest_supported_station(lat: float, lon: float) -> Dict[str, Any]:
    """Finds the closest operational CAAQMS monitoring station to a given coordinate."""
    catalog = load_location_catalog()
    supported = catalog[catalog["forecast_supported"]].reset_index(drop=True)

    if supported.empty:
        raise ValueError("No supported stations found in location catalog.")

    distances = [
        haversine_distance(lat, lon, r["latitude"], r["longitude"])
        for _, r in supported.iterrows()
    ]
    min_idx = int(np.argmin(distances))
    nearest = supported.iloc[min_idx]

    return {
        "station_id": nearest["station_id"],
        "station_name": nearest["station_name"],
        "city": nearest["city"],
        "state": nearest["state"],
        "latitude": float(nearest["latitude"]),
        "longitude": float(nearest["longitude"]),
        "distance_km": round(distances[min_idx], 1),
    }


def load_verified_results() -> pd.DataFrame:
    """Loads the verified benchmark metrics table across all three modeling paradigms."""
    root = get_project_root()
    csv_path = root / "results/multimodal/verified_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Verified results table missing: {csv_path}")
    return pd.read_csv(csv_path)


def load_demo_scenarios() -> pd.DataFrame:
    """Loads pre-extracted 24-hour test scenarios for interactive dashboard demonstration."""
    root = get_project_root()
    demo_path = root / "data/processed/dashboard_demo_sequences.csv"
    if not demo_path.exists():
        return pd.DataFrame()
    return pd.read_csv(demo_path)


def get_scenario_metadata_list(state: Optional[str] = None, city: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns a list of scenario descriptors for dropdown selection in the UI,

    optionally filtered by state or city.
    """
    df = load_demo_scenarios()
    if df.empty:
        return []

    stations_df = load_stations_metadata().set_index("StationId")
    scenarios = []

    for sc_id, group in df.groupby("scenario_id", sort=False):
        first = group.iloc[0]
        sid = first["StationId"]
        st_name = stations_df.loc[sid, "StationName"] if sid in stations_df.index else sid
        st_city = stations_df.loc[sid, "City"] if sid in stations_df.index else "India"
        st_state = stations_df.loc[sid, "State"] if sid in stations_df.index else "India"
        target_time = str(first.get("target_time", ""))
        target_pm25 = float(first.get("target_PM2.5", 0.0))
        target_pm10 = float(first.get("target_PM10", 0.0))

        if state and st_state != state:
            continue
        if city and st_city != city:
            continue

        label = (
            f"[{sid}] {st_city} - {st_name} | Target: {target_time} "
            f"(True PM2.5: {target_pm25:.1f}, PM10: {target_pm10:.1f})"
        )

        scenarios.append({
            "id": sc_id,
            "label": label,
            "station_id": sid,
            "city": st_city,
            "state": st_state,
            "station_name": st_name,
            "target_time": target_time,
            "actual_pm25": target_pm25,
            "actual_pm10": target_pm10,
        })

    return scenarios
