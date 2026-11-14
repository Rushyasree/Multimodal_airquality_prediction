"""
src/services/live_air_quality_service.py

Unified backend service for retrieving actual measured air-quality observations
from official/curated sources (CPCB, OpenAQ, or cached telemetry).
Live measurements are observations, not forecasts.
Integrates ResilientDataIngestionManager for exponential backoff, rate limiting, and cache freshness.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
from src.prediction.current_air_quality_service import CurrentAirQualityService
from src.prediction.data_loader import load_location_catalog
from src.data.resilient_ingestion import get_resilient_ingestion_manager
from src.services.operational_health_service import get_operational_health_service

logger = logging.getLogger("VayuDrishti.LiveService")


class LiveAirQualityService:
    """Service providing verified ground station observations with provenance, resilience, and freshness."""

    def __init__(self):
        self.underlying_service = CurrentAirQualityService()
        self.catalog = load_location_catalog()
        self.ingestion_mgr = get_resilient_ingestion_manager()
        self.health_service = get_operational_health_service()

    def get_recent_hourly_sequence(
        self,
        station_id: Optional[str] = None,
        city: Optional[str] = None,
        past_hours: int = 24,
        force_refresh: bool = False,
    ) -> Optional[Any]:
        """Retrieves recent continuous observation sequence in Asia/Kolkata (IST)."""
        target_city = city or "Amaravati"
        clean_sid = str(station_id or "AP001").strip().upper()
        if station_id:
            match = self.catalog[self.catalog["station_id"] == station_id]
            if not match.empty:
                target_city = match.iloc[0]["city"]
        return self.underlying_service.get_recent_hourly_observations(
            station_id=clean_sid,
            city=target_city,
            past_hours=past_hours,
            force_refresh=force_refresh,
        )

    def get_live_observation(
        self,
        station_id: Optional[str] = None,
        city: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Retrieves actual measured air quality with resilience. Live observations are not forecasts."""
        target_city = city or "Amaravati"
        clean_sid = str(station_id or "AP001").strip().upper()

        if station_id:
            # Match station in catalog to find city if needed
            match = self.catalog[self.catalog["station_id"] == station_id]
            if not match.empty:
                target_city = match.iloc[0]["city"]

        def _fetch_from_provider(sid: str) -> Dict[str, Any]:
            return self.underlying_service.get_current_observations(
                city=target_city,
                force_refresh=force_refresh,
            )

        # Ingest through ResilientDataIngestionManager
        obs = self.ingestion_mgr.get_station_observation(
            station_id=clean_sid,
            fetcher_fn=_fetch_from_provider,
            force_refresh=force_refresh,
        )

        measurements = obs.get("pollutants") or obs.get("measurements", {})
        pm25_val = measurements.get("PM2.5") if isinstance(measurements, dict) else obs.get("pm25")
        pm10_val = measurements.get("PM10") if isinstance(measurements, dict) else obs.get("pm10")

        # Determine freshness status and age
        obs_time = obs.get("observation_time_ist") or obs.get("observation_time") or str(datetime.now())
        freshness_status = obs.get("freshness_status", "Fresh")
        is_cached = obs.get("is_cached", False)
        cache_age = obs.get("cache_age_seconds", 0.0)

        # Update station operational health
        self.health_service.evaluate_station_health(
            station_id=clean_sid,
            observation_timestamp=obs_time,
            has_pm25=bool(pm25_val is not None),
            has_pm10=bool(pm10_val is not None),
            data_source=obs.get("data_source", "CPCB / OpenAQ"),
        )

        return {
            "result_type": "Observation",
            "station_id": clean_sid,
            "station_name": obs.get("station_name", f"{target_city} Central"),
            "city": obs.get("city", target_city),
            "data_source": obs.get("data_source", "CPCB / OpenAQ"),
            "observation_time": obs_time,
            "freshness_status": freshness_status,
            "is_live": obs.get("is_live", True),
            "is_cached": is_cached,
            "cache_age_seconds": cache_age,
            "pm25": float(pm25_val) if pm25_val is not None else None,
            "pm10": float(pm10_val) if pm10_val is not None else None,
            "other_pollutants": {
                k: float(v) for k, v in measurements.items()
                if k not in ["PM2.5", "PM10"] and v is not None and isinstance(v, (int, float, str))
            } if isinstance(measurements, dict) else {},
            "weather": obs.get("weather", {}),
            "unit": "ug/m3",
            "scientific_note": "Live measurements are actual recorded observations, not model forecasts.",
        }
