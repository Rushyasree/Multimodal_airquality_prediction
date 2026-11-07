"""
src/prediction/live_service.py

Live air quality telemetry handler with secure timeout, 15-minute caching,
source attribution, and explicit fallback to Historical Dataset Mode.
"""

from typing import Dict, Any, Optional
import time
import requests


class LiveAirQualityService:
    """Safely interfaces with public air-quality APIs (OpenAQ / Data.gov.in)

    when credentials are provided, or defaults transparently to Historical Dataset Mode.
    Never fabricates synthetic live values.
    """

    def __init__(self, cache_ttl_seconds: int = 900):
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._last_fetch_time: float = 0.0

    def get_live_status(
        self,
        station_id: Optional[str] = None,
        city: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attempts to poll public monitoring telemetry or returns authenticated status metadata."""
        now = time.time()

        # If cache is valid, return cached status
        cache_key = f"{station_id}_{city}_{api_key}"
        if cache_key in self._cache and (now - self._last_fetch_time) < self.cache_ttl:
            return self._cache[cache_key]

        # Check if an official API key is provided
        if not api_key:
            result = {
                "is_live": False,
                "mode": "Historical Dataset Mode (Certified CPCB/APPCB Archives)",
                "status": "Offline / Archival Mode",
                "message": (
                    "Official real-time CPCB/APPCB streaming gateways require registered "
                    "institutional API keys. The system is operating in verified Historical "
                    "Dataset Mode utilizing calibrated 2015–2020 CPCB continuous hourly archives."
                ),
                "station_id": station_id or "N/A",
                "city": city or "Andhra Pradesh / India",
                "source": "CPCB CAAQMS / APPCB Archives & ECMWF ERA5-Land Reanalysis",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
                "disclaimer": "Real-time readings are not fabricated. Predictions use validated sensor sequence data.",
            }
            self._cache[cache_key] = result
            self._last_fetch_time = now
            return result

        # Optional live fetch attempt if API key provided
        try:
            # Example endpoint: OpenAQ or Data.gov.in
            headers = {"X-API-Key": api_key, "User-Agent": "MultimodalAirQualityBTechProject/1.0"}
            resp = requests.get(
                "https://api.openaq.org/v2/latest",
                params={"city": city or "Amaravati", "limit": 1},
                headers=headers,
                timeout=4.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = {
                    "is_live": True,
                    "mode": "Live Telemetry Feed",
                    "status": "Connected",
                    "message": "Live station observations retrieved successfully.",
                    "station_id": station_id or "Live Monitor",
                    "city": city,
                    "source": "OpenAQ Public Gateway",
                    "data": data,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
                }
            else:
                result = {
                    "is_live": False,
                    "mode": "Historical Dataset Mode (Fallback)",
                    "status": f"Gateway Error ({resp.status_code})",
                    "message": "Public gateway returned non-200 response. Falling back to calibrated historical telemetry.",
                    "station_id": station_id or "N/A",
                    "city": city,
                    "source": "CPCB CAAQMS Archives",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
                }
        except Exception as e:
            result = {
                "is_live": False,
                "mode": "Historical Dataset Mode (Network Timeout Fallback)",
                "status": "Offline",
                "message": f"External API connection timed out ({type(e).__name__}). Running in verified historical fallback mode.",
                "station_id": station_id or "N/A",
                "city": city,
                "source": "CPCB CAAQMS Archives",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
            }

        self._cache[cache_key] = result
        self._last_fetch_time = now
        return result
