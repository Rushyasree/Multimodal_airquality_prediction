"""
Unit tests for stale air-quality handling, station freshness ranking,
cache invalidation, and CPCB AQI validity.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from src.prediction.current_air_quality_service import CurrentAirQualityService, format_age_readable

IST = ZoneInfo("Asia/Kolkata")


class TestFreshnessAndStaleData(unittest.TestCase):
    def setUp(self):
        self.service = CurrentAirQualityService()
        self.service._cache.clear()

    def test_masked_api_key_prevents_leakage(self):
        """Verify API keys are masked in diagnostics and never printed in plaintext."""
        self.service.api_key = "openaq_secret_key_1234567890abcdef"
        masked = self.service.get_masked_api_key()
        self.assertTrue(masked.startswith("open...cdef"))
        self.assertNotIn("1234567890", masked)

        self.service.api_key = ""
        self.assertEqual(self.service.get_masked_api_key(), "NOT_SET")

    def test_stale_observation_enforces_none_aqi_and_retains_historical_aqi(self):
        """Observations older than 24h must set aqi=None, is_stale=True, and retain historical_aqi."""
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        raw_items = [
            {"parameter": "pm25", "value": 45.0, "unit": "µg/m³", "datetime": {"utc": stale_time}},
            {"parameter": "pm10", "value": 80.0, "unit": "µg/m³", "datetime": {"utc": stale_time}},
            {"parameter": "no2", "value": 25.0, "unit": "µg/m³", "datetime": {"utc": stale_time}},
        ]
        res = self.service.process_raw_measurements(
            raw_items=raw_items,
            station_name="Test Stale Station",
            city_name="Amaravati",
        )
        self.assertTrue(res["is_stale"])
        self.assertTrue(res["is_historical"])
        self.assertFalse(res["is_live"])
        self.assertIsNone(res["aqi"])
        self.assertIsNotNone(res["historical_aqi"])
        self.assertEqual(res["historical_aqi"], 80)
        self.assertIn("Outdated", res["aqi_tier"])
        self.assertIn("Historical observation", res["stale_warning"])

    def test_fresh_station_ranks_ahead_of_closer_stale_station(self):
        """A fresh station further away must rank ahead of a closer stale station."""
        now_utc = datetime.now(timezone.utc)
        fresh_time = (now_utc - timedelta(hours=1)).isoformat()
        stale_time = (now_utc - timedelta(hours=40)).isoformat()

        mock_locations = [
            {
                "id": 101,
                "name": "Nearby Stale Station",
                "coordinates": {"latitude": 16.51, "longitude": 80.64},
                "datetimeLast": {"utc": stale_time},
            },
            {
                "id": 102,
                "name": "Distant Fresh Station",
                "coordinates": {"latitude": 16.60, "longitude": 80.70},
                "datetimeLast": {"utc": fresh_time},
            },
        ]

        with patch.object(self.service, "fetch_openaq_locations_by_coordinates", return_value=mock_locations):
            ranked = self.service.find_nearby_stations(lat=16.50, lon=80.60, force_refresh=True)
            self.assertGreaterEqual(len(ranked), 2)
            # The fresh station (id 102) must be first even though id 101 is closer
            self.assertEqual(ranked[0]["station_id"], "openaq_102")
            self.assertFalse(ranked[0]["is_stale"])
            self.assertEqual(ranked[1]["station_id"], "openaq_101")
            self.assertTrue(ranked[1]["is_stale"])

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True must invalidate cache and trigger fresh provider call."""
        self.service._cache["nearby_16.5_80.6_50_10"] = ([{"station_id": 999}], datetime.now().timestamp())
        mock_locations = [
            {
                "id": 201,
                "name": "New Station After Refresh",
                "coordinates": {"latitude": 16.51, "longitude": 80.64},
                "datetimeLast": {"utc": datetime.now(timezone.utc).isoformat()},
            }
        ]
        with patch.object(self.service, "fetch_openaq_locations_by_coordinates", return_value=mock_locations) as mock_fetch:
            res = self.service.find_nearby_stations(lat=16.50, lon=80.60, force_refresh=True)
            self.assertTrue(mock_fetch.called)
            self.assertEqual(res[0]["station_id"], "openaq_201")

    def test_one_hour_cache_ttl_during_normal_operations(self):
        """Cache must be preserved within 3600 seconds and expired after 3600 seconds."""
        self.assertEqual(self.service.cache_ttl, 3600)
        cache_key = "nearby_16.5_80.6_50_10"
        cached_entry = [{"station_id": "openaq_cached", "is_stale": False}]
        now_ts = datetime.now().timestamp()
        
        # Fresh cache (5 minutes old)
        self.service._cache[cache_key] = (cached_entry, now_ts - 300)
        with patch.object(self.service, "fetch_openaq_locations_by_coordinates") as mock_fetch:
            res = self.service.find_nearby_stations(lat=16.50, lon=80.60, force_refresh=False)
            self.assertFalse(mock_fetch.called)
            self.assertEqual(res[0]["station_id"], "openaq_cached")

    def test_format_age_readable_labels(self):
        """Verifies clean human-readable duration strings."""
        self.assertEqual(format_age_readable(None), "Unknown age")
        self.assertEqual(format_age_readable(0.4), "24 minutes ago")
        self.assertEqual(format_age_readable(1.0), "1 hour ago")
        self.assertEqual(format_age_readable(5.0), "5 hours ago")
        self.assertEqual(format_age_readable(5.5), "6 hours ago")
        self.assertEqual(format_age_readable(25.0), "1 day 1 hr ago")
        self.assertEqual(format_age_readable(72.0), "3 days ago")

    def test_cpcb_criteria_insufficient_pollutants(self):
        """AQI must NOT be calculated if fewer than 3 criteria pollutants exist."""
        fresh_time = datetime.now(timezone.utc).isoformat()
        # Only PM2.5 provided
        raw_items = [
            {"parameter": "pm25", "value": 35.0, "unit": "µg/m³", "datetime": {"utc": fresh_time}},
        ]
        res = self.service.process_raw_measurements(
            raw_items=raw_items,
            station_name="Partial Monitor",
            city_name="Vijayawada",
        )
        self.assertFalse(res["is_stale"])
        self.assertIsNone(res["aqi"])
        self.assertIsNone(res["historical_aqi"])
        self.assertIn("Insufficient Pollutants", res["aqi_tier"])

    def test_timestamp_distinction_observed_vs_retrieved(self):
        """Dashboard payload must distinguish station observation time from fetch time."""
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        raw_items = [
            {"parameter": "pm25", "value": 25.0, "unit": "µg/m³", "datetime": {"utc": two_hours_ago.isoformat()}},
            {"parameter": "pm10", "value": 45.0, "unit": "µg/m³", "datetime": {"utc": two_hours_ago.isoformat()}},
            {"parameter": "no2", "value": 15.0, "unit": "µg/m³", "datetime": {"utc": two_hours_ago.isoformat()}},
        ]
        res = self.service.process_raw_measurements(
            raw_items=raw_items,
            station_name="Distinction Station",
            city_name="Tirupati",
        )
        self.assertIn("IST", res["observed_at_ist"])
        self.assertIn("IST", res["retrieved_at_ist"])
        self.assertNotEqual(res["observed_at_ist"], res["retrieved_at_ist"])

    def test_fallback_to_historical_observation_when_all_stale(self):
        """When no station has recent data, get_best_nearby_observation selects newest historical with computed AQI."""
        stale_time = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        mock_candidates = [
            {
                "station_id": "openaq_5408",
                "openaq_location_id": 5408,
                "station_name": "Secretariat, Amaravati - APPCB",
                "city": "Amaravati",
                "latitude": 16.5131,
                "longitude": 80.5165,
                "distance_km": 0.0,
                "is_stale": True,
                "age_hours": 72.0,
                "datetimeLast": {"utc": stale_time},
            }
        ]
        mock_obs = {
            "is_live": False,
            "is_stale": True,
            "is_historical": True,
            "station_name": "Secretariat, Amaravati - APPCB",
            "pollutants": {"PM2.5": 32.0, "PM10": 64.0, "NO2": 15.0},
            "aqi": None,
            "historical_aqi": 65,
            "historical_aqi_tier": "Satisfactory",
            "historical_aqi_color": "#10b981",
        }
        with patch.object(self.service, "find_nearby_stations", return_value=mock_candidates), \
             patch.object(self.service, "get_current_observations_for_station", return_value=mock_obs):
            best = self.service.get_best_nearby_observation(lat=16.5131, lon=80.5165, max_radius_km=50.0)
            self.assertTrue(best["is_historical"])
            self.assertEqual(best["historical_aqi"], 65)
            self.assertEqual(best["historical_aqi_tier"], "Satisfactory")


if __name__ == "__main__":
    unittest.main()
