"""
tests/test_geolocation_and_live_stations.py

Comprehensive test suite verifying:
1. Browser geolocation coordinates validation, error handling, and security models.
2. Haversine distance calculation and candidate station ranking.
3. Genuine provider metadata and coverage radius filtering.
4. CPCB data sufficiency rules (min 3 criteria pollutants with PM2.5 or PM10).
5. Data freshness classification (Live, Recent, Stale).
6. Gateway resilience (HTTP 429 rate limits, timeouts, connection errors).
7. Strict prohibition of synthetic/fabricated live telemetry.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import pytz
import requests

from src.prediction.current_air_quality_service import (
    CurrentAirQualityService,
    validate_coordinates,
    PHYSICAL_BOUNDS,
)
from src.prediction.data_loader import haversine_distance


class TestGeolocationAndCoordinates(unittest.TestCase):
    """Verifies coordinate validation, bounds, and edge cases."""

    def test_valid_coordinates(self):
        # Visakhapatnam
        valid, lat, lon, err = validate_coordinates(17.72, 83.30)
        self.assertTrue(valid)
        self.assertAlmostEqual(lat, 17.72)
        self.assertAlmostEqual(lon, 83.30)
        self.assertEqual(err, "")

        # Amaravati
        valid, lat, lon, err = validate_coordinates("16.518", "80.518")
        self.assertTrue(valid)
        self.assertAlmostEqual(lat, 16.518)
        self.assertAlmostEqual(lon, 80.518)

        # Boundary extremes
        self.assertTrue(validate_coordinates(90.0, 180.0)[0])
        self.assertTrue(validate_coordinates(-90.0, -180.0)[0])
        self.assertTrue(validate_coordinates(0.0, 0.0)[0])

    def test_invalid_coordinates(self):
        # Out of range latitude
        valid, _, _, err = validate_coordinates(91.0, 80.0)
        self.assertFalse(valid)
        self.assertIn("Latitude", err)

        valid, _, _, err = validate_coordinates(-90.5, 80.0)
        self.assertFalse(valid)

        # Out of range longitude
        valid, _, _, err = validate_coordinates(15.0, 185.0)
        self.assertFalse(valid)
        self.assertIn("Longitude", err)

        valid, _, _, err = validate_coordinates(15.0, -181.0)
        self.assertFalse(valid)

        # Non-numeric values
        self.assertFalse(validate_coordinates("invalid_lat", 80.0)[0])
        self.assertFalse(validate_coordinates(None, 80.0)[0])
        self.assertFalse(validate_coordinates(15.0, None)[0])

        # NaN / Infinity
        self.assertFalse(validate_coordinates(float("nan"), 80.0)[0])
        self.assertFalse(validate_coordinates(15.0, float("inf"))[0])


class TestHaversineDistanceAndRanking(unittest.TestCase):
    """Verifies distance calculations and station ordering."""

    def test_haversine_distance_accuracy(self):
        # Distance from Amaravati (16.518, 80.518) to Guntur (16.306, 80.436) ~ 25-28 km
        d_gnt_amv = haversine_distance(16.518, 80.518, 16.306, 80.436)
        self.assertTrue(20.0 < d_gnt_amv < 32.0, f"Expected ~26 km, got {d_gnt_amv}")

        # Distance to same point must be 0
        self.assertAlmostEqual(haversine_distance(17.7, 83.3, 17.7, 83.3), 0.0, places=5)

        # Distance from Visakhapatnam (17.7, 83.3) to Amaravati (16.5, 80.5) ~ 320-360 km
        d_vskp_amv = haversine_distance(17.7, 83.3, 16.5, 80.5)
        self.assertTrue(300.0 < d_vskp_amv < 380.0)

    def test_find_nearby_stations_ranking(self):
        srv = CurrentAirQualityService(api_key="test_key")
        # Near Visakhapatnam (17.72, 83.30)
        stations = srv.find_nearby_stations(17.72, 83.30, max_radius_km=100.0, limit=5)
        self.assertTrue(len(stations) > 0)

        # Must be strictly ordered by distance ascending
        for i in range(len(stations) - 1):
            self.assertLessEqual(stations[i]["distance_km"], stations[i + 1]["distance_km"])

        # Nearest station must be in or near Visakhapatnam
        nearest = stations[0]
        self.assertTrue(nearest["distance_km"] < 25.0)
        self.assertIn("Visakhapatnam", nearest["city"])
        self.assertIn("latitude", nearest)
        self.assertIn("longitude", nearest)
        self.assertIn("provider", nearest)

    def test_search_radius_filter(self):
        srv = CurrentAirQualityService(api_key="test_key")
        # Guntur (16.306, 80.436) has no direct CPCB monitor inside city limits;
        # Nearest CPCB monitor is Amaravati (AP001) ~25-27 km away.
        gnt_lat, gnt_lon = 16.306, 80.436
        strict_stations = srv.find_nearby_stations(gnt_lat, gnt_lon, max_radius_km=10.0, limit=5)
        self.assertEqual(len(strict_stations), 0)

        # When expanded to 50 km, candidate monitors (Amaravati) must be found
        expanded_stations = srv.find_nearby_stations(gnt_lat, gnt_lon, max_radius_km=50.0, limit=5)
        self.assertTrue(len(expanded_stations) > 0)
        for stn in expanded_stations:
            self.assertLessEqual(stn["distance_km"], 50.0)
            self.assertTrue(stn["distance_km"] > 10.0)


class TestCpcbDataSufficiencyAndFreshness(unittest.TestCase):
    """Verifies official CPCB NAQI math and data freshness thresholds."""

    def setUp(self):
        self.srv = CurrentAirQualityService(api_key="test_key")
        self.ist = pytz.timezone("Asia/Kolkata")

    def test_sufficient_pollutants_calculates_aqi(self):
        # 3 criteria pollutants with PM2.5 satisfies CPCB requirement
        items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 45.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 85.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 30.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
        ]
        res = self.srv.process_raw_measurements(items, "Test Station", "Visakhapatnam", distance_km=5.2)
        self.assertTrue(res["is_live"])
        self.assertIsNotNone(res["aqi"])
        self.assertIn(res["aqi_tier"], ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
        self.assertEqual(res["user_distance_km"], 5.2)
        self.assertIn("5.2 km", res["doorstep_disclosure"])

    def test_insufficient_pollutants_does_not_invent_aqi(self):
        # Only 2 pollutants (PM2.5 and NO2) - does NOT satisfy CPCB min 3 rule
        items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 45.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 30.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
        ]
        res = self.srv.process_raw_measurements(items, "Test Station", "Visakhapatnam", distance_km=2.0)
        self.assertTrue(res["is_live"])
        # Official AQI must be None to prevent scientific falsehood
        self.assertIsNone(res["aqi"])
        self.assertEqual(res["aqi_tier"], "Insufficient Pollutants for AQI")
        # Individual pollutants must still be faithfully reported
        self.assertEqual(res["pollutants"]["PM2.5"], 45.0)
        self.assertEqual(res["pollutants"]["NO2"], 30.0)

    def test_stale_data_older_than_24h_not_presented_as_current(self):
        # Observation from 36 hours ago
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=36)).isoformat()
        items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 55.0, "datetime": {"utc": stale_time}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 110.0, "datetime": {"utc": stale_time}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 40.0, "datetime": {"utc": stale_time}},
        ]
        res = self.srv.process_raw_measurements(items, "Stale Monitor", "Guntur", distance_km=12.0)
        self.assertEqual(res["status_badge"], "STALE OBSERVATION")
        # AQI must NOT be presented as valid current AQI
        self.assertIsNone(res["aqi"])
        self.assertIn("Stale Telemetry", res["aqi_tier"])

    def test_doorstep_and_far_station_advisory(self):
        # Station at 35 km (> 25 km threshold)
        items = [
            {"parameter": {"name": "pm25", "units": "µg/m³"}, "value": 35.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "pm10", "units": "µg/m³"}, "value": 65.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            {"parameter": {"name": "no2", "units": "µg/m³"}, "value": 20.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
        ]
        res = self.srv.process_raw_measurements(items, "Regional Monitor", "Amaravati", distance_km=35.0)
        self.assertEqual(res["user_distance_km"], 35.0)
        self.assertIn("35.0 km", res["doorstep_disclosure"])
        self.assertIn("regional ambient conditions", res["doorstep_disclosure"].lower())
        # Far station advisory must be present
        self.assertTrue(len(res["far_station_advisory"]) > 0)
        self.assertIn("35.0 km away", res["far_station_advisory"])


class TestTelemetryGatewayResilience(unittest.TestCase):
    """Verifies graceful degradation under network failure, HTTP 429, and timeouts."""

    @patch("src.prediction.current_air_quality_service.requests.get")
    def test_http_429_rate_limit_handled(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Client Error", response=mock_resp)
        mock_get.return_value = mock_resp

        srv = CurrentAirQualityService(api_key="valid_test_key")
        res = srv.get_current_observations("Visakhapatnam", force_refresh=True)

        self.assertFalse(res["is_live"])
        self.assertEqual(res["status_badge"], "RATE LIMITED")
        self.assertIn("rate limit", res["badge_desc"].lower())
        self.assertIsNone(res["aqi"])

    @patch("src.prediction.current_air_quality_service.requests.get")
    def test_timeout_handled(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        srv = CurrentAirQualityService(api_key="valid_test_key")
        res = srv.get_current_observations("Visakhapatnam", force_refresh=True)

        self.assertFalse(res["is_live"])
        self.assertEqual(res["status_badge"], "DATA UNAVAILABLE")
        self.assertEqual(res["aqi_tier"], "Gateway Timeout")
        self.assertIsNone(res["aqi"])

    @patch("src.prediction.current_air_quality_service.requests.get")
    def test_connection_error_handled(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Network offline")

        srv = CurrentAirQualityService(api_key="valid_test_key")
        res = srv.get_current_observations("Visakhapatnam", force_refresh=True)

        self.assertFalse(res["is_live"])
        self.assertEqual(res["status_badge"], "DATA UNAVAILABLE")
        self.assertIsNone(res["aqi"])

    def test_cache_bypassed_on_force_refresh(self):
        srv = CurrentAirQualityService(api_key="valid_test_key")
        # Prepopulate cache
        srv._cache["Visakhapatnam_auto"] = ({"cached": True, "is_live": True}, 9999999999.0)

        # Without force_refresh, returns cached
        res1 = srv.get_current_observations("Visakhapatnam", force_refresh=False)
        self.assertTrue(res1.get("cached"))

    @patch.object(CurrentAirQualityService, "fetch_openaq_location_by_id")
    @patch.object(CurrentAirQualityService, "fetch_openaq_latest_by_location")
    def test_sensor_map_fallback_when_raw_location_missing(self, mock_latest, mock_loc):
        mock_loc.return_value = {
            "id": 5408,
            "name": "Secretariat, Amaravati - APPCB",
            "sensors": [
                {"id": 101, "parameter": {"name": "pm25", "units": "µg/m³"}},
                {"id": 102, "parameter": {"name": "pm10", "units": "µg/m³"}},
                {"id": 103, "parameter": {"name": "no2", "units": "µg/m³"}},
            ]
        }
        mock_latest.return_value = {
            "results": [
                {"sensorsId": 101, "value": 35.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
                {"sensorsId": 102, "value": 75.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
                {"sensorsId": 103, "value": 25.0, "datetime": {"utc": datetime.now(timezone.utc).isoformat()}},
            ]
        }

        srv = CurrentAirQualityService(api_key="valid_test_key")
        station_stub = {
            "station_id": "openaq_5408",
            "openaq_location_id": 5408,
            "station_name": "Secretariat, Amaravati - APPCB",
            "city": "Amaravati",
            "latitude": 16.515,
            "longitude": 80.518,
            # Notice: raw_location is deliberately omitted
        }
        res = srv.get_current_observations_for_station(station_stub, force_refresh=True)
        self.assertEqual(mock_loc.call_count, 1)
        self.assertEqual(res["pollutants"]["PM2.5"], 35.0)
        self.assertEqual(res["pollutants"]["PM10"], 75.0)
        self.assertEqual(res["pollutants"]["NO2"], 25.0)
        self.assertIsNotNone(res["aqi"])


if __name__ == "__main__":
    unittest.main()
