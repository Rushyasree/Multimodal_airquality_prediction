"""
tests/test_greeting_and_aqi_card.py

Unit tests for:
1. Dynamic time-based greeting for Asia/Kolkata timezone with all boundary hours:
   - 05:00 to 11:59 -> "Good Morning!"
   - 12:00 to 16:59 -> "Good Afternoon!"
   - 17:00 to 20:59 -> "Good Evening!"
   - 21:00 to 04:59 -> "Good Night!"
   Tested specifically for boundary hours: 0, 5, 11, 12, 16, 17, 20, 21.

2. Current AQI card rendering:
   - Available AQI layout: centered gauge, separate status/explanation, observation timestamp with no-wrap, bottom scale.
   - Unavailable AQI layout (stale observation): "AQI unavailable", "Latest observation is outdated. Current air quality cannot be confirmed.", "Last observed: [timestamp]".
   - Unavailable AQI layout (insufficient criteria pollutants): "AQI unavailable", "Insufficient Pollutants".
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unittest
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from app import get_time_based_greeting, render_reference_aqi_gauge


class TestGreetingAndAqiCard(unittest.TestCase):
    def setUp(self):
        self.kolkata_tz = ZoneInfo("Asia/Kolkata")

    # -------------------------------------------------------------
    # GREETING BOUNDARY TESTS
    # -------------------------------------------------------------
    def test_greeting_hour_0_good_night(self):
        # 12:00 AM (Midnight)
        dt = datetime(2026, 10, 17, 0, 0, 0, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Night!")

    def test_greeting_hour_4_good_night(self):
        # 04:59 AM
        dt = datetime(2026, 10, 17, 4, 59, 59, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Night!")

    def test_greeting_hour_5_good_morning(self):
        # 05:00 AM boundary
        dt = datetime(2026, 10, 17, 5, 0, 0, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Morning!")

    def test_greeting_hour_11_good_morning(self):
        # 11:59 AM boundary
        dt = datetime(2026, 10, 17, 11, 59, 59, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Morning!")

    def test_greeting_hour_12_good_afternoon(self):
        # 12:00 PM boundary
        dt = datetime(2026, 10, 17, 12, 0, 0, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Afternoon!")

    def test_greeting_hour_16_good_afternoon(self):
        # 04:59 PM boundary
        dt = datetime(2026, 10, 17, 16, 59, 59, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Afternoon!")

    def test_greeting_hour_17_good_evening(self):
        # 05:00 PM boundary
        dt = datetime(2026, 10, 17, 17, 0, 0, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Evening!")

    def test_greeting_hour_20_good_evening(self):
        # 08:59 PM boundary
        dt = datetime(2026, 10, 17, 20, 59, 59, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Evening!")

    def test_greeting_hour_21_good_night(self):
        # 09:00 PM boundary
        dt = datetime(2026, 10, 17, 21, 0, 0, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Night!")

    def test_greeting_hour_23_good_night(self):
        # 11:59 PM boundary
        dt = datetime(2026, 10, 17, 23, 59, 59, tzinfo=self.kolkata_tz)
        self.assertEqual(get_time_based_greeting(dt), "Good Night!")

    def test_greeting_utc_conversion_to_ist(self):
        # UTC 01:00 AM is 06:30 AM IST -> Good Morning!
        dt_utc = datetime(2026, 10, 17, 1, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(get_time_based_greeting(dt_utc), "Good Morning!")

        # UTC 16:00 (4:00 PM) is 21:30 (9:30 PM) IST -> Good Night!
        dt_utc2 = datetime(2026, 10, 17, 16, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(get_time_based_greeting(dt_utc2), "Good Night!")

    # -------------------------------------------------------------
    # CURRENT AQI CARD LAYOUT TESTS
    # -------------------------------------------------------------
    def test_aqi_card_when_available(self):
        gauge_html = render_reference_aqi_gauge(
            aqi_val=82,
            cat_tier="Good",
            cat_color="#10b981",
            dom_pollutant="PM2.5",
            obs_time="16 Oct 2026, 08:30 PM IST",
            is_stale=False,
        )
        # 1. Gauge contains actual value and tier
        self.assertIn(">82<", gauge_html)
        self.assertIn(">Good<", gauge_html)
        # 2. Contains responsive column classes
        self.assertIn("vayu-aqi-card-body", gauge_html)
        self.assertIn("vayu-aqi-gauge-col", gauge_html)
        self.assertIn("vayu-aqi-expl-col", gauge_html)
        # 3. Contains observation timestamp with nowrap
        self.assertIn("Last observed:", gauge_html)
        self.assertIn("16 Oct 2026, 08:30 PM IST", gauge_html)
        self.assertIn("white-space: nowrap", gauge_html)

    def test_aqi_card_when_stale_outdated(self):
        gauge_html = render_reference_aqi_gauge(
            aqi_val=162,
            cat_tier="Outdated",
            cat_color="#ea580c",
            dom_pollutant="PM2.5",
            obs_time="15 Oct 2026, 06:00 PM IST",
            is_stale=True,
        )
        # 1. Gauge displays dashes --, not stale value
        self.assertIn(">--<", gauge_html)
        self.assertNotIn(">162<", gauge_html)
        # 2. Displays "AQI unavailable" and required stale notice
        self.assertIn("AQI unavailable", gauge_html)
        self.assertIn("Latest observation is outdated. Current air quality cannot be confirmed.", gauge_html)
        # 3. Displays observation timestamp
        self.assertIn("Last observed:", gauge_html)
        self.assertIn("15 Oct 2026, 06:00 PM IST", gauge_html)

    def test_aqi_card_when_insufficient_pollutants(self):
        gauge_html = render_reference_aqi_gauge(
            aqi_val=None,
            cat_tier="Insufficient Pollutants for AQI",
            cat_color="#94a3b8",
            dom_pollutant="N/A",
            obs_time="16 Oct 2026, 07:00 PM IST",
            is_stale=False,
        )
        self.assertIn(">--<", gauge_html)
        self.assertIn("AQI unavailable", gauge_html)
        self.assertIn("Insufficient Pollutants", gauge_html)
        self.assertIn("Last observed:", gauge_html)


if __name__ == "__main__":
    unittest.main()
