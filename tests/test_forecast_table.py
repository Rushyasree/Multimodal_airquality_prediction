"""
tests/test_forecast_table.py

Regression tests for forecast-table column selection and missing optional weather fields.
Verifies that:
1. Actual dataset column names ('relative_humidity', 'wind_speed', 'temperature_C') are mapped to friendly display labels.
2. Missing optional weather fields are handled gracefully without KeyError.
3. No fake zeros or fabricated values are substituted for missing fields.
4. Preserves all available criteria pollutants and temperature.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import unittest
import pandas as pd
from src.prediction.lstm_forecaster import prepare_forecast_display_table
from src.prediction.data_loader import load_demo_scenarios


class TestForecastTableColumns(unittest.TestCase):
    """Regression test suite for forecast antecedent feature table rendering."""

    def test_demo_df_columns_map_successfully(self):
        """Verifies real demo scenarios map without KeyError."""
        df = load_demo_scenarios()
        first_sc_id = df["scenario_id"].iloc[0]
        scenario_1 = df[df["scenario_id"] == first_sc_id].copy()
        
        display_df = prepare_forecast_display_table(scenario_1)
        
        self.assertFalse(display_df.empty)
        # Check required friendly labels exist
        self.assertIn("Datetime", display_df.columns)
        self.assertIn("PM2.5 (µg/m³)", display_df.columns)
        self.assertIn("PM10 (µg/m³)", display_df.columns)
        self.assertIn("NO2 (µg/m³)", display_df.columns)
        self.assertIn("SO2 (µg/m³)", display_df.columns)
        self.assertIn("CO (mg/m³)", display_df.columns)
        self.assertIn("O3 (µg/m³)", display_df.columns)
        self.assertIn("Temperature (°C)", display_df.columns)
        self.assertIn("Relative Humidity (%)", display_df.columns)
        self.assertIn("Wind Speed (m/s)", display_df.columns)

        # Check values match underlying source exactly (no fabricated zeros)
        self.assertAlmostEqual(display_df["Temperature (°C)"].iloc[0], scenario_1["temperature_C"].iloc[0])
        self.assertAlmostEqual(display_df["Relative Humidity (%)"].iloc[0], scenario_1["relative_humidity"].iloc[0])
        self.assertAlmostEqual(display_df["Wind Speed (m/s)"].iloc[0], scenario_1["wind_speed"].iloc[0])

    def test_missing_optional_weather_fields_handled_gracefully(self):
        """Verifies table does not raise KeyError when weather columns are missing."""
        minimal_df = pd.DataFrame({
            "Datetime": ["2026-10-16 12:00:00", "2026-10-16 13:00:00"],
            "PM2.5": [45.2, 48.0],
            "PM10": [85.1, 90.0],
            "NO2": [24.0, 26.0],
            "SO2": [8.0, 9.0],
            "CO": [0.8, 0.9],
            "O3": [35.0, 38.0],
            "temperature_C": [28.5, 29.0],
            # Note: relative_humidity, wind_speed, wind_direction, precipitation_mm are OMITTED
        })

        display_df = prepare_forecast_display_table(minimal_df)
        
        # Must not raise KeyError
        self.assertEqual(len(display_df), 2)
        self.assertIn("Temperature (°C)", display_df.columns)
        self.assertNotIn("Relative Humidity (%)", display_df.columns)
        self.assertNotIn("Wind Speed (m/s)", display_df.columns)
        self.assertNotIn("Precipitation (mm)", display_df.columns)

    def test_alternate_column_names_supported(self):
        """Verifies alternate column names (e.g. relative_humidity_pct, wind_speed_ms) map cleanly."""
        alt_df = pd.DataFrame({
            "Datetime": ["2026-10-16 12:00:00"],
            "PM2.5": [50.0],
            "PM10": [95.0],
            "NO2": [30.0],
            "SO2": [10.0],
            "CO": [1.0],
            "O3": [40.0],
            "temp_C": [30.0],
            "relative_humidity_pct": [65.0],
            "wind_speed_ms": [3.5],
        })

        display_df = prepare_forecast_display_table(alt_df)
        self.assertIn("Temperature (°C)", display_df.columns)
        self.assertIn("Relative Humidity (%)", display_df.columns)
        self.assertIn("Wind Speed (m/s)", display_df.columns)
        self.assertEqual(display_df["Relative Humidity (%)"].iloc[0], 65.0)
        self.assertEqual(display_df["Wind Speed (m/s)"].iloc[0], 3.5)

    def test_empty_or_none_dataframe(self):
        """Verifies empty or None inputs return an empty DataFrame rather than crashing."""
        self.assertTrue(prepare_forecast_display_table(None).empty)
        self.assertTrue(prepare_forecast_display_table(pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
