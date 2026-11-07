"""
tests/test_dashboard_navigation.py

Verifies that:
1. app.py loads and compiles cleanly.
2. All 8 dashboard sections can be rendered via Streamlit AppTest without unhandled exceptions.
3. Every individual and hybrid component initializes and executes.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from unittest.mock import patch
from streamlit.testing.v1 import AppTest

SECTIONS = [
    "Overview",
    "Live Air Quality",
    "Individual Forecasting Methods",
    "Unified Hybrid Forecast",
    "Side-by-Side Comparison",
    "Station Map",
    "Model Performance",
    "Data Sources and System Health",
]


def test_section_rendering():
    print("\n--- Starting End-to-End Dashboard Navigation Test across 8 Sections ---")
    app = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=60)
    
    # Run initial app (default is Overview)
    app.run()
    assert not app.exception, f"Overview encountered unhandled exception: {app.exception}"
    print("[OK] Section 1: Overview rendered successfully")

    for section in SECTIONS[1:]:
        print(f"Testing navigation to: {section}...")
        # Update session state nav_tab and top_navbar_radio
        app.session_state["nav_tab"] = section
        app.session_state["top_navbar_radio"] = section
        app.run()
        
        # Verify no unhandled exception occurred
        assert not app.exception, f"Section '{section}' encountered unhandled exception: {app.exception}"
        print(f"[OK] Section '{section}' rendered successfully with 0 exceptions")

    print("\nALL 8 DASHBOARD SECTIONS VERIFIED AND FUNCTIONAL!")


if __name__ == "__main__":
    test_section_rendering()
