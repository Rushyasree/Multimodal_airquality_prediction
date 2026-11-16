"""
tests/test_dashboard_navigation.py

Verifies that:
1. app.py loads and compiles cleanly.
2. All 6 active dashboard sections can be rendered via Streamlit AppTest without unhandled exceptions.
3. The 4 removed sections (Accuracy & Benchmarks, Monitoring Map, National CPCB & APPCB Network, System Health)
   no longer appear anywhere in visible navigation (top bar, sidebar buttons, or sidebar labels).
4. Spatial Prediction & Heatmap remains fully accessible and operational.
5. Legacy navigation requests (Station Map, Model Performance, System Health) gracefully fall back to Overview.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from streamlit.testing.v1 import AppTest

ACTIVE_SECTIONS = [
    "Overview",
    "Live Air Quality",
    "Individual Forecasting Methods",
    "Unified Hybrid Forecast",
    "Side-by-Side Comparison",
    "Spatial Prediction & Heatmap",
]

REMOVED_SECTIONS = [
    "Station Map",
    "Model Performance",
    "Data Sources and System Health",
]

REMOVED_UI_TERMS = [
    "Accuracy & Benchmarks",
    "Monitoring Map",
    "National CPCB & APPCB Network",
    "System Health",
    "🏆 Accuracy",
    "🗺️ Map",
    "🛡️ Health",
]


def test_dashboard_navigation_and_cleanup():
    print("\n--- Starting End-to-End Dashboard Navigation & UI Cleanup Verification ---")
    app = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=60)
    
    # Run initial app (default is Overview)
    app.run()
    assert not app.exception, f"Overview encountered unhandled exception: {app.exception}"
    print("[OK] Active Section 1: Overview rendered successfully")

    # 1. Verify Top Navigation Bar does not contain removed sections
    top_nav = app.radio("top_navbar_radio")
    assert top_nav is not None, "Top navigation radio widget not found"
    top_nav_options = top_nav.options
    print(f"Top nav options in UI: {top_nav_options}")

    for removed in REMOVED_SECTIONS:
        assert removed not in top_nav_options, f"Removed section '{removed}' still found in top navigation options!"
    print("[OK] Verified top navigation bar excludes all removed sections.")

    # 2. Verify Sidebar Buttons and Labels do not contain removed UI terms
    sidebar_buttons = [btn.label for btn in app.sidebar.button]
    sidebar_markdown = " ".join([m.value for m in app.sidebar.markdown])
    print(f"Sidebar buttons found: {sidebar_buttons}")

    for term in REMOVED_UI_TERMS:
        assert not any(term in btn_label for btn_label in sidebar_buttons), (
            f"Removed UI term '{term}' still found in sidebar buttons: {sidebar_buttons}"
        )
        assert term not in sidebar_markdown, (
            f"Removed UI term '{term}' still found in sidebar markdown text!"
        )
    print("[OK] Verified sidebar drawer navigation excludes all 4 removed sections and subtitles.")

    # 3. Test navigation across all 6 active sections
    for section in ACTIVE_SECTIONS[1:]:
        print(f"Testing navigation to active section: {section}...")
        app.session_state["nav_tab"] = section
        app.session_state["top_navbar_radio"] = section
        app.run()
        assert not app.exception, f"Active section '{section}' encountered unhandled exception: {app.exception}"
        print(f"[OK] Section '{section}' rendered cleanly with 0 exceptions")

    # 4. Specifically verify Spatial Prediction & Heatmap page
    print("Verifying Spatial Prediction & Heatmap accessibility...")
    app.session_state["nav_tab"] = "Spatial Prediction & Heatmap"
    app.session_state["top_navbar_radio"] = "Spatial Prediction & Heatmap"
    app.run()
    assert not app.exception, f"Spatial Prediction encountered unhandled exception: {app.exception}"
    print("[OK] Spatial Prediction & Heatmap verified accessible and fully functional.")

    # 5. Verify legacy fallback behavior for removed sections
    for legacy_section in REMOVED_SECTIONS:
        print(f"Verifying legacy fallback for removed route '{legacy_section}'...")
        app.session_state["nav_tab"] = legacy_section
        app.run()
        assert not app.exception, f"Legacy fallback for '{legacy_section}' caused exception: {app.exception}"
        assert app.session_state["nav_tab"] == "Overview", (
            f"Expected fallback to 'Overview' for '{legacy_section}', got '{app.session_state['nav_tab']}'"
        )
        print(f"[OK] Legacy route '{legacy_section}' safely normalized to 'Overview'")

    print("\nALL NAVIGATION & REMOVAL TESTS PASSED SUCCESSFULLY WITH ZERO ERRORS!")


if __name__ == "__main__":
    test_dashboard_navigation_and_cleanup()
