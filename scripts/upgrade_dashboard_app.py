"""
scripts/upgrade_dashboard_app.py

Upgrades app.py to support both Mode A and Mode B across the 8 required sections:
1. Overview
2. Live Air Quality
3. Individual Forecasting Methods
4. Unified Hybrid Forecast
5. Side-by-Side Comparison
6. Station Map
7. Model Performance
8. Data Sources and System Health
"""

from pathlib import Path
import re

app_path = Path("app.py")
content = app_path.read_text(encoding="utf-8")

# 1. Add imports
import_target = "from src.utils.i18n import t\n"
new_imports = """from src.utils.i18n import t

# Unified Services & Modular Section Renderers
from src.components.hybrid_dashboard_pages import (
    render_overview_dual_mode_card,
    render_unified_hybrid_page,
    render_side_by_side_comparison_page,
    render_station_map_page,
    render_data_sources_system_health_page,
)
from src.services.live_air_quality_service import LiveAirQualityService as UnifiedLiveService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.image_estimation_service import ImageEstimationService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.model_comparison_service import ModelComparisonService
"""

assert import_target in content, "Import target not found!"
content = content.replace(import_target, new_imports, 1)

# 2. Update CSS for navbar radio to support wrapping and compact font
css_target = """    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) > div[role="radiogroup"] {
        gap: 0.45rem !important;
        align-items: center !important;
        justify-content: flex-end !important;
        flex-wrap: nowrap !important;
    }"""

new_css = """    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) > div[role="radiogroup"] {
        gap: 0.35rem !important;
        align-items: center !important;
        justify-content: flex-end !important;
        flex-wrap: wrap !important;
    }"""

assert css_target in content, "CSS target 1 not found!"
content = content.replace(css_target, new_css, 1)

css_target2 = """    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label p {
        color: #8DA4BE !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
    }"""

new_css2 = """    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label p {
        color: #8DA4BE !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
    }"""

assert css_target2 in content, "CSS target 2 not found!"
content = content.replace(css_target2, new_css2, 1)

# 3. Update Session State Navigation initialization
session_target = """# -------------------------------------------------------------
# 4. SESSION STATE & NAVIGATION INITIALIZATION
# -------------------------------------------------------------
if "nav" in st.query_params:
    _url_nav = st.query_params.pop("nav", None)
    if _url_nav in ("Home", "Live Data", "Forecast", "AI Image Analysis", "Historical Trends", "About"):
        st.session_state["nav_tab"] = _url_nav
        st.session_state["top_navbar_radio"] = _url_nav
        st.session_state["_synced_nav"] = _url_nav

if "nav_tab" not in st.session_state:
    st.session_state["nav_tab"] = "Home"

if "top_navbar_radio" not in st.session_state:
    st.session_state["top_navbar_radio"] = st.session_state["nav_tab"]

if "_synced_nav" not in st.session_state:
    st.session_state["_synced_nav"] = st.session_state["nav_tab"]

if st.session_state.get("nav_tab") == "Environmental Vision":
    st.session_state["nav_tab"] = "AI Image Analysis"
    st.session_state["top_navbar_radio"] = "AI Image Analysis"
    st.session_state["_synced_nav"] = "AI Image Analysis"
"""

new_session = """# -------------------------------------------------------------
# 4. SESSION STATE & NAVIGATION INITIALIZATION (8 SECTIONS)
# -------------------------------------------------------------
NAV_PAGES = [
    ("Overview", "🏠", "Overview", "Dual-Mode Executive Summary"),
    ("Live Air Quality", "📡", "Live Air Quality", "Station Telemetry & Observations"),
    ("Individual Forecasting Methods", "📈", "Individual Methods", "Mode A: Sensor, Weather & Vision"),
    ("Unified Hybrid Forecast", "🔮", "Unified Hybrid", "Mode B: Fused Model with Gating"),
    ("Side-by-Side Comparison", "⚖️", "Side-by-Side", "Comparative Evaluation (Section 4)"),
    ("Station Map", "🗺️", "Station Map", "90 CPCB Monitoring Stations"),
    ("Model Performance", "📊", "Performance", "Verified Metrics & Ablation"),
    ("Data Sources and System Health", "🩺", "System Health", "Pipelines, APIs & Integrity"),
]
top_nav_pages = [p[0] for p in NAV_PAGES]

NAV_ALIASES = {
    "Home": "Overview",
    "Live Data": "Live Air Quality",
    "Forecast": "Individual Forecasting Methods",
    "AI Image Analysis": "Individual Forecasting Methods",
    "Environmental Vision": "Individual Forecasting Methods",
    "Historical Trends": "Model Performance",
    "Trends & Models": "Model Performance",
    "Air Danger Advisor": "Individual Forecasting Methods",
    "About": "Data Sources and System Health",
}

if "nav" in st.query_params:
    _url_nav = st.query_params.pop("nav", None)
    if _url_nav in NAV_ALIASES:
        _url_nav = NAV_ALIASES[_url_nav]
    if _url_nav in top_nav_pages:
        st.session_state["nav_tab"] = _url_nav
        st.session_state["top_navbar_radio"] = _url_nav
        st.session_state["_synced_nav"] = _url_nav

if "nav_tab" not in st.session_state:
    st.session_state["nav_tab"] = "Overview"

if st.session_state.get("nav_tab") in NAV_ALIASES:
    st.session_state["nav_tab"] = NAV_ALIASES[st.session_state["nav_tab"]]

if "top_navbar_radio" not in st.session_state or st.session_state.get("top_navbar_radio") in NAV_ALIASES:
    st.session_state["top_navbar_radio"] = st.session_state["nav_tab"]

if "_synced_nav" not in st.session_state:
    st.session_state["_synced_nav"] = st.session_state["nav_tab"]
"""

assert session_target in content, "Session target not found!"
content = content.replace(session_target, new_session, 1)

# 4. Update Top Navbar column layout and Sidebar Pages
nav_bar_target = """current_nav = st.session_state.get("nav_tab", "Home")
if current_nav in ("Explore",):
    current_nav = "Live Data"
elif current_nav in ("Air Danger Advisor", "Trends", "Trends & Models"):
    current_nav = "Historical Trends"
elif current_nav in ("Environmental Vision",):
    current_nav = "AI Image Analysis"
elif current_nav in ("Model Performance",):
    current_nav = "About"
st.session_state["nav_tab"] = current_nav
st.session_state["top_navbar_radio"] = current_nav
st.session_state["_synced_nav"] = current_nav

top_nav_pages = ["Home", "Live Data", "Forecast", "AI Image Analysis", "Historical Trends", "About"]

# Top desktop navigation bar
top_col_brand, top_col_nav = st.columns([1.05, 2.95], gap="small")"""

new_nav_bar = """current_nav = st.session_state.get("nav_tab", "Overview")
if current_nav in NAV_ALIASES:
    current_nav = NAV_ALIASES[current_nav]

st.session_state["nav_tab"] = current_nav
st.session_state["top_navbar_radio"] = current_nav
st.session_state["_synced_nav"] = current_nav

top_nav_pages = [p[0] for p in NAV_PAGES]

# Top desktop navigation bar
top_col_brand, top_col_nav = st.columns([0.70, 3.30], gap="small")"""

assert nav_bar_target in content, "Navbar target not found!"
content = content.replace(nav_bar_target, new_nav_bar, 1)

# Update sidebar list
sidebar_target = """    pages = [
        ("Home", "🏠", "Home", "Current Air Quality"),
        ("Live Data", "📡", "Live Data", "Station Telemetry"),
        ("Forecast", "☀️", "Forecast", "PM2.5 & PM10"),
        ("AI Image Analysis", "🖼️", "AI Image Analysis", "Upload a Photo"),
        ("Historical Trends", "📊", "Historical Trends", "History & Evaluation"),
        ("About", "ℹ️", "About", "Data, Methodology"),
    ]"""

new_sidebar = """    pages = NAV_PAGES"""

assert sidebar_target in content, "Sidebar target not found!"
content = content.replace(sidebar_target, new_sidebar, 1)

# 5. Update Overview page check and insert Dual-Mode card
overview_target = 'if st.session_state["nav_tab"] == "Home":'
new_overview = 'if st.session_state["nav_tab"] in ("Overview", "Home"):'
assert overview_target in content, "Overview target not found!"
content = content.replace(overview_target, new_overview, 1)

# Insert dual-mode card right between Row 1 and Row 2 of Overview
dual_mode_target = '    render_html("<div style=\'margin-bottom: 1.15rem;\'></div>")\n\n    # ---------------- ROW 2: 2 CARDS (POLLUTANT LEVELS | 24-HOUR AQI TREND) ----------------'
new_dual_mode = """    render_html("<div style='margin-bottom: 0.65rem;'></div>")

    # Dual-Mode Executive Summary (Mode A & Mode B)
    render_overview_dual_mode_card(station_id=target_station_id, city=active_city_name)

    render_html("<div style='margin-bottom: 0.85rem;'></div>")

    # ---------------- ROW 2: 2 CARDS (POLLUTANT LEVELS | 24-HOUR AQI TREND) ----------------"""

assert dual_mode_target in content, "Dual mode target not found!"
content = content.replace(dual_mode_target, new_dual_mode, 1)

# 6. Update Live Air Quality page check
live_target = 'elif st.session_state["nav_tab"] == "Live Data":'
new_live = 'elif st.session_state["nav_tab"] in ("Live Air Quality", "Live Data"):'
assert live_target in content, "Live target not found!"
content = content.replace(live_target, new_live, 1)

# 7. Add Mode B, Comparison, and Station Map page blocks
# Let's see what is right after AI Image Analysis page or Forecast page
forecast_target = 'elif st.session_state["nav_tab"] == "Forecast":'
# We will wrap Forecast and AI Image Analysis under "Individual Forecasting Methods"
# Let's inspect where AI Image Analysis ends: line 2650: elif st.session_state["nav_tab"] in ("Historical Trends",
trends_target = 'elif st.session_state["nav_tab"] in ("Historical Trends", "Trends & Models", "Air Danger Advisor", "Trends"):'
new_pages_code = """elif st.session_state["nav_tab"] in ("Unified Hybrid Forecast", "Unified Hybrid"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Unified Hybrid Forecast")
    render_unified_hybrid_page()
    logger.info("FINISH: Render Page: Unified Hybrid Forecast | elapsed=%.4fs", time.perf_counter() - _t_page_start)

elif st.session_state["nav_tab"] in ("Side-by-Side Comparison", "Comparison"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Side-by-Side Comparison")
    render_side_by_side_comparison_page()
    logger.info("FINISH: Render Page: Side-by-Side Comparison | elapsed=%.4fs", time.perf_counter() - _t_page_start)

elif st.session_state["nav_tab"] in ("Station Map", "Map"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Station Map")
    render_station_map_page()
    logger.info("FINISH: Render Page: Station Map | elapsed=%.4fs", time.perf_counter() - _t_page_start)

elif st.session_state["nav_tab"] in ("Model Performance", "Historical Trends", "Trends & Models", "Air Danger Advisor", "Trends"):"""

assert trends_target in content, "Trends target not found!"
content = content.replace(trends_target, new_pages_code, 1)

# 8. Update About / System Health page
about_target = 'elif st.session_state["nav_tab"] in ("About", "Model Performance"):'
new_about = 'elif st.session_state["nav_tab"] in ("Data Sources and System Health", "System Health", "About"):\n    render_data_sources_system_health_page()'
assert about_target in content, "About target not found!"
content = content.replace(about_target, new_about, 1)

# 9. Update Forecast check to also match "Individual Forecasting Methods"
new_forecast_check = 'elif st.session_state["nav_tab"] in ("Individual Forecasting Methods", "Individual Methods", "Forecast"):'
content = content.replace(forecast_target, new_forecast_check, 1)

app_path.write_text(content, encoding="utf-8")
print("SUCCESS: app.py upgraded to support all 8 dashboard sections and both Mode A & Mode B!")
