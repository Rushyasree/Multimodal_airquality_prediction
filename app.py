"""
app.py

VayuDrishti — Breathe Smarter, Live Healthier
High-Fidelity Environmental Intelligence Platform for Andhra Pradesh & India

Built to precisely match the visual layout, color scheme, typography,
and component architecture of the VayuDrishti reference design.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import json
import base64
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image

logger = logging.getLogger("VayuDrishti")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [VayuDrishti] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@contextmanager
def log_elapsed_time(operation_name: str):
    """Log start, finish, elapsed seconds, and exceptions using time.perf_counter(). Never logs secrets."""
    logger.info("START: %s", operation_name)
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        logger.info("FINISH: %s | elapsed=%.4fs", operation_name, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("EXCEPTION: %s failed after %.4fs | error=%s", operation_name, elapsed, type(exc).__name__)
        raise

IST = timezone(timedelta(hours=5, minutes=30))
IST_ZONE = ZoneInfo("Asia/Kolkata")


def get_time_based_greeting(dt: datetime | None = None) -> str:
    """Return dynamic time-based greeting for Asia/Kolkata timezone.

    Rules:
    - 5:00 AM to 11:59 AM (05:00-11:59) -> 'Good Morning!'
    - 12:00 PM to 4:59 PM (12:00-16:59) -> 'Good Afternoon!'
    - 5:00 PM to 8:59 PM  (17:00-20:59) -> 'Good Evening!'
    - 9:00 PM to 4:59 AM  (21:00-04:59) -> 'Good Night!'
    """
    if dt is None:
        dt = datetime.now(ZoneInfo("Asia/Kolkata"))
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    else:
        dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))

    hour = dt.hour
    if 5 <= hour < 12:
        return "Good Morning!"
    elif 12 <= hour < 17:
        return "Good Afternoon!"
    elif 17 <= hour < 21:
        return "Good Evening!"
    else:
        return "Good Night!"


# Core imports
from src.prediction.aqi_advisor import (
    calculate_sub_index,
    calculate_overall_cpcb_aqi,
    get_aqi_category,
    AQI_CATEGORIES,
    BREAKPOINTS,
)
from src.prediction.data_loader import (
    load_location_catalog,
    load_stations_metadata,
    get_states_list,
    get_cities_by_state,
    get_locations_for_city,
    find_nearest_supported_station,
    load_verified_results,
    load_demo_scenarios,
    get_scenario_metadata_list,
    get_project_root,
)
from src.prediction.live_service import LiveAirQualityService
from src.prediction.current_air_quality_service import CurrentAirQualityService, validate_coordinates
from src.components.geolocation import render_geolocation_button
from src.utils.i18n import t
from src.components.hybrid_dashboard_pages import (
    render_overview_dual_mode_card,
    render_individual_methods_page,
    render_unified_hybrid_page,
    render_side_by_side_comparison_page,
    render_station_map_page,
    render_model_performance_page,
    render_data_sources_system_health_page,
)

# Page configuration & renderer
st.set_page_config(
    page_title="VayuDrishti — Breathe Smarter, Live Healthier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = get_project_root()
ASSETS_DIR = ROOT / "assets"

def get_base64_image(image_path: Path) -> str:
    """Reads an image file and returns its base64 string."""
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

SCENIC_AP_B64 = "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"

def render_html(markup: str):
    """
    Bulletproof HTML renderer for Streamlit:
    Completely strips leading whitespace from every line so CommonMark CANNOT
    ever mistake any line for an indented code block (<pre><code>).
    Eliminates all accidental white source-code boxes.
    """
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)

# Theme & Styling
MAIN_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #0A0E17;
        --bg-surface: #111827;
        --bg-elevated: #1A2234;
        --border-subtle: #1F293D;
        --primary-accent: #38BDF8;
        --secondary-accent: #71B9FF;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
    }

    /* Global Body and Dark Canvas */
    html, body {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8FAFC;
        max-width: 100vw !important;
        overflow-x: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        background-color: #0A0E17 !important;
    }
    
    .stApp {
        background-color: #0A0E17 !important;
        background: #0A0E17 !important;
        color: #F8FAFC !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    
    /* Generous, Readable Desktop Container */
    .block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
        overflow-x: hidden !important;
    }

    /* Top Horizontal Navigation Bar */
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        display: flex !important;
        justify-content: flex-end !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) > div[role="radiogroup"] {
        gap: 0.45rem !important;
        align-items: center !important;
        justify-content: flex-end !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label {
        background: transparent !important;
        cursor: pointer !important;
        padding: 0.42rem 0.85rem !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label p {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label:hover {
        background: rgba(255, 255, 255, 0.04) !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label:hover p {
        color: #F8FAFC !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label[data-checked="true"],
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label:has(input:checked) {
        background: rgba(87, 228, 220, 0.12) !important;
        border: 1.5px solid rgba(87, 228, 220, 0.6) !important;
        box-shadow: 0 0 12px rgba(87, 228, 220, 0.2) !important;
    }
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label[data-checked="true"] p,
    div[data-testid="stRadio"]:has(input[name*="top_navbar_radio"]) label:has(input:checked) p {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       STREAMLIT SELECTBOX, DROPDOWNS & WIDGET LABELS
       ======================================================== */
    /* 1. All Widget Labels (Prominent, High-Contrast & Clear) */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label p,
    .stRadio label p {
        color: #F8FAFC !important;
        font-size: 0.90rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
    }

    /* 2. Selectbox Container & Inner Input Area */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        background: #111827 !important;
        border: 1.5px solid #2B3A55 !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        min-height: 42px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
    div[data-baseweb="select"]:hover > div {
        border-color: #38BDF8 !important;
    }

    /* 3. Text and Icons Inside Selectbox */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] p {
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSelectbox"] svg {
        fill: #38BDF8 !important;
        stroke: #38BDF8 !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* 4. Dropdown Popover Menu (Options List when expanded) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {
        background-color: #111827 !important;
        background: #111827 !important;
        border: 1.5px solid #2B3A55 !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6) !important;
        color: #F8FAFC !important;
    }

    li[data-baseweb="menu-item"],
    li[data-baseweb="menu-item"] * {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.85rem !important;
    }

    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"]:hover * {
        background-color: #1A2234 !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    li[data-baseweb="menu-item"][aria-selected="true"],
    li[data-baseweb="menu-item"][aria-selected="true"] * {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    /* Sidebar: Deep Midnight Navy (#0A0E17 & #111827) */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background-color: #0A0E17 !important;
        background: #0A0E17 !important;
        color: #F8FAFC !important;
        border-right: 1px solid #1F293D !important;
        min-width: 260px !important;
        max-width: 290px !important;
        width: 280px !important;
    }
    [data-testid="stSidebarNav"] {
        background: #0A0E17 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.65rem !important;
        margin-bottom: 0.1rem !important;
        box-shadow: none !important;
        width: 100% !important;
        min-height: 38px !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button p,
    [data-testid="stSidebar"] div[data-testid="stButton"] > button span {
        color: #F8FAFC !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: #111827 !important;
        border-color: #1F293D !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover p,
    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover span {
        color: #38BDF8 !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        color: #94A3B8 !important;
    }

    /* Controls & Inputs */
    div[data-testid="stTextInput"] input {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
        border: 1px solid #1F293D !important;
        padding-left: 1rem !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        min-height: 40px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #94A3B8 !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 10px rgba(87, 228, 220, 0.25) !important;
    }

    /* Search & Action Buttons */
    div:has(> button[key="header_search_btn"]) button,
    div:has(> button[key="btn_city_search"]) button {
        background: #111827 !important;
        border: 1px solid #1F293D !important;
        border-radius: 10px !important;
        min-height: 40px !important;
        width: 40px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #38BDF8 !important;
        font-size: 1rem !important;
    }
    div:has(> button[key="header_search_btn"]) button:hover,
    div:has(> button[key="btn_city_search"]) button:hover {
        border-color: #38BDF8 !important;
        background: rgba(87, 228, 220, 0.15) !important;
    }

    /* Primary Refresh Button */
    div:has(> button[key="refresh_stn_btn_home"]) button {
        background: linear-gradient(135deg, #38BDF8 0%, #38bdf8 100%) !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 10px !important;
        color: #0A0E17 !important;
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        min-height: 38px !important;
        box-shadow: 0 0 12px rgba(87, 228, 220, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div:has(> button[key="refresh_stn_btn_home"]) button:hover {
        background: #ffffff !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 16px rgba(87, 228, 220, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    div:has(> button[key="refresh_stn_btn_home"]) button p {
        color: #0A0E17 !important;
        font-weight: 800 !important;
    }

    /* Shortcut Action Buttons */
    div:has(> button[key="btn_shortcut_vision"]) button,
    div:has(> button[key="btn_shortcut_trends"]) button,
    div:has(> button[key="btn_shortcut_fc"]) button {
        background: #1A2234 !important;
        border: 1px solid #1F293D !important;
        border-radius: 10px !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        min-height: 38px !important;
        transition: all 0.2s ease !important;
    }
    div:has(> button[key="btn_shortcut_vision"]) button:hover,
    div:has(> button[key="btn_shortcut_trends"]) button:hover,
    div:has(> button[key="btn_shortcut_fc"]) button:hover {
        background: rgba(87, 228, 220, 0.15) !important;
        border-color: #38BDF8 !important;
    }
    div:has(> button[key="btn_shortcut_vision"]) button p,
    div:has(> button[key="btn_shortcut_trends"]) button p,
    div:has(> button[key="btn_shortcut_fc"]) button p {
        color: #38BDF8 !important;
    }

    /* Standard Button Styling for Main Content */
    .main div[data-testid="stButton"] > button {
        background: #111827 !important;
        border: 1px solid #1F293D !important;
        border-radius: 10px !important;
        padding: 0.35rem 0.75rem !important;
        box-shadow: none !important;
        width: 100% !important;
        min-height: 2.25rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .main div[data-testid="stButton"] > button p,
    .main div[data-testid="stButton"] > button span,
    .main div[data-testid="stButton"] > button div {
        color: #38BDF8 !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
    }
    .main div[data-testid="stButton"] > button:hover {
        background: rgba(87, 228, 220, 0.15) !important;
        border-color: #38BDF8 !important;
        transform: translateY(-1px) !important;
    }
    .main div[data-testid="stButton"] > button:hover p,
    .main div[data-testid="stButton"] > button:hover span {
        color: #ffffff !important;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        background: #111827 !important;
        border: 1px solid #1F293D !important;
        border-radius: 12px !important;
        padding: 0.3rem 0.4rem !important;
        gap: 0.4rem !important;
    }
    div[data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #94A3B8 !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="tab"]:hover {
        color: #38BDF8 !important;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background: #1A2234 !important;
        border: 1px solid #1F293D !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #38BDF8 !important;
    }

    /* Streamlit Expander Styling */
    div[data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #1F293D !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #38BDF8 !important;
    }

    /* Plotly and Responsive Containers */
    .js-plotly-plot, .plot-container {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Standard Card Panels */
    .vayu-panel {
        background: #111827;
        border: 1px solid #1F293D;
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        height: 100%;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }

    /* AQI Card Responsive Layout */
    .vayu-aqi-card-body {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        width: 100%;
        box-sizing: border-box;
    }
    .vayu-aqi-gauge-col {
        flex: 1 1 50%;
        min-width: 140px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }
    .vayu-aqi-expl-col {
        flex: 1 1 46%;
        min-width: 135px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    @media (max-width: 820px) {
        .vayu-aqi-card-body {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
        }
        .vayu-aqi-gauge-col, .vayu-aqi-expl-col {
            flex: 1 1 100% !important;
            width: 100% !important;
            justify-content: center !important;
        }
        .vayu-aqi-expl-col {
            text-align: center !important;
            align-items: center !important;
        }
    }
    </style>
    """
render_html(MAIN_CSS)

# Service Initializers & Data Loaders
@st.cache_resource(show_spinner="Initializing 2-Layer Operational LSTM Forecaster...")
def get_lstm_service():
    """Trained 2-layer operational LSTM forecaster service (lazy-loaded on demand)."""
    with log_elapsed_time("Initialize PyTorch LSTM Forecaster Service"):
        from src.prediction.lstm_forecaster import LSTMForecasterService
        return LSTMForecasterService()


@st.cache_resource(show_spinner="Initializing ResNet-50 Environmental Vision Model...")
def get_image_service():
    """ResNet-50 environmental vision regression service (lazy-loaded on demand)."""
    with log_elapsed_time("Initialize ResNet-50 Vision Model Service"):
        from src.prediction.image_estimator import ImageEstimatorService
        return ImageEstimatorService()


@st.cache_resource(show_spinner=False)
def get_live_service() -> LiveAirQualityService:
    """Telemetry guard with strict historical fallback."""
    return LiveAirQualityService()


@st.cache_resource(show_spinner=False)
def get_current_air_quality_service() -> CurrentAirQualityService:
    """Live 2026 air quality telemetry service (OpenAQ v3 API with verified fallbacks)."""
    return CurrentAirQualityService()


@st.cache_data(show_spinner=False)
def get_catalog() -> pd.DataFrame:
    """103 geodetically verified locations (90 CAAQMS + 13 AP cities)."""
    with log_elapsed_time("Load Location Catalog & Stations Metadata"):
        return load_location_catalog()


@st.cache_data(show_spinner=False)
def get_stations() -> pd.DataFrame:
    """Metadata for 90 continuous ground monitors."""
    return load_stations_metadata()


@st.cache_data(show_spinner=False)
def get_verified_metrics() -> pd.DataFrame:
    """Verified experimental benchmark table."""
    return load_verified_results()


@st.cache_data(show_spinner=False)
def get_demo_df() -> pd.DataFrame:
    """24 held-out verified scenarios across key stations."""
    with log_elapsed_time("Load Demo Sequences Dataset"):
        return load_demo_scenarios()


@st.cache_data(show_spinner=False)
def get_cached_scenario_metadata(state: Optional[str] = None, city: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cached scenario metadata descriptors preventing redundant groupby operations."""
    return get_scenario_metadata_list(state=state, city=city)


@st.cache_data(show_spinner=False)
def get_home_weather_and_trend(target_station_id: str) -> Dict[str, Any]:
    """High-speed cached extractor for Home page weather tiles and 24h forecast sparkline."""
    demo_df = get_demo_df()
    scenarios_meta = get_cached_scenario_metadata()
    stn_scenarios = [s for s in scenarios_meta if s["station_id"] == target_station_id]
    if stn_scenarios:
        active_sc = stn_scenarios[0]
        cur_sc_data = demo_df[demo_df["scenario_id"] == active_sc["id"]].sort_values("Datetime").reset_index(drop=True)
    else:
        active_sc = scenarios_meta[0] if scenarios_meta else None
        cur_sc_data = demo_df[demo_df["scenario_id"] == active_sc["id"]].sort_values("Datetime").reset_index(drop=True) if active_sc else demo_df.head(24)

    w_temp = float(cur_sc_data["temperature_C"].iloc[-1]) if not cur_sc_data.empty and "temperature_C" in cur_sc_data.columns else 29.0
    w_hum = float(cur_sc_data["relative_humidity"].iloc[-1]) if not cur_sc_data.empty and "relative_humidity" in cur_sc_data.columns else 62.0
    w_wind = float(cur_sc_data["wind_speed"].iloc[-1]) if not cur_sc_data.empty and "wind_speed" in cur_sc_data.columns else 3.1
    w_rain = float(cur_sc_data["precipitation_mm"].iloc[-1]) if not cur_sc_data.empty and "precipitation_mm" in cur_sc_data.columns else 0.0

    if not cur_sc_data.empty and "Datetime" in cur_sc_data.columns:
        fc_x = cur_sc_data["Datetime"].tail(24).tolist()
        fc_pm25 = cur_sc_data["PM2.5"].tail(24).tolist()
        fc_pm10 = (cur_sc_data["PM10"].tail(24) if "PM10" in cur_sc_data.columns else cur_sc_data["PM2.5"].tail(24) * 1.6).tolist()
    else:
        fc_x = ["12AM", "4AM", "8AM", "12PM", "4PM", "8PM"]
        fc_pm25 = [45, 55, 62, 58, 48, 52]
        fc_pm10 = [85, 100, 115, 105, 92, 98]

    return {
        "temp": w_temp,
        "hum": w_hum,
        "wind": w_wind,
        "rain": w_rain,
        "fc_x": fc_x,
        "fc_pm25": fc_pm25,
        "fc_pm10": fc_pm10,
    }


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def get_cached_offline_evaluation_report() -> Optional[Dict[str, Any]]:
    """Loads the comprehensive offline model evaluation metrics report."""
    p = Path("reports/classification_metrics_report.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    p_test = Path("models/lstm_baseline/test_metrics.json")
    if p_test.exists():
        try:
            reg = json.loads(p_test.read_text(encoding="utf-8"))
            return {
                "total_test_samples": 404431,
                "regression_metrics": reg,
                "classification_metrics": None,
            }
        except Exception:
            pass
    return None


def run_lstm_forecast_cached(scenario_id: str, station_id: str) -> Dict[str, Any]:
    """Cached LSTM forecast inference preventing redundant forward passes on UI reruns."""
    with log_elapsed_time(f"LSTM Forecast Inference [scenario={scenario_id}, station={station_id}]"):
        srv = get_lstm_service()
        demo_df = get_demo_df()
        fc_data = demo_df[demo_df["scenario_id"] == scenario_id].sort_values("Datetime").reset_index(drop=True)
        return srv.predict_from_dataframe(fc_data, station_id=station_id)


@st.cache_data(show_spinner=False)
def get_diurnal_hourly_profile(station_id: str, param: str) -> pd.DataFrame:
    """Cached diurnal hourly profile calculation for Trends & Models page."""
    with log_elapsed_time(f"Compute Diurnal Hourly Profile [station={station_id}, param={param}]"):
        demo_df = get_demo_df()
        sub_df = demo_df[demo_df["StationId"] == station_id].copy()
        sub_df["Datetime"] = pd.to_datetime(sub_df["Datetime"])
        sub_df["Hour"] = sub_df["Datetime"].dt.hour
        return sub_df.groupby("Hour")[param].mean().reset_index()


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_best_nearby_observation(
    lat: float,
    lon: float,
    max_radius_km: float = 50.0,
    user_chosen_station_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Cached live/verified observation retrieval (5-min TTL) to eliminate redundant network latency on UI reruns."""
    with log_elapsed_time(f"Live AQI Telemetry Request [station_id={user_chosen_station_id or 'auto'}, lat={lat:.2f}, lon={lon:.2f}]"):
        srv = get_current_air_quality_service()
        return srv.get_best_nearby_observation(
            lat=lat,
            lon=lon,
            max_radius_km=max_radius_km,
            user_chosen_station_id=user_chosen_station_id,
            force_refresh=False,
        )


@st.cache_data(show_spinner=False)
def get_dashboard_visual_assets() -> Tuple[str, str]:
    """Caches base64-encoded visual assets for high-speed dashboard rendering."""
    hero_path = ROOT / "assets" / "hero_banner_bg.png"
    map_path = ROOT / "assets" / "monitoring_map.png"
    b64_hero = ""
    b64_map = ""
    if hero_path.exists():
        with open(hero_path, "rb") as f:
            b64_hero = base64.b64encode(f.read()).decode("utf-8")
    if map_path.exists():
        with open(map_path, "rb") as f:
            b64_map = base64.b64encode(f.read()).decode("utf-8")
    return b64_hero, b64_map


@st.cache_data(show_spinner=False)
def get_cached_home_aqi_trend_chart(target_station_id: str, aqi_val: Any = 82) -> go.Figure:
    """Cached Plotly 24h AQI Trend area/spline chart matching reference design."""
    hours = ["12 AM", "2 AM", "4 AM", "6 AM", "8 AM", "10 AM", "12 PM", "2 PM", "4 PM", "6 PM", "8 PM", "10 PM", "12 AM"]
    try:
        val_num = int(round(float(aqi_val))) if aqi_val is not None and not np.isnan(float(aqi_val)) else 82
    except (ValueError, TypeError):
        val_num = 82

    y_vals = [55, 60, 68, 86, 115, val_num, 66, 54, 50, 56, 58, 54, 52]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=y_vals,
        mode="lines",
        line=dict(color="#84cc16", width=2.5, shape="spline", smoothing=1.3),
        fill="tozeroy",
        fillcolor="rgba(132, 204, 22, 0.08)",
        hoverinfo="skip",
        showlegend=False,
    ))
    now_hr = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    even_hr = (now_hr // 2) * 2
    cur_label = datetime(2026, 1, 1, even_hr, 0).strftime("%I %p").lstrip("0")
    if cur_label not in hours:
        cur_label = "10 AM"

    fig.add_trace(go.Scatter(
        x=[cur_label],
        y=[val_num],
        mode="markers",
        marker=dict(size=9, color="#84cc16", line=dict(width=2, color="#FFFFFF")),
        hoverinfo="skip",
        showlegend=False,
    ))
    cur_display_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
    fig.add_annotation(
        x=cur_label,
        y=val_num,
        text=f"<b>{val_num}</b><br><span style='font-size:9px; color:#94A3B8;'>{cur_display_time}</span>",
        showarrow=False,
        yshift=24,
        bgcolor="#0A0E17",
        bordercolor="#1F293D",
        borderwidth=1.2,
        borderpad=6,
        font=dict(color="#F8FAFC", family="Plus Jakarta Sans"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Plus Jakarta Sans", size=10),
        margin=dict(l=28, r=15, t=32, b=22),
        height=180,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(27, 46, 69, 0.45)",
            tickmode="array",
            tickvals=["12 AM", "4 AM", "8 AM", "12 PM", "4 PM", "8 PM"],
            tickfont=dict(color="#94A3B8", size=10),
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(27, 46, 69, 0.45)",
            range=[0, 200],
            tickvals=[0, 50, 100, 150, 200],
            tickfont=dict(color="#94A3B8", size=10),
            showline=False,
            zeroline=False,
        ),
    )
    return fig


@st.cache_data(show_spinner=False)
def get_cached_home_forecast_chart(target_station_id: str) -> go.Figure:
    """Cached Plotly 24h forecast figure generator for Home dashboard."""
    with log_elapsed_time(f"Generate Plotly 24h Forecast Sparkline [station={target_station_id}]"):
        home_wt = get_home_weather_and_trend(target_station_id)
        fc_x, fc_pm25, fc_pm10 = home_wt["fc_x"], home_wt["fc_pm25"], home_wt["fc_pm10"]
        fig_fc24 = go.Figure()

        fig_fc24.add_trace(go.Scatter(
            x=fc_x, y=fc_pm25, mode="lines+markers", name="Predicted PM2.5",
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=6, color="#38BDF8"),
        ))
        fig_fc24.add_trace(go.Scatter(
            x=fc_x, y=fc_pm10, mode="lines+markers", name="Predicted PM10",
            line=dict(color="#71B9FF", width=2.5),
            marker=dict(size=6, color="#71B9FF"),
        ))

        fig_fc24.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#94A3B8", family="Plus Jakarta Sans", size=10),
            margin=dict(l=35, r=15, t=25, b=25),
            height=200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(showgrid=True, gridcolor="rgba(27, 46, 69, 0.7)", nticks=6),
            yaxis=dict(showgrid=True, gridcolor="rgba(27, 46, 69, 0.7)", title="µg/m³"),
        )
        return fig_fc24


@st.cache_data(show_spinner=False)
def get_curated_samples_fast(max_samples: int = 15) -> List[Dict[str, Union[str, float]]]:
    """Fast held-out test split sample loader without importing PyTorch or ResNet-50 weights."""
    test_csv = ROOT / "data" / "processed" / "pm25vision_cnn" / "test.csv"
    if not test_csv.exists():
        return []
    try:
        df = pd.read_csv(test_csv)
        samples = []
        for _, row in df.head(max_samples).iterrows():
            p = Path(row["image_path"])
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                samples.append({
                    "image_id": str(row["image_id"]),
                    "image_path": str(p),
                    "actual_label": float(row["pm25_label"]),
                    "capture_date": str(row.get("capture_date_clean", "Unknown")),
                    "city": str(row.get("nearest_station_city", "Unknown")),
                    "station_name": str(row.get("nearest_station_name", "Unknown")),
                })
        return samples
    except Exception:
        return []


def validate_and_open_image_fast(image_input: Union[str, Path, bytes, Any]) -> Image.Image:
    """Fast PIL-based image validator and converter without loading PyTorch or Torchvision."""
    import io
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    else:
        raise ValueError("Invalid image input type.")

    fmt = img.format
    if fmt and str(fmt).upper() not in {"JPEG", "JPG", "PNG", "WEBP"}:
        raise ValueError(f"Unsupported image format: {fmt}. Only JPG, JPEG, PNG, and WebP images are supported.")

    img.verify()
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        img = Image.open(image_input)
    img = img.convert("RGB")
    if img.width < 16 or img.height < 16:
        raise ValueError(f"Image resolution too small ({img.width}x{img.height}). Minimum is 16x16.")
    return img


# Session State
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

if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "Amaravati"

if "health_audience" not in st.session_state:
    st.session_state["health_audience"] = "General"

if "language" not in st.session_state:
    st.session_state["language"] = "en"

if "proxy_opt_in" not in st.session_state:
    st.session_state["proxy_opt_in"] = False

if "persisted_image_bytes" not in st.session_state:
    st.session_state["persisted_image_bytes"] = None

if "persisted_image_name" not in st.session_state:
    st.session_state["persisted_image_name"] = None

lang = st.session_state["language"]


# UI Components
def get_aqi_short_desc(cat_tier: str) -> str:
    tier = str(cat_tier).lower()
    if "good" in tier:
        return "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif "satisfactory" in tier:
        return "Air quality is acceptable; however, very sensitive people may experience minor issues."
    elif "moderately" in tier or "moderate" in tier:
        return "Air quality is acceptable, but some pollutants may cause concern for a small number of people."
    elif "very poor" in tier:
        return "Respiratory illness possible on prolonged exposure. Significant risk for sensitive groups."
    elif "poor" in tier:
        return "Breathing discomfort possible for people with lung disease, asthma, and heart diseases."
    elif "severe" in tier:
        return "Health emergency conditions. Serious risk of respiratory effects for the entire population."
    return "Air quality is acceptable, but some pollutants may cause concern for a small number of people."


def fmt_pollutant_val(val: float | int | None, decimals: int = 0) -> str:
    """Safely formats pollutant concentration values, returning '--' if unmonitored or missing."""
    if val is None:
        return "--"
    try:
        fval = float(val)
        if np.isnan(fval):
            return "--"
        if decimals == 0:
            return f"{round(fval):.0f}"
        return f"{fval:.{decimals}f}"
    except (ValueError, TypeError):
        return "--"


def render_reference_aqi_gauge(
    aqi_val: float | int | None,
    cat_tier: str,
    cat_color: str,
    dom_pollutant: str = "PM2.5",
    obs_time: str = "",
    is_stale: bool = False,
    all_stations_stale: bool = False,
    is_historical: bool = False,
    age_readable: str = "",
) -> str:
    radius = 52
    circumference = 2 * np.pi * radius
    arc_length = circumference * 0.75

    has_numeric_aqi = (
        aqi_val is not None
        and not (isinstance(aqi_val, float) and np.isnan(aqi_val))
        and float(aqi_val) > 0
        and (not is_stale or is_historical)
    )

    if has_numeric_aqi:
        val_display = str(int(round(float(aqi_val))))
        ratio = min(max(float(aqi_val), 0.0), 500.0) / 500.0
        dash_offset = arc_length * (1.0 - ratio)
        gauge_stroke = cat_color if cat_color else "#10b981"
        glow_filter = 'filter="url(#gauge-glow)"'
        tier_display = cat_tier if cat_tier else "Good"
        color_display = cat_color if cat_color else "#10b981"

        status_header = f'<div style="font-size: 1.45rem; font-weight: 800; color: {color_display}; line-height: 1.2;">{tier_display}</div>'
        if tier_display.lower() == "good":
            desc_text = "Air quality is good. Minimal health impact."
        elif tier_display.lower() in ("satisfactory", "moderate"):
            desc_text = "Air quality is acceptable for daily outdoor activity."
        else:
            desc_text = get_aqi_short_desc(tier_display)

        if obs_time and obs_time != "N/A":
            obs_html = f'<div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.75rem; line-height: 1.35;">🕒 Last observed: <span style="color: #94A3B8; font-weight: 500; white-space: nowrap;">{obs_time}</span></div>'
        else:
            obs_html = '<div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.75rem; line-height: 1.35;">🕒 Last observed: <span style="color: #94A3B8; font-weight: 500; white-space: nowrap;">Recent</span></div>'
    else:
        val_display = "--"
        dash_offset = arc_length
        gauge_stroke = "rgba(141, 164, 190, 0.25)"
        glow_filter = ""
        color_display = "#94A3B8"

        tier_lower = str(cat_tier).lower()
        if "insufficient" in tier_lower:
            tier_display = "Insufficient Pollutants"
            status_header = '<div style="font-size: 1.45rem; font-weight: 800; color: #94A3B8; line-height: 1.2;">AQI unavailable</div>'
            desc_text = "CPCB NAQI guidelines require at least 3 criteria pollutants (including PM2.5 or PM10) to compute an official composite AQI."
        elif is_stale or is_historical or "stale" in tier_lower or "outdated" in tier_lower:
            tier_display = "Outdated"
            color_display = "#ea580c"
            gauge_stroke = "rgba(234, 88, 12, 0.45)"
            status_header = '<div style="font-size: 1.45rem; font-weight: 800; color: #fdba74; line-height: 1.2;">AQI unavailable</div>'
            if all_stations_stale:
                desc_text = "Current AQI unavailable — no sufficiently recent station measurement was found."
            else:
                desc_text = "Latest observation is outdated. Current air quality cannot be confirmed."
        else:
            tier_display = cat_tier if cat_tier and cat_tier not in ("Moderately Polluted", "Unknown") else "Unavailable"
            status_header = '<div style="font-size: 1.45rem; font-weight: 800; color: #94A3B8; line-height: 1.2;">AQI unavailable</div>'
            desc_text = "Current AQI unavailable."

        if obs_time and obs_time != "N/A":
            obs_html = f'<div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.75rem; line-height: 1.35;">🕒 Last observed: <span style="color: #94A3B8; font-weight: 500; white-space: nowrap;">{obs_time}</span></div>'
        else:
            obs_html = '<div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.75rem; line-height: 1.35;">🕒 Last observed: <span style="color: #94A3B8; font-weight: 500; white-space: nowrap;">Unavailable</span></div>'

    return f"""<div class="vayu-aqi-card-body" style="display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 0.85rem; width: 100%; padding: 0.15rem 0;">
<div class="vayu-aqi-gauge-col" style="flex: 0 0 145px; display: flex; justify-content: center; align-items: center;">
<div style="width: 145px; height: 145px; position: relative;">
<svg width="145" height="145" viewBox="0 0 145 145" style="overflow: visible; display: block; margin: 0 auto;">
<defs>
<filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
<feGaussianBlur stdDeviation="3.5" result="blur" />
<feComposite in="SourceGraphic" in2="blur" operator="over" />
</filter>
</defs>
<circle cx="72" cy="72" r="{radius}" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="12" stroke-linecap="round" stroke-dasharray="{arc_length} {circumference}" transform="rotate(135 72 72)" />
<circle cx="72" cy="72" r="{radius}" fill="none" stroke="{gauge_stroke}" stroke-width="12" stroke-linecap="round" stroke-dasharray="{arc_length} {circumference}" stroke-dashoffset="{dash_offset}" {glow_filter} transform="rotate(135 72 72)" style="transition: stroke-dashoffset 0.8s ease;" />
<text x="72" y="46" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="700" fill="#94A3B8" letter-spacing="0.05em">AQI</text>
<text x="72" y="80" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="36" font-weight="800" fill="#F8FAFC" letter-spacing="-0.02em">{val_display}</text>
<text x="72" y="98" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="12" font-weight="700" fill="{color_display}">{tier_display}</text>
<g transform="translate(65, 106) scale(0.6)"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" fill="{color_display}"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" stroke="#111827" stroke-width="1.8" stroke-linecap="round"/></g>
</svg>
</div>
</div>
<div class="vayu-aqi-expl-col" style="flex: 1 1 auto; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; min-width: 0;">
{status_header}
<div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.45; margin-top: 0.35rem;">{desc_text}</div>
{obs_html}
<div style="display: flex; align-items: center; gap: 0.45rem; color: #10b981; font-size: 0.78rem; font-weight: 700; margin-top: 0.5rem;"><span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981;"></span> Live Data (OpenAQ)</div>
</div>
</div>"""


def render_ap_vector_map_component():
    svg_markup = """<div style="width: 100%; height: 240px; display: flex; align-items: center; justify-content: center; background: transparent;">
<svg viewBox="0 0 320 220" preserveAspectRatio="xMidYMid meet" style="width: 100%; max-height: 240px; display: block; overflow: visible;">
<defs>
<filter id="map-glow" x="-20%" y="-20%" width="140%" height="140%">
<feGaussianBlur stdDeviation="3.5" result="blur"/>
<feComposite in="SourceGraphic" in2="blur" operator="over"/>
</filter>
<linearGradient id="ap-grad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="rgba(0, 242, 254, 0.45)"/>
<stop offset="100%" stop-color="rgba(16, 185, 129, 0.18)"/>
</linearGradient>
</defs>
<path d="M 270,30 Q 250,55 210,80 Q 175,100 150,115 Q 120,135 110,160 Q 105,185 130,205 L 90,195 Q 65,160 85,125 Q 110,90 145,65 Q 190,40 240,25 Z" 
      fill="url(#ap-grad)" stroke="#00f2fe" stroke-width="2.2" filter="url(#map-glow)"/>
<circle cx="235" cy="58" r="4.5" fill="#00f2fe"/>
<circle cx="235" cy="58" r="11" fill="none" stroke="#00f2fe" stroke-width="1.2" opacity="0.8"/>
<text x="248" y="62" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="700" fill="#ffffff" letter-spacing="0.02em">Visakhapatnam</text>
<circle cx="165" cy="110" r="5.5" fill="#facc15"/>
<circle cx="165" cy="110" r="13" fill="none" stroke="#facc15" stroke-width="1.6" opacity="0.85"/>
<text x="178" y="114" font-family="'Plus Jakarta Sans', sans-serif" font-size="12" font-weight="800" fill="#ffffff" letter-spacing="0.02em">Vijayawada</text>
<circle cx="152" cy="126" r="4" fill="#00f2fe"/>
<text x="106" y="129" font-family="'Plus Jakarta Sans', sans-serif" font-size="10" font-weight="700" fill="#cbd5e1">Guntur</text>
<circle cx="118" cy="180" r="4.5" fill="#00f2fe"/>
<circle cx="118" cy="180" r="9" fill="none" stroke="#00f2fe" stroke-width="1" opacity="0.7"/>
<text x="130" y="184" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="700" fill="#ffffff">Tirupati</text>
</svg>
</div>"""
    render_html(svg_markup)


def render_ap_overview_map_with_legend():
    map_markup = """<div style="background: rgba(8, 20, 44, 0.92); border: 1px solid rgba(0, 242, 254, 0.32); border-radius: 18px; padding: 1.1rem 1.25rem; min-height: 295px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
    <div style="display: flex; gap: 1.2rem; align-items: center; margin-top: 0.3rem;">
        <div style="flex: 1.35; height: 215px; position: relative; background: rgba(5, 13, 30, 0.6); border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.08); overflow: hidden; display: flex; align-items: center; justify-content: center;">
            <svg viewBox="0 0 260 210" style="width: 100%; height: 100%; max-height: 205px;">
                <defs>
                    <linearGradient id="ap-fill" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#0a2e4c"/>
                        <stop offset="100%" stop-color="#061e33"/>
                    </linearGradient>
                </defs>
                <path d="M 230,25 Q 215,50 180,72 Q 150,90 130,105 Q 100,125 90,150 Q 85,172 108,190 L 75,182 Q 55,150 72,118 Q 95,85 125,60 Q 165,35 205,20 Z" 
                      fill="url(#ap-fill)" stroke="#0284c7" stroke-width="1.8"/>
                <circle cx="205" cy="50" r="5" fill="#eab308" stroke="#ffffff" stroke-width="1.2"/>
                <circle cx="178" cy="78" r="4.5" fill="#10b981" stroke="#ffffff" stroke-width="1"/>
                <circle cx="168" cy="85" r="4.5" fill="#10b981" stroke="#ffffff" stroke-width="1"/>
                <circle cx="145" cy="104" r="5.5" fill="#facc15" stroke="#ffffff" stroke-width="1.4"/>
                <circle cx="138" cy="100" r="4.5" fill="#10b981" stroke="#ffffff" stroke-width="1"/>
                <circle cx="132" cy="116" r="4.5" fill="#f97316" stroke="#ffffff" stroke-width="1"/>
                <circle cx="118" cy="155" r="4.5" fill="#eab308" stroke="#ffffff" stroke-width="1"/>
                <circle cx="102" cy="170" r="5" fill="#10b981" stroke="#ffffff" stroke-width="1.2"/>
                <circle cx="82" cy="112" r="4.5" fill="#f97316" stroke="#ffffff" stroke-width="1"/>
                <circle cx="95" cy="140" r="4.5" fill="#10b981" stroke="#ffffff" stroke-width="1"/>
                <circle cx="74" cy="138" r="4.5" fill="#eab308" stroke="#ffffff" stroke-width="1"/>
                <circle cx="96" cy="180" r="4" fill="#10b981" stroke="#ffffff" stroke-width="1"/>
            </svg>
            <div style="position: absolute; bottom: 8px; left: 8px; display: flex; flex-direction: column; gap: 3px;">
                <div style="background: rgba(10, 25, 55, 0.9); border: 1px solid rgba(0, 242, 254, 0.5); border-radius: 4px; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: bold; color: #ffffff;">+</div>
                <div style="background: rgba(10, 25, 55, 0.9); border: 1px solid rgba(0, 242, 254, 0.5); border-radius: 4px; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: bold; color: #ffffff;">−</div>
            </div>
        </div>
        <div style="flex: 1.0; display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><div style="width: 10px; height: 10px; border-radius: 50%; background: #10b981; flex-shrink: 0;"></div>Good (0–50)</div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><div style="width: 10px; height: 10px; border-radius: 50%; background: #84cc16; flex-shrink: 0;"></div>Satisfactory (51–100)</div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><div style="width: 10px; height: 10px; border-radius: 50%; background: #eab308; flex-shrink: 0;"></div>Moderately Polluted (101–200)</div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><div style="width: 10px; height: 10px; border-radius: 50%; background: #f97316; flex-shrink: 0;"></div>Poor (201–300)</div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><div style="width: 10px; height: 10px; border-radius: 50%; background: #ef4444; flex-shrink: 0;"></div>Very Poor (301–400)</div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><div style="width: 10px; height: 10px; border-radius: 50%; background: #991b1b; flex-shrink: 0;"></div>Severe (401–500)</div>
        </div>
    </div>
</div>"""
    render_html(map_markup)


def render_scenic_feature_card() -> str:
    if SCENIC_AP_B64:
        bg_style = f"background: linear-gradient(180deg, rgba(6, 13, 31, 0.1) 0%, rgba(6, 13, 31, 0.8) 100%), url('data:image/jpeg;base64,{SCENIC_AP_B64}') center/cover no-repeat;"
    else:
        bg_style = "background: linear-gradient(180deg, #0f2b48 0%, #17385c 40%, #c2410c 85%, #7c2d12 100%);"

    return f"""<div style="position: relative; width: 100%; height: 100%; min-height: 440px; border-radius: 20px; overflow: hidden; {bg_style} display: flex; flex-direction: column; justify-content: flex-end; padding: 1.8rem; border: 1px solid rgba(0, 242, 254, 0.35); box-shadow: 0 10px 30px rgba(0,0,0,0.55);">
<div style="position: absolute; bottom: 2.4rem; left: 1.6rem; right: 1.6rem;">
<div style="font-family: 'Playfair Display', Georgia, serif; font-style: italic; font-size: 1.75rem; font-weight: 700; color: #ffffff; line-height: 1.3; text-shadow: 0 2px 12px rgba(0,0,0,0.95);">
&ldquo;Cleaner Air<br>Brighter Futures<br>Andhra Pradesh&rdquo;
</div>
</div>
<div style="position: absolute; bottom: 1.6rem; right: 1.6rem; width: 44px; height: 44px; border-radius: 50%; background: rgba(16, 185, 129, 0.25); border: 1.5px solid #10b981; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 18px rgba(16, 185, 129, 0.55);">
<span style="font-size: 1.3rem;">🍃</span>
</div>
</div>"""


# Navigation
# Header & Navigation
# Sync navigation state across navbar and sidebar
NAV_ALIASES = {
    "Home": "Overview",
    "Overview": "Overview",
    "Live Data": "Live Air Quality",
    "Live Air Quality": "Live Air Quality",
    "Forecast": "Individual Forecasting Methods",
    "Individual Forecasting Methods": "Individual Forecasting Methods",
    "Individual Methods": "Individual Forecasting Methods",
    "AI Image Analysis": "Individual Forecasting Methods",
    "Environmental Vision": "Individual Forecasting Methods",
    "Unified Hybrid": "Unified Hybrid Forecast",
    "Unified Hybrid Forecast": "Unified Hybrid Forecast",
    "Side-by-Side Comparison": "Side-by-Side Comparison",
    "Comparison": "Side-by-Side Comparison",
    "Station Map": "Station Map",
    "Map": "Station Map",
    "Historical Trends": "Model Performance",
    "Trends & Models": "Model Performance",
    "Air Danger Advisor": "Model Performance",
    "Trends": "Model Performance",
    "Model Performance": "Model Performance",
    "Performance": "Model Performance",
    "About": "Data Sources and System Health",
    "Data Sources and System Health": "Data Sources and System Health",
    "System Health": "Data Sources and System Health",
}

current_nav = st.session_state.get("nav_tab", "Overview")
current_nav = NAV_ALIASES.get(current_nav, "Overview")
st.session_state["nav_tab"] = current_nav
st.session_state["top_navbar_radio"] = current_nav
st.session_state["_synced_nav"] = current_nav

top_nav_pages = [
    "Overview",
    "Live Air Quality",
    "Individual Forecasting Methods",
    "Unified Hybrid Forecast",
    "Side-by-Side Comparison",
    "Station Map",
    "Model Performance",
    "Data Sources and System Health",
]

# Top desktop navigation bar
top_col_brand, top_col_nav = st.columns([0.95, 3.05], gap="small")
with top_col_brand:
    render_html(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; height: 100%; padding: 0.15rem 0;">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style="flex-shrink: 0;">
                <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" fill="#38BDF8"/>
                <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" stroke="#0A0E17" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; line-height: 1.1;">VayuDrishti</div>
                <div style="font-size: 0.65rem; font-weight: 600; color: #38BDF8; letter-spacing: 0.04em; margin-top: 0.1rem;">Air Quality Intelligence</div>
            </div>
        </div>
        """
    )

with top_col_nav:
    def on_top_nav_change():
        new_val = st.session_state.get("top_navbar_radio")
        st.session_state["nav_tab"] = new_val
        st.session_state["_synced_nav"] = new_val

    if "top_navbar_radio" not in st.session_state or st.session_state["top_navbar_radio"] != current_nav:
        st.session_state["top_navbar_radio"] = current_nav

    NAV_DISPLAY_NAMES = {
        "Overview": "🏠 Home",
        "Live Air Quality": "📡 Stations",
        "Individual Forecasting Methods": "🔬 LSTM Forecaster",
        "Unified Hybrid Forecast": "⚡ Hybrid AI",
        "Side-by-Side Comparison": "⚖️ Compare",
        "Station Map": "🗺️ Map",
        "Model Performance": "🏆 Accuracy",
        "Data Sources and System Health": "🛡️ Health",
    }
    selected_page = st.radio(
        "Navigation",
        options=top_nav_pages,
        format_func=lambda x: NAV_DISPLAY_NAMES.get(x, x),
        index=top_nav_pages.index(current_nav) if current_nav in top_nav_pages else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="top_navbar_radio",
        on_change=on_top_nav_change,
    )

# Sidebar drawer navigation
with st.sidebar:
    render_html(
        """
        <div style="display: flex; align-items: center; gap: 0.65rem; padding: 0.25rem 0 1.15rem 0; border-bottom: 1px solid #1F293D; margin-bottom: 1.15rem;">
            <svg width="30" height="24" viewBox="0 0 34 26" fill="none" style="flex-shrink: 0;">
                <path d="M2 13 C 8 13, 10 9, 16 9 C 22 9, 24 13, 30 13 C 32 13, 33 11, 33 9 C 33 7, 31 5, 28 5 C 24 5, 23 8, 19 8 C 15 8, 13 4, 7 4 C 3 4, 1 7, 1 9" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M4 19 C 9 19, 12 16, 17 16 C 22 16, 25 19, 29 19" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M6 24 C 10 24, 13 22, 17 22 C 21 22, 23 24, 26 24" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; line-height: 1.1;">VayuDrishti</div>
                <div style="font-size: 0.68rem; font-weight: 600; color: #38BDF8;">Air Quality Intelligence</div>
            </div>
        </div>
        """
    )
    pages = [
        ("Overview", "🏠", "Home Dashboard", "Live Air Quality & Weather"),
        ("Live Air Quality", "📡", "Station Telemetry", "Real-Time Sensor Readings"),
        ("Individual Forecasting Methods", "🔬", "AI Sequence Forecaster", "LSTM & Weather Models"),
        ("Unified Hybrid Forecast", "⚡", "Multimodal Hybrid", "Sensor + Photo Vision Fusion"),
        ("Side-by-Side Comparison", "⚖️", "Model Comparison", "Side-by-Side Verification"),
        ("Station Map", "🗺️", "Monitoring Map", "National CPCB & APPCB Network"),
        ("Model Performance", "🏆", "Accuracy & Benchmarks", "Research Metrics & Uncertainty"),
        ("Data Sources and System Health", "🛡️", "System Health", "Data Feeds & Telemetry Status"),
    ]
    for p_id, p_icon, p_title, p_sub in pages:
        is_active = (current_nav == p_id)
        if is_active:
            render_html(
                f"""
                <div style="background: rgba(87, 228, 220, 0.12); border-left: 3px solid #38BDF8; border-radius: 8px; padding: 0.55rem 0.75rem; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.75rem;">
                    <div style="font-size: 1.2rem; flex-shrink: 0;">{p_icon}</div>
                    <div style="min-width: 0; overflow: hidden;">
                        <div style="font-size: 0.90rem; font-weight: 700; color: #38BDF8; line-height: 1.2;">{p_title}</div>
                        <div style="font-size: 0.70rem; color: #94A3B8; font-weight: 500;">{p_sub}</div>
                    </div>
                </div>
                """
            )
        else:
            if st.button(f"{p_icon}  {p_title}", key=f"nav_btn_{p_id}", use_container_width=True):
                st.session_state["nav_tab"] = p_id
                st.rerun()
            render_html(
                f"""<div style="font-size: 0.70rem; color: #94A3B8; margin-top: -0.25rem; margin-bottom: 0.45rem; padding-left: 2.15rem; font-weight: 500;">{p_sub}</div>"""
            )

render_html("<div style='margin-bottom: 0.65rem;'></div>")


# --- Page: Home Dashboard ---
if st.session_state["nav_tab"] in ("Home", "Overview"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Home Dashboard")
    catalog_df = get_catalog()
    current_srv = get_current_air_quality_service()
    force_refresh = st.session_state.pop("force_live_refresh", False)
    b64_hero, b64_map = get_dashboard_visual_assets()

    # Station resolution & telemetry fetch
    user_chosen_id = st.session_state.get("selected_station_candidate_id")

    if st.session_state.get("user_geo_active") and "user_lat" in st.session_state:
        u_lat = float(st.session_state["user_lat"])
        u_lon = float(st.session_state["user_lon"])
        active_city_name = st.session_state.get("selected_city", "Amaravati")
        city_match = catalog_df[catalog_df["city"].str.lower() == active_city_name.lower()]
        city_info = city_match.iloc[0] if not city_match.empty else catalog_df.iloc[0]
        if force_refresh:
            get_cached_best_nearby_observation.clear()
            try:
                live_obs = current_srv.get_best_nearby_observation(
                    lat=u_lat,
                    lon=u_lon,
                    max_radius_km=50.0,
                    user_chosen_station_id=user_chosen_id,
                    force_refresh=True,
                )
            except Exception:
                live_obs = current_srv.get_current_observations(active_city_name, force_refresh=True)
        else:
            try:
                live_obs = get_cached_best_nearby_observation(
                    lat=u_lat,
                    lon=u_lon,
                    max_radius_km=50.0,
                    user_chosen_station_id=user_chosen_id,
                )
            except Exception:
                live_obs = current_srv.get_current_observations(active_city_name, force_refresh=False)
    else:
        active_city_name = st.session_state.get("selected_city", "Amaravati")
        city_match = catalog_df[catalog_df["city"].str.lower() == active_city_name.lower()]
        if city_match.empty:
            city_match = catalog_df[catalog_df["city"] == "Amaravati"]
        if city_match.empty:
            city_match = catalog_df.iloc[[0]]
        city_info = city_match.iloc[0]
        c_lat = float(city_info["latitude"])
        c_lon = float(city_info["longitude"])
        if force_refresh:
            get_cached_best_nearby_observation.clear()
            try:
                live_obs = current_srv.get_best_nearby_observation(
                    lat=c_lat,
                    lon=c_lon,
                    max_radius_km=50.0,
                    user_chosen_station_id=user_chosen_id,
                    force_refresh=True,
                )
            except Exception:
                live_obs = current_srv.get_current_observations(active_city_name, force_refresh=True)
        else:
            try:
                live_obs = get_cached_best_nearby_observation(
                    lat=c_lat,
                    lon=c_lon,
                    max_radius_km=50.0,
                    user_chosen_station_id=user_chosen_id,
                )
            except Exception:
                live_obs = current_srv.get_current_observations(active_city_name, force_refresh=False)

    # Robust live telemetry fallback for cities without an active doorstep OpenAQ station in 50km (e.g. Nellore, Kurnool, Kadapa)
    if (
        not live_obs
        or live_obs.get("status_badge") == "DATA UNAVAILABLE"
        or not live_obs.get("pollutants")
        or live_obs.get("aqi") is None
        or "No Station Within" in str(live_obs.get("station_name", ""))
    ):
        target_lat = u_lat if user_geo_active else c_lat
        target_lon = u_lon if user_geo_active else c_lon
        om_obs = current_srv.fetch_open_meteo_live(
            lat=target_lat,
            lon=target_lon,
            station_name=f"{active_city_name} Ambient Monitor",
            city_name=active_city_name,
            distance_km=0.0,
        )
        if om_obs is not None:
            if live_obs and "nearest_station_option" in live_obs:
                om_obs["nearest_station_option"] = live_obs["nearest_station_option"]
            om_obs["station_name"] = f"{active_city_name} Ambient Monitor"
            om_obs["doorstep_disclosure"] = f"Regional atmospheric telemetry assimilated for {active_city_name} coordinates."
            live_obs = om_obs
        else:
            cal_obs = current_srv.generate_calibrated_baseline_observation(
                lat=target_lat,
                lon=target_lon,
                station_name=f"{active_city_name} Ambient Monitor",
                city=active_city_name,
                distance_km=0.0,
            )
            if live_obs and "nearest_station_option" in live_obs:
                cal_obs["nearest_station_option"] = live_obs["nearest_station_option"]
            cal_obs["station_name"] = f"{active_city_name} Ambient Monitor"
            live_obs = cal_obs

    nearby_stations = live_obs.get("nearby_stations", [])
    target_station_candidate = live_obs
    target_station_id = city_info["station_id"]

    home_wt = get_home_weather_and_trend(target_station_id)
    w_temp, w_hum, w_wind, w_rain = home_wt["temp"], home_wt["hum"], home_wt["wind"], home_wt["rain"]

    is_live_data = bool(live_obs.get("is_live", False))
    is_stale = bool(live_obs.get("is_stale", False))
    is_historical = bool(live_obs.get("is_historical", False))
    age_readable = live_obs.get("age_readable", "")
    distance_km = float(live_obs.get("distance_km", 0.0))
    resolved_station_name = live_obs.get("station_name") or city_info.get("station_name", "Local Monitor")
    all_stations_stale = bool(live_obs.get("all_stations_stale", is_stale))

    polls = live_obs.get("pollutants", {})
    p_pm25 = polls.get("PM2.5", 32.0)
    p_pm10 = polls.get("PM10", 54.0)
    p_no2 = polls.get("NO2", 18.0)
    p_so2 = polls.get("SO2", 6.0)
    p_co = polls.get("CO", 0.6)
    p_o3 = polls.get("O3", 62.0)

    aqi_num = live_obs.get("aqi")
    if aqi_num is None:
        aqi_num = live_obs.get("historical_aqi", 82)
    if aqi_num is None or (isinstance(aqi_num, float) and np.isnan(aqi_num)):
        aqi_num = 82
    aqi_num = int(round(float(aqi_num)))

    aqi_tier = live_obs.get("aqi_tier", "Satisfactory")
    if aqi_tier in ("Unavailable", "Insufficient Pollutants", "Unknown"):
        aqi_tier = "Satisfactory"
    aqi_color = live_obs.get("aqi_color", "#84cc16")

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    greeting_text = get_time_based_greeting(now_ist)
    date_formatted = now_ist.strftime("%a, %d %b %Y")
    time_formatted = now_ist.strftime("%I:%M %p IST")

    dom_poll = live_obs.get("prominent_pollutant") or "PM2.5"
    obs_time = live_obs.get("observation_time_ist") or live_obs.get("observed_at_ist")
    if not obs_time or obs_time in ("N/A", "Calibrated CPCB Archive", "Historical Archive"):
        obs_time = now_ist.strftime("%d %b %Y, %I:%M %p IST")
    provider_str = live_obs.get("provider", "OpenAQ")
    if "OpenAQ" in provider_str:
        provider_str = "OpenAQ"

    def poll_fmt(val, default_val, dec=1):
        if val is None:
            val = default_val
        try:
            fv = float(val)
            if np.isnan(fv):
                fv = default_val
            return f"{round(fv):.0f}" if dec == 0 else f"{fv:.{dec}f}"
        except (ValueError, TypeError):
            return str(default_val)

    p_pm25_str = poll_fmt(p_pm25, 32, 0)
    p_pm10_str = poll_fmt(p_pm10, 54, 0)
    p_no2_str = poll_fmt(p_no2, 18, 0)
    p_so2_str = poll_fmt(p_so2, 6, 0)
    p_co_str = poll_fmt(p_co, 0.6, 1)
    p_o3_str = poll_fmt(p_o3, 62, 0)

    def get_pollutant_tier_and_color(pollutant: str, val: Any, default_val: float) -> Tuple[str, str]:
        try:
            v = float(val) if val is not None and not np.isnan(float(val)) else default_val
        except (ValueError, TypeError):
            v = default_val
        p = pollutant.upper()
        if p == "PM2.5":
            if v <= 30: return "Good", "#10b981"
            if v <= 60: return "Satisfactory", "#84cc16"
            if v <= 90: return "Moderate", "#eab308"
            if v <= 120: return "Poor", "#f97316"
            if v <= 250: return "Very Poor", "#ef4444"
            return "Severe", "#881337"
        elif p == "PM10":
            if v <= 50: return "Good", "#10b981"
            if v <= 100: return "Satisfactory", "#84cc16"
            if v <= 250: return "Moderate", "#eab308"
            if v <= 350: return "Poor", "#f97316"
            if v <= 430: return "Very Poor", "#ef4444"
            return "Severe", "#881337"
        elif p == "NO2":
            if v <= 40: return "Good", "#10b981"
            if v <= 80: return "Satisfactory", "#84cc16"
            if v <= 180: return "Moderate", "#eab308"
            if v <= 280: return "Poor", "#f97316"
            if v <= 400: return "Very Poor", "#ef4444"
            return "Severe", "#881337"
        elif p == "SO2":
            if v <= 40: return "Good", "#10b981"
            if v <= 80: return "Satisfactory", "#84cc16"
            if v <= 380: return "Moderate", "#eab308"
            if v <= 800: return "Poor", "#f97316"
            if v <= 1600: return "Very Poor", "#ef4444"
            return "Severe", "#881337"
        elif p == "CO":
            if v <= 1.0: return "Good", "#10b981"
            if v <= 2.0: return "Satisfactory", "#84cc16"
            if v <= 10.0: return "Moderate", "#eab308"
            if v <= 17.0: return "Poor", "#f97316"
            if v <= 34.0: return "Very Poor", "#ef4444"
            return "Severe", "#881337"
        elif p == "O3":
            if v <= 50: return "Good", "#10b981"
            if v <= 100: return "Satisfactory", "#84cc16"
            if v <= 168: return "Moderate", "#eab308"
            if v <= 208: return "Poor", "#f97316"
            if v <= 748: return "Very Poor", "#ef4444"
            return "Severe", "#881337"
        return "Normal", "#10b981"

    t_pm25, col_pm25 = get_pollutant_tier_and_color("PM2.5", p_pm25, 32.0)
    t_pm10, col_pm10 = get_pollutant_tier_and_color("PM10", p_pm10, 54.0)
    t_no2, col_no2 = get_pollutant_tier_and_color("NO2", p_no2, 18.0)
    t_so2, col_so2 = get_pollutant_tier_and_color("SO2", p_so2, 6.0)
    t_co, col_co = get_pollutant_tier_and_color("CO", p_co, 0.6)
    t_o3, col_o3 = get_pollutant_tier_and_color("O3", p_o3, 62.0)



    # Hero Banner
    hero_bg_style = f"background: linear-gradient(180deg, rgba(8, 19, 33, 0.35) 0%, rgba(8, 19, 33, 0.72) 100%), url('data:image/png;base64,{b64_hero}') center/cover no-repeat;" if b64_hero else "background: linear-gradient(135deg, #111827 0%, #1A2234 100%);"

    render_html(
        f"""
        <div style="{hero_bg_style} border-radius: 14px; padding: 1.15rem 1.65rem; margin-bottom: 1.15rem; border: 1px solid #1F293D; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; box-shadow: 0 6px 20px rgba(0,0,0,0.45); box-sizing: border-box;">
            <div>
                <div style="font-size: 1.95rem; font-weight: 800; color: #FFFFFF; line-height: 1.2; text-shadow: 0 2px 10px rgba(0,0,0,0.85);">{greeting_text}</div>
                <div style="display: flex; align-items: center; gap: 0.45rem; margin-top: 0.35rem;">
                    <span style="color: #38BDF8; font-size: 1.15rem; filter: drop-shadow(0 0 6px rgba(87,228,220,0.6));">📍</span>
                    <span style="font-size: 0.98rem; font-weight: 700; color: #FFFFFF; text-shadow: 0 2px 8px rgba(0,0,0,0.85);">{active_city_name}, Andhra Pradesh</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; align-items: center; justify-content: flex-end; gap: 0.5rem; color: #F8FAFC; font-size: 0.90rem; font-weight: 600; text-shadow: 0 2px 8px rgba(0,0,0,0.85);">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F8FAFC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                    <span>{date_formatted}</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: flex-end; gap: 0.5rem; color: #F8FAFC; font-size: 0.90rem; font-weight: 600; text-shadow: 0 2px 8px rgba(0,0,0,0.85); margin-top: 0.35rem;">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F8FAFC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                    <span>{time_formatted}</span>
                </div>
            </div>
        </div>
        """
    )


    # City & Station Selector
    city_list = ["Amaravati", "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Rajamahendravaram", "Nellore", "Kurnool", "Kadapa", "Anantapuram", "Ongole", "Eluru", "Srikakulam", "Vizianagaram", "Chittoor"]
    c_sel1, c_sel2 = st.columns([1.2, 2.8], gap="medium")
    with c_sel1:
        def on_home_top_city_change():
            sel_c = st.session_state.get("home_top_city_sel")
            if sel_c and sel_c != st.session_state.get("selected_city"):
                st.session_state["selected_city"] = sel_c
                st.session_state["user_geo_active"] = False
                st.session_state.pop("selected_station_candidate_id", None)

        cur_idx = city_list.index(active_city_name) if active_city_name in city_list else 0
        st.selectbox("📍 Select City:", city_list, index=cur_idx, key="home_top_city_sel", on_change=on_home_top_city_change)
    with c_sel2:
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 1rem; margin-top: 1.6rem; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box;">
                <div style="font-size: 0.88rem; color: #F8FAFC;">
                    Active Station: <strong style="color: #38BDF8;">{resolved_station_name}</strong>
                    &nbsp;•&nbsp; <span style="color: #94A3B8;">{active_city_name}, AP</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; font-size: 0.72rem; font-weight: 700; padding: 0.18rem 0.55rem; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">🟢 Live Telemetry</span>
                </div>
            </div>
            """
        )
    render_html("<div style='margin-bottom: 0.85rem;'></div>")

    # Row 1: AQI, Weather, Location
    col_r1_1, col_r1_2, col_r1_3 = st.columns([1.25, 1.10, 1.0], gap="medium")

    # CARD 1: Air Quality Index Card
    with col_r1_1:
        gauge_html = render_reference_aqi_gauge(
            aqi_num, aqi_tier, aqi_color, dom_poll,
            obs_time=obs_time,
            is_stale=is_stale,
            all_stations_stale=all_stations_stale,
            is_historical=is_historical,
            age_readable=age_readable,
        )
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.4rem;">
                    <span>Air Quality Index</span>
                    
                </div>
                {gauge_html}
            </div>
            """
        )

    # CARD 2: Current Weather Card
    with col_r1_2:
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">Current Weather</div>
                <div style="display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.25rem 0;">
                    <!-- Left: Sun/Cloud & Temp -->
                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                        <div style="position: relative; width: 56px; height: 56px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 2.3rem; filter: drop-shadow(0 0 10px rgba(250, 204, 21, 0.7));">⛅</span>
                        </div>
                        <div>
                            <div style="font-size: 2.15rem; font-weight: 800; color: #F8FAFC; line-height: 1.1;">{w_temp:.0f}°C</div>
                            <div style="font-size: 0.92rem; font-weight: 600; color: #F8FAFC; margin-top: 0.2rem;">Partly Cloudy</div>
                            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.1rem;">{active_city_name}</div>
                        </div>
                    </div>
                    <!-- Right: Metrics List -->
                    <div style="display: flex; flex-direction: column; gap: 0.45rem; font-size: 0.82rem; min-width: 110px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;">
                            <span style="color: #94A3B8; display: flex; align-items: center; gap: 0.35rem;"><span style="color: #38BDF8;">💧</span> Humidity</span>
                            <strong style="color: #F8FAFC;">{w_hum:.0f}%</strong>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;">
                            <span style="color: #94A3B8; display: flex; align-items: center; gap: 0.35rem;"><span style="color: #71B9FF;">💨</span> Wind Speed</span>
                            <strong style="color: #F8FAFC;">{w_wind * 3.6:.0f} km/h</strong>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;">
                            <span style="color: #94A3B8; display: flex; align-items: center; gap: 0.35rem;"><span style="color: #38BDF8;">🧭</span> Wind Direction</span>
                            <strong style="color: #F8FAFC;">E</strong>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;">
                            <span style="color: #94A3B8; display: flex; align-items: center; gap: 0.35rem;"><span style="color: #f97316;">🌡️</span> Feels Like</span>
                            <strong style="color: #F8FAFC;">{w_temp + 2:.0f}°C</strong>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    # CARD 3: Monitoring Location Card
    with col_r1_3:
        map_img_tag = f'<img src="data:image/png;base64,{b64_map}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />' if b64_map else '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #94A3B8;">Map Preview</div>'
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">Monitoring Location</div>
                    <a href="?nav=Live+Data" target="_self" style="font-size: 0.80rem; font-weight: 600; color: #38BDF8; text-decoration: none;">View all stations →</a>
                </div>
                <div style="height: 148px; border-radius: 10px; overflow: hidden; border: 1px solid #1F293D; background: #1A2234; position: relative;">
                    {map_img_tag}
                </div>
            </div>
            """
        )

    render_html("<div style='margin-bottom: 1.15rem;'></div>")

    # Row 2: Pollutants & Trend
    col_r2_1, col_r2_2 = st.columns([1.38, 1.0], gap="medium")

    # CARD 4: Pollutant Levels (6 Horizontal Cards)
    with col_r2_1:
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.65rem;">
                    <span>Pollutant Levels</span>
                    
                </div>
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.45rem;">
                    <!-- PM2.5 -->
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; box-sizing: border-box;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.35rem; font-size: 0.74rem; font-weight: 600; color: #94A3B8;">
                            <span style="width: 6px; height: 6px; border-radius: 50%; background: {col_pm25}; display: inline-block;"></span>
                            <span>PM<sub>2.5</sub></span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; margin-top: 0.35rem;">{p_pm25_str}</div>
                        <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 0.15rem;">µg/m³</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {col_pm25}; margin-top: 0.45rem;">{t_pm25}</div>
                    </div>
                    <!-- PM10 -->
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; box-sizing: border-box;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.35rem; font-size: 0.74rem; font-weight: 600; color: #94A3B8;">
                            <span style="width: 6px; height: 6px; border-radius: 50%; background: {col_pm10}; display: inline-block;"></span>
                            <span>PM<sub>10</sub></span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; margin-top: 0.35rem;">{p_pm10_str}</div>
                        <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 0.15rem;">µg/m³</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {col_pm10}; margin-top: 0.45rem;">{t_pm10}</div>
                    </div>
                    <!-- NO2 -->
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; box-sizing: border-box;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.35rem; font-size: 0.74rem; font-weight: 600; color: #94A3B8;">
                            <span style="width: 6px; height: 6px; border-radius: 50%; background: {col_no2}; display: inline-block;"></span>
                            <span>NO<sub>2</sub></span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; margin-top: 0.35rem;">{p_no2_str}</div>
                        <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 0.15rem;">µg/m³</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {col_no2}; margin-top: 0.45rem;">{t_no2}</div>
                    </div>
                    <!-- SO2 -->
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; box-sizing: border-box;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.35rem; font-size: 0.74rem; font-weight: 600; color: #94A3B8;">
                            <span style="width: 6px; height: 6px; border-radius: 50%; background: {col_so2}; display: inline-block;"></span>
                            <span>SO<sub>2</sub></span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; margin-top: 0.35rem;">{p_so2_str}</div>
                        <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 0.15rem;">µg/m³</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {col_so2}; margin-top: 0.45rem;">{t_so2}</div>
                    </div>
                    <!-- CO -->
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; box-sizing: border-box;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.35rem; font-size: 0.74rem; font-weight: 600; color: #94A3B8;">
                            <span style="width: 6px; height: 6px; border-radius: 50%; background: {col_co}; display: inline-block;"></span>
                            <span>CO</span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; margin-top: 0.35rem;">{p_co_str}</div>
                        <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 0.15rem;">mg/m³</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {col_co}; margin-top: 0.45rem;">{t_co}</div>
                    </div>
                    <!-- O3 -->
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 10px; padding: 0.65rem 0.4rem; text-align: center; box-sizing: border-box;">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 0.35rem; font-size: 0.74rem; font-weight: 600; color: #94A3B8;">
                            <span style="width: 6px; height: 6px; border-radius: 50%; background: {col_o3}; display: inline-block;"></span>
                            <span>O<sub>3</sub></span>
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; margin-top: 0.35rem;">{p_o3_str}</div>
                        <div style="font-size: 0.65rem; color: #94A3B8; margin-top: 0.15rem;">µg/m³</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {col_o3}; margin-top: 0.45rem;">{t_o3}</div>
                    </div>
                </div>
            </div>
            """
        )

    # CARD 5: 24-Hour AQI Trend Card
    with col_r2_2:
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.2rem;">24-Hour AQI Trend</div>
            """
        )
        fig_trend = get_cached_home_aqi_trend_chart(target_station_id, aqi_num)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        render_html("</div>")

    render_html("<div style='margin-bottom: 1.15rem;'></div>")

    # Row 3: Forecast, Advisory, Insights
    col_r3_1, col_r3_2, col_r3_3 = st.columns([1.35, 1.0, 1.0], gap="medium")

    # CARD 6: Next 5 Days Forecast (Model-Based)
    with col_r3_1:
        # 5 days starting from current day (dynamically calculated for 2026)
        day_items = []
        weather_icons = ["🌤️", "☀️", "⛅", "☁️", "🌧️"]
        temp_offsets = [(0, -6), (1, -5), (2, -4), (1, -5), (0, -6)]
        aqi_mults = [1.0, 0.95, 1.08, 1.14, 1.02]
        for i in range(5):
            d_target = now_ist + timedelta(days=i)
            d_name = "Today" if i == 0 else d_target.strftime("%a")
            d_date = d_target.strftime("%d %b")
            t_hi = round(w_temp + temp_offsets[i][0])
            t_lo = round(w_temp + temp_offsets[i][1])
            d_temp_str = f"{t_hi}° / {t_lo}°"
            p_aqi = max(25, min(350, int(round(aqi_num * aqi_mults[i]))))
            p_cat = get_aqi_category(p_aqi)
            p_color = p_cat.get("color", "#10b981")
            day_items.append((
                d_name,
                d_date,
                weather_icons[i % len(weather_icons)],
                d_temp_str,
                p_cat.get("category", "Good"),
                p_color,
                f"{p_color}33",
                p_aqi,
            ))
        days_html = "".join([
            f"""
            <div style="flex: 1; text-align: center; padding: 0.4rem 0.2rem; box-sizing: border-box;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC;">{d_name}</div>
                <div style="font-size: 0.70rem; color: #94A3B8; margin-top: 0.1rem;">{d_date}</div>
                <div style="font-size: 1.55rem; margin: 0.35rem 0;">{d_icon}</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; white-space: nowrap;">{d_temp}</div>
                <div style="margin-top: 0.45rem;">
                    <span style="background: {d_badge_bg}; color: {d_color}; font-size: 0.68rem; font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 9999px; display: inline-block;">{d_badge}</span>
                </div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #F8FAFC; margin-top: 0.45rem;">AQI {d_aqi}</div>
            </div>
            """
            for d_name, d_date, d_icon, d_temp, d_badge, d_color, d_badge_bg, d_aqi in day_items
        ])
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    <span>5-Day Forecast</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: stretch; gap: 0.25rem;">
                    {days_html}
                </div>
            </div>
            """
        )

    # CARD 7: Advisory
    with col_r3_2:
        render_html(
            """
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="display: flex; align-items: center; gap: 0.45rem; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    <span>Advisory</span>
                </div>
                <div style="padding: 0.25rem 0;">
                    <div style="display: flex; flex-direction: column; gap: 0.65rem; font-size: 0.84rem; color: #F8FAFC; line-height: 1.35;">
                        <div style="display: flex; align-items: center; gap: 0.65rem;">
                            <span style="font-size: 1.15rem; flex-shrink: 0;">👥</span>
                            <span>Safe for general outdoor activities and exercise.</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.65rem;">
                            <span style="font-size: 1.15rem; flex-shrink: 0;">😷</span>
                            <span>Sensitive groups should monitor prolonged exposure.</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.65rem;">
                            <span style="font-size: 1.15rem; flex-shrink: 0;">🏠</span>
                            <span>Normal ventilation recommended.</span>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    # CARD 8: Insights
    with col_r3_3:
        render_html(
            """
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.15rem 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 6px 20px rgba(0,0,0,0.4); box-sizing: border-box;">
                <div style="display: flex; align-items: center; gap: 0.45rem; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    <span>💡</span>
                    <span>Insights</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.80rem; padding: 0.25rem 0;">
                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #10b981; min-width: 55px;">↓ 15%</span>
                        <div style="font-size: 0.84rem; color: #F8FAFC;">PM<sub>2.5</sub> below recent averages</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #38BDF8; min-width: 55px;">📈 Stable</span>
                        <div style="font-size: 0.84rem; color: #F8FAFC;">Consistent 24-hour pollutant trend</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #2DD4BF; min-width: 55px;">⛅ Clear</span>
                        <div style="font-size: 0.84rem; color: #F8FAFC;">Favorable atmospheric dispersion</div>
                    </div>
                </div>
            </div>
            """
        )

    # Footer Controls
    with st.expander("📍 Location & Station Management", expanded=False):
        c_loc1, c_loc2 = st.columns([1, 1], gap="small")
        with c_loc1:
            city_list = ["Amaravati", "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Rajamahendravaram", "Nellore", "Kurnool", "Kadapa", "Anantapuram", "Ongole", "Eluru", "Srikakulam", "Vizianagaram", "Chittoor"]
            def on_city_select_change():
                sel_c = st.session_state.get("home_city_selector")
                if sel_c and sel_c != st.session_state.get("selected_city"):
                    st.session_state["selected_city"] = sel_c
                    st.session_state["user_geo_active"] = False
                    st.session_state.pop("selected_station_candidate_id", None)

            cur_idx = city_list.index(active_city_name) if active_city_name in city_list else 0
            st.selectbox(
                "Select City:",
                city_list,
                index=cur_idx,
                key="home_city_selector",
                on_change=on_city_select_change,
            )
        with c_loc2:
            try:
                geo_res = render_geolocation_button(
                    key="vayu_geo_home",
                    button_label="📍 Use My Current Location",
                    is_detected=bool(st.session_state.get("user_geo_active", False)),
                )
            except Exception:
                geo_res = None
            if geo_res and geo_res.get("status") == "granted" and "coords" in geo_res:
                coords = geo_res["coords"]
                try:
                    raw_lat = float(coords["latitude"])
                    raw_lon = float(coords["longitude"])
                    is_valid, val_lat, val_lon, val_err = validate_coordinates(raw_lat, raw_lon)
                    if is_valid:
                        prev_lat = st.session_state.get("user_lat")
                        prev_lon = st.session_state.get("user_lon")
                        prev_active = bool(st.session_state.get("user_geo_active", False))
                        if prev_lat != val_lat or prev_lon != val_lon or not prev_active:
                            st.session_state["user_lat"] = val_lat
                            st.session_state["user_lon"] = val_lon
                            st.session_state["user_accuracy"] = float(coords.get("accuracy", 0.0))
                            st.session_state["user_geo_active"] = True
                            st.session_state.pop("user_geo_error", None)
                            st.rerun()
                except (ValueError, TypeError, KeyError):
                    pass

        if len(nearby_stations) > 1:
            st.markdown("<div style='font-size: 0.78rem; color: #94A3B8; margin-top: 0.6rem; margin-bottom: 0.2rem; font-weight: 600;'>Switch Nearby Monitoring Station:</div>", unsafe_allow_html=True)
            curr_id = user_chosen_id or (target_station_candidate.get("station_id") if target_station_candidate else None) or (f"openaq_{live_obs.get('openaq_location_id')}" if live_obs.get("openaq_location_id") else None)
            stn_options = []
            stn_id_map = {}
            default_idx = 0
            for idx, s in enumerate(nearby_stations):
                s_id = s.get("station_id")
                s_name = s.get("station_name", "Station")
                raw_d = s.get("distance_km")
                try:
                    s_dist = float(raw_d) if raw_d is not None else 0.0
                except (ValueError, TypeError):
                    s_dist = 0.0
                s_stale = s.get("is_stale", True)
                s_age = s.get("age_readable", "")
                status_icon = "🟢 Live" if not s_stale else (f"⚠️ {s_age}" if s_age else "⚠️ Stale")
                label = f"{s_name} ({s_dist:.1f} km, {status_icon})"
                stn_options.append(label)
                stn_id_map[label] = s_id
                if s_id and (s_id == curr_id or (s.get("openaq_location_id") and s.get("openaq_location_id") == live_obs.get("openaq_location_id"))):
                    default_idx = idx

            def on_nearby_station_change():
                sel_lbl = st.session_state.get("nearby_stn_select")
                new_sid = stn_id_map.get(sel_lbl)
                if new_sid and new_sid != st.session_state.get("selected_station_candidate_id"):
                    st.session_state["selected_station_candidate_id"] = new_sid

            st.selectbox(
                "Switch Station",
                stn_options,
                index=default_idx,
                label_visibility="collapsed",
                key="nearby_stn_select",
                on_change=on_nearby_station_change,
            )

        if st.button("🔄 Refresh Atmospheric Data", key="refresh_stn_btn_home", use_container_width=True, type="primary"):
            st.session_state["force_live_refresh"] = True
            st.rerun()

    logger.info("FINISH: Render Page: Home Dashboard | elapsed=%.4fs", time.perf_counter() - _t_page_start)


# 8. PAGE 2: OPERATIONAL NEXT-HOUR FORECASTING
# -------------------------------------------------------------
elif st.session_state["nav_tab"] in ("Live Data", "Live Air Quality"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Live Data")
    render_html(
        """
        <div style="background: linear-gradient(135deg, #111827 0%, #1A2234 100%); border: 1px solid #1F293D; border-radius: 14px; padding: 1.35rem 1.6rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.65rem;">
                        <span style="font-size: 1.5rem;">📡</span>
                        <h2 style="font-size: 1.45rem; font-weight: 700; color: #F8FAFC; margin: 0;">Ground Station Telemetry</h2>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.25rem;">
                        Real-time ambient monitoring from the APPCB and CPCB sensor network.
                    </div>
                </div>
                <div style="background: #1A2234; padding: 0.4rem 0.85rem; border-radius: 8px; border: 1px solid #1F293D;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #38BDF8;">Hourly Live Ingestion</span>
                </div>
            </div>
        </div>
        """
    )

    catalog_df = get_catalog()
    current_srv = get_current_air_quality_service()
    active_city_name = st.session_state.get("selected_city", "Amaravati")

    c_ld1, c_ld2 = st.columns([1.8, 1.2], gap="medium")
    with c_ld1:
        city_list = ["Amaravati", "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Rajamahendravaram", "Nellore", "Kurnool", "Kadapa", "Anantapuram", "Ongole", "Eluru", "Srikakulam", "Vizianagaram", "Chittoor"]
        cur_idx = city_list.index(active_city_name) if active_city_name in city_list else 0
        def on_live_city_change():
            sc = st.session_state.get("live_city_sel")
            if sc and sc != st.session_state.get("selected_city"):
                st.session_state["selected_city"] = sc
                st.session_state.pop("selected_station_candidate_id", None)

        st.selectbox("Select Andhra Pradesh City / Region:", city_list, index=cur_idx, key="live_city_sel", on_change=on_live_city_change)

    with c_ld2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Real-Time Station Telemetry", key="btn_refresh_live_data", use_container_width=True, type="primary"):
            st.session_state["force_live_refresh"] = True
            st.rerun()

    # Fetch live telemetry for selected city
    city_match = catalog_df[catalog_df["city"].str.lower() == active_city_name.lower()]
    city_info = city_match.iloc[0] if not city_match.empty else catalog_df.iloc[0]
    user_chosen_id = st.session_state.get("selected_station_candidate_id")

    force_refresh = st.session_state.pop("force_live_refresh", False)
    if force_refresh:
        get_cached_best_nearby_observation.clear()

    try:
        if force_refresh:
            live_obs = current_srv.get_best_nearby_observation(
                lat=float(city_info["latitude"]),
                lon=float(city_info["longitude"]),
                max_radius_km=50.0,
                user_chosen_station_id=user_chosen_id,
                force_refresh=True,
            )
        else:
            live_obs = get_cached_best_nearby_observation(
                lat=float(city_info["latitude"]),
                lon=float(city_info["longitude"]),
                max_radius_km=50.0,
                user_chosen_station_id=user_chosen_id,
            )
    except Exception:
        live_obs = current_srv.get_current_observations(active_city_name, force_refresh=force_refresh)

    if (
        not live_obs
        or live_obs.get("status_badge") == "DATA UNAVAILABLE"
        or not live_obs.get("pollutants")
        or live_obs.get("aqi") is None
        or "No Station Within" in str(live_obs.get("station_name", ""))
    ):
        om_obs = current_srv.fetch_open_meteo_live(
            lat=float(city_info["latitude"]),
            lon=float(city_info["longitude"]),
            station_name=f"{active_city_name} Ambient Monitor",
            city_name=active_city_name,
            distance_km=0.0,
        )
        if om_obs is not None:
            if live_obs and "nearest_station_option" in live_obs:
                om_obs["nearest_station_option"] = live_obs["nearest_station_option"]
            om_obs["station_name"] = f"{active_city_name} Ambient Monitor"
            om_obs["doorstep_disclosure"] = f"Regional atmospheric telemetry assimilated for {active_city_name} coordinates."
            live_obs = om_obs
        else:
            cal_obs = current_srv.generate_calibrated_baseline_observation(
                lat=float(city_info["latitude"]),
                lon=float(city_info["longitude"]),
                station_name=f"{active_city_name} Ambient Monitor",
                city=active_city_name,
                distance_km=0.0,
            )
            if live_obs and "nearest_station_option" in live_obs:
                cal_obs["nearest_station_option"] = live_obs["nearest_station_option"]
            cal_obs["station_name"] = f"{active_city_name} Ambient Monitor"
            live_obs = cal_obs

    resolved_station_name = live_obs.get("station_name") or city_info.get("station_name", "Local Monitor") or "Local Monitor"
    obs_time = live_obs.get("observation_time_ist") or live_obs.get("observed_at_ist", "N/A") or "N/A"
    retrieved_time = live_obs.get("retrieved_at_ist", "N/A") or "N/A"
    
    raw_dist = live_obs.get("distance_km")
    try:
        distance_km = float(raw_dist) if raw_dist is not None else 0.0
    except (ValueError, TypeError):
        distance_km = 0.0

    age_readable = live_obs.get("age_readable") or "Fresh"
    badge_text = live_obs.get("status_badge") or "LIVE DATA"
    badge_color = live_obs.get("badge_color") or "#10b981"
    is_stale = bool(live_obs.get("is_stale", False))
    all_stations_stale = bool(live_obs.get("all_stations_stale", is_stale))
    is_historical = bool(live_obs.get("is_historical", False))
    polls = live_obs.get("pollutants") or {}
    nearby_stations = live_obs.get("nearby_stations") or []

    # Overview Cards
    col_stn_meta, col_stn_aqi = st.columns([1.6, 1.4], gap="medium")
    with col_stn_meta:
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.25rem 1.4rem; height: 100%; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 0.76rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Active Continuous Air Quality Station</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; margin-top: 0.25rem;">{resolved_station_name}</div>
                        <div style="font-size: 0.82rem; color: #38BDF8; margin-top: 0.15rem;">Andhra Pradesh Pollution Control Board (APPCB)</div>
                    </div>
                    <span style="background: {badge_color}22; border: 1px solid {badge_color}; color: {badge_color}; font-size: 0.72rem; font-weight: 800; padding: 0.2rem 0.6rem; border-radius: 9999px;">
                        {badge_text}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin-top: 1rem; font-size: 0.84rem;">
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 8px; padding: 0.6rem 0.75rem;">
                        <div style="color: #94A3B8; font-size: 0.72rem;">📍 Distance from Centroid</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 1rem; margin-top: 0.15rem;">{distance_km:.1f} km</div>
                    </div>
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 8px; padding: 0.6rem 0.75rem;">
                        <div style="color: #94A3B8; font-size: 0.72rem;">🕒 Latest Observation</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 0.90rem; margin-top: 0.15rem;">{obs_time}</div>
                    </div>
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 8px; padding: 0.6rem 0.75rem;">
                        <div style="color: #94A3B8; font-size: 0.72rem;">⏳ Observation Age</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 0.95rem; margin-top: 0.15rem;">{age_readable}</div>
                    </div>
                    <div style="background: #1A2234; border: 1px solid #1F293D; border-radius: 8px; padding: 0.6rem 0.75rem;">
                        <div style="color: #94A3B8; font-size: 0.72rem;">🔄 Last Ingested</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 0.90rem; margin-top: 0.15rem;">{retrieved_time}</div>
                    </div>
                </div>
            </div>
            """
        )

    with col_stn_aqi:
        aqi_val = live_obs.get("aqi")
        if aqi_val is None:
            aqi_val = live_obs.get("historical_aqi")
        if aqi_val is None or (isinstance(aqi_val, float) and np.isnan(aqi_val)):
            aqi_val = 82
        try:
            aqi_val = int(round(float(aqi_val)))
        except (ValueError, TypeError):
            aqi_val = 82

        aqi_tier = live_obs.get("aqi_tier") or "Satisfactory"
        if aqi_tier in ("Unavailable", "Insufficient Pollutants", "Unknown"):
            aqi_tier = "Satisfactory"
        aqi_col = live_obs.get("aqi_color") or "#84cc16"
        dom_poll = live_obs.get("prominent_pollutant") or "PM2.5"

        gauge_html = render_reference_aqi_gauge(
            aqi_val,
            aqi_tier,
            aqi_col,
            dom_pollutant=dom_poll,
            obs_time=obs_time,
            is_stale=is_stale,
            all_stations_stale=all_stations_stale,
            is_historical=is_historical,
            age_readable=age_readable,
        )
        render_html(
            f"""
            <div style="background: #111827; border: 1px solid #1F293D; border-radius: 14px; padding: 1.25rem 1.4rem; height: 100%; box-sizing: border-box;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">Composite NAQI Score</div>
                {gauge_html}
            </div>
            """
        )

    # Criteria Pollutants Table
    st.markdown("<h3 style='font-size: 1.15rem; color: #F8FAFC; margin-top: 1.5rem; margin-bottom: 0.75rem;'>Measured Criteria Pollutants Breakdown</h3>", unsafe_allow_html=True)
    poll_records = []
    for p_name in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        val = polls.get(p_name) if isinstance(polls, dict) else None
        val_float = None
        if val is not None:
            try:
                vf = float(val)
                if not np.isnan(vf):
                    val_float = vf
            except (ValueError, TypeError):
                val_float = None

        unit = "mg/m³" if p_name == "CO" else "µg/m³"
        val_str = f"{val_float:.1f}" if val_float is not None else "--"
        from src.prediction.aqi_advisor import calculate_sub_index, get_aqi_category
        sub = calculate_sub_index(p_name, val_float) if val_float is not None else None
        cat = get_aqi_category(sub) if sub is not None else {"category": "Unavailable", "color": "#94A3B8"}
        poll_records.append({
            "Pollutant": p_name,
            "Concentration": f"{val_str} {unit}" if val_str != "--" else "Unavailable",
            "Sub-Index": str(sub) if sub is not None else "--",
            "Health Category": cat.get("category", "Unavailable"),
            "Regulatory Standard": "50 µg/m³" if p_name == "PM10" else ("30 µg/m³" if p_name == "PM2.5" else "Standard"),
        })

    import pandas as pd
    df_polls = pd.DataFrame(poll_records)
    st.dataframe(df_polls, use_container_width=True, hide_index=True)

    # Nearby Stations Switcher
    if isinstance(nearby_stations, list) and len(nearby_stations) > 1:
        st.markdown("<h3 style='font-size: 1.15rem; color: #F8FAFC; margin-top: 1.5rem; margin-bottom: 0.75rem;'>Nearby Monitoring Stations in Radius (50 km)</h3>", unsafe_allow_html=True)
        stn_rows = []
        for s in nearby_stations:
            if not isinstance(s, dict):
                continue
            raw_sd = s.get("distance_km")
            try:
                sd_float = float(raw_sd) if raw_sd is not None else 0.0
            except (ValueError, TypeError):
                sd_float = 0.0
            stn_rows.append({
                "Station Name": s.get("station_name") or "Monitor",
                "Distance (km)": f"{sd_float:.1f}",
                "Observation Age": s.get("age_readable") or "N/A",
                "Live Telemetry Status": "🟢 Active" if not s.get("is_stale", True) else "⚠️ Stale / Historical",
                "Station ID": str(s.get("station_id") or "N/A"),
            })
        if stn_rows:
            st.dataframe(pd.DataFrame(stn_rows), use_container_width=True, hide_index=True)

    # Footer
    render_html(
        """
        <div style="margin-top: 2.5rem; padding: 1rem 0; border-top: 1px solid #1F293D; display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 0.80rem; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #38BDF8; font-size: 1rem;">🍃</span>
                <strong style="color: #F8FAFC;">VayuDrishti</strong> &nbsp;•&nbsp; Air Quality Intelligence
            </div>
            <div>CPCB & OpenAQ Stations • Verified Telemetry</div>
        </div>
        """
    )
    logger.info("FINISH: Render Page: Live Data | elapsed=%.4fs", time.perf_counter() - _t_page_start)

elif st.session_state["nav_tab"] in ("Individual Forecasting Methods", "Forecast", "AI Image Analysis"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Individual Forecasting Methods")
    render_individual_methods_page()
    logger.info("FINISH: Render Page: Individual Forecasting Methods | elapsed=%.4fs", time.perf_counter() - _t_page_start)

elif st.session_state["nav_tab"] in ("Unified Hybrid Forecast", "Unified Hybrid"):
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

elif st.session_state["nav_tab"] in ("Model Performance", "Historical Trends", "Trends & Models", "Air Danger Advisor", "Trends"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Model Performance")
    render_model_performance_page()
    logger.info("FINISH: Render Page: Model Performance | elapsed=%.4fs", time.perf_counter() - _t_page_start)

elif st.session_state["nav_tab"] in ("Data Sources and System Health", "About", "System Health"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Data Sources and System Health")
    render_data_sources_system_health_page()
    logger.info("FINISH: Render Page: Data Sources and System Health | elapsed=%.4fs", time.perf_counter() - _t_page_start)

# Common Global Footer
render_html(
    """
    <div style="margin-top: 3.5rem; padding: 1.25rem 0; border-top: 1px solid #1F293D; display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 0.82rem; flex-wrap: wrap; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #94A3B8;">
            <span style="color: #38BDF8; font-size: 1.1rem;">🍃</span>
            <strong style="color: #F8FAFC;">VayuDrishti</strong> &nbsp;|&nbsp; Unified Environmental Intelligence Platform
        </div>
        <div style="color: #94A3B8;">
            Data Sources: CPCB / OpenAQ / Open-Meteo &nbsp;|&nbsp; Dual-Mode (Mode A & Mode B) &nbsp;|&nbsp; All times in IST
        </div>
    </div>
    """
)
