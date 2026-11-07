"""
scripts/patch_app.py

Upgrades app.py to wire all 8 dashboard sections cleanly:
1. Overview (Dual-Mode Summary)
2. Live Air Quality
3. Individual Forecasting Methods
4. Unified Hybrid Forecast
5. Side-by-Side Comparison
6. Station Map
7. Model Performance
8. Data Sources and System Health
"""

import sys

def main():
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add import
    old_imp = "from src.utils.i18n import t\n"
    new_imp = """from src.utils.i18n import t
from src.components.hybrid_dashboard_pages import (
    render_overview_dual_mode_card,
    render_individual_methods_page,
    render_unified_hybrid_page,
    render_side_by_side_comparison_page,
    render_station_map_page,
    render_model_performance_page,
    render_data_sources_system_health_page,
)
"""
    assert old_imp in content, "old_imp not found in app.py"
    content = content.replace(old_imp, new_imp, 1)

    # 2. Update navigation block
    nav_start = "# Reconcile navigation state across top radio navbar"
    nav_end = 'render_html("<div style=\'margin-bottom: 0.65rem;\'></div>")'

    pos_nav_start = content.find(nav_start)
    pos_nav_end = content.find(nav_end) + len(nav_end)
    assert pos_nav_start != -1 and pos_nav_end != -1, f"nav markers not found: {pos_nav_start}, {pos_nav_end}"

    new_nav = """# Reconcile navigation state across top radio navbar, sidebar, and shortcut cards
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
        \"\"\"
        <div style="display: flex; align-items: center; gap: 0.75rem; height: 100%; padding: 0.15rem 0;">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style="flex-shrink: 0;">
                <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" fill="#57E4DC"/>
                <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" stroke="#081321" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-size: 1.30rem; font-weight: 800; color: #FFFFFF; line-height: 1.1; letter-spacing: -0.01em;">VayuDrishti</div>
                <div style="font-size: 0.58rem; font-weight: 700; color: #57E4DC; letter-spacing: 0.12em; margin-top: 0.15rem;">CLEANER AIR &nbsp; BRIGHTER TOMORROW</div>
            </div>
        </div>
        \"\"\"
    )

with top_col_nav:
    def on_top_nav_change():
        new_val = st.session_state.get("top_navbar_radio")
        st.session_state["nav_tab"] = new_val
        st.session_state["_synced_nav"] = new_val

    if "top_navbar_radio" not in st.session_state or st.session_state["top_navbar_radio"] != current_nav:
        st.session_state["top_navbar_radio"] = current_nav

    selected_page = st.radio(
        "Navigation",
        options=top_nav_pages,
        index=top_nav_pages.index(current_nav) if current_nav in top_nav_pages else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="top_navbar_radio",
        on_change=on_top_nav_change,
    )

# Sidebar drawer navigation
with st.sidebar:
    render_html(
        \"\"\"
        <div style="display: flex; align-items: center; gap: 0.65rem; padding: 0.25rem 0 1.15rem 0; border-bottom: 1px solid #1B2E45; margin-bottom: 1.15rem;">
            <svg width="30" height="24" viewBox="0 0 34 26" fill="none" style="flex-shrink: 0;">
                <path d="M2 13 C 8 13, 10 9, 16 9 C 22 9, 24 13, 30 13 C 32 13, 33 11, 33 9 C 33 7, 31 5, 28 5 C 24 5, 23 8, 19 8 C 15 8, 13 4, 7 4 C 3 4, 1 7, 1 9" stroke="#57E4DC" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M4 19 C 9 19, 12 16, 17 16 C 22 16, 25 19, 29 19" stroke="#57E4DC" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M6 24 C 10 24, 13 22, 17 22 C 21 22, 23 24, 26 24" stroke="#57E4DC" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #F0F4F8; line-height: 1.1;">VayuDrishti</div>
                <div style="font-size: 0.68rem; font-weight: 600; color: #57E4DC;">Environmental Intelligence</div>
            </div>
        </div>
        \"\"\"
    )
    pages = [
        ("Overview", "🏠", "Overview", "Dual-Mode Intelligence"),
        ("Live Air Quality", "📡", "Live Air Quality", "Station Telemetry"),
        ("Individual Forecasting Methods", "🔬", "Individual Methods", "Mode A: LSTM, Weather, Vision"),
        ("Unified Hybrid Forecast", "⚡", "Unified Hybrid", "Mode B: Sensor-Vision Fusion"),
        ("Side-by-Side Comparison", "⚖️", "Side-by-Side", "Multi-Method Comparison"),
        ("Station Map", "🗺️", "Station Map", "90 CPCB/APPCB Stations"),
        ("Model Performance", "🏆", "Model Performance", "404k Benchmarks & Validation"),
        ("Data Sources and System Health", "🛡️", "Data & Health", "CPCB, OpenAQ, Models"),
    ]
    for p_id, p_icon, p_title, p_sub in pages:
        is_active = (current_nav == p_id)
        if is_active:
            render_html(
                f\"\"\"
                <div style="background: rgba(87, 228, 220, 0.12); border-left: 3px solid #57E4DC; border-radius: 8px; padding: 0.55rem 0.75rem; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.75rem;">
                    <div style="font-size: 1.2rem; flex-shrink: 0;">{p_icon}</div>
                    <div style="min-width: 0; overflow: hidden;">
                        <div style="font-size: 0.90rem; font-weight: 700; color: #57E4DC; line-height: 1.2;">{p_title}</div>
                        <div style="font-size: 0.70rem; color: #8DA4BE; font-weight: 500;">{p_sub}</div>
                    </div>
                </div>
                \"\"\"
            )
        else:
            if st.button(f"{p_icon}  {p_title}", key=f"nav_btn_{p_id}", use_container_width=True):
                st.session_state["nav_tab"] = p_id
                st.rerun()
            render_html(
                f\"\"\"<div style="font-size: 0.70rem; color: #8DA4BE; margin-top: -0.25rem; margin-bottom: 0.45rem; padding-left: 2.15rem; font-weight: 500;">{p_sub}</div>\"\"\"
            )

render_html("<div style='margin-bottom: 0.65rem;'></div>")"""

    content = content[:pos_nav_start] + new_nav + content[pos_nav_end:]

    # 3. Update Home check
    content = content.replace('if st.session_state["nav_tab"] == "Home":', 'if st.session_state["nav_tab"] in ("Home", "Overview"):', 1)

    # 4. Insert Overview Dual Mode Card right above ROW 1
    row1_marker = "# ---------------- ROW 1: 3 CARDS (AQI | CURRENT WEATHER | MONITORING LOCATION) ----------------"
    dual_card_call = """    # Dual-Mode Summary Card (Mode A vs Mode B)
    render_overview_dual_mode_card(station_id=city_info.get("station_id", "AP001"), city=active_city_name)

    # ---------------- ROW 1: 3 CARDS (AQI | CURRENT WEATHER | MONITORING LOCATION) ----------------"""
    assert row1_marker in content, "row1_marker not found"
    content = content.replace(row1_marker, dual_card_call, 1)

    # 5. Update Live Data check
    content = content.replace('elif st.session_state["nav_tab"] == "Live Data":', 'elif st.session_state["nav_tab"] in ("Live Data", "Live Air Quality"):', 1)

    # 6. Replace Forecast and remaining pages
    fc_marker = 'elif st.session_state["nav_tab"] == "Forecast":'
    pos_fc = content.find(fc_marker)
    assert pos_fc != -1, "fc_marker not found"

    remaining_pages = """elif st.session_state["nav_tab"] in ("Individual Forecasting Methods", "Forecast", "AI Image Analysis"):
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
    \"\"\"
    <div style="margin-top: 3.5rem; padding: 1.25rem 0; border-top: 1px solid #1B2E45; display: flex; justify-content: space-between; align-items: center; color: #8DA4BE; font-size: 0.82rem; flex-wrap: wrap; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #8DA4BE;">
            <span style="color: #57E4DC; font-size: 1.1rem;">🍃</span>
            <strong style="color: #F0F4F8;">VayuDrishti</strong> &nbsp;|&nbsp; Unified Environmental Intelligence Platform
        </div>
        <div style="color: #8DA4BE;">
            Data Sources: CPCB / OpenAQ / Open-Meteo &nbsp;|&nbsp; Dual-Mode (Mode A & Mode B) &nbsp;|&nbsp; All times in IST
        </div>
    </div>
    \"\"\"
)
"""

    content = content[:pos_fc] + remaining_pages

    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)

    print("SUCCESS: app.py updated!")

if __name__ == "__main__":
    main()
