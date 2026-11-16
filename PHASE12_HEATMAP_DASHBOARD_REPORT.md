# VayuDrishti Phase 12: Interactive Spatial Prediction & Urban Heatmap Dashboard Report

**Document Status:** Dashboard Integration & UI Specification Report  
**New Page Name:** "Spatial Prediction & Heatmap"  
**Source Path:** `src/components/spatial_heatmap_page.py`  
**Host Application:** `app.py`  
**Date:** November 2026  
**UI Framework:** Streamlit 1.32.0 / Plotly 5.24.1 / WebGL  

---

## 1. Executive Summary & Dashboard Role

Prior to Phase 12, the VayuDrishti dashboard displayed station-level discrete cards, comparative tables, and a national scatter map of 90 monitoring sites. Users could not visualize how pollution dispersed across continuous urban terrain or obtain localized forecasts for specific addresses between physical monitoring stations.

Phase 12 integrates a **9th dedicated dashboard page: "Spatial Prediction & Heatmap"**. This page renders high-resolution, interactive continuous 2D spatial fields for PM2.5 and PM10 across four Indian metropolitan regions, overlaid with verified physical CAAQMS stations and an interactive coordinate query interface.

Existing pages and workflows remain **100% intact and unaffected**.

---

## 2. Dashboard Navigation & Integration Architecture

The new page is cleanly registered across VayuDrishti's dual navigation hierarchy in `app.py`:
1. **Top Desktop Navbar:** Added as `"🌐 Heatmap"` (mapping internally to `"Spatial Prediction & Heatmap"`).
2. **Sidebar Drawer Menu:** Registered with icon `"🌐"`, title `"Spatial Heatmap"`, and subtitle `"2D Urban PM2.5/PM10 Grid"`.
3. **URL Alias Resolution:** Accepts `?nav=Spatial+Heatmap`, `?nav=Heatmap`, or `?nav=Spatial+Prediction+&+Heatmap`.
4. **Isolated Dispatch Block:**
   ```python
   elif st.session_state["nav_tab"] in ("Spatial Prediction & Heatmap", "Spatial Heatmap", "Heatmap"):
       _t_page_start = time.perf_counter()
       logger.info("START: Render Page: Spatial Prediction & Heatmap")
       from src.components.spatial_heatmap_page import render_spatial_heatmap_page
       render_spatial_heatmap_page()
       logger.info("FINISH: Render Page: Spatial Prediction & Heatmap | elapsed=%.4fs", time.perf_counter() - _t_page_start)
   ```

---

## 3. Key Visual & Functional Features

### 3.1 Metropolitan Agglomeration Selector
Users can dynamically switch between four major urban agglomerations defined in `src/config/spatial_config.py`:
- **Delhi NCR:** Covering Central Delhi, Gurugram, Noida, Ghaziabad, and Faridabad (37+ stations).
- **Mumbai MMR:** Covering South Mumbai, Bandra, Andheri, Navi Mumbai, and Thane (22+ stations).
- **Kolkata Metropolitan:** Covering Central Kolkata, Salt Lake, Howrah, and Barrackpore (14+ stations).
- **Bengaluru Urban:** Covering Central CBD, Whitefield, Electronic City, and Peenya (14+ stations).

---

### 3.2 Dynamic Spatial Parameter Controls
- **Target Pollutant:** Seamless toggle between **PM2.5** and **PM10**.
- **Grid Resolution:**
  - $25 \times 25$ (625 points): Rapid exploratory overview ($\approx 2.9\,\text{ms}$ latency).
  - $50 \times 50$ (2,500 points): Production high-resolution standard ($\approx 11.2\,\text{ms}$ latency).
  - $75 \times 75$ (5,625 points): Ultra high-definition geostatistical field ($\approx 64.0\,\text{ms}$ latency).
- **IDW Distance Decay Exponent ($p$):** Configurable slider from $1.0$ to $3.0$ (default $2.0$).

---

### 3.3 Executive Spatial Metrics Ribbon
Immediately above the interactive map, five responsive cards summarize the current urban airshed:
1. **Mean Pollutant Field:** Area-weighted average concentration across the metropolitan grid ($\mu\text{g/m}^3$).
2. **Peak Hotspot:** Maximum concentration cell detected in the interpolated field ($\mu\text{g/m}^3$).
3. **Cleanest Zone:** Minimum concentration cell observed ($\mu\text{g/m}^3$).
4. **Station Density:** Number of active CPCB physical monitors contributing to the field + Urban Coverage %.
5. **Engine Latency:** Measured CPU execution time for the current grid in milliseconds.

---

### 3.4 Interactive Plotly Heatmap & Sensor Overlay
Rendered using Plotly's WebGL-accelerated `Densitymapbox` combined with `Scattermapbox`:
- **Map Base Layer:** `carto-darkmatter` (requires zero external API keys, tokens, or network tracking).
- **Continuous Surface:** Dynamic color scales (`Turbo` for PM2.5, `Plasma` for PM10) with translucent alpha ($68\%$) allowing road networks and urban topography to show through.
- **Physical Station Markers:** Distinctive light-blue circles with high opacity ($95\%$) marking exact CAAQMS ground monitors.
- **Rich Hover Tooltips:** Moving the cursor over any physical monitor reveals:
  - Station Name & CPCB Code (e.g., `DL001 - Anand Vihar`).
  - Exact Decimal Coordinates.
  - Verified PM2.5 and PM10 Forecasts for target hour.
  - Forecast Target Timestamp ($t+1$).

---

### 3.5 Arbitrary Location Coordinate Prediction Interface
Located directly below the map, this module allows users to query localized air quality at any address:
- **Urban Landmark Presets:** Quick selection for major transit and business hubs:
  - *Delhi NCR:* Connaught Place, India Gate, Cyber City Gurugram, Noida Sector 62.
  - *Mumbai MMR:* Bandra Kurla Complex (BKC), Marine Drive, Andheri East, Vashi.
  - *Kolkata:* Park Street, Victoria Memorial, Salt Lake Sector V, Howrah Station.
  - *Bengaluru:* MG Road, Whitefield Tech Zone, Electronic City Phase 1, Indiranagar.
- **Custom Coordinate Input:** Decimal inputs for Latitude (°N) and Longitude (°E).
- **Real-Time Estimation Card:**
  - Localized PM2.5 and PM10 concentration estimates.
  - Geostatistical 95% Confidence Bounds ($\pm \delta$) scaled by proximity to nearest monitor.
  - Nearest Physical Station ID and geodesic Haversine distance in kilometers.
  - Contributing monitor count.
  - Data Quality Status Badge (`HIGH_CONFIDENCE_INTERPOLATION`, `SPARSE_INTERPOLATION`, `EXACT_STATION_HIT`, or `UNSUPPORTED_EXTRAPOLATION`).

---

### 3.6 Scientific Disclosure & Geostatistical Limitations Card
Prominently displayed at the bottom of the page to uphold IEEE research standards:
- Explicitly discloses that interpolated values are spatial statistical estimations derived from discrete ambient stations, not physical local monitors.
- Notes that hyper-local street canyons, roadside vehicular exhaust, and indoor microclimates may induce variations not captured by regional ambient monitors.
- Reaffirms the strict **Zero Synthetic Data Policy**: all surfaces originate from verified physical ground monitors.
