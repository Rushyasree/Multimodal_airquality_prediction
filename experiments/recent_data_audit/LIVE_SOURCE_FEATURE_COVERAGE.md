# VayuDrishti Phase 6B — Live Source Feature Coverage & Station Mapping Audit

**Document Version:** 1.0  
**Date:** November 2026  
**Auditor:** Independent Research & Deployment Verification Team  
**Scope:** Exhaustive audit of model input requirements, OpenAQ API v3 live telemetry coverage, XKDR database status, and CAAQMS station mapping.

---

## 1. Model Input Requirements Audit

The VayuDrishti forecasting system employs two distinct modeling branches:
1. **The Historical Production Default:** 2-Layer LSTM (`best_lstm_model.pt`) and static / severity-gated GBDT (`best_gbdt_pm25.joblib`, `best_gbdt_pm10.joblib`).
2. **The Post-2020 Adapted Ensemble:** Fine-tuned LSTM (`best_finetuned_lstm.pt`) and 49-feature GBDT (`adapted_gbdt_pm25.joblib`, `adapted_gbdt_pm10.joblib`).

### 1.1 Production LSTM Input Tensor Specification
- **Tensor Dimensions:** $\mathbf{X} \in \mathbb{R}^{B \times 24 \times 27}$ where:
  - $B$: Batch size (typically 1 at inference time).
  - $24$: 24 contiguous preceding hours $[t-23, \dots, t]$.
  - $27$: 27 input features in exact canonical ordering.
- **Scaling:** Features are standardized using training set means ($\mu_{\text{train}}$) and standard deviations ($\sigma_{\text{train}}$) saved in `data/processed/model_sequences/preprocessing_metadata.json`:
  $$x_{\text{scaled}} = \frac{x_{\text{raw}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- **Target Inversion:** Predictions are unscaled via $\mu_{\text{target}} = [79.514, 172.107]$ and $\sigma_{\text{target}} = [77.417, 137.471]\ \mu\text{g/m}^3$.
- **Missing Value Handling in Code:** In `src/prediction/lstm_forecaster.py`, missing or NaN features are imputed using training-period medians (`imputation_medians`).

### 1.2 Adapted GBDT Feature Specification (49 Features)
- **Input Matrix:** $\mathbf{X}_{\text{GBDT}} \in \mathbb{R}^{B \times 49}$.
- **Features Extracted:** Strictly causal features derived from $[t-23 \dots t]$:
  - Instantaneous $\text{PM}_{2.5}(t)$ and $\text{PM}_{10}(t)$ (2)
  - Cyclical calendar encodings ($\sin/\cos$ for hour, month, day of week) (6)
  - Station latitude and longitude (2)
  - 12 autoregressive lags for $\text{PM}_{2.5}$ and $\text{PM}_{10}$ ($t-1, t-2, t-3, t-5, t-11, t-23$)
  - 16 rolling window statistics for $\text{PM}_{2.5}$ and $\text{PM}_{10}$ (3h, 6h, 12h, 24h mean & std)
  - 10 differencing/momentum operators (1h, 3h, 6h, 24h diff, acceleration)
  - 1 ratio: $\text{PM}_{2.5}(t) / (\text{PM}_{10}(t) + 10^{-4})$
- **Excluded Channels (18):** All trace gases ($\text{NO}, \text{NO}_2, \text{NO}_x, \text{NH}_3, \text{CO}, \text{SO}_2, \text{O}_3$), VOCs ($\text{Benzene}, \text{Toluene}, \text{Xylene}$), and meteorological channels ($\text{Temp}, \text{Dewpoint}, \text{RH}, \text{Wind Speed}, \text{Precipitation}, \text{Wind Direction}$) are **excluded**.
- **Missing Value Handling:** Native tree branching in Scikit-Learn's `HistGradientBoostingRegressor` (which handles NaNs as an informative missing category without requiring artificial imputation).

---

## 2. Channel-by-Channel Reconciliation Matrix (27 Channels)

The table below reconciles each of the 27 historical LSTM channels against live telemetry genuinely provided by OpenAQ API v3:

| Pos | Canonical Feature | Historical Source | Historical Units | Available via Live OpenAQ v3? | OpenAQ Units / Conversion | Missingness Handling in System |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **0** | `PM2.5` | CPCB Monitor | $\mu\text{g/m}^3$ | **YES** | $\mu\text{g/m}^3$ (Direct) | **Mandatory**; no imputation permitted |
| **1** | `PM10` | CPCB Monitor | $\mu\text{g/m}^3$ | **YES** | $\mu\text{g/m}^3$ (Direct) | **Mandatory**; no imputation permitted |
| **2** | `NO` | CPCB Gas Analyzer | $\mu\text{g/m}^3$ | Partial / Rare | Often reported in ppb | Imputed with training median ($17.8 \ \mu\text{g/m}^3$) |
| **3** | `NO2` | CPCB Gas Analyzer | $\mu\text{g/m}^3$ | **YES** (some stations) | $\mu\text{g/m}^3$ or ppb $\times 1.88$ | Imputed with training median ($32.4 \ \mu\text{g/m}^3$) |
| **4** | `NOx` | CPCB Gas Analyzer | ppb | Partial / Rare | ppb | Imputed with training median ($33.1\text{ ppb}$) |
| **5** | `NH3` | CPCB Gas Analyzer | $\mu\text{g/m}^3$ | **NO** (Not reported) | N/A | Imputed with training median ($28.4 \ \mu\text{g/m}^3$) |
| **6** | `CO` | CPCB Gas Analyzer | $\text{mg/m}^3$ | **YES** (some stations) | ppb $\times 0.001145$ or $\text{mg/m}^3$ | Imputed with training median ($1.08\text{ mg/m}^3$) |
| **7** | `SO2` | CPCB Gas Analyzer | $\mu\text{g/m}^3$ | **YES** (some stations) | $\mu\text{g/m}^3$ or ppb $\times 2.62$ | Imputed with training median ($13.2 \ \mu\text{g/m}^3$) |
| **8** | `O3` | CPCB Photometer | $\mu\text{g/m}^3$ | **YES** (some stations) | $\mu\text{g/m}^3$ or ppb $\times 1.96$ | Imputed with training median ($34.5 \ \mu\text{g/m}^3$) |
| **9** | `Benzene` | Offline / GC-MS | $\mu\text{g/m}^3$ | **NO** (Absent in live feed) | N/A | Imputed with training median ($2.65 \ \mu\text{g/m}^3$) |
| **10** | `Toluene` | Offline / GC-MS | $\mu\text{g/m}^3$ | **NO** (Absent in live feed) | N/A | Imputed with training median ($8.42 \ \mu\text{g/m}^3$) |
| **11** | `Xylene` | Offline / GC-MS | $\mu\text{g/m}^3$ | **NO** (Absent in live feed) | N/A | Imputed with training median ($1.89 \ \mu\text{g/m}^3$) |
| **12** | `temperature_C` | AWS / Weather Sensor | $^\circ\text{C}$ | **YES** (some stations) | Celsius ($^\circ\text{C}$) | Imputed with training median ($27.4^\circ\text{C}$) |
| **13** | `dewpoint_C` | ERA5 / Hygrometer | $^\circ\text{C}$ | **NO** (Not reported) | N/A | Imputed with training median ($18.6^\circ\text{C}$) |
| **14** | `relative_humidity` | AWS / Weather Sensor | $\%$ | **YES** (some stations) | Percent ($\%$) | Imputed with training median ($64.2\%$) |
| **15** | `wind_speed` | Anemometer | $\text{m/s}$ | **YES** (some stations) | $\text{m/s}$ | Imputed with training median ($1.65\text{ m/s}$) |
| **16** | `precipitation_mm` | Rain Gauge / ERA5 | $\text{mm}$ | **NO** (Rarely reported) | N/A | Imputed with training median ($0.0\text{ mm}$) |
| **17** | `wind_direction_sin`| Derived from WD | $[-1.0, 1.0]$ | **YES** (if WD present) | $\sin(\text{deg} \cdot \pi / 180)$ | Imputed with training median ($-0.12$) |
| **18** | `wind_direction_cos`| Derived from WD | $[-1.0, 1.0]$ | **YES** (if WD present) | $\cos(\text{deg} \cdot \pi / 180)$ | Imputed with training median ($0.45$) |
| **19** | `hour_sin` | Clock Transform | $[-1.0, 1.0]$ | **YES** (Deterministic) | $\sin(2\pi \cdot \text{hour} / 24)$ | Causal deterministic transform |
| **20** | `hour_cos` | Clock Transform | $[-1.0, 1.0]$ | **YES** (Deterministic) | $\cos(2\pi \cdot \text{hour} / 24)$ | Causal deterministic transform |
| **21** | `month_sin` | Calendar Transform | $[-1.0, 1.0]$ | **YES** (Deterministic) | $\sin(2\pi \cdot (\text{month}-1)/12)$ | Causal deterministic transform |
| **22** | `month_cos` | Calendar Transform | $[-1.0, 1.0]$ | **YES** (Deterministic) | $\cos(2\pi \cdot (\text{month}-1)/12)$ | Causal deterministic transform |
| **23** | `day_of_week_sin` | Calendar Transform | $[-1.0, 1.0]$ | **YES** (Deterministic) | $\sin(2\pi \cdot \text{dow} / 7)$ | Causal deterministic transform |
| **24** | `day_of_week_cos` | Calendar Transform | $[-1.0, 1.0]$ | **YES** (Deterministic) | $\cos(2\pi \cdot \text{dow} / 7)$ | Causal deterministic transform |
| **25** | `Latitude` | Station Metadata | Degrees N | **YES** (Catalog) | Degrees | Station metadata |
| **26** | `Longitude` | Station Metadata | Degrees E | **YES** (Catalog) | Degrees | Station metadata |

### Key Architectural Finding:
- **Particulates ($\text{PM}_{2.5}$ and $\text{PM}_{10}$):** Genuinely available across active CAAQMS stations in OpenAQ.
- **Trace Gases & Weather:** Partially available depending on whether the station has active gas analyzer and AWS hardware.
- **Trace VOCs ($\text{Benzene}, \text{Toluene}, \text{Xylene}$), Ammonia ($\text{NH}_3$), Dewpoint ($\text{dewpoint\_C}$), and Precipitation ($\text{precipitation\_mm}$):** **Completely unobserved in real-time OpenAQ live feeds**.
- **Model Consequences:**
  - The **Post-2020 Adapted GBDT model** was explicitly designed to handle this reality: it operates exclusively on 49 features derived from $\text{PM}_{2.5}$, $\text{PM}_{10}$, calendar features, and coordinates, completely ignoring the 18 missing channels.
  - The **Production LSTM model** cannot run on live OpenAQ data without falling back on its documented median imputation for the 18 unobserved channels.

---

## 3. Empirical Live Telemetry Inspection (Location #5628)

An empirical query was executed against OpenAQ API v3 for **Location #5628 (`GVM Corporation, Visakhapatnam - APPCB`)**, matching VayuDrishti station `AP005`:

### 3.1 Live Sensor Manifest Returned
```
Location ID: 5628
Location Name: GVM Corporation, Visakhapatnam - APPCB
Coordinates: 17.722682°N, 83.308197°E (Distance from AP005: 0.89 km)
Total Active Sensors: 12
  - Sensor #12235460: pm25 (µg/m³) -> Active reading: 39.0 µg/m³
  - Sensor #12235459: pm10 (µg/m³) -> Active reading: 175.0 µg/m³
  - Sensor #12235458: o3 (µg/m³)   -> Active reading: 19.2 µg/m³
  - Sensor #12235455: co (ppb)     -> Active reading: 1.21 ppb
  - Sensor #12235462: so2 (ppb)    -> Active reading: 12.5 ppb
  - Sensor #12235456: no (ppb)     -> Active reading: 108.2 ppb
  - Sensor #12235457: no2 (ppb)    -> Active reading: 58.4 ppb
  - Sensor #14340842: nox (ppb)    -> Active reading: 0.119 ppb
  - Sensor #12235463: temperature (c) -> Active reading: 27.2 °C
  - Sensor #12235461: relativehumidity (%) -> Active reading: 73.0%
  - Sensor #14340844: wind_speed (m/s) -> Active reading: 0.3 m/s
  - Sensor #14340843: wind_direction (deg) -> Active reading: 322.0°
```

### 3.2 Missing Parameter Confirmation
Confirmed absent on location #5628: `benzene`, `toluene`, `xylene`, `nh3`, `dewpoint`, `precipitation`.

---

## 4. Station Mapping & Geographic Proximity Audit

A rigorous audit was conducted mapping VayuDrishti's 90 CAAQMS stations against OpenAQ API v3 location identifiers:

### 4.1 Danger of Proximity-Based Interchangeability
The audit revealed a critical operational finding regarding coordinate-based station discovery:
- For `AP005` (Visakhapatnam), OpenAQ location #5628 is an exact name and coordinate match (`GVM Corporation, Visakhapatnam - APPCB`, 0.89 km).
- For `AP001` (Amaravati), OpenAQ locations #5408 and #5596 are exact matches (`Secretariat, Amaravati - APPCB`).
- However, for `DL001` (**Alipur, Delhi**: $28.8153^\circ\text{N}, 77.1530^\circ\text{E}$):
  - A 10 km radius search returns OpenAQ location #13 and #5626 (**DTU, Delhi - CPCB**).
  - In VayuDrishti's catalog, DTU is station **`DL008`**, located 8.4 km south of Alipur.
  - **Conclusion:** A nearby monitor is **not interchangeable** with the target station. Blindly matching stations by nearest coordinates would assign DTU readings to Alipur, distorting localized spatial evaluation.
  - **Remedy Implemented:** `StationRollingBuffer` strictly enforces explicit station ID mapping (`station_id == self.station_id`) and cross-references `location_catalog.csv` station metadata.

---

## 5. Feasibility of Real 24-Hour Sequence Construction

### 5.1 The Telemetry Acquisition Dilemma
1. **Endpoint Granularity:** The `/locations/{id}/latest` endpoint provides only the **latest single instantaneous reading** per sensor. It does not return historical 24-hour time series in a single request.
2. **Sub-Hourly Irregularity:** Live readings arrive at variable minute offsets (e.g. 06:15, 06:45), requiring rolling accumulation and clock-hour aggregation.
3. **Hourly Query Rate Limiting:** The alternative endpoint `/sensors/{sensor_id}/hours` requires **1 separate HTTP query per individual sensor**. For 85 stations with an average of 12 sensors each, querying 24 hours of history would require $85 \times 12 = 1,020$ API calls per hour, which immediately triggers HTTP 429 rate limits on public OpenAQ tiers (rate limit: 10–20 req/min).
4. **Conclusion on Live 24-Hour Input Construction:**
   A real 24-hour contiguous input sequence can only be constructed live by **running a dedicated background ingestion service over an uninterrupted 24-hour observation period** using `/locations/{id}/latest`, or by securing an authorized bulk streaming telemetry feed directly from CPCB / state pollution control boards.
