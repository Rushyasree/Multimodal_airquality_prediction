# Phase 17.2 — Initial Upstream Source Architecture & Provenance Audit

> **Audit Date**: November 12, 2026  
> **Repository Baseline**: Phase 17.1 (Commit `22ef0bb6f8dd65e854a095a62c84c7f8d3f6c7bb`)  
> **Scope**: Pre-implementation audit of upstream air quality data sources, temporal resolution, observation schemas, live buffer architecture, and provenance propagation.

---

## 1. Executive Summary & Purpose

Phase 17.2 establishes a scientifically rigorous, end-to-end live forecasting pathway using authentic physical monitoring station measurements. The purpose of this initial audit is to inspect the existing data acquisition layer, identify upstream data providers and schemas, document timestamp characteristics and irregularities, and expose scientific boundaries before creating the strict live sequence builder.

### Key Audit Findings
1. **Upstream Source Heterogeneity**: The platform currently accesses three distinct data tiers:
   - **Physical CAAQMS Inlets via OpenAQ API v3**: Delivers genuine physical sensor readings from CPCB, APPCB, and DPCC continuous ambient monitors. Observations are often instantaneous or sub-hourly, with potential transmission delays and reporting dropouts.
   - **Atmospheric Model Reanalysis via Open-Meteo**: Delivers numerical reanalysis estimates derived from ECMWF / Copernicus Atmosphere Monitoring Service (CAMS). While strictly regular and hourly, these are **modeled regional reanalysis estimates**, not physical station measurements.
   - **Deterministic Offline Baseline**: A mathematical diurnal sinusoidal model used strictly for offline resilience when networks are unreachable.
2. **Current Limitations in Forecast Sequence Acquisition**:
   - In Phase 17.1, when `df_24h=None` was passed to `SensorForecastService`, `HybridPredictionService`, or `ModelComparisonService`, the pipeline defaulted to `get_verified_24h_sequence_for_station()`. This returned static curated historical benchmark sequences from `data/processed/dashboard_demo_sequences.csv`.
   - While Phase 17.1 successfully ensured that these historical sequences are labeled `HISTORICAL DEMO FORECAST` and never masqueraded as live data, there is currently **no end-to-end pathway that pulls a live 24-hour sequence of measured observations directly into the forecasting models**.
3. **Sequence Validator Provenance Limitation**:
   - In `src/prediction/sequence_validator.py`, the validator classifies a sequence as `LIVE_MEASURED` based solely on age ($\text{age} \le \text{max\_age\_hours}$) and station matching. It lacks knowledge of the upstream acquisition origin. A synthetic or reanalysis sequence with fresh timestamps would be erroneously accepted as `LIVE_MEASURED` without an upstream provenance contract.

---

## 2. Upstream Data Sources & Schema Analysis

### 2.1 OpenAQ API v3 (Physical CAAQMS Monitoring Inlets)
* **API Endpoint**: `https://api.openaq.org/v3`
* **Coverage**: Official continuous monitoring stations (CAAQMS) operated by Central Pollution Control Board (CPCB) and Andhra Pradesh Pollution Control Board (APPCB).
* **Observation Nature**: Physical instrument measurements (Beta Attenuation Monitor for PM2.5/PM10, UV Fluorescence for SO2, Chemiluminescence for NOx, NDIR for CO, UV Photometry for O3).
* **Payload Structure (`/locations/{id}/latest`)**:
  ```json
  {
    "results": [
      {
        "datetime": {
          "utc": "2026-11-12T03:00:00Z",
          "local": "2026-11-12T08:30:00+05:30"
        },
        "parameter": {
          "id": 2,
          "name": "pm25",
          "units": "µg/m³",
          "displayName": "PM2.5"
        },
        "value": 46.2,
        "coordinates": {"latitude": 16.515083, "longitude": 80.518167}
      }
    ]
  }
  ```
* **Cadence & Irregularities**:
  - Upstream CPCB telemetry is uploaded at irregular intervals (often sub-hourly, e.g. 15-minute bursts, or delayed by 1–3 hours).
  - Stations frequently experience channel-specific sensor dropouts (e.g., PM2.5 reporting while SO2 or CO is offline for calibration).
  - Telemetry is **not** pre-interpolated into a contiguous 24-hour matrix by OpenAQ.

### 2.2 Open-Meteo Air Quality API (Atmospheric Reanalysis)
* **API Endpoint**: `https://air-quality-api.open-meteo.com/v1/air-quality`
* **Coverage**: Global numerical gridded atmospheric model (ECMWF IFS / CAMS reanalysis).
* **Observation Nature**: Numerically simulated regional atmospheric chemistry model.
* **Payload Structure**:
  ```json
  {
    "current": {
      "time": "2026-11-12T03:00",
      "pm10": 74.0,
      "pm2_5": 38.5,
      "carbon_monoxide": 450.0,
      "nitrogen_dioxide": 21.0,
      "sulphur_dioxide": 8.0,
      "ozone": 52.0
    }
  }
  ```
* **Scientific Boundary**: Provides reliable regional atmospheric estimates when physical station inlets are down or beyond 50 km, but **must never be badged as a physical station inlet measurement**.

### 2.3 Curated Benchmark Sequences (`dashboard_demo_sequences.csv`)
* **File Location**: `data/processed/dashboard_demo_sequences.csv`
* **Content**: 576 rows across 12 stations, 24 scenarios of 24 contiguous hours each.
* **Source**: Curated from historical CPCB/APPCB 2026/2018 test partitions.
* **Scientific Boundary**: Static benchmark sequences for reproducible offline demonstration and evaluation. Validated and labeled strictly as `HISTORICAL DEMO FORECAST`.

---

## 3. Data Availability & Upstream Latency Realities

| Data Source | Provider / Origin | Measurement Nature | Frequency | Latency / Freshness | Live Forecast Suitability |
|:---|:---|:---|:---:|:---:|:---|
| **CPCB / OpenAQ Inlets** | CPCB / APPCB / DPCC via OpenAQ API v3 | Direct physical inlet sensor | Irregular / 15-min to 1-hr | 0.5h to 3.0h (often delayed) | **Eligible for `LIVE_MEASURED`** when 24h contiguous window is assembled |
| **Open-Meteo Regional** | ECMWF / CAMS Atmospheric Reanalysis | Gridded mathematical reanalysis | Regular hourly | Real-time | **`MODELED_REANALYSIS` only**; ineligible for `LIVE_MEASURED` |
| **Calibrated Baseline** | VayuDrishti Climatological Model | Deterministic diurnal calculation | Instantaneous | Current clock | **`SYNTHETIC_FALLBACK` only**; diagnostic baseline |
| **Dashboard Sequences** | Static CSV Holdout | Historical physical measurements | Regular hourly | Fixed in past | **`HISTORICAL_DEMO` only**; evaluation benchmark |

---

## 4. Phase 17.2 Implementation Roadmap

Based on this source audit, Phase 17.2 requires:

1. **Explicit Data Provenance Schema (`src/prediction/observation_provenance.py`)**:
   - Define formal data classes: `ObservationRecord`, `SequenceProvenanceMeta`, and `ObservationProvenanceCategory`.
   - Enforce 7 mutually exclusive categories: `LIVE_MEASURED`, `LIVE_PARTIAL`, `HISTORICAL_DEMO`, `REGIONAL_PROXY`, `MODELED_REANALYSIS`, `SYNTHETIC_FALLBACK`, `INVALID`.
   - Track upstream provider, station ID, measurement timestamps, retrieval timestamps, raw values, and aggregation method.
2. **Strict Live Sequence Builder (`src/prediction/live_sequence_builder.py`)**:
   - Ingest physical measurements from OpenAQ or station buffers.
   - Enforce deterministic clock-hour aggregation ($H:00$ to $H:59$ arithmetic mean) with sample count tracking.
   - Verify that all 24 hours contain non-null PM2.5 and PM10 measurements.
   - Zero silent synthetic replacement: if $< 24$ measured hours exist, reject live forecast mode and return `LIVE_PARTIAL` or `LIVE INPUT UNAVAILABLE`.
3. **Strict Live Validation (`src/prediction/sequence_validator.py`)**:
   - Enhance `validate_live_forecast_sequence` to separately report:
     - `sequence_structural_validity`
     - `station_identity_validity`
     - `source_authenticity` (verified upstream physical inlet)
     - `measurement_completeness`
     - `freshness_validity` (relative to real forecast issue time)
     - `scaler_compatibility`
     - `live_forecast_eligibility`
4. **Service Integration**:
   - Connect `SensorForecastService`, `HybridPredictionService`, and `ModelComparisonService` to the live sequence builder.
   - Expose explicit `live=True` inference with full provenance return.
5. **Model Comparison Integrity**:
   - Assert SHA-256 tensor parity and reject live synchronization if any model ingests historical or synthetic fallback input.
6. **Dashboard UI Refinement**:
   - Explicit badges: `LIVE MEASURED OBSERVATION`, `LIVE FORECAST — VALIDATED INPUT`, `HISTORICAL DEMO FORECAST`, `REGIONAL MODEL ESTIMATE`, `IMAGE-BASED EXPERIMENTAL ESTIMATE`, `LIVE INPUT UNAVAILABLE`.
