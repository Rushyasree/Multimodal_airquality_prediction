# VayuDrishti Phase 6B — Telemetry Buffer & Live Readiness Audit Report

**Document Version:** 1.0  
**Date:** November 2026  
**Auditor:** Independent Research & Deployment Verification Team  
**Evaluation Status:** **BLOCKED BY MISSING LIVE FEATURES**

---

## 1. Executive Summary & Core Verdict

Phase 6B completed an exhaustive architectural and empirical readiness evaluation of live telemetry ingestion, model input requirements, and rolling buffer mechanics for the VayuDrishti air quality forecasting system.

Following empirical inspection of OpenAQ API v3 feeds, station coordinate mappings, and historical vs adapted model requirements, the deployment status is formally designated as:

$$\mathbf{BLOCKED \ BY \ MISSING \ LIVE \ FEATURES}$$

### Verdict Justification:
1. **Missing Trace Gas & Meteorological Channels in Public Live Telemetry:**
   - The production baseline LSTM (`best_lstm_model.pt`) and the fine-tuned adapted LSTM (`best_finetuned_lstm.pt`) both possess an input dimension of **27 channels** ($\mathbf{X} \in \mathbb{R}^{24 \times 27}$).
   - Live telemetry from OpenAQ API v3 delivers particulate matter ($\text{PM}_{2.5}, \text{PM}_{10}$) and sporadic criteria gases ($\text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$), but **completely lacks volatile organic compounds ($\text{Benzene}, \text{Toluene}, \text{Xylene}$), Ammonia ($\text{NH}_3$), Dewpoint ($\text{dewpoint\_C}$), and Precipitation ($\text{precipitation\_mm}$)**.
   - Running the historical production LSTM on live OpenAQ data requires substituting historical training-period medians for all 18 unobserved channels. In accordance with strict research rigor, **such substitutions must not be silently masked or claimed as genuine live multivariate observations**.
2. **Buffer Infrastructure Fully Built & Verified:**
   - A station-specific, timestamp-safe rolling buffer (`src/data/live_rolling_buffer.py`) was engineered and verified with 100% test pass rates.
   - The buffer guarantees strict UTC normalization, deterministic sub-hourly to hourly aggregation ($H:00$ to $H:59$ arithmetic mean), zero future leakage (quarantines $t+1$), physical plausibility filtering, and exact 24-hour contiguous sequence verification.
   - If any required channel or hourly step is absent, the buffer halts forecasting and records explicit reason codes: `insufficient_history`, `missing_required_channels`, `stale_data`, or `station_mapping_failed`.
3. **Contiguous 24-Hour Input Construction Constraint:**
   - OpenAQ's `/locations/{id}/latest` endpoint returns only single instantaneous snapshots. It does not provide historical 24-hour contiguous sequences in a single query.
   - Real-time 24-hour contiguous sequences can only be populated by maintaining an active rolling buffer running continuously for $\ge 24$ hours or by acquiring an authorized institutional bulk telemetry feed from the Central Pollution Control Board (CPCB).

---

## 2. Summary of Project Changes

The following files were created, enhanced, and validated during Phase 6B:

| File Path | Nature of Change | Purpose & Scope |
| :--- | :---: | :--- |
| `src/data/live_rolling_buffer.py` | **NEW** | Production implementation of `StationRollingBuffer` and `LiveRollingBufferRegistry`. Implements UTC normalization, sub-hourly aggregation, deduplication, contiguous 24h verification, and multi-station isolation. |
| `src/services/shadow_observation_runner.py` | **ENHANCED** | Added `check_live_readiness()` and `record_forecast_from_buffer()` methods to enforce station mapping, 24h input completeness, freshness, and dry-run inference before any shadow forecast can be logged. |
| `tests/test_live_rolling_buffer.py` | **NEW** | Comprehensive unit test suite with 12 synthetic tests covering 24h completeness, missing hours, deduplication, sub-hourly aggregation, timezone conversions, staleness, unphysical readings, station isolation, future leakage prevention, and runner integration. |
| `tests/test_live_openaq_connectivity.py` | **NEW** | Read-only live OpenAQ API v3 connectivity test verifying credential masking, location metadata parsing, sensor parameter inspection, and shadow scoring isolation. |
| `experiments/recent_data_audit/LIVE_BUFFER_DESIGN.md` | **NEW** | Architectural specification of the rolling buffer, mathematical aggregation formalisms, physical plausibility bounds, and multi-model schema separation. |
| `experiments/recent_data_audit/LIVE_SOURCE_FEATURE_COVERAGE.md` | **NEW** | Exhaustive channel-by-channel reconciliation matrix across all 27 features, empirical OpenAQ sensor inspection, XKDR database audit, and station mapping caveats. |
| `experiments/recent_data_audit/PHASE6B_READINESS_REPORT.md` | **NEW** | This canonical audit report detailing verification results, telemetry gaps, and the definitive readiness status. |

---

## 3. Automated Verification & Test Results

All test suites were executed on the local Python 3.10 host environment with zero errors:

### 3.1 Live Rolling Buffer Test Suite (`tests/test_live_rolling_buffer.py`)
```
Ran 12 tests in 20.891s
OK (100% Passed)
- test_01_exactly_24_contiguous_hours: PASS
- test_02_missing_hours_reported_as_insufficient_history: PASS
- test_03_deduplication_of_readings: PASS
- test_04_sub_hourly_irregular_timestamp_aggregation: PASS
- test_05_timezone_conversions: PASS
- test_06_stale_data_rejection: PASS
- test_07_missing_required_channels: PASS
- test_08_unphysical_sensor_readings_rejected: PASS
- test_09_station_id_mismatch_rejected: PASS
- test_10_future_data_leakage_prevention: PASS
- test_11_multi_station_isolation_and_registry: PASS
- test_12_shadow_runner_readiness_integration: PASS
```

### 3.2 Live OpenAQ Read-Only Connectivity Suite (`tests/test_live_openaq_connectivity.py`)
```
Ran 4 tests in 1.714s (Outside Sandbox)
OK (100% Passed)
- test_01_credential_protection_and_masking: PASS (Key masked as a2b2...936f)
- test_02_read_only_location_metadata_and_sensor_schema: PASS (Location 5628 AP005 verified)
- test_03_read_only_latest_measurements_parsing: PASS (Validated values and ISO datetimes)
- test_04_shadow_scoring_remains_disabled: PASS (Runner remains disabled by default)
```

### 3.3 System Regression & Health Check
- **Shadow Observation Runner Suite (`tests/test_shadow_observation_runner.py`):** `6/6 PASSED`.
- **Shadow Mode Integration Suite (`tests/test_shadow_mode_integration.py`):** `8/8 PASSED`.
- **Phase 5B End-to-End Suite (`tests/test_phase5b_e2e_verification.py`):** `7/7 PASSED`.
- **Dashboard Eight-Section Smoke Suite (`tests/test_eight_sections_smoke.py`):** `ALL 8 SECTIONS PASSED`.

---

## 4. Telemetry Source Coverage & Gap Analysis

### 4.1 Channels Genuinely Available via Live OpenAQ
- **Particulates:** $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$), $\text{PM}_{10}$ ($\mu\text{g/m}^3$).
- **Criteria Gases:** $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, $\text{NO}$, $\text{NO}_x$ (available on select stations, often reported in ppb requiring unit conversion).
- **Meteorology:** Ambient Temperature ($^\circ\text{C}$), Relative Humidity ($\%$), Wind Speed ($\text{m/s}$), Wind Direction (degrees).

### 4.2 Channels Permanently Absent in Live OpenAQ
- **Volatile Organic Compounds (VOCs):** $\text{Benzene}$, $\text{Toluene}$, $\text{Xylene}$ (absent in live public streams across all Indian stations).
- **Ammonia ($\text{NH}_3$):** Not reported in live OpenAQ feeds.
- **Derived Meteorological Physics:** Dewpoint Temperature ($\text{dewpoint\_C}$) and Precipitation ($\text{precipitation\_mm}$).

### 4.3 Model Alignment & Imputation Disclosure
- **Adapted GBDT Forecaster (`adapted_gbdt_pm25.joblib`, `adapted_gbdt_pm10.joblib`):**
  - Uses strictly **49 tabular features** derived from $\text{PM}_{2.5}$, $\text{PM}_{10}$, calendar transforms, and station coordinates.
  - Excludes all 18 missing trace gas and meteorological channels.
  - Can operate cleanly on live particulate telemetry without artificial imputation.
- **Production Baseline LSTM (`best_lstm_model.pt`):**
  - Requires all 27 features in its input tensor.
  - Can run on live data only by imputing the 18 missing channels with training-set medians.
  - `StationRollingBuffer.check_readiness()` explicitly reports these 18 channels in `unobserved_channels_imputed`, ensuring full disclosure to downstream consumers.

---

## 5. Station Mapping & Proximity Verification

An audit of 90 CAAQMS monitoring stations against OpenAQ locations demonstrated that **geographic proximity does not guarantee station identity**:
- In dense metropolitan areas such as Delhi, querying OpenAQ by coordinates for `DL001` (Alipur) returns monitors for `DL008` (DTU), which is 8.4 km away.
- Merging data from a neighboring monitor under the label of another station would distort spatial accuracy benchmarks.
- `StationRollingBuffer` prevents cross-station contamination by enforcing strict station ID checks on every ingested reading.

---

## 6. Operational Prerequisites for Future Live Shadow Trial

Before live shadow observation can transition from `BLOCKED BY MISSING LIVE FEATURES` to `READY FOR CONTROLLED LIVE SHADOW TRIAL`:
1. **Telemetry Feed Resolution:**
   - Either establish an authorized streaming integration directly with CPCB's CAAQMS telemetry gateway to obtain the complete criteria pollutant spectrum, OR
   - Formally accept and document the policy that the production baseline operates in **Median-Imputed Sensor Mode** for the 18 unobserved channels while the adapted GBDT operates in **Particulate-Only Mode**.
2. **24-Hour Buffer Accumulation:**
   - Run a scheduled telemetry accumulator for an uninterrupted 24-hour period to populate contiguous 24-hour sequences across active target stations.
3. **Explicit Opt-In Execution:**
   - Enable shadow recording strictly via `shadow_trial_config.json` (`"shadow_trial_enabled": true`) after the 24-hour buffer readiness checks pass.
