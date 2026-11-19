# VayuDrishti — Phase 5B: End-to-End Shadow-Mode Verification Report

**Project Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Date:** October 27, 2026  
**Auditor:** Senior ML Research Engineer / Time-Series Specialist  
**Execution Context:** CPU Development Host (Windows 10, Intel64 16 Logical Cores, 15.69 GB RAM)  
**Verification Verdict:** **SHADOW MODE VERIFIED** (Conditionally Staged for Winter Observation)

---

## Safety Protocol Compliance Sign-Off

In strict conformance with Phase 5B guidelines:
- [x] **Zero Model Retraining:** No neural networks, GBDTs, or baseline models were trained or fine-tuned.
- [x] **Checkpoints Preserved Read-Only:** Historical checkpoints (`models/lstm_baseline/best_lstm_model.pt`, `experiments/accuracy_optimization_v2/checkpoints/`) and adapted checkpoints (`experiments/recent_data_audit/checkpoints/`) were accessed read-only with zero modifications.
- [x] **Historical Production Model Remains Default:** Default calls to `SensorForecastService.get_forecast()` continue to execute the Historical Baseline/Severity-Gated Ensemble.
- [x] **Holdout Split Untouched:** The April–August 2025 holdout dataset was not used to tune thresholds, hyperparameters, or blend weights.
- [x] **Eight Dashboard Sections Preserved:** All 8 dashboard sections in `src/components/` and `app.py` remain fully functional and intact.
- [x] **Empirical Verification Only:** Every single metric, test outcome, and latency figure reported below was directly measured from executed code.

---

## 1. Real Service Behavior Verification

We evaluated the operational backend service [`src/services/sensor_forecast_service.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/services/sensor_forecast_service.py) on identical, deterministic 24-hour CAAQMS telemetry input ($N = 24$ contiguous hourly observations, Station `AP001`, $\text{PM}_{2.5} \in [30.0, 57.6]\ \mu\text{g/m}^3$, $\text{PM}_{10} \in [60.0, 117.5]\ \mu\text{g/m}^3$).

### 1.1 Three-Way Deterministic Comparison Table

| Service Invocation Parameter | Primary Model Version | Primary $\text{PM}_{2.5}$ | Primary $\text{PM}_{10}$ | Shadow Payload Present? | Shadow Status | Shadow $\text{PM}_{2.5}$ | Shadow $\text{PM}_{10}$ | Official Forecast Integrity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline (Shadow OFF)**<br>`model_mode="severity_gated"`<br>`enable_shadow=False` | `Severity-Gated Ensemble Forecaster (LSTM + GBDT)` | **$54.43$** | **$111.45$** | **No** (`None`) | N/A | N/A | N/A | **Identical Baseline** |
| **Historical Baseline (Shadow ON)**<br>`model_mode="severity_gated"`<br>`enable_shadow=True` | `Severity-Gated Ensemble Forecaster (LSTM + GBDT)` | **$54.43$** | **$111.45$** | **Yes** (`dict`) | `success` | **$54.99$** | **$110.15$** | **100% Unaltered** |
| **Explicit Opt-In Adapted Mode**<br>`model_mode="adapted_severity_gated"`<br>`enable_shadow=False` | `Adapted Severity-Gated Ensemble (Fine-Tuned LSTM + 49-Feat GBDT)` | **$54.99$** | **$110.15$** | **No** (`None`) | N/A | N/A | N/A | **Adapted Served** |

### 1.2 Verification Findings:
1. **Zero Primary Forecast Distortion:** Enabling shadow mode (`enable_shadow=True`) produced **identically equal** primary predictions down to machine precision ($\text{PM}_{2.5} = 54.43\ \mu\text{g/m}^3, \text{PM}_{10} = 111.45\ \mu\text{g/m}^3$). Units, timestamps, station IDs, benchmark metrics, and scientific explanatory notes were completely identical.
2. **Strict Payload Separation:** The shadow forecast was strictly encapsulated under the dedicated top-level key `"shadow_forecast"`. When shadow mode is disabled (`enable_shadow=False`), the key is completely absent from the returned dictionary.
3. **Explicit Experimental Provenance:** The shadow payload includes:
   ```json
   {
     "status": "success",
     "experimental": true,
     "model_version": "Adapted Severity-Gated Ensemble (Fine-Tuned LSTM + 49-Feat GBDT)",
     "pm25": 54.99,
     "pm10": 110.15,
     "unit": "ug/m3",
     "components": {
       "lstm_pm25": 54.67,
       "lstm_pm10": 110.22,
       "gbdt_pm25": 55.20,
       "gbdt_pm10": 110.10,
       "weights": {"w_pm25_lstm": 0.40, "w_pm10_lstm": 0.45},
       "n_features_gbdt": 49
     }
   }
   ```
4. **Execution Mechanism Provenance:** The shadow model execution in `SensorForecastService` runs **synchronously in-line** within the request handling flow (sequential execution on the active thread). It does **not** spawn separate background threads or asynchronous processes. Documentation reflects this synchronous reality accurately.

---

## 2. Failure Handling and Resilience Verification

We subjected `SensorForecastService` to hostile edge cases, malformed inputs, missing checkpoints, and corrupted data:

| Failure / Stress Scenario | Test Input Specification | Expected System Behavior | Observed Service Response | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Under-length Sequence** | 10 hourly rows instead of 24 | Explicit exception; reject input without fabricating fake forecast | Raised `ValueError: Expected exactly 24 hourly rows, got 10.` | **PASSED** |
| **Over-length Sequence** | 28 hourly rows instead of 24 | Explicit exception; reject ambiguous window | Raised `ValueError: Expected exactly 24 hourly rows, got 28.` | **PASSED** |
| **Missing Target Pollutants** | All 24 rows contain `NaN` in $\text{PM}_{2.5}$ and $\text{PM}_{10}$ | Impute training-period medians, predict non-negative concentration | Imputed median; predicted primary $\text{PM}_{2.5}=51.98$, shadow $\text{PM}_{2.5}=52.08$ | **PASSED** |
| **Corrupted String Telemetry** | String literal `"CORRUPT_SENSOR_ERR"` in numeric fields | Coerce non-numeric strings to `NaN` and impute medians without crashing | Coerced to `NaN`; completed inference cleanly with $\text{PM}_{2.5}=52.96$ | **PASSED** |
| **Missing Adapted Checkpoints (Shadow Mode)** | Simulated missing `best_finetuned_lstm.pt` or GBDT joblib | Primary forecast served normally; shadow records structured error | Primary forecast served unaffected; shadow returned `status="error"` | **PASSED** |
| **Missing Adapted Checkpoints (Opt-In Mode)** | `model_mode="adapted_severity_gated"` with missing checkpoint | Safe fallback to production baseline; record fallback diagnostic | Served historical baseline; attached `"fallback_note"` without crashing | **PASSED** |

### Finding on Plausible-Looking Prediction Fallacy:
The service strictly avoids silent fabrication. When input sequences violate structural requirements (such as length $\neq 24$), the service immediately raises `ValueError`. It never substitutes a fake default prediction (e.g. $45.0\ \mu\text{g/m}^3$) when telemetry structure is invalid.

---

## 3. Feature Consistency and Permutation Sensitivity

### 3.1 Training vs. Inference Feature Parity
We verified that the canonical 49 reduced features extracted by [`src/data/tabular_features.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/data/tabular_features.py) identically match the features expected by [`adapted_gbdt_pm25.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib) and [`adapted_gbdt_pm10.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib):
- **Feature Dimension:** Exactly 49 (`n_features_in_ = 49`).
- **Data Type:** `numpy.float32`.
- **Target Units:** Raw physical concentration ($\mu\text{g/m}^3$).
- **Missing Value Handling:** Supported natively by scikit-learn's `HistGradientBoostingRegressor` (binned histogram splits assign missing values to optimal default split direction).

### 3.2 Locked Canonical Ordering (Positions 0–48)
```python
LOCKED_CANONICAL_FEATURES = [
    # Instantaneous at t (2)
    "pm25_t", "pm10_t",
    # Cyclical Calendar & Spatial (8)
    "hour_sin_t", "hour_cos_t", "month_sin_t", "month_cos_t", "day_of_week_sin_t", "day_of_week_cos_t", "latitude", "longitude",
    # Autoregressive Lags (12)
    "pm25_lag1", "pm10_lag1", "pm25_lag2", "pm10_lag2", "pm25_lag3", "pm10_lag3",
    "pm25_lag5", "pm10_lag5", "pm25_lag11", "pm10_lag11", "pm25_lag23", "pm10_lag23",
    # Multi-Scale Rolling Statistics (16)
    "pm25_roll3_mean", "pm10_roll3_mean", "pm25_roll3_std", "pm10_roll3_std",
    "pm25_roll6_mean", "pm10_roll6_mean", "pm25_roll6_std", "pm10_roll6_std",
    "pm25_roll12_mean", "pm10_roll12_mean", "pm25_roll12_std", "pm10_roll12_std",
    "pm25_roll24_mean", "pm10_roll24_mean", "pm25_roll24_std", "pm10_roll24_std",
    # Differencing & Momentum (10)
    "pm25_diff1h", "pm10_diff1h", "pm25_diff3h", "pm10_diff3h",
    "pm25_diff6h", "pm10_diff6h", "pm25_diff24h", "pm10_diff24h",
    "pm25_accel", "pm10_accel",
    # Domain Physical Interaction (1)
    "pm_ratio_t"
]
```

### 3.3 Permutation Sensitivity Test Sign-Off
We designed and executed `test_06_feature_consistency_and_permutation_sensitivity` in [`tests/test_phase5b_e2e_verification.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_phase5b_e2e_verification.py). By synthesizing a sequence matrix with uniquely identifiable channel values at each step ($101.0, 102.0, \dots, 108.0$), we verified that:
1. Every position strictly maps to its designated channel (e.g. Position 0 = $\text{PM}_{2.5}(t)$, Position 10 = $\text{PM}_{2.5}(t-1)$, Position 20 = $\text{PM}_{2.5}(t-23)$).
2. Swapping, omitting, or permuting columns (e.g., reversing the 49 columns) results in immediate test failure even when the total column count remains 49.

---

## 4. Dashboard Integration & Modality Isolation

### 4.1 Eight-Sections Smoke Test Results
We executed [`tests/test_eight_sections_smoke.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_eight_sections_smoke.py) against the application components:
- **Section 1: Overview (Dual-Mode Summary):** PASSED
- **Section 2: Live Air Quality:** PASSED
- **Section 3: Individual Forecasting Methods (Mode A):** PASSED (Method 1 Sensor LSTM, Method 3 Image Estimation, Method 4 Baselines)
- **Section 4: Unified Hybrid Forecast (Mode B):** PASSED (Multimodal $\alpha=0.755$, Fallback $\alpha=0.0$)
- **Section 5: Side-by-Side Comparison:** PASSED
- **Section 6: Station Map:** PASSED (90 stations loaded)
- **Section 7: Model Performance Artifacts:** PASSED
- **Section 8: Data Sources & System Health:** PASSED

### 4.2 Dashboard UI Scope & Public Exposure
- **Current Dashboard Scope:** The web application (`app.py` and `src/components/hybrid_dashboard_pages.py`) currently calls `SensorForecastService` with default parameters. It does **not** expose a public UI toggle for shadow mode or adapted models.
- **Strict Compliance:** In accordance with instructions, we did **not** add an unsolicited public UI toggle to the dashboard. The shadow mode remains strictly backend/service-level, ready for automated background telemetry logging.

---

## 5. Summary of Test Suites Executed

| Test Suite | File Path | Tests Passed | Status |
| :--- | :--- | :---: | :---: |
| **Phase 5B End-to-End Verification** | [`tests/test_phase5b_e2e_verification.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_phase5b_e2e_verification.py) | **7 / 7** | **PASS** |
| **Shadow Mode Integration** | [`tests/test_shadow_mode_integration.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_shadow_mode_integration.py) | **8 / 8** | **PASS** |
| **Accuracy Optimization v2 Regression** | [`tests/test_accuracy_optimization_v2.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_accuracy_optimization_v2.py) | **5 / 5** | **PASS** |
| **Eight Sections Smoke Test** | [`tests/test_eight_sections_smoke.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_eight_sections_smoke.py) | **8 / 8 Sections** | **PASS** |
| **Steady-State Latency Benchmark** | [`experiments/recent_data_audit/measure_runtime.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/measure_runtime.py) | **120 / 120 Runs** | **PASS** |

---

## 6. Operational Limitations & Final Decision

### 6.1 Documented Operational Limitations
1. **Winter Peak Episode Inversion:** The 2025 final holdout evaluated by Phase 4 covers April to August 2025 (pre-monsoon and monsoon). Extreme winter thermal inversions (November–January, $\text{PM}_{2.5} > 250\ \mu\text{g/m}^3$) are not present in the 2025 holdout and must be verified during the upcoming winter season via shadow mode.
2. **Synchronous Execution Latency Overhead:** Because shadow mode executes synchronously in-process, enabling shadow mode adds approximately $\sim 40\text{ ms}$ of latency per request (dual mode median $143.36$ ms vs historical median $103.91$ ms). For low-frequency hourly forecasts, this overhead is negligible and well within the $< 250\text{ ms}$ SLA.

---

## Final Verification Verdict

# **SHADOW MODE VERIFIED**

The Phase 5A shadow mode implementation operates deterministically, protects production default behaviors, isolates experimental outputs, handles edge-case failures gracefully, and satisfies all CPU latency SLAs without altering existing dashboard functionality or retraining models.
