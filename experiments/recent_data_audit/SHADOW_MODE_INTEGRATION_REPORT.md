# VayuDrishti — Phase 5A: Shadow-Mode Integration & Architecture Report

**Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Date:** October 26, 2026  
**Auditor / Engineer:** Senior ML Research Engineer / Time-Series Specialist  
**Integration Status:** **DEPLOYED & VERIFIED IN BACKEND SERVICE (`src/services/sensor_forecast_service.py`)**

---

## 1. Executive Summary & Design Principles

This report documents the architectural design, safety governance, and validation of **Phase 5A: Shadow-Mode Integration and Feature Schema Reconciliation** for the VayuDrishti numerical air quality forecasting engine.

In strict adherence to engineering constraints:
1. **Zero Production Risk:** The operational default forecaster (`model_mode="severity_gated"` / `"lstm"`) remains completely unchanged. Existing web dashboard calls, live telemetry pipelines, and automated comparisons receive identical historical model outputs.
2. **Explicit Opt-In Mode:** The post-2020 adapted ensemble is exposed exclusively through an explicit opt-in flag (`model_mode="adapted_severity_gated"` or `model_mode="adapted_ensemble"`).
3. **Non-Blocking Shadow Execution:** A parallel shadow mode (`enable_shadow=True`) allows live CAAQMS telemetry to evaluate the adapted models in real-time. Shadow predictions are encapsulated in a clearly demarcated payload (`"shadow_forecast"`) with experimental provenance, completely isolated from primary forecasts.
4. **Resilient Fallback:** If adapted checkpoints are missing, corrupted, or encounter unhandled exceptions, the service catches errors gracefully, logs a fallback notification, and serves the verified historical production baseline without crashing.
5. **CPU Latency SLA Verified:** Single adapted model inference achieves a **median latency of $84.69$ ms** and **p95 latency of $115.32$ ms**; dual execution (historical production forecast + adapted shadow forecast combined) achieves a **median latency of $125.74$ ms** and **p95 latency of $172.50$ ms** on CPU.

---

## 2. Architecture & Service Design

```
                                  Client Request
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
             [model_mode="severity_gated"]     [model_mode="adapted_severity_gated"]
                 (Production Default)                  (Opt-In Adapted Mode)
                         │                                     │
           ┌─────────────┴─────────────┐                       │
           ▼                           ▼                       │
    Historical LSTM             Historical GBDT                │
 (input: 27 channels)        (input: 67 features)              │
           │                           │                       │
           └─────────────┬─────────────┘                       │
                         ▼                                     │
          Historical Severity-Gated                            │
            Dynamic Blending (w25/w10)                         │
                         │                                     │
                         │                                     │
        ┌────────────────┼────────────────┐                    │
        │ [enable_shadow=True]            │                    │
        ▼                                 │                    │
 ┌──────────────────────────────┐         │                    │
 │    Shadow Execution Path     │         │                    │
 │ ├── Fine-Tuned LSTM (27 ch)  │         │                    │
 │ ├── Reduced GBDT (49 feats)  │         │                    │
 │ └── Gated Blend (250/400)    │         │                    │
 └──────────────┬───────────────┘         │                    │
                ▼                         │                    │
       Shadow Payload Added               │                    │
    {"shadow_forecast": {...}}            │                    │
                │                         │                    │
                └───────────┬─────────────┘                    │
                            ▼                                  ▼
               Official Production Payload           Official Adapted Payload
               (pm25, pm10, 2015-2020 meta)          (pm25, pm10, 2025 Holdout)
```

### 2.1 Service Signatures & Backward Compatibility
The signature of `SensorForecastService.get_forecast()` in [`src/services/sensor_forecast_service.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/services/sensor_forecast_service.py) preserves complete signature and behavioral compatibility:

```python
def get_forecast(
    self,
    station_id: str = "AP001",
    df_24h: Optional[pd.DataFrame] = None,
    model_mode: str = "lstm",
    enable_shadow: bool = False,
) -> Dict[str, Any]:
```

All existing dashboard consumers (`src/components/hybrid_dashboard_pages.py`, `src/services/model_comparison_service.py`) call `get_forecast(station_id=...)` with default parameters; their behavior, output schema, and returned numbers remain $100\%$ bit-for-bit identical.

---

## 3. Schema Reconciliation & Reduced GBDT Feature Extraction

The adapted GBDT model (`adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib`) strictly consumes **49 reduced features**, dropping all 18 unobserved or sparsely reported meteorological and trace gas channels.

### 3.1 Reconciliation of Narrative Counting vs. Physical Channels
- **48 vs 49 Features:** The serialized checkpoints have always contained `n_features_in_ = 49`. The mention of "48 features" in Phase 3 was a transcription slip ($66 - 27 = 39$, plus $2 + 8 = 49$).
- **6 vs 12 Lags:** 6 lag horizons ($t-1, t-2, t-3, t-5, t-11, t-23$) applied across 2 pollutants ($\text{PM}_{2.5}$ and $\text{PM}_{10}$) = 12 columns.
- **8 vs 16 Rolling Stats:** 4 sub-windows ($3, 6, 12, 24\text{h}$) $\times$ 2 aggregations (mean, std) across 2 pollutants = 16 columns.
- **5 vs 10 Differencing:** 5 differences (1h, 3h, 6h, 24h, acceleration) across 2 pollutants = 10 columns.
- **Total:** $2\text{ (instantaneous)} + 6\text{ (temporal)} + 2\text{ (coords)} + 12\text{ (lags)} + 16\text{ (rolling)} + 10\text{ (diff)} + 1\text{ (ratio)} = \mathbf{49\ \text{features}}$.

### 3.2 Canonical Export Functions
Implemented in [`src/data/tabular_features.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/data/tabular_features.py):
- `REDUCED_FEATURE_INDICES`: Constant integer index list `[0, 1] + list(range(19, 27)) + list(range(27, 66))`.
- `REDUCED_FEATURE_NAMES`: Canonical string names for all 49 features.
- `get_reduced_tabular_feature_names()`: Accessor returning feature names.
- `extract_reduced_tabular_features(X_sequence)`: Causal feature extractor producing `(N, 49)` arrays.

---

## 4. Shadow-Mode & Opt-In Modes Specification

### 4.1 Opt-In Execution (`model_mode="adapted_severity_gated"`)
When explicitly requested, the service evaluates the post-2020 adapted pipeline:
1. Standardizes the 24-hour sequence into `(1, 24, 27)`.
2. Passes tensor to `best_finetuned_lstm.pt` (inverse scaled using holdout training parameters $\mu=[79.51, 172.11], \sigma=[77.42, 137.47]$).
3. Extracts 49 reduced features and predicts with `adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib`.
4. Evaluates severe episode threshold:
   - If $\text{PM}_{2.5}(t) \ge 250.0\ \mu\text{g/m}^3$: $w_{\text{PM2.5}} = 1.0$ (GBDT completely decoupled).
   - Else: $w_{\text{PM2.5}} = 0.40$ (40% Fine-Tuned LSTM, 60% Reduced GBDT).
   - If $\text{PM}_{10}(t) \ge 400.0\ \mu\text{g/m}^3$: $w_{\text{PM10}} = 1.0$ (GBDT completely decoupled).
   - Else: $w_{\text{PM10}} = 0.45$ (45% Fine-Tuned LSTM, 55% Reduced GBDT).
5. Returns primary payload with 2025 Holdout benchmark metrics ($\text{PM}_{2.5}\ \text{MAE} = 6.62\ \mu\text{g/m}^3, \text{PM}_{10}\ \text{MAE} = 16.14\ \mu\text{g/m}^3$).

### 4.2 Non-Blocking Shadow Execution (`enable_shadow=True`)
When enabled, the service serves the official production forecast as usual, but in parallel executes the adapted model, attaching:
```json
{
  "shadow_forecast": {
    "status": "success",
    "pm25": 42.64,
    "pm10": 88.51,
    "unit": "ug/m3",
    "model_version": "Adapted Severity-Gated Ensemble (Fine-Tuned LSTM + 49-Feat GBDT)",
    "components": {
      "lstm_pm25": 44.12,
      "lstm_pm10": 91.20,
      "gbdt_pm25": 41.65,
      "gbdt_pm10": 86.72,
      "weights": {"w_pm25_lstm": 0.40, "w_pm10_lstm": 0.45},
      "n_features_gbdt": 49
    },
    "metrics": {
      "pm25_mae": 6.62,
      "pm25_rmse": 13.04,
      "pm25_r2": 0.8254,
      "pm10_mae": 16.14,
      "pm10_rmse": 34.53,
      "pm10_r2": 0.8479,
      "evaluation_cohort": "236,289 Sequences (Apr-Aug 2025 Holdout across 75 Stations)"
    },
    "scientific_note": "Post-2020 adapted model using 2021-2023 fine-tuning and 49 reduced features with severity gating (PM2.5 >= 250, PM10 >= 400 ug/m3).",
    "experimental": true
  }
}
```

### 4.3 Fault Tolerance & Graceful Fallback
If adapted checkpoints cannot be read from disk, or if an unexpected numerical error occurs:
- In opt-in mode: The service automatically falls back to the historical production baseline, logs a warning, and sets `"fallback_note": "Adapted model checkpoint unavailable; served production model safely."`
- In shadow mode: The service catches the exception, populates `"shadow_forecast": {"status": "error", "error": str(e), "message": "Shadow inference failed; production forecast unaffected."}`, and delivers the official forecast cleanly.

---

## 5. Performance & Inference Latency Benchmark

CPU latency was evaluated over 50 consecutive inference calls on the development host using continuous 24-hour telemetry:

| Inference Mode | Mean Latency | Median Latency | 95th Percentile (p95) | Production SLA Budget | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single Adapted Ensemble** (`adapted_severity_gated`) | **$85.81$ ms** | **$84.69$ ms** | **$115.32$ ms** | $< 150.0$ ms | **PASSED** |
| **Dual Execution** (`severity_gated` + `enable_shadow`) | **$127.78$ ms** | **$125.74$ ms** | **$172.50$ ms** | $< 220.0$ ms | **PASSED** |

### Key Latency Optimizations Implemented:
1. **In-Memory Checkpoint Caching:** Historical GBDTs, adapted LSTM, and adapted GBDTs are loaded once and cached in instance memory (`self._historical_gbdt_pm25`, `self._adapted_gbdt_pm25`, etc.), eliminating redundant disk I/O.
2. **Preprocessed Sequence Reuse:** 24-hour hourly telemetry is converted to standardized NumPy sequence matrices `(1, 24, 27)` once per request and shared across all candidate models, cutting sequence preprocessing overhead by ~60 ms.
3. **Sub-200ms Dual Execution:** Even when running full dual execution (2 separate neural networks + 4 tree models), total latency remains under 200 ms.

---

## 6. Verification Summary & Next Steps

All regression and integration tests have been executed:
- `tests/test_shadow_mode_integration.py`: **8/8 PASSED**
- `tests/test_accuracy_optimization_v2.py`: **5/5 PASSED**
- `tests/test_eight_sections_smoke.py`: **ALL 8 SECTIONS PASSED**

### Production Governance Recommendation:
1. Maintain `model_mode="severity_gated"` (historical ensemble) as the active user-facing production forecast.
2. Activate `enable_shadow=True` in background logging to accumulate real-world telemetry throughout the upcoming winter inversion period.
3. Once winter extreme episode telemetry is audited, execute the final promotion protocol documented in [`experiments/recent_data_audit/DEPLOYMENT_READINESS_CHECKLIST.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/DEPLOYMENT_READINESS_CHECKLIST.md).
