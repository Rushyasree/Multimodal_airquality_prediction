# VayuDrishti Phase 12: Forecast Horizon Verification & Audit Report

**Document Status:** Complete Audit & Verification  
**Investigation Date:** November 2026  
**Workspace:** `e:\Documents\multimodal_air_quality_vscode_starter`  
**Execution Environment:** Python 3.10.11 / PyTorch 2.12.1+cpu / LightGBM 4.6.0  

---

## 1. Executive Summary: Clarifying the Horizon Confusion

Prior documentation, presentation materials, and colloquial descriptions in the repository frequently referenced a "24-hour forecast". This audit investigated every single model, dataset preparation script, sequence generator, evaluation pipeline, and service in the codebase to determine the ground-truth prediction horizon.

### Definitive Finding:
1. **Input Sequence Horizon:** 24 contiguous historical hourly observations ($t-23, t-22, \dots, t-1, t$).
2. **Forecast Target Horizon:** **Next 1 Hour ($t+1$)**.
3. **Ground-Truth Consensus:** **100% of the forecasting models** (Historical Sensor-Only LSTM, Environmental CNN Hybrid, Standalone GBDT, Residual GBDT, and 1-Hour Persistence Baseline) are trained, evaluated, and served in production to predict **the immediate next hour ($t+1$)**.
4. **Discrepancy Resolution:** There is **NO model in the codebase that predicts $t+24$ hours ahead**. The phrase "24h forecast" was an informal shorthand for *"forecast driven by a 24-hour antecedent sequence"* or *"24-hour historical sparkline display"*.
5. **Spatial Heatmap Compatibility:** Because all models share the exact same $t+1$ target horizon, their predictions are mathematically synchronized and can be safely interpolated spatially across stations at time $t+1$.

---

## 2. Component-by-Component Horizon Verification

### 2.1 Historical Sequence Dataset Generation (`prepare_model_sequences.py`)
- **Source File:** `prepare_model_sequences.py` (Lines 183–193, Lines 310–330)
- **Input Window:**
  ```python
  seq["history_end"] - seq["history_start"] == pd.Timedelta(hours=WINDOW_SIZE - 1)  # 24 hours (t-23 to t)
  ```
- **Target Timestamp Definition:**
  ```python
  assert (seq["history_end"] + pd.Timedelta(hours=1) == seq["target_time"]).all(), \
      "Some targets are not exactly one hour after history."
  ```
- **Target Value Source:** `merged.loc[df["Datetime"] == row.target_time, ["PM2.5", "PM10"]]`
- **Verified Horizon:** **$t+1$ hour**.

---

### 2.2 Historical Sensor LSTM Baseline (`train_lstm.py` & `src/prediction/lstm_forecaster.py`)
- **Source Files:**
  - `train_lstm.py` (Lines 60–140)
  - `src/prediction/lstm_forecaster.py` (Lines 55, Lines 440–445)
- **Model Signature:**
  - Input: `(batch_size, 24, 27)` $\to$ 24-hour historical sequence of 27 features.
  - Target: `(batch_size, 2)` $\to$ `[PM2.5, PM10]` loaded from `y_train.npy` (derived from `seq["target_time"]`).
- **Configuration Attribute:**
  ```python
  # src/prediction/lstm_forecaster.py line 55:
  "forecast_horizon_hours": 1
  # src/prediction/lstm_forecaster.py line 443:
  "forecast_horizon": "t+1 (next 1 hour)"
  ```
- **Verified Horizon:** **$t+1$ hour**.

---

### 2.3 Post-2020 Adapted Standalone GBDT (`scripts/run_recent_adaptation_experiment.py`)
- **Source File:** `scripts/run_recent_adaptation_experiment.py` (Lines 236–253)
- **Sliding Window Extraction:**
  ```python
  # Windows of length 24
  windows = np.lib.stride_tricks.sliding_window_view(F, window_shape=24, axis=0)
  # Target extracted immediately at index 24 (1 step ahead of window index 23)
  t25_cand = raw_pm25[24 : 24 + n_cand]
  t10_cand = raw_pm10[24 : 24 + n_cand]
  # Issue-time observation at index 23
  iss25_cand = raw_pm25[23 : 23 + n_cand]
  ```
- **Verified Horizon:** Index 23 is issue time $t$; Index 24 is target time $t+1$. **Exact $t+1$ hour horizon**.

---

### 2.4 Residual GBDT & Severity-Gated Ensemble (`src/services/sensor_forecast_service.py`)
- **Source File:** `src/services/sensor_forecast_service.py` (Lines 410–415, Lines 530–535)
- **Service Output Dictionary:**
  ```python
  return {
      "forecast_horizon": "Next 1 Hour (t + 1)",
      "target_time": target_time,
      "predicted_pm25": y_pred_25,
      "predicted_pm10": y_pred_10,
      ...
  }
  ```
- **Target Time Calculation:**
  ```python
  target_time = (end_dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
  ```
- **Verified Horizon:** **$t+1$ hour**.

---

### 2.5 Unified Multimodal Hybrid Model (`src/services/hybrid_prediction_service.py`)
- **Source File:** `src/services/hybrid_prediction_service.py` (Lines 118–122, Lines 284–287)
- **Service Output Dictionary:**
  ```python
  return {
      "forecast_horizon": "Next 1 Hour (t + 1)",
      "target_time": target_time,
      "predicted_pm25": round(pred_pm25, 2),
      "predicted_pm10": round(pred_pm10, 2),
      ...
  }
  ```
- **Verified Horizon:** **$t+1$ hour**.

---

### 2.6 Persistence Baseline (`evaluate_persistence_baseline.py`)
- **Source File:** `evaluate_persistence_baseline.py` (Lines 52–68)
- **Implementation:** Uses observation at `history_end` as prediction for `target_time = history_end + 1 hour`.
- **Verified Horizon:** **$t+1$ hour**.

---

## 3. Horizon Audit Summary Matrix

| Model / Pipeline Name | Input Window (History) | Target Index / Offset | Output Target Timestamp | Nominal Claim in Docs | Ground-Truth Code Horizon | Horizon Consistency Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical LSTM Baseline** | 24 Hours ($t-23 \to t$) | $t+1$ hour | `history_end + 1h` | "24h forecast" (ambiguous) | **$t+1$ Hour** | **CONSISTENT ($t+1$)** |
| **Environmental CNN Hybrid** | 24 Hours ($t-23 \to t$) + Image | $t+1$ hour | `history_end + 1h` | "Next-day / 24h" | **$t+1$ Hour** | **CONSISTENT ($t+1$)** |
| **Standalone LightGBM** | 24 Hours ($t-23 \to t$) | $t+1$ hour | `history_end + 1h` | "1h lead" | **$t+1$ Hour** | **CONSISTENT ($t+1$)** |
| **Residual GBDT Ensemble** | 24 Hours ($t-23 \to t$) | $t+1$ hour | `history_end + 1h` | "Next hour" | **$t+1$ Hour** | **CONSISTENT ($t+1$)** |
| **Persistence Baseline** | 1 Hour ($t$) | $t+1$ hour | `history_end + 1h` | "1h persistence" | **$t+1$ Hour** | **CONSISTENT ($t+1$)** |
| **Live Telemetry Pipeline** | 24 Hours ($t-23 \to t$) | $t+1$ hour | `end_dt + 1h` | "Next 1 Hour" | **$t+1$ Hour** | **CONSISTENT ($t+1$)** |

---

## 4. Policy for Spatial Interpolation & Heatmap

Since **all models produce $t+1$ forecasts**, there is zero temporal incompatibility between models regarding lead time. However, to maintain strict scientific integrity:
1. **Single-Source Heatmap Enforcement:** The spatial interpolation layer must never blend predictions from different model families into the same grid (e.g., blending LSTM with GBDT).
2. **Synchronous Target Timestamp:** All station predictions ingested by the spatial interpolator must share the identical `target_time` ISO string.
3. **Explicit Labeling:** The spatial interpolation output and dashboard UI must explicitly state:
   - **Forecast Issue Time ($t$):** Observation time of input sequence.
   - **Forecast Target Time ($t+1$):** Prediction target hour.
   - **Lead Time / Horizon:** **1 Hour Ahead ($t+1$)**.
