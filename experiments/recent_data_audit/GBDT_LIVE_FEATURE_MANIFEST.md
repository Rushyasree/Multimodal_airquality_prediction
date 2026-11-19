# VayuDrishti Phase 7 — Canonical 49-Feature Standalone GBDT Manifest

**Document Version:** 1.0  
**Date:** November 2026  
**Status:** **CANONICAL REFERENCE SPECIFICATION (VERIFIED & TESTED)**  
**Target Models:**  
- `experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib` (`n_features_in_ = 49`)  
- `experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib` (`n_features_in_ = 49`)  

---

## 1. Executive Summary & Independence Proof

Phase 6B established that public live telemetry gateways (OpenAQ API v3) cannot provide the 27 continuous telemetry channels required by the 2-layer LSTM architecture. Specifically, VOCs ($\text{Benzene}, \text{Toluene}, \text{Xylene}$), Ammonia ($\text{NH}_3$), Dewpoint ($\text{dewpoint\_C}$), and Precipitation ($\text{precipitation\_mm}$) are unobserved in real-time feeds.

This manifest establishes that the **Post-2020 Adapted Gradient-Boosted Decision Tree (GBDT) Forecaster** operates **completely independently of the LSTM** and **requires zero trace gas or meteorological observations**:
1. **Mathematical Independence:** Every single one of the 49 features is computed exclusively from historical particulate measurements ($\text{PM}_{2.5}, \text{PM}_{10}$), station geographical coordinates ($\text{Latitude}, \text{Longitude}$), and the forecast issue timestamp $t$.
2. **Zero Channel Leakage:** Slicing verification confirms that setting trace gas and weather channels to zero or NaN results in a maximum absolute difference of exactly **$0.000000$** across all 49 features.
3. **Target Invariance:** The GBDT models were trained directly on ground-truth physical targets $y_{\text{PM2.5}}$ and $y_{\text{PM10}}$ in raw $\mu\text{g/m}^3$. They do not predict residuals and do not receive LSTM predictions, hidden states, or embeddings.
4. **Causality:** Every feature is strictly evaluated at forecast issue time $t$ using historical steps $k \in \{0, \dots, 23\}$ ($t-23\text{h}$ to $t$). Zero features access the target hour $t+1$ or any future timestep.

---

## 2. Exhaustive Audit of All 49 Features

The table below documents all 49 canonical features in their exact physical inference order expected by `adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib`:

| Reduced Idx | Feature Name | Mathematical Definition / Formula | Source Channel | Physical Unit / Scale | Temporal Horizon | Available at Issue Time $t$? | Constructible from OpenAQ? | Native Missingness Handling |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | `pm25_t` | $x_t^{(\text{PM2.5})}$ (Step index 23) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t$ (Current) | **YES** | **YES** | Native tree branch |
| **1** | `pm10_t` | $x_t^{(\text{PM10})}$ (Step index 23) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t$ (Current) | **YES** | **YES** | Native tree branch |
| **2** | `hour_sin_t` | $\sin(2\pi \cdot \text{hour}_t / 24.0)$ | Clock Transform | $[-1.0, 1.0]$ | $t$ (Current) | **YES** | **YES** (Deterministic) | Non-missing |
| **3** | `hour_cos_t` | $\cos(2\pi \cdot \text{hour}_t / 24.0)$ | Clock Transform | $[-1.0, 1.0]$ | $t$ (Current) | **YES** | **YES** (Deterministic) | Non-missing |
| **4** | `month_sin_t` | $\sin(2\pi \cdot (\text{month}_t - 1) / 12.0)$ | Calendar Transform | $[-1.0, 1.0]$ | $t$ (Current) | **YES** | **YES** (Deterministic) | Non-missing |
| **5** | `month_cos_t` | $\cos(2\pi \cdot (\text{month}_t - 1) / 12.0)$ | Calendar Transform | $[-1.0, 1.0]$ | $t$ (Current) | **YES** | **YES** (Deterministic) | Non-missing |
| **6** | `day_of_week_sin_t` | $\sin(2\pi \cdot \text{dow}_t / 7.0)$ | Calendar Transform | $[-1.0, 1.0]$ | $t$ (Current) | **YES** | **YES** (Deterministic) | Non-missing |
| **7** | `day_of_week_cos_t` | $\cos(2\pi \cdot \text{dow}_t / 7.0)$ | Calendar Transform | $[-1.0, 1.0]$ | $t$ (Current) | **YES** | **YES** (Deterministic) | Non-missing |
| **8** | `latitude` | Station geographic latitude | Station Metadata | Degrees N | Static | **YES** | **YES** (Catalog) | Non-missing |
| **9** | `longitude` | Station geographic longitude | Station Metadata | Degrees E | Static | **YES** | **YES** (Catalog) | Non-missing |
| **10** | `pm25_lag1` | $x_{t-1}^{(\text{PM2.5})}$ (Step index 22) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-1\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **11** | `pm10_lag1` | $x_{t-1}^{(\text{PM10})}$ (Step index 22) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-1\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **12** | `pm25_lag2` | $x_{t-2}^{(\text{PM2.5})}$ (Step index 21) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **13** | `pm10_lag2` | $x_{t-2}^{(\text{PM10})}$ (Step index 21) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **14** | `pm25_lag3` | $x_{t-3}^{(\text{PM2.5})}$ (Step index 20) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-3\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **15** | `pm10_lag3` | $x_{t-3}^{(\text{PM10})}$ (Step index 20) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-3\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **16** | `pm25_lag5` | $x_{t-5}^{(\text{PM2.5})}$ (Step index 18) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **17** | `pm10_lag5` | $x_{t-5}^{(\text{PM10})}$ (Step index 18) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **18** | `pm25_lag11` | $x_{t-11}^{(\text{PM2.5})}$ (Step index 12) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-11\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **19** | `pm10_lag11` | $x_{t-11}^{(\text{PM10})}$ (Step index 12) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-11\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **20** | `pm25_lag23` | $x_{t-23}^{(\text{PM2.5})}$ (Step index 0) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **21** | `pm10_lag23` | $x_{t-23}^{(\text{PM10})}$ (Step index 0) | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\text{h}$ | **YES** | **YES** (Buffer) | Native tree branch |
| **22** | `pm25_roll3_mean` | $\frac{1}{3}\sum_{k=21}^{23} x_k^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **23** | `pm10_roll3_mean` | $\frac{1}{3}\sum_{k=21}^{23} x_k^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **24** | `pm25_roll3_std` | $\text{std}(x_{21:23}^{(\text{PM2.5})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **25** | `pm10_roll3_std` | $\text{std}(x_{21:23}^{(\text{PM10})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **26** | `pm25_roll6_mean` | $\frac{1}{6}\sum_{k=18}^{23} x_k^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **27** | `pm10_roll6_mean` | $\frac{1}{6}\sum_{k=18}^{23} x_k^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **28** | `pm25_roll6_std` | $\text{std}(x_{18:23}^{(\text{PM2.5})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **29** | `pm10_roll6_std` | $\text{std}(x_{18:23}^{(\text{PM10})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **30** | `pm25_roll12_mean` | $\frac{1}{12}\sum_{k=12}^{23} x_k^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-11\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **31** | `pm10_roll12_mean` | $\frac{1}{12}\sum_{k=12}^{23} x_k^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-11\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **32** | `pm25_roll12_std` | $\text{std}(x_{12:23}^{(\text{PM2.5})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-11\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **33** | `pm10_roll12_std` | $\text{std}(x_{12:23}^{(\text{PM10})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-11\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **34** | `pm25_roll24_mean` | $\frac{1}{24}\sum_{k=0}^{23} x_k^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **35** | `pm10_roll24_mean` | $\frac{1}{24}\sum_{k=0}^{23} x_k^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **36** | `pm25_roll24_std` | $\text{std}(x_{0:23}^{(\text{PM2.5})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **37** | `pm10_roll24_std` | $\text{std}(x_{0:23}^{(\text{PM10})})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **38** | `pm25_diff1h` | $x_t^{(\text{PM2.5})} - x_{t-1}^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-1\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **39** | `pm10_diff1h` | $x_t^{(\text{PM10})} - x_{t-1}^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-1\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **40** | `pm25_diff3h` | $x_t^{(\text{PM2.5})} - x_{t-3}^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-3\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **41** | `pm10_diff3h` | $x_t^{(\text{PM10})} - x_{t-3}^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-3\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **42** | `pm25_diff6h` | $x_t^{(\text{PM2.5})} - x_{t-5}^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **43** | `pm10_diff6h` | $x_t^{(\text{PM10})} - x_{t-5}^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-5\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **44** | `pm25_diff24h` | $x_t^{(\text{PM2.5})} - x_{t-23}^{(\text{PM2.5})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **45** | `pm10_diff24h` | $x_t^{(\text{PM10})} - x_{t-23}^{(\text{PM10})}$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-23\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **46** | `pm25_accel` | $(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **47** | `pm10_accel` | $(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$ | Telemetry | $\mu\text{g/m}^3$ (raw) | $t-2\dots t$ | **YES** | **YES** (Buffer) | Native tree branch |
| **48** | `pm_ratio_t` | $\text{clip}(x_t^{(\text{PM2.5})} / (x_t^{(\text{PM10})} + 10^{-4}), -5, 5)$ | Telemetry | Unitless ratio | $t$ (Current) | **YES** | **YES** | Native tree branch |

---

## 3. Detailed Operator Breakdown

### 3.1 The 12 Autoregressive Lag Features
- **Horizons Sampled:** $t-1\text{h}, t-2\text{h}, t-3\text{h}, t-5\text{h}, t-11\text{h}, t-23\text{h}$.
- **Multiplier:** 6 temporal horizons $\times$ 2 particulate pollutants = **12 features**.
- **Diurnal Seasonality:** Lag $t-23\text{h}$ corresponds to sequence index 0, capturing the exact 24-hour diurnal cycle relative to forecast target $t+1\text{h}$.
- **Telemetry Feasibility:** Readily available from `StationRollingBuffer` whenever 24 contiguous clock hours have accumulated.

### 3.2 The 16 Rolling Window Statistics
- **Sub-Windows:** 3-hour ($t-2\dots t$), 6-hour ($t-5\dots t$), 12-hour ($t-11\dots t$), and 24-hour ($t-23\dots t$).
- **Operators:** Arithmetic Mean ($\mu$) and Standard Deviation ($\sigma$).
- **Multiplier:** 4 window durations $\times$ 2 aggregation operators $\times$ 2 particulate pollutants = **16 features**.
- **Physical Interpretation:** Captures short-term plume turbulence (3h std), medium-term diurnal buildup (6h/12h mean), and daily background baseline level (24h mean/std).

### 3.3 The 10 Momentum & Differencing Features
- **Temporal Spans:** 1-hour difference, 3-hour difference, 6-hour difference, 24-hour diurnal difference, and discrete second derivative (acceleration).
- **Multiplier:** 5 operators $\times$ 2 particulate pollutants = **10 features**.
- **Physical Interpretation:** First-order difference captures instantaneous rate of pollution accumulation; acceleration captures rapid onset of severe stagnation or sudden dispersion episodes.

### 3.4 Calendar & Coordinate Features (8 Features)
- **Cyclical Trigonometric Encodings:**
  - $\sin(2\pi \cdot \text{hour} / 24), \cos(2\pi \cdot \text{hour} / 24)$
  - $\sin(2\pi \cdot (\text{month}-1) / 12), \cos(2\pi \cdot (\text{month}-1) / 12)$
  - $\sin(2\pi \cdot \text{dow} / 7), \cos(2\pi \cdot \text{dow} / 7)$
- **Station Coordinates:**
  - `latitude`, `longitude`
- **Feasibility:** Deterministically computable from timestamp and static station registry with zero sensor dependencies.

---

## 4. Verification of Standalone Viability

| Requirement | Audit Result | Evidence |
| :--- | :---: | :--- |
| **Independent Checkpoint Loading** | **VERIFIED** | Loaded `adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib` with zero errors |
| **Exact Feature Count** | **VERIFIED** | Both checkpoints confirm `n_features_in_ = 49` |
| **Zero Trace Gas / Weather Dependency** | **VERIFIED** | Setting channels 2-18 to zero produces $\Delta = 0.000000$ difference in 49 features |
| **Zero LSTM Dependency** | **VERIFIED** | Standalone inference executes without instantiating or evaluating PyTorch models |
| **Causal Integrity** | **VERIFIED** | All features use step indices $k \le 23$. Step $k=24$ ($t+1$) is never accessed |
| **Native Missing Value Handling** | **VERIFIED** | `HistGradientBoostingRegressor` branches NaNs without throwing exceptions |
