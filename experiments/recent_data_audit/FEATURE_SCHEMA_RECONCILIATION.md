# VayuDrishti — Phase 5A: Feature Schema Reconciliation Manifest

**Project Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Date:** October 26, 2026  
**Auditor:** Senior ML Research Engineer / Time-Series Specialist  
**Status:** **CANONICAL REFERENCE SPECIFICATION (VERIFIED & LOCKED)**

---

## 1. Executive Summary & Audit Resolution

During Phase 4 independent verification of the recent-telemetry adaptation experiment, minor nomenclature and counting differences were observed across historical reports:
1. **Feature Count:** Phase 3 text informally referred to "48 features", while Phase 4 audit verified exactly **49 features** in the checkpoint architecture (`n_features_in_ = 49`).
2. **Lag Count:** Summaries cited "6 lags" versus "12 lag feature columns".
3. **Rolling Statistics:** Summaries cited "8 rolling statistics" versus "16 rolling feature columns".
4. **Differencing / Momentum:** Summaries cited "5 differencing terms" versus "10 momentum columns".

This document provides the definitive, mathematically rigorous reconciliation based on the physical model checkpoints and code inspection of `src/data/tabular_features.py`, `experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib`, `experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib`, and `experiments/recent_data_audit/checkpoints/best_finetuned_lstm.pt`.

### Key Findings
- **Checkpoints Have Always Been 49 Features:** Both serialized HistGradientBoosting models possess `n_features_in_ = 49`. Zero retraining or model modification has occurred. The phrase "48 features" was a casual counting typo in narrative documentation ($66 - 27 = 39$, plus $2 + 8 = 49$, but was mistakenly transcribed in text as $38 + 10 = 48$).
- **Per-Pollutant vs. Total Column Counting:** The shorthand counting in Phase 3 described temporal operators (6 lag horizons, 8 rolling window aggregations, 5 differencing operations), whereas the actual feature matrix extracts each operator for **both** particulate pollutants ($\text{PM}_{2.5}$ and $\text{PM}_{10}$), doubling the column count.
- **Strict Causality Confirmed:** Every single feature in the 49-feature manifest is strictly computable at forecast issue time $t$ from historical sequence indices $0 \le k \le 23$. No feature accesses target step $t+1$.

---

## 2. Quantitative Feature Reconciliation

The table below reconciles the shorthand operator descriptions with the physical matrix columns:

| Component Category | Narrative Shorthand (Phase 3) | Physical Columns (Phase 4 / 5A) | Multiplier Rationale |
| :--- | :---: | :---: | :--- |
| **Instantaneous Particulates** | 2 | **2** | $\text{PM}_{2.5}(t)$ and $\text{PM}_{10}(t)$ observed at forecast issue time $t$ |
| **Cyclical Calendar Encodings**| 6 | **6** | $\sin/\cos$ pairs for Hour (2), Month (2), Day of Week (2) |
| **Geospatial Coordinates** | 2 | **2** | Station Latitude and Longitude |
| **Autoregressive Lags** | 6 lag horizons | **12** | 6 temporal horizons ($t-1, t-2, t-3, t-5, t-11, t-23$) $\times$ 2 pollutants |
| **Rolling Window Statistics** | 8 stats | **16** | 4 sub-windows (3h, 6h, 12h, 24h) $\times$ 2 aggregations (mean, std) $\times$ 2 pollutants |
| **Differencing / Momentum** | 5 differences | **10** | 5 variations (1h, 3h, 6h, 24h difference, acceleration) $\times$ 2 pollutants |
| **Domain Ratio** | 1 ratio | **1** | $\text{PM}_{2.5}(t) / (\text{PM}_{10}(t) + 10^{-4})$ |
| **Total Features** | **"48" (narrative typo)** | **49 (exact)** | **Sum: $2 + 6 + 2 + 12 + 16 + 10 + 1 = 49$ columns** |

---

## 3. Canonical 49-Feature Manifest (Exact Inference Order)

The 49 features must be fed to `adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib` in this exact column sequence. This order is produced by `src.data.tabular_features.extract_reduced_tabular_features(X_sequence)`:

| Reduced Pos | Orig Tabular Idx | Canonical Feature Name | Mathematical Definition / Operator | Units / Scale | Causality Check |
| :---: | :---: | :--- | :--- | :---: | :---: |
| **0** | 0 | `pm25_t` | Sequence step index 23, channel 0: $x_{t}^{(\text{PM2.5})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t$) |
| **1** | 1 | `pm10_t` | Sequence step index 23, channel 1: $x_{t}^{(\text{PM10})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t$) |
| **2** | 19 | `hour_sin_t` | $\sin(2\pi \cdot \text{hour}_t / 24.0)$ | $[-1.0, 1.0]$ | Causal ($t$) |
| **3** | 20 | `hour_cos_t` | $\cos(2\pi \cdot \text{hour}_t / 24.0)$ | $[-1.0, 1.0]$ | Causal ($t$) |
| **4** | 21 | `month_sin_t` | $\sin(2\pi \cdot (\text{month}_t - 1) / 12.0)$ | $[-1.0, 1.0]$ | Causal ($t$) |
| **5** | 22 | `month_cos_t` | $\cos(2\pi \cdot (\text{month}_t - 1) / 12.0)$ | $[-1.0, 1.0]$ | Causal ($t$) |
| **6** | 23 | `day_of_week_sin_t` | $\sin(2\pi \cdot \text{dow}_t / 7.0)$ | $[-1.0, 1.0]$ | Causal ($t$) |
| **7** | 24 | `day_of_week_cos_t` | $\cos(2\pi \cdot \text{dow}_t / 7.0)$ | $[-1.0, 1.0]$ | Causal ($t$) |
| **8** | 25 | `latitude` | Station geographic latitude | Degrees N | Static metadata |
| **9** | 26 | `longitude` | Station geographic longitude | Degrees E | Static metadata |
| **10** | 27 | `pm25_lag1` | Step index 22 ($t-1$ hour) channel 0 | $\mu\text{g/m}^3$ (raw) | Causal ($t-1$) |
| **11** | 28 | `pm10_lag1` | Step index 22 ($t-1$ hour) channel 1 | $\mu\text{g/m}^3$ (raw) | Causal ($t-1$) |
| **12** | 29 | `pm25_lag2` | Step index 21 ($t-2$ hours) channel 0 | $\mu\text{g/m}^3$ (raw) | Causal ($t-2$) |
| **13** | 30 | `pm10_lag2` | Step index 21 ($t-2$ hours) channel 1 | $\mu\text{g/m}^3$ (raw) | Causal ($t-2$) |
| **14** | 31 | `pm25_lag3` | Step index 20 ($t-3$ hours) channel 0 | $\mu\text{g/m}^3$ (raw) | Causal ($t-3$) |
| **15** | 32 | `pm10_lag3` | Step index 20 ($t-3$ hours) channel 1 | $\mu\text{g/m}^3$ (raw) | Causal ($t-3$) |
| **16** | 33 | `pm25_lag5` | Step index 18 ($t-5$ hours) channel 0 | $\mu\text{g/m}^3$ (raw) | Causal ($t-5$) |
| **17** | 34 | `pm10_lag5` | Step index 18 ($t-5$ hours) channel 1 | $\mu\text{g/m}^3$ (raw) | Causal ($t-5$) |
| **18** | 35 | `pm25_lag11` | Step index 12 ($t-11$ hours) channel 0 | $\mu\text{g/m}^3$ (raw) | Causal ($t-11$) |
| **19** | 36 | `pm10_lag11` | Step index 12 ($t-11$ hours) channel 1 | $\mu\text{g/m}^3$ (raw) | Causal ($t-11$) |
| **20** | 37 | `pm25_lag23` | Step index 0 ($t-23$ hours, 24h diurnal lag to $t+1$) | $\mu\text{g/m}^3$ (raw) | Causal ($t-23$) |
| **21** | 38 | `pm10_lag23` | Step index 0 ($t-23$ hours, 24h diurnal lag to $t+1$) | $\mu\text{g/m}^3$ (raw) | Causal ($t-23$) |
| **22** | 39 | `pm25_roll3_mean` | Mean of steps 21..23 ($t-2 \dots t$) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-2 \dots t$) |
| **23** | 40 | `pm10_roll3_mean` | Mean of steps 21..23 ($t-2 \dots t$) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-2 \dots t$) |
| **24** | 41 | `pm25_roll3_std` | Std dev of steps 21..23 ($t-2 \dots t$) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-2 \dots t$) |
| **25** | 42 | `pm10_roll3_std` | Std dev of steps 21..23 ($t-2 \dots t$) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-2 \dots t$) |
| **26** | 43 | `pm25_roll6_mean` | Mean of steps 18..23 ($t-5 \dots t$) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-5 \dots t$) |
| **27** | 44 | `pm10_roll6_mean` | Mean of steps 18..23 ($t-5 \dots t$) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-5 \dots t$) |
| **28** | 45 | `pm25_roll6_std` | Std dev of steps 18..23 ($t-5 \dots t$) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-5 \dots t$) |
| **29** | 46 | `pm10_roll6_std` | Std dev of steps 18..23 ($t-5 \dots t$) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-5 \dots t$) |
| **30** | 47 | `pm25_roll12_mean` | Mean of steps 12..23 ($t-11 \dots t$) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-11 \dots t$) |
| **31** | 48 | `pm10_roll12_mean` | Mean of steps 12..23 ($t-11 \dots t$) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-11 \dots t$) |
| **32** | 49 | `pm25_roll12_std` | Std dev of steps 12..23 ($t-11 \dots t$) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-11 \dots t$) |
| **33** | 50 | `pm10_roll12_std` | Std dev of steps 12..23 ($t-11 \dots t$) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-11 \dots t$) |
| **34** | 51 | `pm25_roll24_mean` | Mean of full 24h window (steps 0..23) for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-23 \dots t$) |
| **35** | 52 | `pm10_roll24_mean` | Mean of full 24h window (steps 0..23) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-23 \dots t$) |
| **36** | 53 | `pm25_roll24_std` | Std dev of full 24h window (steps 0..23) for $\text{PM}_{2.5}$| $\mu\text{g/m}^3$ (raw) | Causal ($t-23 \dots t$) |
| **37** | 54 | `pm10_roll24_std` | Std dev of full 24h window (steps 0..23) for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-23 \dots t$) |
| **38** | 55 | `pm25_diff1h` | $x_t^{(\text{PM2.5})} - x_{t-1}^{(\text{PM2.5})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-1 \dots t$) |
| **39** | 56 | `pm10_diff1h` | $x_t^{(\text{PM10})} - x_{t-1}^{(\text{PM10})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-1 \dots t$) |
| **40** | 57 | `pm25_diff3h` | $x_t^{(\text{PM2.5})} - x_{t-3}^{(\text{PM2.5})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-3 \dots t$) |
| **41** | 58 | `pm10_diff3h` | $x_t^{(\text{PM10})} - x_{t-3}^{(\text{PM10})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-3 \dots t$) |
| **42** | 59 | `pm25_diff6h` | $x_t^{(\text{PM2.5})} - x_{t-6}^{(\text{PM2.5})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-6 \dots t$) |
| **43** | 60 | `pm10_diff6h` | $x_t^{(\text{PM10})} - x_{t-6}^{(\text{PM10})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-6 \dots t$) |
| **44** | 61 | `pm25_diff24h` | $x_t^{(\text{PM2.5})} - x_{t-23}^{(\text{PM2.5})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-23 \dots t$) |
| **45** | 62 | `pm10_diff24h` | $x_t^{(\text{PM10})} - x_{t-23}^{(\text{PM10})}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-23 \dots t$) |
| **46** | 63 | `pm25_accel` | $(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$ for $\text{PM}_{2.5}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-2 \dots t$) |
| **47** | 64 | `pm10_accel` | $(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$ for $\text{PM}_{10}$ | $\mu\text{g/m}^3$ (raw) | Causal ($t-2 \dots t$) |
| **48** | 65 | `pm_ratio_t` | $\text{clip}(x_t^{(\text{PM2.5})} / (x_t^{(\text{PM10})} + 10^{-4}), -5.0, 5.0)$ | Unitless ratio | Causal ($t$) |

---

## 4. Excluded Unobserved Features Manifest (18 Channels)

The following 18 channels from the historical 67-feature set are explicitly omitted from the adapted GBDT model:

| Tabular Idx | Channel Name | Category | Exclusion Reason |
| :---: | :--- | :--- | :--- |
| **2** | `no_t` | Instantaneous Gas | Nitrogen oxide: missing/unreported in > 65% of recent CPCB CAAQMS stations |
| **3** | `no2_t` | Instantaneous Gas | Nitrogen dioxide: missing/unreported in > 40% of recent stations |
| **4** | `nox_t` | Instantaneous Gas | Total NOx: high missingness; redundant with NO/NO2 |
| **5** | `nh3_t` | Instantaneous Gas | Ammonia: sensor degradation and > 70% missingness across Indian stations |
| **6** | `co_t` | Instantaneous Gas | Carbon monoxide: sparse and erratic calibration across network |
| **7** | `so2_t` | Instantaneous Gas | Sulfur dioxide: missing in > 50% of recent monitoring periods |
| **8** | `o3_t` | Instantaneous Gas | Ozone: high diurnal and photochemical drift; sparse availability |
| **9** | `benzene_t` | VOC | Volatile organic compound: offline laboratory or rare automated monitor |
| **10** | `toluene_t` | VOC | Volatile organic compound: rarely reported in live CAAQMS streams |
| **11** | `xylene_t` | VOC | Volatile organic compound: rarely reported in live CAAQMS streams |
| **12** | `temperature_t` | Meteorology | Ambient temperature: often absent on particulate-only CAAQMS stations |
| **13** | `dewpoint_t` | Meteorology | Dewpoint temperature: requires external hygrometer or ERA5 reanalysis |
| **14** | `relative_humidity_t`| Meteorology | Humidity: frequently missing or uncalibrated on low-cost sensor nodes |
| **15** | `wind_speed_t` | Meteorology | Anemometer: frequently offline or obstructed at urban ground stations |
| **16** | `precipitation_t` | Meteorology | Rain gauge: typically zero and sparse in non-monsoon periods |
| **17** | `wind_dir_sin_t` | Meteorology | Wind direction sine encoding: unavailable when anemometer is offline |
| **18** | `wind_dir_cos_t` | Meteorology | Wind direction cosine encoding: unavailable when anemometer is offline |
| **66** | `dewpoint_depression_t` | Derived Physics | Derived from Temperature and Dewpoint: cannot be computed without both |

---

## 5. Model Scaling, Target Alignment, and Missing-Value Protocol

### 5.1 Target Scale & Physical Units
- **Target Scale:** All predictions represent physical ambient air pollutant concentrations in **$\mu\text{g/m}^3$**.
- **GBDT Processing:** `adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib` are trained directly on unscaled target values. Outputs are in $\mu\text{g/m}^3$ and require no inverse transformation.
- **LSTM Processing:** `best_finetuned_lstm.pt` uses internal z-score scaling:
  $$\hat{y}_{\text{physical}} = \hat{y}_{\text{norm}} \cdot \sigma_{\text{target}} + \mu_{\text{target}}$$
  where:
  - $\mu_{\text{target}} = [79.51448, 172.1068]\ \mu\text{g/m}^3$
  - $\sigma_{\text{target}} = [77.41743, 137.4705]\ \mu\text{g/m}^3$

### 5.2 Missing-Value Handling
- The underlying `HistGradientBoostingRegressor` provides native support for `NaN` values. Missing inputs are treated as informative missingness during tree building without requiring artificial constant imputation.
- When an individual station telemetry channel has momentary dropout, feature extraction proceeds without throwing exceptions.
- Physical clamping ensures non-negativity: $\hat{y} \leftarrow \max(0.0, \hat{y})$.

---

## 6. Implementation Verification Checklist

- [x] Canonical slice indices verified: `[0, 1] + list(range(19, 27)) + list(range(27, 66))` produces exactly 49 elements.
- [x] Helper functions `get_reduced_tabular_feature_names()` and `extract_reduced_tabular_features()` exported in `src/data/tabular_features.py`.
- [x] Checkpoint metadata confirms `n_features_in_ = 49` on both PM2.5 and PM10 GBDT models.
- [x] Frozen LSTM input size confirmed at 27 continuous telemetry channels.
- [x] Verification test suite implemented and passing.
