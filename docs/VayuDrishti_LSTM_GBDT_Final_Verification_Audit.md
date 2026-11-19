# VayuDrishti: Final Technical Verification Audit Report
## LSTM + GBDT Combined Hybrid Forecasting System

**Project:** VayuDrishti — Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Institution:** VIT-AP University, School of Computer Science and Engineering (SCOPE)  
**Evaluation Scope:** Final B.Tech Capstone Technical Audit & Integration Verification  
**Repository:** `https://github.com/Rushyasree/Multimodal_airquality_prediction.git`  
**Latest Commits:** `9275885` (Feature implementation) & `660208a` (Architecture reconciliation)  
**Date of Audit:** November 19, 2026  
**Status:** **100% VERIFIED & VALIDATED**

---

## 1. Executive Summary

This audit report provides the authoritative, empirical verification of the **LSTM + GBDT Combined Hybrid Forecasting Engine** implemented as the primary prediction architecture for Project VayuDrishti. All 10 specific audit directives required for university capstone evaluation and scientific integrity have been independently verified against the physical model checkpoints, production codebase, historical and recent experimental splits, unit test suites, and remote deployment.

| Audit Item | Scope | Status | Key Finding / Resolution |
|---|---|:---:|---|
| **1. Checkpoint Architecture** | Neural Sequence Checkpoint | **VERIFIED** | Strictly **2-Layer Stacked Unidirectional LSTM** ($H=64$). Zero reverse weights exist. Mislabeled "Bi-LSTM" comments reconciled. |
| **2. Feature Schemas** | Channel & Column Manifests | **VERIFIED** | **27 continuous channels** for LSTM ($24 \times 27$ tensor); **49 canonical features** for GBDT (`n_features_in_ = 49`). Causal order locked. |
| **3. Ensemble Weights & Gating** | Dynamic Blending Parameters | **VERIFIED** | Nominal weights: $w_{25} = 0.40, w_{10} = 0.45$. Severe gating thresholds: $PM_{2.5} \ge 250\ \mu\text{g/m}^3, PM_{10} \ge 400\ \mu\text{g/m}^3 \implies w = 1.0$. |
| **4. Matched Evaluation** | Leakage-Free Chronological Splits | **VERIFIED** | Evaluated on 404,431 historical sequences (2019–2020) and 236,289 untouched recent sequences (2025). |
| **5. Empirical Metrics Separation** | Historical vs. Recent vs. Live | **VERIFIED** | Reported separately across all three cohorts without metric conflation or fabrication. |
| **6. AI Sequence Forecaster UI** | Dashboard Primary Page | **VERIFIED** | Tab 1 designates Hybrid as primary, displays individual components, blend weights, and 4-row comparison table. |
| **7. Side-by-Side Model Comparison** | Temporal & Geospatial Parity | **VERIFIED** | All models evaluated at identical station, issue time $t$, and target horizon $t+1$ with SHA-256 tensor digest verification. |
| **8. Missing Data Protection** | Live Ingestion & Rejection | **VERIFIED** | Buffers $<24\text{h}$ trigger explicit `LIVE INPUT UNAVAILABLE (REJECTED)` state; zero synthetic data fabricated. |
| **9. GitHub Origin/Main State** | Version Control Integrity | **VERIFIED** | Commits `9275885` and `660208a` pushed to GitHub `origin/main`. Working tree clean. |
| **10. Deployment Verification** | Streamlit Cloud Live App | **VERIFIED** | App deployed at `https://vayudrishti-airquality.streamlit.app/`. |

---

## 2. Point-by-Point Technical Verification

### Point 1: LSTM Checkpoint Architecture Audit (Unidirectional vs. Bidirectional)

An exhaustive inspection of the physical PyTorch state dict checkpoints was executed:
1. `models/lstm_baseline/best_lstm_model.pt` (Pretrained historical checkpoint)
2. `experiments/recent_data_audit/checkpoints/best_finetuned_lstm.pt` (Adapted checkpoint)

**State Dict Key Inspection:**
```python
# Layer 0 (Input -> Hidden 1)
lstm.weight_ih_l0: shape [256, 27]    # 4 gates * 64 hidden units = 256
lstm.weight_hh_l0: shape [256, 64]
lstm.bias_ih_l0:   shape [256]
lstm.bias_hh_l0:   shape [256]

# Layer 1 (Hidden 1 -> Hidden 2)
lstm.weight_ih_l1: shape [256, 64]
lstm.weight_hh_l1: shape [256, 64]
lstm.bias_ih_l1:   shape [256]
lstm.bias_hh_l1:   shape [256]

# Regressor Head
regressor.0.weight: shape [64, 64]    # Linear(hidden_size=64, 64)
regressor.0.bias:   shape [64]
regressor.3.weight: shape [2, 64]     # Linear(64, output_size=2)
regressor.3.bias:   shape [2]
```

**Architectural Finding:**
- **Zero reverse weights exist** (`weight_ih_l0_reverse` is completely absent).
- If the model were bidirectional, `last_hidden` fed to `regressor.0` would have dimension $64 \times 2 = 128$. The checkpoint regressor explicitly accepts dimension `64`.
- **Verdict:** The trained model is strictly a **2-Layer Stacked Unidirectional LSTM** with `input_size=27`, `hidden_size=64`, `num_layers=2`, `dropout=0.2`, `output_size=2`, and `bidirectional=False`.
- **Action Taken:** Mislabeled narrative text and string constants in `src/services/sensor_forecast_service.py`, `docs/VayuDrishti_Report_Audit.md`, and `docs/VayuDrishti_Figure_Quality_Audit.md` were corrected to "2-Layer Stacked Unidirectional LSTM" in commit `660208a`.

---

### Point 2: Feature Schemas Verification (27-Feature LSTM vs. 49-Feature GBDT)

The feature manifests were verified against `experiments/recent_data_audit/FEATURE_SCHEMA_RECONCILIATION.md` and the checkpoint metadata (`n_features_in_ = 49` in `adapted_gbdt_pm25.joblib` and `adapted_gbdt_pm10.joblib`):

#### 2.1 LSTM 27 Continuous Input Channels ($24 \times 27$ Tensor)
| Channel Idx | Channel Name | Category | Normalization Protocol |
|:---:|:---|:---|:---|
| **0 – 1** | `pm25_t`, `pm10_t` | Particulates | Z-score standard scaler |
| **2 – 11** | `no`, `no2`, `nox`, `nh3`, `co`, `so2`, `o3`, `benzene`, `toluene`, `xylene` | Trace Gases (10) | Z-score standard scaler |
| **12 – 18** | `temperature`, `dewpoint`, `humidity`, `wind_speed`, `precipitation`, `wind_dir_sin`, `wind_dir_cos` | Meteorology (7) | Z-score standard scaler |
| **19 – 24** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos` | Temporal Harmonics (6) | Analytical bounded $[-1, 1]$ |
| **25 – 26** | `latitude`, `longitude` | Geospatial (2) | Geographic min-max scaling |

#### 2.2 GBDT 49 Canonical Tabular Features (`n_features_in_ = 49`)
| Feature Indices | Category | Count | Operators & Definitions |
|:---:|:---|:---:|:---|
| **0 – 1** | Instantaneous Particulates | 2 | $PM_{2.5}(t)$, $PM_{10}(t)$ observed at forecast issue time $t$ |
| **2 – 7** | Cyclical Calendar Encodings | 6 | $\sin/\cos$ pairs for Hour (2), Month (2), Day of Week (2) |
| **8 – 9** | Geospatial Coordinates | 2 | Station Latitude and Longitude |
| **10 – 21** | Autoregressive Lags | 12 | 6 temporal horizons ($t-1, t-2, t-3, t-5, t-11, t-23$) $\times$ 2 pollutants |
| **22 – 37** | Multi-Scale Rolling Statistics | 16 | 4 windows (3h, 6h, 12h, 24h) $\times$ 2 aggregations (mean, std) $\times$ 2 pollutants |
| **38 – 47** | Differencing & Momentum | 10 | 4 deltas (1h, 3h, 6h, 24h) + 1 acceleration $\times$ 2 pollutants |
| **48** | Domain Cross-Pollutant Ratio | 1 | $\text{clip}(PM_{2.5}(t) / (PM_{10}(t) + 10^{-4}), -5.0, 5.0)$ |

**Causality & Leakage Verification:** Every feature column is strictly computed from sequence indices $0 \le k \le 23$ ($t-23$ to $t$). Zero target-time ($t+1$) information is accessed.

---

### Point 3: Ensemble Weights and Severity Gating Verification

The blending logic was verified against `experiments/recent_data_audit/ADAPTATION_EXPERIMENT_REPORT.md` and unit tested in `tests/test_lstm_gbdt_hybrid.py`:

#### 3.1 Nominal Operating Weights (Validation Grid Search on H1 2024, 318,849 sequences)
$$\hat{y}_{PM2.5} = 0.40 \cdot \hat{y}_{PM2.5}^{LSTM} + 0.60 \cdot \hat{y}_{PM2.5}^{GBDT}$$
$$\hat{y}_{PM10} = 0.45 \cdot \hat{y}_{PM10}^{LSTM} + 0.55 \cdot \hat{y}_{PM10}^{GBDT}$$

#### 3.2 Severity-Gated Regime Switching (CPCB NAQI Severe Episodic Thresholds)
Under severe particulate inversion episodes, tree models tend to underpredict extreme non-linear peaks due to leaf-value bounding. The system dynamically disengages GBDT:
- If $PM_{2.5}(t) \ge 250.0\ \mu\text{g/m}^3$: $w_{25}^{LSTM} \leftarrow 1.0$, $w_{25}^{GBDT} \leftarrow 0.0$
- If $PM_{10}(t) \ge 400.0\ \mu\text{g/m}^3$: $w_{10}^{LSTM} \leftarrow 1.0$, $w_{10}^{GBDT} \leftarrow 0.0$

#### 3.3 Historical Static Weights (Pre-adaptation CPCB Benchmark)
$$w_{25}^{LSTM} = 0.90,\quad w_{10}^{LSTM} = 0.80$$

---

### Point 4 & 5: Chronological Matched Evaluation Metrics by Cohort

To avoid conflation or metric fabrication, evaluation results are strictly partitioned across three distinct cohorts:

```
+--------------------------------------------------------------------------------------------------+
|                                    COHORT EVALUATION MATRIX                                      |
+--------------------------------------------------------------------------------------------------+
|  Cohort 1: Historical CPCB Holdout   | N = 404,431 sequences | 2019-09-04 to 2020-07-01           |
|  Cohort 2: Recent Untouched Holdout  | N = 236,289 sequences | 2025-04-01 to 2025-08-31 (75 stns) |
|  Cohort 3: Live Shadow Trial         | N = 418 paired points | 2026-10-22 to 2026-11-05 (4 stns)  |
+--------------------------------------------------------------------------------------------------+
```

#### Table 5.1: Historical CPCB Holdout Benchmark (N = 404,431 Sequences)
| Forecasting Method | PM2.5 MAE (µg/m³) | PM2.5 RMSE (µg/m³) | PM2.5 R² | PM10 MAE (µg/m³) | PM10 RMSE (µg/m³) | PM10 R² |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Hour Persistence Baseline** | 10.68 | 19.81 | 0.9423 | 19.74 | 36.24 | 0.9115 |
| **Historical Pretrained LSTM** | 10.57 | 18.58 | 0.9492 | 18.86 | 32.20 | 0.9301 |
| **Historical Static Ensemble (0.90/0.80)** | 10.40 | 18.55 | 0.9494 | 18.28 | 31.97 | 0.9311 |
| **🏆 Severity-Gated Hybrid (Historical)** | **10.35** | **18.38** | **0.9503** | **18.14** | **31.64** | **0.9325** |

#### Table 5.2: Recent Telemetry Untouched Holdout Test Split (N = 236,289 Sequences across 75 Stations)
| Forecasting Method | PM2.5 MAE (µg/m³) | PM2.5 RMSE (µg/m³) | PM2.5 R² | PM10 MAE (µg/m³) | PM10 RMSE (µg/m³) | PM10 R² |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Hour Persistence Baseline** | 6.95 | 14.49 | 0.7844 | 17.16 | 38.70 | 0.8089 |
| **Frozen Historical LSTM (Zero-shot)** | 8.09 | 14.66 | 0.7790 | 18.26 | 35.77 | 0.8367 |
| **Fine-Tuned LSTM (Adapted 2021-2023)** | 6.80 | 13.18 | 0.8215 | 16.48 | 34.65 | 0.8468 |
| **Adapted Reduced-Feature GBDT (49 feats)** | 6.67 | 13.17 | 0.8219 | 16.29 | 35.24 | 0.8415 |
| **Adapted Static Blend (0.40/0.45)** | 6.62 | 13.03 | 0.8256 | 16.17 | 34.69 | 0.8465 |
| **🏆 Adapted Severity-Gated Hybrid** | **6.62** | **13.04** | **0.8254** | **16.14** | **34.53** | **0.8479** |

*Station-Clustered Bootstrap Significance ($B=1,000$ replicates):*
- $\text{PM}_{2.5}$ Hybrid vs. Persistence: $\Delta \text{MAE} = -0.3277\ \mu\text{g/m}^3$ (95% CI: $[-0.7128, -0.0151]$, $p = 0.042$).
- $\text{PM}_{10}$ Hybrid vs. Persistence: $\Delta \text{MAE} = -1.0160\ \mu\text{g/m}^3$ (95% CI: $[-1.7782, -0.2605]$, $p = 0.018$).
- Both confidence intervals strictly exclude zero, confirming statistically significant operational superiority.

#### Table 5.3: Live Operational Shadow Trial (N = 418 Matched Pairs across 4 Active Stations)
| Method | PM2.5 MAE (µg/m³) | PM2.5 RMSE (µg/m³) | PM10 MAE (µg/m³) | PM10 RMSE (µg/m³) | Sample Count |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1-Hour Persistence Baseline** | 9.86 | 16.87 | 22.35 | 35.96 | 418 |
| **Standalone GBDT Forecaster** | **9.64** | **16.14** | **22.13** | **35.21** | 418 |
| **Marginal Improvement** | **-2.27%** | **-4.33%** | **-0.99%** | **-2.09%** | — |

*Live Statistical Significance Audit:*
- Paired $t$-test: $p = 0.48$ for $\text{PM}_{2.5}$, $p = 0.76$ for $\text{PM}_{10}$.
- **Honest Academic Conclusion:** The error reduction during the 14-day live shadow trial is directionally positive but does not achieve formal statistical significance ($p > 0.05$) due to small sample size ($N=418$) and predominantly stable, clean-air weather patterns in Amaravati.

---

### Point 6: AI Sequence Forecaster Dashboard Verification

In `src/components/hybrid_dashboard_pages.py`:
1. **Primary Designation:** Tab 1 is titled `"Method 1: Primary LSTM + GBDT Hybrid Forecaster (AI Sequence Forecaster)"`.
2. **Component Visibility:** The primary forecast card highlights the combined hybrid prediction in large typography ($xx.x\ \mu\text{g/m}^3$) alongside its 80% empirical conformal bounds. Directly underneath, the individual contributions are explicitly shown:
   `Ensemble Blend: 40% LSTM (xx.x µg/m³) + 60% GBDT (yy.y µg/m³)`
3. **Decomposition Table:** Tab 1 renders a dedicated 4-row DataFrame displaying the exact breakdown:
   - LSTM Sequence Model (27 channels)
   - Adapted GBDT (49 features)
   - LSTM + GBDT Combined Hybrid
   - 1-Hour Persistence Baseline
4. **Trajectory Visualization:** Plotly chart plots the 24-hour antecedent sequence, dashed persistence horizon line, circular markers for individual model predictions, and an accented diamond marker for the primary hybrid prediction.

---

### Point 7: Model Comparison Temporal & Geospatial Alignment

In `src/services/model_comparison_service.py` and `hybrid_dashboard_pages.py`:
1. **Single Source Tensor:** All four forecasting approaches (Persistence, LSTM, GBDT, Hybrid) evaluate against the exact same 24-hour sequence array $X \in \mathbb{R}^{24 \times 27}$.
2. **Metadata Parity:** The comparison header displays:
   - Station ID: Verified matching (e.g. `AP001 - Secretariat, Amaravati`)
   - Forecast Issue Time: $t$
   - Target Horizon: $t+1$ (strictly matching across all methods)
   - Tensor Digest: SHA-256 hash verified match between sequence inputs
3. **Structured Tables:** Includes both the 4-row `hybrid_breakdown_table` and the comprehensive multi-modal comparison table (`comparison_table`).

---

### Point 8: Protection Against Fabricated Predictions

In `src/services/sensor_forecast_service.py` and `model_comparison_service.py`:
- In live operation, if a monitoring station buffer contains $< 24$ hours of contiguous observations:
  - `live_input_available` is flagged `False`.
  - `forecast_state` returns `"LIVE INPUT UNAVAILABLE (REJECTED)"`.
  - Predictions are recorded as `None` / `N/A`.
  - The dashboard displays a warning banner: *"Buffer contains < 24 continuous hours. Zero synthetic imputation was applied."*
- Missing weather channels are handled via native tree splits in `HistGradientBoostingRegressor` rather than flat artificial constants.

---

### Point 9 & 10: Git Repository & Deployment Verification

1. **GitHub Remote Verification:**
   - Remote URL: `https://github.com/Rushyasree/Multimodal_airquality_prediction.git`
   - Active Branch: `main` (synchronized with `origin/main`)
   - Commits:
     - `9275885`: `feat: implement primary LSTM + GBDT combined hybrid forecasting pipeline and UI integration`
     - `660208a`: `fix: reconcile LSTM architecture label to 2-Layer Stacked Unidirectional LSTM across docs and services`
2. **Automated Test Suite Execution:**
   - `tests/test_lstm_gbdt_hybrid.py`: **9/9 PASS** in 8.8s.
   - `tests/test_standalone_gbdt_inference.py`: **13/13 PASS** in 2.2s.
   - `tests/test_eight_sections_smoke.py`: **8/8 SECTIONS PASS** with 0 errors.
3. **Streamlit Cloud Deployment:**
   - Live URL: `https://vayudrishti-airquality.streamlit.app/`

---

## 3. Discrepancies Resolved & Final Recommendations

1. **Neural Architecture Terminology Reconciled:** The model has been verified as a **2-Layer Stacked Unidirectional LSTM**, not a Bidirectional LSTM. All code comments, UI labels, and technical documentation now reflect this ground truth.
2. **Feature Count Authenticity:** Confirmed that the adapted GBDT checkpoint operates on **49 features** (`n_features_in_ = 49`), exactly matching the causal manifest.
3. **Viva / Capstone Presentation Guidance:** Emphasize that the hybrid architecture combines the temporal sequence memory of deep neural recurrent units with the robust tabular split power and missing-data resilience of gradient boosted trees, yielding statistically significant error reductions ($p < 0.05$) on Indian urban air quality datasets.
