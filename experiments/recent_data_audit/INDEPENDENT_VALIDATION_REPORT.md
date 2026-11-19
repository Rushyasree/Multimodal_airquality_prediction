# VayuDrishti — Phase 4: Independent Validation and Deployment Readiness Audit Report

**Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Audit Completion Date:** October 25, 2026  
**Auditor:** Senior ML Research Engineer / Time-Series Specialist  
**Evaluation Cohort:** Final Forward Chronological Holdout (April 1, 2025 to August 31, 2025)  
**Evaluated Sample Size:** $N = 236,289$ contiguous 24-hour sequence forecasts across 75 active CPCB CAAQMS monitoring stations  
**Canonical Predictions File:** [`experiments/recent_data_audit/final_predictions.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/final_predictions.csv) (64.2 MB, 236,289 rows)  
**Paired Bootstrap Audit Manifest:** [`experiments/recent_data_audit/paired_recent_bootstrap.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/paired_recent_bootstrap.json)  
**Deployment Checklist:** [`experiments/recent_data_audit/DEPLOYMENT_READINESS_CHECKLIST.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/DEPLOYMENT_READINESS_CHECKLIST.md)  

---

## Executive Summary & Independent Audit Verdict

This audit independently assesses the predictive validity, methodological integrity, statistical rigor, and production deployment readiness of the **Phase 3 Recent-Data Adaptation Experiment** for the VayuDrishti air quality forecasting system.

In strict compliance with audit protocols:
- **Zero Retraining Executed:** No models, weights, thresholds, or hyperparameters were re-trained or modified.
- **Checkpoints Preserved:** All historical and adapted checkpoints were loaded read-only.
- **Production Isolation:** The existing production dashboard and backend services were strictly preserved with zero modifications to their operational default behaviors.
- **Independent Recalculation:** All metrics, predictions, confidence intervals, and p-values were recomputed from scratch using frozen checkpoints and the raw telemetry sequence pipeline.

### Audit Verdict: **CONDITIONALLY READY FOR DEPLOYMENT**

1. **Reproduction Accuracy: 100.0% Match (0.0000 Discrepancy)**  
   Every reported metric on the 2025 holdout—including MAE, RMSE, $R^2$, and Mean Bias for all 8 candidate architectures—reproduced with exact numerical precision down to four decimal places.
2. **Defeating 1-Hour Persistence Confirmed**:  
   The **Adapted Severity-Gated Ensemble** genuinely outperforms the 1-Hour Persistence baseline on the 2025 holdout across both particulate targets:
   - $\text{PM}_{2.5}$ MAE: **$6.6199\ \mu\text{g/m}^3$** vs **$6.9476\ \mu\text{g/m}^3$** ($\Delta = -0.3277\ \mu\text{g/m}^3$, $p = 0.042$)
   - $\text{PM}_{10}$ MAE: **$16.1425\ \mu\text{g/m}^3$** vs **$17.1584\ \mu\text{g/m}^3$** ($\Delta = -1.0160\ \mu\text{g/m}^3$, $p = 0.018$)
   - $\text{PM}_{10}$ RMSE: **$34.5278\ \mu\text{g/m}^3$** vs **$38.6982\ \mu\text{g/m}^3$** ($10.8\%$ reduction in quadratic error)
3. **Statistical Robustness Verified Across Three Bootstrap Methodologies**:  
   Paired bootstrap testing confirms that the ensemble's improvement over persistence is statistically significant ($p < 0.05$) under block-subsampling, full station-clustered bootstrap, and macro-station bootstrap.
4. **Causality & Zero-Leakage Confirmed**:  
   All 49 features used by the adapted models are strictly computable at forecast issue time $t$. No future observations ($t+1$) or cross-station information leak into the sequence matrices.
5. **Operational Constraint Flagged**:  
   The 2025 final holdout covers **April 1 to August 31, 2025** (warm, pre-monsoon, and monsoon months). Because winter extreme inversion months (November–January) are absent in the 2025 holdout, full production promotion is **conditioned** on shadow validation through the upcoming winter season.

---

## 1. Independent Metric Reproduction (2025 Holdout)

We generated canonical model predictions across all **236,289 sequence instances** and exported them to [`final_predictions.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/final_predictions.csv). All metrics were recalculated directly from the unscaled ground truth targets and saved predictions.

### Side-by-Side Verification Table ($N = 236,289$ Sequences, 75 Stations):

| Model Architecture | Recomputed PM2.5 MAE | Reported PM2.5 MAE | PM2.5 Delta | Recomputed PM2.5 RMSE | Recomputed PM2.5 R² | Recomputed PM10 MAE | Reported PM10 MAE | PM10 Delta | Recomputed PM10 RMSE | Recomputed PM10 R² | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Hour Persistence Baseline** | **6.9476** | 6.9476 | **0.0000** | 14.4869 | 0.7844 | **17.1584** | 17.1584 | **0.0000** | 38.6982 | 0.8089 | **EXACT MATCH** |
| **Frozen Historical LSTM** | **8.0870** | 8.0870 | **0.0000** | 14.6640 | 0.7790 | **18.2638** | 18.2638 | **0.0000** | 35.7690 | 0.8367 | **EXACT MATCH** |
| **Historical Static Ensemble (0.90/0.80)** | **7.8797** | 7.8797 | **0.0000** | 14.4109 | 0.7866 | **17.6677** | 17.6677 | **0.0000** | 35.3211 | 0.8408 | **EXACT MATCH** |
| **Historical Severity-Gated Ensemble** | **7.8815** | 7.8815 | **0.0000** | 14.4763 | 0.7847 | **17.6700** | 17.6700 | **0.0000** | 35.3920 | 0.8402 | **EXACT MATCH** |
| **Fine-Tuned LSTM (Adapted 2021-2023)** | **6.8011** | 6.8011 | **0.0000** | 13.1813 | 0.8215 | **16.4791** | 16.4791 | **0.0000** | 34.6463 | 0.8468 | **EXACT MATCH** |
| **Adapted Reduced-Feature GBDT** | **6.6716** | 6.6716 | **0.0000** | 13.1663 | 0.8219 | **16.2906** | 16.2906 | **0.0000** | 35.2406 | 0.8415 | **EXACT MATCH** |
| **Adapted Ensemble (FT-LSTM + GBDT)** | **6.6216** | 6.6216 | **0.0000** | 13.0285 | 0.8256 | **16.1660** | 16.1660 | **0.0000** | 34.6875 | 0.8465 | **EXACT MATCH** |
| **Adapted Severity-Gated Ensemble** | **6.6199** | 6.6199 | **0.0000** | 13.0366 | 0.8254 | **16.1425** | 16.1425 | **0.0000** | 34.5278 | 0.8479 | **EXACT MATCH** |

*All evaluated models share identical target rows, target timestamps, station coordinates, and physical units ($\mu\text{g/m}^3$).*

---

## 2. Leakage and Chronological Causality Audit

We conducted an exhaustive audit of feature engineering, sequence construction, and target alignment:

### 2.1 Forecast Horizon & Step Alignment
- **Input Horizon:** Sequence indices $0$ to $23$ represent exactly 24 contiguous historical hours ($t-23, \dots, t$).
- **Issue Time:** Index $23$ represents forecast issue time $t$. The persistence baseline reads strictly from index $23$.
- **Target Time:** Target $y_{t+1}$ corresponds to index $24$ ($t+1$ hour ahead).
- **Finding:** Target hour alignment is strictly causal. There is zero overlap between the historical input window $[t-23, t]$ and the prediction target $t+1$.

### 2.2 Sequence Boundary Integrity
- In `extract_split_sequences`, sequences are constructed strictly station-by-station.
- Reindexing to a 1-hour regular grid occurs per station.
- Sequences are explicitly prevented from crossing station boundaries.
- **Finding:** PASSED with 0 cross-station sequence contaminations.

### 2.3 Reduced-Feature GBDT Feature Causality Audit
The report described the tabular model as using "48 features". Our code audit confirmed that the feature selection slicing `[0, 1] + list(range(19, 27)) + list(range(27, 66))` produces **exactly 49 features** (`Shape: (N, 49)`). We inspected every single feature:

| Feature Group | Indices | Dimension | Description | Provenance & Causality Check |
| :--- | :---: | :---: | :--- | :--- |
| **Instantaneous Observations** | 0, 1 | 2 | $\text{PM}_{2.5}(t)$, $\text{PM}_{10}(t)$ | Causal; observed at issue time $t$ |
| **Cyclical Temporal Encodings** | 19–24 | 6 | Hour, Month, Day-of-Week (sin & cos) | Causal; deterministic function of timestamp $t$ |
| **Station Geospatial Coordinates** | 25, 26 | 2 | Latitude, Longitude | Static station metadata |
| **Autoregressive Lags** | 27–38 | 12 | Lags at $t-1, t-2, t-3, t-5, t-11, t-23$ | Causal; strictly past observations ($k \le 22$) |
| **Rolling Window Statistics** | 39–54 | 16 | Means & Stds over 3h, 6h, 12h, 24h past sub-windows | Causal; computed strictly over historical steps $\le 23$ |
| **Differencing & Momentum** | 55–64 | 10 | 1h, 3h, 6h, 24h rate-of-change & acceleration | Causal; backward differences from step $23$ |
| **Particulate Ratio** | 65 | 1 | $\text{PM}_{2.5}(t) / (\text{PM}_{10}(t) + \epsilon)$ | Causal; computed from observed particulate values at $t$ |

- **Excluded Unobserved Channels:** Channels 2–18 (trace gases and weather) and channel 66 (`dewpoint_depression_t`) were completely excluded from the reduced feature set.
- **Finding:** All 49 features are strictly causal and computable in production at issue time $t$. The minor documentation discrepancy ("48" vs "49" features due to an off-by-one counting error in the report text) does not affect mathematical validity.

### 2.4 Split Isolation & Normalization Integrity
- **Training Range:** `2021-01-01 00:00:00` to `2023-12-31 23:00:00` ($1,901,006$ sequences).
- **Validation Range:** `2024-01-01 00:00:00` to `2024-06-30 23:00:00` ($318,849$ sequences).
- **Final Holdout Range:** `2025-04-01 00:00:00` to `2025-08-31 23:00:00` ($236,289$ sequences).
- **Finding:** The splits are mutually disjoint with zero chronological overlap. The ensemble weights ($w_{\text{PM2.5}} = 0.40, w_{\text{PM10}} = 0.45$) were tuned exclusively on H1 2024 and remained frozen before 2025 holdout evaluation. Scalers were loaded read-only from historical metadata; no holdout information leaked into normalization.

---

## 3. Statistical Uncertainty & Bootstrap Audit

To independently test whether the observed improvements are statistically significant, we evaluated the paired error differences ($\Delta \text{MAE} = \text{MAE}_{\text{Model}} - \text{MAE}_{\text{Reference}}$) under three distinct bootstrap methodologies using $B = 1,000$ replicates with fixed seed $42$:

### Paired Uncertainty Results Across Methodologies:

| Comparison | Target | Observed $\Delta$MAE (µg/m³) | Method A: Block Subsample 95% CI | Method B: Weighted Station Cluster 95% CI | Method C: Macro Station 95% CI | $p$-value (Method B) | Significance Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adapted Gated Ens vs. Persistence** | $\text{PM}_{2.5}$ | **$-0.3277$** | $[-0.7128,\ -0.0151]$ | **$[-0.4283,\ -0.2270]$** | **$[-0.4125,\ -0.2053]$** | **$p < 0.0001$** | **STATISTICALLY SIGNIFICANT** |
| **Adapted Gated Ens vs. Persistence** | $\text{PM}_{10}$ | **$-1.0160$** | $[-1.8656,\ -0.1912]$ | **$[-1.2861,\ -0.7664]$** | **$[-1.2197,\ -0.7046]$** | **$p < 0.0001$** | **STATISTICALLY SIGNIFICANT** |
| **Fine-Tuned LSTM vs. Frozen LSTM** | $\text{PM}_{2.5}$ | **$-1.2859$** | $[-1.9119,\ -0.8908]$ | **$[-1.5980,\ -1.0024]$** | **$[-1.7061,\ -1.0433]$** | **$p < 0.0001$** | **STATISTICALLY SIGNIFICANT** |
| **Fine-Tuned LSTM vs. Frozen LSTM** | $\text{PM}_{10}$ | **$-1.7848$** | $[-2.7868,\ -1.0248]$ | **$[-2.4321,\ -1.2407]$** | **$[-2.4540,\ -1.2894]$** | **$p < 0.0001$** | **STATISTICALLY SIGNIFICANT** |
| **Fine-Tuned LSTM vs. Persistence** | $\text{PM}_{2.5}$ | $-0.1465$ | $[-0.4674,\ +0.2299]$ | $[-0.2549,\ -0.0274]$ | $[-0.2419,\ -0.0197]$ | $p = 0.014$ | Statistically significant under full cluster |
| **Fine-Tuned LSTM vs. Persistence** | $\text{PM}_{10}$ | $-0.6794$ | $[-1.5167,\ +0.1951]$ | $[-0.9753,\ -0.4032]$ | $[-0.9052,\ -0.3422]$ | $p < 0.0001$ | Statistically significant under full cluster |

### Methodological Audit of Bootstrap Techniques:
1. **Method A (Phase 3 Subsample):** Resamples 75 stations with one 24-hour block per station ($1,800$ observations per replicate). This conservative subsampling approach yields wider intervals ($[-0.7128, -0.0151]$ for $\text{PM}_{2.5}$, $p = 0.042$), but still strictly excludes zero.
2. **Method B (Full Station-Clustered Bootstrap):** Resamples station clusters with replacement while weighting each station proportionally to its valid observation count ($N = 236,289$). This accounts for cross-station clustering and temporal dependence. Under Method B, the 95% CI is **$[-0.4283, -0.2270]$**, confirming statistical significance with $p < 0.0001$.
3. **Method C (Macro-Average Station Bootstrap):** Weights all 75 stations equally regardless of sample size. The resulting 95% CI is **$[-0.4125, -0.2053]$**, proving that the ensemble's advantage is not driven by a few disproportionate high-volume stations.

---

## 4. Evaluation Design & Chronology Audit

### 4.1 Chronological Split Assessment
- **Adaptation Buffer (2021–2023):** 3 full years capturing post-lockdown emission normalization.
- **Validation Split (H1 2024):** January to June 2024.  
  *Audit Note:* Because full-year 2024 was explored in Phase 2 zero-shot benchmark experiments, H1 2024 was an observed cohort in aggregate. Therefore, H1 2024 served appropriately for hyperparameter tuning and model selection, while the final holdout was strictly shifted forward into 2025.
- **Final Holdout Split (April to August 2025):** Genuinely untouched, forward-chronological 5-month test cohort. Zero 2025 data was observed during model training or threshold selection.

### 4.2 Network Composition & Missing-Month Coverage
- **Station Attrition:** 
  - Canonical network: 90 stations.
  - Matched XKDR stations: 85 stations.
  - Active in 2021–2023 train: 85 stations.
  - Active in H1 2024 val: 82 stations.
  - Active in 2025 holdout: 75 stations.
  - *Audit Finding:* Ten stations exhibited telemetry gaps or were decommissioned by CPCB in 2025. The 75 remaining stations maintain representative geographic coverage across North, South, East, West, and Central India.
- **Temporal Gaps in 2025:** 
  - Months 1–3 (January–March 2025) and late September 2025 were absent in the XKDR archive partitions. The holdout period is restricted to April 1 through August 31, 2025.

---

## 5. Robustness, Monthly, and Station-Level Analysis

### 5.1 Month-by-Month Consistency
We audited whether performance gains were localized to specific weather patterns:

| Month | Sequences ($N$) | Mean $\text{PM}_{2.5}$ | Persist $\text{PM}_{2.5}$ MAE | Adapted Ens $\text{PM}_{2.5}$ MAE | $\text{PM}_{2.5}$ Delta | Mean $\text{PM}_{10}$ | Persist $\text{PM}_{10}$ MAE | Adapted Ens $\text{PM}_{10}$ MAE | $\text{PM}_{10}$ Delta | Month Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **April 2025** | 46,210 | 55.58 | 9.3780 | **8.5212** | **-0.8568** | 159.56 | 24.8672 | **22.1027** | **-2.7645** | **Adapted Ensemble** |
| **May 2025** | 47,787 | 45.33 | 8.0890 | **7.7270** | **-0.3620** | 124.07 | 21.3792 | **20.1419** | **-1.2373** | **Adapted Ensemble** |
| **June 2025** | 47,353 | 36.81 | 6.5176 | **6.3436** | **-0.1740** | 104.64 | 16.5231 | **15.8116** | **-0.7115** | **Adapted Ensemble** |
| **July 2025** | 47,563 | 24.67 | 5.3697 | **5.2403** | **-0.1294** | 63.52 | 11.4571 | **11.2246** | **-0.2325** | **Adapted Ensemble** |
| **August 2025** | 47,376 | 26.46 | 5.4398 | **5.3101** | **-0.1297** | 69.10 | 11.7409 | **11.5629** | **-0.1780** | **Adapted Ensemble** |

**Finding:** The Adapted Severity-Gated Ensemble **strictly beats 1-Hour Persistence in all 5 calendar months** on both pollutants. The margin of improvement scales with baseline pollution levels (strongest in pre-monsoon April/May; lower during wet monsoon precipitation washout in July/August).

### 5.2 Station-Level Heterogeneity
- **$\text{PM}_{2.5}$ Station Win Rate:** The ensemble outperforms 1-hour persistence at **53 of 75 stations (70.7%)**.
- **$\text{PM}_{10}$ Station Win Rate:** The ensemble outperforms 1-hour persistence at **55 of 75 stations (73.3%)**.
- **Station $\Delta\text{MAE}$ Range:** Mean $\text{PM}_{2.5}$ delta is $-0.3093\ \mu\text{g/m}^3$ (median: $-0.2678$, min: $-1.3596$, max: $+0.7561$).
- **Finding:** Gains are distributed broadly across the CAAQMS network rather than concentrated in an isolated regional cluster.

---

## 6. Application Integration & Feasibility Check

### 6.1 CPU Inference Latency
Benchmarked on the target CPU host over 100 consecutive forward passes:
- Fine-Tuned LSTM latency: **$3.12$ ms** per sequence.
- Adapted Reduced GBDT latency: **$24.78$ ms** per sequence.
- Total Combined Pipeline latency: **$27.89$ ms** per sequence.
- **Finding:** End-to-end CPU inference executes in under **30 ms**, far below the 1-hour operational dispatch cycle.

### 6.2 System Health & Regression Smoke Tests
- [`tests/test_eight_sections_smoke.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_eight_sections_smoke.py): **ALL 8 SECTIONS PASSED WITH ZERO ERRORS** (Overview, Live Air Quality, Mode A Individual Methods, Mode B Unified Hybrid, Side-by-Side Comparison, Station Map, Model Performance, and System Health).
- [`tests/test_accuracy_optimization_v2.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_accuracy_optimization_v2.py): **5/5 UNIT TESTS PASSED**.

### 6.3 Current Production Default Status
- **Current Production Default:** The operational service [`src/services/sensor_forecast_service.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/services/sensor_forecast_service.py) continues to point to the canonical Historical Severity-Gated Ensemble (`models/lstm_baseline/best_lstm_model.pt` + `experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib`).
- **Experimental Adapted Checkpoints:** Located in `experiments/recent_data_audit/checkpoints/` (`best_finetuned_lstm.pt`, `adapted_gbdt_pm25.joblib`, `adapted_gbdt_pm10.joblib`).
- **Safety Protocol:** Zero production files were overwritten. Production defaults will not switch until explicit deployment sign-off.

---

## 7. Known Limitations & Audit Findings Summary

| Audit Item | Status | Verified Finding / Action Required |
| :--- | :---: | :--- |
| **Numerical Reproduction** | **PASS (100%)** | All 8 models match reported 2025 holdout MAE down to 0.0000. |
| **Data & Feature Leakage** | **PASS** | Strict causality verified; targets are $t+1$; input features are strictly $\le t$. |
| **Reduced Feature Set** | **RESOLVED** | Model trained on 49 causal features (report referred to 48 due to a counting typo). |
| **Statistical Significance** | **PASS** | Significant under all 3 bootstrap methods ($p = 0.042$ subsample; $p < 0.0001$ full cluster). |
| **Monthly Consistency** | **PASS** | Ensemble beats persistence across all 5 test months (April–August 2025). |
| **Station Consistency** | **PASS** | $70.7\%$ station win rate for $\text{PM}_{2.5}$; $73.3\%$ for $\text{PM}_{10}$. |
| **CPU Feasibility** | **PASS** | $27.89$ ms end-to-end CPU latency per forecast. |
| **Application Integrity** | **PASS** | All 8 dashboard sections pass smoke tests with zero errors. |
| **Seasonal Window Limit** | **FLAGGED** | Holdout covers warm & monsoon seasons (Apr–Aug 2025); winter inversion months missing. |

---

## 8. Final Deployment Recommendation

### Overall Status: **CONDITIONALLY READY FOR PRODUCTION DEPLOYMENT**

The adapted severity-gated model has been scientifically validated. It resolves the zero-shot degradation observed in Phase 2 and outperforms 1-hour persistence on both particulate criteria.

### Phased Deployment Roadmap:
1. **Phase A (Immediate Shadow Mode):** Deploy the adapted checkpoints (`experiments/recent_data_audit/checkpoints/`) in parallel with the production model in `src/services/sensor_forecast_service.py` under an opt-in flag (`model_mode="adapted_severity_gated"`).
2. **Phase B (Winter Telemetry Benchmark):** Once XKDR or CPCB releases full winter telemetry for November 2025 – January 2026, evaluate the severe-tier performance on the winter inversion cohort.
3. **Phase C (Full Default Promotion):** Upon confirming that winter severe underprediction remains bounded, promote the adapted ensemble to the dashboard default.
