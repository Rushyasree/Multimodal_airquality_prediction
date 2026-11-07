# VayuDrishti — Research Audit & Discrepancy Resolution Report

**Project Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Evaluation Scope:** Accuracy Optimization v2 Benchmark & Research Integrity Audit  
**Date:** October 2026  
**Auditor:** Senior ML Research Engineer / Time-Series Specialist  

---

## Executive Summary

This audit report formally resolves conflicting statements, timestamp discrepancies, and model behavior across the **Accuracy Optimization v2** benchmark artifacts. Every finding presented herein has been verified directly against raw sequence indices, unscaled ground-truth arrays (`y_test.npy`), model checkpoints, and reproducible bootstrap resampling.

### Key Audit Conclusions:
1. **Timestamp Discrepancy Resolved**: The empirical held-out test split spans **`2019-09-04 20:00:00` to `2020-07-01 00:00:00`** ($N = 404,431$ sequences across 90 stations). The raw merged dataset terminates on `2020-07-01 00:00:00`; earlier mentions of `2019-10-31` to `2020-12-31` were informal calendar projections written prior to fixing contiguous sequence boundaries. All reports and documentation have been harmonized.
2. **Statistically Validated Severity-Gated Ensemble**: Dynamic gating ($\theta_{25} = 250\ \mu\text{g/m}^3, \theta_{10} = 400\ \mu\text{g/m}^3$) eliminates the $-5.49\ \mu\text{g/m}^3$ severe underprediction drag caused by GBDT tree saturation. Across all 404,431 test sequences, it achieves **PM2.5 MAE $10.3508$ µg/m³** and **RMSE $18.3792$ µg/m³**, achieving **statistically significant RMSE improvement over the LSTM ($p < 0.0001$)** while restoring LSTM-equivalent severe-tier bias.
3. **Severe-Tier Evaluation on Identical Rows**: In the severe regime (PM2.5 $\ge 250\ \mu\text{g/m}^3$, $N = 18,938$; PM10 $\ge 430\ \mu\text{g/m}^3$, $N = 15,561$), all candidate models were evaluated on the exact same row indices. Gating recovers the severe bias from $-21.97$ µg/m³ (static ensemble) to $-16.59$ µg/m³ (matching LSTM $-16.48$ µg/m³).
4. **Strict Deployment Policy**:
   - The dashboard and inference services maintain **selectable access** to: (a) Pretrained LSTM, (b) Validated Static Ensemble ($0.90/0.10$), and (c) Severity-Gated Ensemble.
   - We **explicitly reject** adding arbitrary $+20\ \mu\text{g/m}^3$ uncalibrated offsets to predictions.
   - We **strictly delineate** tabular sensor ensembling (high-frequency temporal modeling) from multimodal visual hybrid estimation (ambient photographic calibration). The tabular ensemble is never referred to as an image model.

---

## 1. Resolution of the Test-Period Timestamp Discrepancy

### 1.1 Empirical Sequence Index and Dataset Audit

A comprehensive programmatic audit of [`data/processed/sequences/air_quality_24h_sequence_index.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/data/processed/sequences/air_quality_24h_sequence_index.csv) and [`data/processed/air_quality_weather_merged.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/data/processed/air_quality_weather_merged.csv) reveals the exact ground truth:

| Split | Sequence Count ($N$) | Station Count | History Start Range | History End Range | Target Timestamp Range ($\Delta t = 1\text{h}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 336,641 | 55 | `2015-01-01 05:00:00` to `2018-11-05 14:00:00` | `2015-01-02 04:00:00` to `2018-11-06 13:00:00` | **`2015-01-02 05:00:00` to `2018-11-06 14:00:00`** |
| **VALIDATION** | 326,423 | 78 | `2018-11-06 15:00:00` to `2019-09-02 19:00:00` | `2018-11-07 14:00:00` to `2019-09-03 18:00:00` | **`2018-11-07 15:00:00` to `2019-09-03 19:00:00`** |
| **HELD-OUT TEST** | **404,431** | **90** | `2019-09-03 20:00:00` to `2020-06-30 00:00:00` | `2019-09-04 19:00:00` to `2020-06-30 23:00:00` | **`2019-09-04 20:00:00` to `2020-07-01 00:00:00`** |

### 1.2 Underlying Dataset Boundary
The raw consolidated CPCB dataset `data/processed/air_quality_weather_merged.csv` contains $1,369,779$ rows spanning 90 stations. Its absolute temporal span is:
$$\text{Earliest Datetime: } \mathbf{2015\text{-}01\text{-}01\ 01:00:00} \quad \longleftrightarrow \quad \text{Latest Datetime: } \mathbf{2020\text{-}07\text{-}01\ 00:00:00}$$
**There are zero observations in the repository after July 1, 2020, 00:00:00.**

### 1.3 Provenance of the Conflicting Dates
1. **Origin of `2020-12-31` & `2019-10-31`**: In early exploratory outlines and proposal drafts, the test period was conceptually targeted as "late 2019 to end of 2020". These round dates were copied into draft markdown files as narrative placeholders before sequence generation scripts extracted the contiguous 24-hour non-overlapping windows.
2. **Empirical Fact**: When sequences were strictly generated requiring 24 contiguous preceding hours without station boundary crossing, the actual chronological window for the 404,431 test sequences became `2019-09-04 20:00:00` to `2020-07-01 00:00:00`.
3. **Harmonization Status**: All benchmark reports ([`ACCURACY_OPTIMIZATION_V2_REPORT.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/accuracy_optimization_v2/ACCURACY_OPTIMIZATION_V2_REPORT.md), [`FINAL_ENSEMBLE_VALIDATION_REPORT.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/accuracy_optimization_v2/FINAL_ENSEMBLE_VALIDATION_REPORT.md), and [`FINAL_RESEARCH_READINESS_REPORT.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/accuracy_optimization_v2/FINAL_RESEARCH_READINESS_REPORT.md)) have now been updated with the verified empirical timestamps.

---

## 2. Statistical Validation of the Severity-Gated Ensemble

### 2.1 Full Historical Test Benchmark (Identical $N = 404,431$ Sequences)

All models were evaluated on the exact same 404,431 test sequences against unscaled physical targets with zero leakage:

| Model Architecture | PM2.5 MAE (µg/m³) | PM2.5 RMSE (µg/m³) | PM2.5 R² | PM10 MAE (µg/m³) | PM10 RMSE (µg/m³) | PM10 R² | Operational Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1-Hour Persistence Baseline** | 10.6783 | 19.8072 | 0.9423 | 19.7384 | 36.2374 | 0.9115 | Physical continuity ($y_{t+1} = y_t$) |
| **Pretrained 2-Layer LSTM** | 10.5719 | 18.5789 | 0.9492 | 18.8643 | 32.2027 | 0.9301 | Recurrent state (27 channels) |
| **Static Ensemble Forecaster** | 10.3950 | 18.5498 | 0.9494 | 18.2809 | 31.9708 | 0.9311 | Fixed blend ($0.90 / 0.10$ PM2.5; $0.80 / 0.20$ PM10) |
| **⚡ Severity-Gated Ensemble** | **10.3508** | **18.3792** | **0.9503** | **18.1369** | **31.6385** | **0.9325** | Dynamic gating ($\theta = 250 / 400\ \mu\text{g/m}^3$) |

### 2.2 Paired 7-Day Block Bootstrap Analysis ($B = 2,000, K = 3,370$ Blocks)

To account for temporal autocorrelation and spatial station clustering, we performed paired 7-day contiguous station-block bootstrap resampling:

| Comparison | Target | Mean Difference | 95% Bootstrap CI | $p$-value | Significance Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Static vs. LSTM** | PM2.5 MAE | $+0.1769$ µg/m³ | $[+0.1596, +0.1929]$ | $< 0.0001$ | **Statistically Significant** |
| | PM2.5 RMSE | $+0.0297$ µg/m³ | $[-0.0375, +0.0957]$ | $0.1930$ | *Not Significant* (Spans Zero) |
| | PM10 MAE | $+0.5834$ µg/m³ | $[+0.5326, +0.6321]$ | $< 0.0001$ | **Statistically Significant** |
| | PM10 RMSE | $+0.2319$ µg/m³ | $[+0.1204, +0.3469]$ | $< 0.0001$ | **Statistically Significant** |
| **Gated vs. LSTM** | **PM2.5 MAE** | **$+0.2210$ µg/m³** | **$[+0.2112, +0.2298]$** | **$< 0.0001$** | **Statistically Significant** |
| | **PM2.5 RMSE** | **$+0.1999$ µg/m³** | **$[+0.1849, +0.2149]$** | **$< 0.0001$** | **Statistically Significant (Resolves Ambiguity)** |
| | **PM10 MAE** | **$+0.7269$ µg/m³** | **$[+0.6936, +0.7592]$** | **$< 0.0001$** | **Statistically Significant** |
| | **PM10 RMSE** | **$+0.5636$ µg/m³** | **$[+0.5161, +0.6112]$** | **$< 0.0001$** | **Statistically Significant** |
| **Gated vs. Static** | PM2.5 MAE | $+0.0441$ µg/m³ | $[+0.0321, +0.0572]$ | $< 0.0001$ | **Statistically Significant** |
| | PM2.5 RMSE | $+0.1702$ µg/m³ | $[+0.1092, +0.2317]$ | $< 0.0001$ | **Statistically Significant** |
| | PM10 MAE | $+0.1435$ µg/m³ | $[+0.1190, +0.1702]$ | $< 0.0001$ | **Statistically Significant** |
| | PM10 RMSE | $+0.3317$ µg/m³ | $[+0.2505, +0.4137]$ | $< 0.0001$ | **Statistically Significant** |

**Crucial Research Finding**: While the static ensemble failed to achieve statistically significant improvement in PM2.5 RMSE ($p = 0.193$), the **Severity-Gated Ensemble achieves strictly positive, statistically significant improvement in BOTH MAE and RMSE across both pollutants ($p < 0.0001$)**.

---

## 3. Severe-Tier Performance Analysis on Identical Held-Out Rows

### 3.1 Identical Row Evaluation

The severe cohorts represent the extreme tails of the empirical test distribution:
- **PM2.5 Severe** ($\ge 250\ \mu\text{g/m}^3$): $N = 18,938$ rows ($4.68\%$ of test set), Ground Truth Mean $= 355.11\ \mu\text{g/m}^3$, Max $= 960.00\ \mu\text{g/m}^3$.
- **PM10 Severe** ($\ge 430\ \mu\text{g/m}^3$): $N = 15,561$ rows ($3.85\%$ of test set), Ground Truth Mean $= 558.45\ \mu\text{g/m}^3$, Max $= 998.00\ \mu\text{g/m}^3$.

All models were evaluated on the exact same row indices:

| Cohort | Metric | 1h Persistence | Pretrained LSTM | Standalone GBDT | Static Ensemble (0.90/0.80) | Severity-Gated Ensemble |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PM2.5 Severe**<br>($\ge 250\ \mu\text{g/m}^3$, $N=18,938$) | **MAE (µg/m³)** | 38.3183 | **37.0812** | 90.8000 | 38.1299 | **37.0928** |
| | **RMSE (µg/m³)** | 56.7316 | **56.2559** | 121.9155 | 57.6679 | **56.2410** |
| | **Mean Bias (µg/m³)** | **-9.6737** | -16.4757 | -71.4182 | -21.9700 | **-16.5928** |
| | **Underprediction Rate** | **57.71%** | 61.00% | 85.62% | 66.88% | **61.10%** |
| **PM10 Severe**<br>($\ge 430\ \mu\text{g/m}^3$, $N=15,561$) | **MAE (µg/m³)** | 66.7507 | **61.8232** | 128.4542 | 67.0962 | **62.3524** |
| | **RMSE (µg/m³)** | 98.6215 | **94.6917** | 159.3510 | 98.5569 | **95.0548** |
| | **Mean Bias (µg/m³)** | **-24.8821** | -35.2314 | -120.4578 | -52.2767 | **-36.2215** |
| | **Underprediction Rate** | **59.80%** | 66.16% | 91.25% | 77.76% | **66.96%** |

### 3.2 Algorithmic Root Cause of Underprediction
1. **Tree Partition Saturation**: Decision tree split algorithms cannot extrapolate outside the bounding hyperplanes of their training features. When pollution surges into historically extreme unrepresented regimes, the GBDT predicts the mean of the training leaf partition (saturating at $\sim 283\ \mu\text{g/m}^3$, producing a massive $-71.42\ \mu\text{g/m}^3$ negative bias).
2. **Ensemble Drag**: In the static ensemble, the $10\%$ weight on GBDT drags the LSTM prediction down by $-5.49\ \mu\text{g/m}^3$ (from $-16.48$ to $-21.97\ \mu\text{g/m}^3$).
3. **Gating Solution**: By dynamically transitioning the blend weight $w \to 1.0$ (pure LSTM) when antecedent telemetry $y_t \ge 250\ \mu\text{g/m}^3$ for PM2.5 and $y_t \ge 400\ \mu\text{g/m}^3$ for PM10, the saturating tree is decoupled, restoring LSTM severe performance while maintaining ensemble superiority across the remaining $95.3\%$ of operational conditions.

---

## 4. Production Deployment & Modality Governance

### 4.1 Model Selectability
The VayuDrishti dashboard and backend services maintain selectable access to:
1. **LSTM Baseline** (`model_mode="lstm"`): Unaltered historical neural benchmark.
2. **Static Ensemble** (`model_mode="ensemble"`): Validated $0.90/0.10$ and $0.80/0.20$ blend.
3. **Severity-Gated Ensemble** (`model_mode="severity_gated"`): State-of-the-art dynamic gating forecaster.

### 4.2 Explicit Rejection of Ad-Hoc Offsets
- **Prohibition of $+20\ \mu\text{g/m}^3$ Offset**: Applying an uncalibrated $+20\ \mu\text{g/m}^3$ constant additive offset to severe-tier alerts is **strictly prohibited**. It creates severe overprediction spikes during transient episodic peaks, destroys calibrated uncertainty, and violates physical mass conservation. Calibrated quantile uncertainty intervals or dynamic gating are the only sanctioned mitigations.

### 4.3 Delineation of Tabular Ensemble vs Multimodal Hybrid
- **Tabular Sensor Ensemble (Mode A Forecaster)**: Operates on 24 antecedent hours of 27 telemetry and meteorological features with tabular gradient boosting. This is an atmospheric time-series ensemble.
- **Multimodal Visual Hybrid (Mode B Predictor)**: Integrates sensor telemetry with daytime outdoor photographs via ResNet-50 visual feature extraction and quality gating. It provides daylight atmospheric verification and sensor fallback.
- **Scientific Communication Rule**: The tabular ensemble must **never** be described as an image model or camera fusion model. The two pipelines address distinct operational questions and are maintained as complementary modes.

---

## 5. Final Scientific Verdict: Verified Findings vs Exploratory Mitigations

| Component / Finding | Classification | Empirical Basis & Recommendation |
| :--- | :---: | :--- |
| **Historical Test Set (404,431 rows, 2019-09 to 2020-07)** | **VERIFIED FACT** | Directly proven from raw dataset index; zero sequence leakage. Primary historical benchmark. |
| **Static Ensemble MAE Improvement** | **VERIFIED FACT** | Statistically significant ($p < 0.0001$) across 3-day, 7-day, 14-day, and station clusters. |
| **Static Ensemble PM2.5 RMSE Ambiguity** | **VERIFIED FACT** | 95% CI spans zero ($p = 0.193$); static blend cannot claim quadratic superiority over LSTM. |
| **GBDT Severe-Tier Saturation Drag** | **VERIFIED FACT** | Tree partitions produce $-71.42\ \mu\text{g/m}^3$ bias; drags ensemble down by $-5.49\ \mu\text{g/m}^3$. |
| **Severity-Gated Blending Formulation** | **VALIDATED MITIGATION** | Thresholds tuned strictly on validation set ($\theta = 250 / 400$); achieves $p < 0.0001$ on RMSE and eliminates severe drag. |
| **Dashboard Architecture (Dual-Mode A & B)** | **OPERATIONAL STANDARD** | Preserves all 8 sections, decoupled independent methods, and robust model selectability. |
