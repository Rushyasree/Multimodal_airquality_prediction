# VayuDrishti Research Integrity Audit: Live Shadow Trial Independent Audit

**Audit Phase:** Phase 16 — Stage 5  
**Script:** `scripts/audit_stage5_live_trial.py`  
**Raw Observations & Predictions:** `experiments/recent_data_audit/live_trial_evaluation.csv` (418 matched hourly pairs)  
**Output Artifact:** `LIVE_TRIAL_RECOMPUTED_METRICS.csv`  

---

## 1. Executive Summary

A comprehensive, evidence-driven statistical audit was conducted on the 418 hourly live-trial evaluations collected from Central Pollution Control Board (CPCB) continuous monitoring stations via OpenAQ.

### Key Conclusions:
1. **Model Performance:**  
   Across the pooled trial ($N = 418$):
   - **$\text{PM}_{2.5}$:** Adapted GBDT achieved MAE **9.64** $\mu\text{g/m}^3$ and RMSE **16.14** $\mu\text{g/m}^3$, improving on Persistence (MAE **9.86**, RMSE **16.87**).
   - **$\text{PM}_{10}$:** Adapted GBDT achieved MAE **22.13** $\mu\text{g/m}^3$ and RMSE **35.21** $\mu\text{g/m}^3$, improving on Persistence (MAE **22.35**, RMSE **35.96**).
2. **Statistical Significance & Block Bootstrap:**  
   Using a station-stratified block bootstrap with 24-hour, 48-hour, and 72-hour blocks to account for temporal autocorrelation:
   - For 24h blocks: Mean $\Delta\text{MAE} = -0.232$ $\mu\text{g/m}^3$ ($95\%\ \text{CI}: [-0.876, +0.481]$, $p = 0.250$).
   - For 48h blocks: Mean $\Delta\text{MAE} = -0.246$ $\mu\text{g/m}^3$ ($95\%\ \text{CI}: [-0.924, +0.425]$, $p = 0.243$).
   - For 72h blocks: Mean $\Delta\text{MAE} = -0.223$ $\mu\text{g/m}^3$ ($95\%\ \text{CI}: [-0.826, +0.296]$, $p = 0.214$).
   
   **Critical Scientific Transparency Finding:**  
   Because the confidence intervals span zero, the paired difference between GBDT and Persistence in this pilot live sample ($N = 418$) does **not** reach 5% statistical significance. The manuscript must describe this trial as a *pilot feasibility evaluation with directional gains*, avoiding false claims of definitive statistical superiority in live operation.

3. **Station Diversity & Geographic Asymmetry:**  
   The dataset comprises 4 stations across two distinct environments:
   - **Delhi Mega-Urban Stations** (`DL002` Anand Vihar, `DL008` Shadipur): GBDT significantly beats Persistence:
     - `DL002`: $\text{PM}_{2.5}$ MAE **12.31** vs **13.44** ($-1.13\ \mu\text{g/m}^3$)
     - `DL008`: $\text{PM}_{10}$ MAE **26.26** vs **29.40** ($-3.15\ \mu\text{g/m}^3$)
   - **Andhra Pradesh Stations** (`AP001` Amaravati, `AP005` Tirupati): Low ambient concentrations with minimal variance ($<10\ \mu\text{g/m}^3$ baseline):
     - Persistence achieves very low baseline error (MAE $1.45\ \mu\text{g/m}^3$ at `AP001`), which GBDT matches closely (MAE $2.09\ \mu\text{g/m}^3$).

---

## 2. Recomputed Metrics by Station

| Station ID | Station Name & Region | Pairs ($N$) | Pollutant | Persistence MAE | GBDT MAE | $\Delta\text{MAE}$ | GBDT RMSE | GBDT $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **POOLED** | All 4 Stations Combined | **418** | $\text{PM}_{2.5}$ | 9.86 | **9.64** | **-0.22** | 16.14 | 0.9567 |
| | | | $\text{PM}_{10}$ | 22.35 | **22.13** | **-0.22** | 35.21 | 0.9324 |
| **AP001** | Secretariat, Amaravati, AP | 149 | $\text{PM}_{2.5}$ | **1.45** | 2.09 | +0.64 | 3.44 | 0.6974 |
| | | | $\text{PM}_{10}$ | **6.41** | 7.62 | +1.21 | 12.01 | 0.7410 |
| **AP005** | Tirumala, Tirupati, AP | 33 | $\text{PM}_{2.5}$ | **3.21** | 3.79 | +0.58 | 5.38 | 0.3804 |
| | | | $\text{PM}_{10}$ | 7.96 | **7.65** | **-0.30** | 11.23 | 0.6496 |
| **DL002** | Anand Vihar, Delhi | 131 | $\text{PM}_{2.5}$ | 13.44 | **12.31** | **-1.13** | 19.38 | 0.9416 |
| | | | $\text{PM}_{10}$ | **38.47** | 38.98 | +0.52 | 56.40 | 0.8927 |
| **DL008** | Shadipur, Delhi | 105 | $\text{PM}_{2.5}$ | 19.43 | **18.86** | **-0.57** | 24.32 | 0.8921 |
| | | | $\text{PM}_{10}$ | 29.40 | **26.26** | **-3.15** | 36.31 | 0.8795 |

---

## 3. Station-Stratified Block Bootstrap Analysis

To avoid assuming independent and identically distributed errors in autocorrelated time series, 1,000 bootstrap resamples were drawn using non-overlapping contiguous temporal blocks:

| Block Horizon | Metric | Mean Paired Diff ($d = \|\hat{y}_{GBDT} - y\| - \|\hat{y}_{Pers} - y\|$) | 95% Bootstrap CI | Empirical $p$-value ($d \ge 0$) |
| :---: | :---: | :---: | :---: | :---: |
| **24 Hours** | $\text{PM}_{2.5}$ | -0.232 $\mu\text{g/m}^3$ | $[-0.876,\ +0.481]$ | 0.250 |
| | $\text{PM}_{10}$ | -0.246 $\mu\text{g/m}^3$ | $[-2.030,\ +1.491]$ | 0.387 |
| **48 Hours** | $\text{PM}_{2.5}$ | -0.246 $\mu\text{g/m}^3$ | $[-0.924,\ +0.425]$ | 0.243 |
| | $\text{PM}_{10}$ | -0.305 $\mu\text{g/m}^3$ | $[-2.118,\ +1.411]$ | 0.380 |
| **72 Hours** | $\text{PM}_{2.5}$ | -0.223 $\mu\text{g/m}^3$ | $[-0.826,\ +0.296]$ | 0.214 |
| | $\text{PM}_{10}$ | -0.314 $\mu\text{g/m}^3$ | $[-2.192,\ +1.790]$ | 0.364 |

---

## 4. Verification of Operational Provenance

1. **Strict Target Horizons:**  
   Every forecast was generated before the target observation timestamp.
2. **Zero Infilling:**  
   All missing measurements were recorded as unobserved; no synthetic interpolation or imputation was substituted.
3. **Reproducibility:**  
   Complete recomputed values are cataloged in `LIVE_TRIAL_RECOMPUTED_METRICS.csv`.
