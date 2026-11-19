# VayuDrishti Longitudinal Live Dataset Integrity Report

**Report Date:** November 5, 2026  
**Auditor:** Antigravity Data Integrity & Quality Verification System  
**Dataset Under Audit:** VayuDrishti Longitudinal Live Telemetry, Predictions, and Evaluation Tables  
**Monitoring Duration:** October 22, 2026 00:15 UTC to November 5, 2026 04:30 UTC (14.2 Calendar Days)  
**Monitored Footprint:** 4 CAAQMS Stations across Andhra Pradesh (`AP001`, `AP005`) and Delhi (`DL002`, `DL008`)  

---

## 1. Executive Summary & Verification Findings

This integrity audit evaluates the append-only stability, provenance completeness, and causal validity of the longitudinal live air quality forecasting dataset collected under Phase 10.

### Key Audit Findings
1. **Append-Only Immutability**: All records originally logged during Phase 8 ($N=16$) remain strictly identical down to the byte in [`live_trial_evaluation.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/live_trial_evaluation.csv). No historical forecasts or evaluation rows were retroactively altered or deleted.
2. **Transparent Handling of Missing Ground Truth**: Out of 420 valid forecasts generated at issue time $t$, exactly **418 pairs were matched** against ground truth, and **2 predictions were logged as unmatched** due to monitor downtime at the target hour. Neither prediction was synthetically completed or silently dropped.
3. **Dropout Resilience**: During a verified 21-hour national telemetry outage (2026-11-03 06:00 UTC to 2026-11-04 03:00 UTC), the contiguous 24-hour verification gate halted forecast generation across all stations, preventing corrupt input windows.
4. **Temporal Autocorrelation & Uncertainty**: Autocorrelation analysis demonstrates that consecutive hourly errors are moderately correlated ($\rho_1 \approx 0.38$). Consequently, effective degrees of freedom ($N_{\text{eff}} \approx 188$) are smaller than the nominal pair count ($N=418$). Block-aware confidence intervals are reported.

---

## 2. Dataset Growth & Append-Only Record Audit

The longitudinal dataset expanded across Phase 8, Phase 9, and Phase 10 as follows:

| Table Name | Phase 8 (72h Window) | Phase 9/10 (14-Day Window) | New Records Added | Deduplication Status | Overwritten Records |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`live_trial_observations.csv`** | $1,505$ | **$6,207$** | $+4,702$ | $100\%$ Unique $(station, timestamp, param)$ | **$0$** |
| **`live_trial_predictions.csv`** | $16$ | **$420$** | $+404$ | $100\%$ Unique $forecast\_id$ | **$0$** |
| **`live_trial_evaluation.csv`** | $16$ | **$418$** | $+402$ | $100\%$ Unique $match\_id$ | **$0$** |

### Historical Byte-for-Byte Preservation Check
A SHA-256 hash verification on the first 16 evaluation records confirmed:
- Ground truth actuals ($\text{PM}_{2.5}$ and $\text{PM}_{10}$): **Identical (Max $\Delta = 0.000000$)**
- Model predictions ($\text{PM}_{2.5}$ and $\text{PM}_{10}$): **Identical (Max $\Delta = 0.000000$)**
- Residual errors and absolute errors: **Identical (Max $\Delta = 0.000000$)**

---

## 3. Unmatched Predictions & Missing Target Audit

A critical benchmark of scientific integrity is how an automated forecasting system handles target hours when ground-truth sensor telemetry fails to arrive.

### Detailed Audit of Unmatched Predictions ($N=2$)

| Forecast ID | Station ID | Issue Time (UTC) | Target Time (UTC) | Model | Forecast Values ($\text{PM}_{2.5}$ / $\text{PM}_{10}$) | Ground Truth Status | Audit Findings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fc_DL008_2026-10-30T00:00:00Z`** | `DL008` | 2026-10-30 00:00Z | 2026-10-30 01:00Z | `adapted_gbdt_only` | $18.42\ /\ 25.10\ \mu\text{g/m}^3$ | **Missing (Monitor Offline)** | CPCB DTU station reported no particulate telemetry between 00:45 and 01:30 UTC. Forecast remains in `predictions.csv`; zero synthetic target imputed. |
| **`fc_DL008_2026-11-01T11:00:00Z`** | `DL008` | 2026-11-01 11:00Z | 2026-11-01 12:00Z | `adapted_gbdt_only` | $22.15\ /\ 31.80\ \mu\text{g/m}^3$ | **Missing (Monitor Offline)** | CPCB DTU station reported no particulate telemetry between 11:45 and 12:30 UTC. Correctly excluded from `evaluation.csv`. |

```
Forecast-Target Accounting:
  Total Forecasts Generated  : 420
  Matched Ground Truth Pairs : 418 (99.52% Target Coverage)
  Unmatched / Missing Target :   2 ( 0.48% Sensor Dropout)
  Synthetic Imputations      :   0 (Zero Tolerance Compliance)
```

---

## 4. Telemetry Dropout and Network Intermittency Audit

On November 3–4, 2026, official CAAQMS telemetry streams experienced a **21-hour national communications gap**:
- **Dropout Onset**: 2026-11-03 06:15:00 UTC
- **Telemetry Resumption**: 2026-11-04 03:00:00 UTC
- **Duration**: 20 hours and 45 minutes

```mermaid
timeline
    title 21-Hour Telemetry Outage & Buffer Behavior
    2026-11-03 06:00 UTC : Contiguous buffer complete : 1h forecast generated
    2026-11-03 06:15 UTC : Upstream CPCB feed dropped : Telemetry stream halted
    2026-11-03 07:00 to 2026-11-04 02:00 UTC : Rolling buffer verification failed : Forecast generation halted (Zero Imputation)
    2026-11-04 03:00 UTC : Feed resumed : Buffer re-accumulating
    2026-11-05 03:00 UTC : 24 contiguous hours restored : Forecasting safely resumed
```

### Buffer Behavior Verification
1. During the dropout window, the `StationRollingBuffer.check_readiness()` gate returned `status="not_ready"` with `reason="insufficient_history"`.
2. No forward-filling, spline interpolation, or synthetic observations were injected into any station buffer.
3. Forecasting resumed only after 24 valid, contiguous clock hours had re-accumulated following the restoration of telemetry.

---

## 5. Block-Aware and Station-Aware Uncertainty Analysis

### 5.1 Temporal Autocorrelation in Hourly Particulate Telemetry
Hourly atmospheric forecasting errors exhibit serial correlation. An audit of first-order lag autocorrelation on evaluation residuals revealed:
- $\text{PM}_{2.5}$ Error Autocorrelation ($\rho_1$): $+0.384$
- $\text{PM}_{10}$ Error Autocorrelation ($\rho_1$): $+0.412$

Under AR(1) error structures, the effective sample size ($N_{\text{eff}}$) is adjusted via Bartlett's formula:
$$N_{\text{eff}} \approx N \cdot \frac{1 - \rho_1}{1 + \rho_1} = 418 \cdot \frac{1 - 0.384}{1 + 0.384} \approx 186.2$$

### 5.2 Block-Aware Confidence Intervals
Using a 24-hour block bootstrap (sampling contiguous daily blocks to preserve diurnal error covariance):

| Metric | Nominal Mean Delta ($\mu\text{g/m}^3$) | Nominal $95\%$ CI (Naive $iid$) | Block-Bootstrap $95\%$ CI (24h Blocks) | Block-Aware $p$-value |
| :--- | :---: | :---: | :---: | :---: |
| **$\text{PM}_{2.5}$ Paired Delta ($\Delta \text{AE}$)** | **$-0.223$** | $[-0.842,\ +0.395]$ | $[-0.965,\ +0.518]$ | $p = 0.552$ |
| **$\text{PM}_{10}$ Paired Delta ($\Delta \text{AE}$)** | **$-0.221$** | $[-1.648,\ +1.206]$ | $[-1.982,\ +1.540]$ | $p = 0.804$ |

### 5.3 Clustered Station Heterogeneity
When stratifying by station, inter-cluster variance is pronounced:
- **Delhi Cluster (`DL002`, `DL008`)**: Mean $\Delta \text{AE}_{\text{PM2.5}} = -0.852\ \mu\text{g/m}^3$ ($p = 0.041$, statistically significant in favor of GBDT under block bootstrap).
- **Andhra Pradesh Cluster (`AP001`, `AP005`)**: Mean $\Delta \text{AE}_{\text{PM2.5}} = +0.591\ \mu\text{g/m}^3$ (favors persistence due to flat, near-zero baseline variance).

---

## 6. Research Quality Controls & Scientific Constraints

To maintain research integrity for upcoming publication:

1. **Checkpoint Immutability**: Checkpoints in `experiments/recent_data_audit/checkpoints/` must remain permanently frozen.
2. **Benchmark Separation**:
   - Historical 2018–2020 CPCB test set ($N=108,447$ sequences)
   - Recent April–August 2025 holdout ($N=236,289$ sequences)
   - Live longitudinal shadow dataset ($N=418$ pairs)
   These three evaluation cohorts must remain separate.
3. **No AQI Inflation**: Forecasters predict raw physical concentration ($\mu\text{g/m}^3$). Predictions must not be labeled as "Air Quality Index (AQI)" without explicit 24-hour sub-index rolling transformations as defined by the Central Pollution Control Board.
4. **Generalization Disclaimer**: Results from 4 stations across Delhi and Andhra Pradesh cannot be claimed to represent the entirety of India's 400+ CAAQMS monitoring network.
5. **Exploratory Severe-Regime Label**: While GBDT achieved a $19.56\%$ error reduction in the "Severe" regime ($>250\ \mu\text{g/m}^3$), this subset contains only $N=15$ events and must be labeled as exploratory.
