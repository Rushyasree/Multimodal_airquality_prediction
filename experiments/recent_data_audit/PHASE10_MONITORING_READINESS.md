# VayuDrishti — Phase 10: Longitudinal Monitoring Readiness Report

**Execution Date:** November 5, 2026  
**Auditor:** Antigravity Longitudinal Monitoring & Governance System  
**System Under Audit:** VayuDrishti Persistent Collector & Live Research Dataset  
**Production Baseline Status:** Frozen & Unmodified (Historical Severity-Gated Default)  
**Shadow Evaluator:** `adapted_gbdt_only` (49 Canonical Tabular Features)  

---

## 1. Executive Summary & Readiness Verdict

### Operational Verdict: `LONGITUDINAL MONITORING VERIFIED & RESEARCH-READY`

The VayuDrishti persistent collection and evaluation architecture has been established, tested, and validated under Phase 10. The collector operates in a strict append-only configuration that:
- Automatically acquires genuine particulate telemetry ($\text{PM}_{2.5}$ and $\text{PM}_{10}$) from verified OpenAQ API v3 CAAQMS stations.
- Bins raw readings into deterministic clock hours without synthetic interpolation.
- Requires 24 contiguous preceding hours before issuing a 1-hour-ahead forecast.
- Enriches prediction records with full provenance (model checkpoint SHA-256 hashes, input window bounds, feature schema versions).
- Matches elapsed targets post-horizon and maintains transparent audit logs for unmatched/missing target hours.

---

## 2. Longitudinal Dataset Status & Key Metrics

```
========================================================================================
LONGITUDINAL MONITORING DATASET SUMMARY (OCTOBER 22 – NOVEMBER 5, 2026)
========================================================================================
Monitoring Calendar Window  : 2026-10-22T00:15:00Z to 2026-11-05T04:30:00Z (14.2 Days)
Total Raw Observations      : 6,207 validated sensor readings
Total Clock-Hour Bins       : 844 station-hours
Total Forecasts Generated   : 420 1-hour-ahead predictions
Total Matched Target Pairs  : 418 pairs (99.52% target match coverage)
Unmatched Target Dropouts   : 2 pairs (0.48% upstream sensor downtime on DL008)
Synthetic Imputations       : 0 (Zero tolerance compliance)
========================================================================================
```

### Station Coverage Breakdown

| Station ID | Station Name & City | Raw Readings | Aggregated Clock Hours | Predictions ($t$) | Matched Ground Truth ($t+1$) | Unmatched Dropouts | Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AP001`** | Secretariat, Amaravati | 1,633 | 225 | 149 | 149 | 0 | **100.0%** |
| **`AP005`** | GVM Corp, Visakhapatnam | 1,081 | 148 | 33 | 33 | 0 | **100.0%** |
| **`DL002`** | Anand Vihar, Delhi | 1,752 | 236 | 131 | 131 | 0 | **100.0%** |
| **`DL008`** | DTU, Delhi | 1,741 | 235 | 107 | 105 | 2 | **98.13%** |
| **Total** | *Across All 4 Stations* | **6,207** | **844** | **420** | **418** | **2** | **99.52%** |

---

## 3. Data Integrity & Causal Matching Verification

The longitudinal pipeline enforces strict scientific safeguards verified by the automated test suite:

1. **Strict Temporal Horizon ($t \rightarrow t+1$)**:
   - Every prediction issue time $t$ precedes the target time $t+1$ by exactly 60 minutes.
   - The target observation at $t+1$ is completely absent from the 24-hour sequence $[t-23, \dots, t]$ used for feature extraction.
2. **Missingness vs. Synthetic Data**:
   - The two unmatched forecasts on `DL008` (`fc_DL008_2026-10-30T00:00:00Z` and `fc_DL008_2026-11-01T11:00:00Z`) were logged with status `forecast_generated` and excluded from `live_trial_evaluation.csv`.
   - No synthetic or forward-filled values were generated to artificially complete the pairs.
3. **Dropout Resilience**:
   - A 21-hour national CPCB communications dropout occurred on Nov 3–4, 2026. The rolling buffer gate detected the interruption, returned `insufficient_history`, and stopped forecast generation until 24 contiguous clock hours re-accumulated.

---

## 4. Operational Commands for Ongoing Longitudinal Monitoring

The persistent collector is fully operational and can be executed via the following reproducible CLI commands:

### 1. View Current Telemetry & Dataset Status
```powershell
python scripts/run_longitudinal_collector.py --status
```
*Outputs current observation counts, date spans, prediction totals, matched evaluation counts, and recent API outages from `collector_state.json`.*

### 2. Execute an Incremental Collection & Evaluation Cycle
```powershell
python scripts/run_longitudinal_collector.py --hours-back 48
```
*Connects to OpenAQ API v3, retrieves new particulate telemetry from the past 48 hours, appends new non-duplicate observations, issues 1-hour forecasts for newly verified 24h windows, matches elapsed targets, and updates the evaluation table.*

### 3. Local Dataset Synchronization (Offline / Sandboxed)
```powershell
python scripts/run_longitudinal_collector.py --sync-only
```
*Processes all existing observations in `live_trial_observations.csv`, rebuilds rolling buffers, evaluates un-matched predictions against newly arrived observations, and synchronizes state counters without making external API calls.*

### 4. Background Recurring Automation (Windows Task Scheduler / Cron)
To run the collector on an hourly or daily recurring schedule:
```powershell
# Windows Task Scheduler (runs daily at midnight UTC)
schtasks /create /tn "VayuDrishti_Collector" /tr "python E:\Documents\multimodal_air_quality_vscode_starter\scripts\run_longitudinal_collector.py --hours-back 48" /sc daily /st 00:05
```

---

## 5. Remaining Operational Limitations & Research Governance

1. **Geographic Footprint**: Currently encompasses 4 CAAQMS stations across Andhra Pradesh and Delhi. Expanding coverage to all 85 historical stations requires OpenAQ to register active, verified modern sensor endpoints for each location.
2. **Missing Meteorological Channels**: OpenAQ endpoints provide particulate readings ($\text{PM}_{2.5}$ and $\text{PM}_{10}$) but lack reliable co-located wind, temperature, and planetary boundary layer height data. Consequently, the multimodal severity-gated ensemble remains in historical evaluation, while the standalone GBDT serves live shadow telemetry.
3. **Seasonal Generalization Window**: The current 14.2-day trial demonstrates operational stability, but seasonal generalization across winter inversions, pre-monsoon dust storms, and monsoon washout requires a **minimum 90-day continuous trial**.
4. **AQI Scope**: All predictions and targets are physical concentrations ($\mu\text{g/m}^3$). They must not be advertised as official CPCB Air Quality Index (AQI) values without implementing the statutory 24-hour truncated sub-index piecewise algorithm.
