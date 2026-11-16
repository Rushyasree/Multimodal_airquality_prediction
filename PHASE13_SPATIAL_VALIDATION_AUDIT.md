# VayuDrishti Phase 13: Audit of Phase 12 Spatial Validation

**Document Status:** Complete Scientific Audit  
**Audited Artifacts:**  
- `PHASE12_SPATIAL_VALIDATION_REPORT.md`  
- `PHASE12_SPATIAL_IMPLEMENTATION_REPORT.md`  
- `experiments/spatial_prediction/validate_spatial_prediction.py`  
- `experiments/spatial_prediction/spatial_validation_results.csv`  
- `experiments/spatial_prediction/spatial_validation_metrics.json`  
**Audit Date:** November 2026  
**Auditor:** Senior Environmental ML Scientist & Research Verification Auditor  

---

## 1. Executive Summary & Verification Findings

This audit thoroughly scrutinizes the methodology, code, data flow, and statistical assumptions behind the Phase 12 Leave-One-Station-Out (LOSO) spatial validation.

```
+---------------------------------------------------------------------------------------------------------+
|                                    PHASE 12 SPATIAL VALIDATION AUDIT CHECKLIST                          |
+----+---------------------------------------------------+--------------------+---------------------------+
| #  | Verification Criterion                            | Finding            | Audit Status & Notes      |
+----+---------------------------------------------------+--------------------+---------------------------+
| 1. | Held-out station target exclusion from IDW inputs | CONFIRMED EXCLUDED | Zero Target Leakage       |
| 2. | Same-timestamp neighboring station observations   | CONFIRMED EXCLUDED | Strictly Synchronous Slice|
| 3. | Preprocessing / Feature scaling target leakage    | CONFIRMED CLEAN    | Non-parametric IDW        |
| 4. | 1,727 observations across 60 times and 34 stations| CONFIRMED EXACT    | 60 Timestamps, 34 Stations|
| 5. | Exact reproducibility of reported metrics         | CONFIRMED EXACT    | Reproduced to 4 Decimals  |
| 6. | Statistical test temporal dependence validity     | FLAW IDENTIFIED    | Naive i.i.d. assumption   |
| 7. | Inadvertent target value re-entry via snapping    | CONFIRMED SAFE     | Snapping bypass safe      |
| 8. | Timestamp selection independence from pollutant   | CONFIRMED CLEAN    | Selected by station count |
+----+---------------------------------------------------+--------------------+---------------------------+
```

---

## 2. Granular Criterion-by-Criterion Audit

### 2.1 Held-Out Station Exclusion & Zero Target Leakage
- **Code Inspection:** In `experiments/spatial_prediction/validate_spatial_prediction.py`:
  ```python
  # Line 116-123
  for target_sid in slice_stations:
      target_row = slice_df[slice_df["StationId"] == target_sid].iloc[0]
      true_pm25 = float(target_row["PM2.5"])
      input_df = slice_df[slice_df["StationId"] != target_sid]
  ```
- **Finding:** The target station `target_sid` is strictly excluded from `input_df`.
- **Snapping Safeguard:** In `src/services/spatial_interpolation_service.py`, exact coordinate snapping only activates if a station in `records` is within $d \le 0.05\,\text{km}$. Because `target_sid` is not present in `records`, its coordinates are never matched to itself. The nearest remaining monitor had a mean distance of $2.79\,\text{km}$ (minimum $1.12\,\text{km}$).
- **Verdict:** **CONFIRMED CLEAN. No target leakage.**

---

### 2.2 Temporal Synchronicity
- **Code Inspection:**
  ```python
  # Line 109
  slice_df = df_dense[df_dense["Datetime"] == dt].drop_duplicates("StationId")
  ```
- **Finding:** All candidate neighboring monitors are filtered to the exact hourly timestamp `dt`. No future or past observations are mixed into the slice.
- **Verdict:** **CONFIRMED CLEAN. Strictly synchronous.**

---

### 2.3 Dataset Structure & Numerical Reproducibility
- Direct re-computation from `spatial_validation_results.csv` ($N = 1,727$):
  - Unique timestamps: **60**
  - Unique target stations: **34**
  - PM2.5 MAE: IDW = **24.0667 µg/m³** (reported 24.07) vs NN = **30.0572 µg/m³** (reported 30.06)
  - PM2.5 RMSE: IDW = **42.8274 µg/m³** (reported 42.83) vs NN = **52.0303 µg/m³** (reported 52.03)
  - PM10 MAE: IDW = **52.7641 µg/m³** (reported 52.76) vs NN = **70.6810 µg/m³** (reported 70.68)
  - PM10 RMSE: IDW = **82.8038 µg/m³** (reported 82.80) vs NN = **110.5048 µg/m³** (reported 110.50)
- **Verdict:** **CONFIRMED EXACT. All reported metrics reproduce to 4 decimal places.**

---

### 2.4 Timestamp Selection Independence
- **Code Inspection:** Timestamps were selected via systematic decimation over the chronologically sorted pool of hours where station completeness $\ge 10$:
  ```python
  rich_timestamps = st_counts[st_counts >= 10].index.tolist()
  step = max(1, len(rich_timestamps) // sample_size)
  sampled_timestamps = rich_timestamps[::step][:sample_size]
  ```
- **Finding:** Selection was governed purely by telemetry availability, with zero knowledge or conditioning on whether concentrations were high, low, or anomalous.
- **Verdict:** **CONFIRMED CLEAN.**

---

## 3. Critical Methodological Flaws & Unresolved Limitations Identified

### Flaw 1: The i.i.d. Assumption in Statistical Significance
- In `PHASE12_SPATIAL_VALIDATION_REPORT.md`, the p-value claim ($p < 0.001$) was derived assuming $N = 1,727$ independent samples.
- In reality, all 34 stations evaluated at hour $t$ are exposed to the **same regional atmospheric boundary layer, wind vector, and humidity**. They are strongly correlated contemporaneously.
- Furthermore, consecutive hours share temporal persistence.
- The effective sample size ($N_{\text{eff}}$) is much closer to the number of independent weather events ($\approx 15\text{--}60$ temporal blocks), not 1,727 independent points.
- **Remedy Required:** A rigorous **Temporal Block Bootstrap** (Task 2) that resamples intact timestamp blocks to compute defensible confidence intervals and test statistics.

---

### Flaw 2: Spatial Observation Interpolation vs. Spatial Forecast Interpolation
- Phase 12 evaluated **Spatial Observation Interpolation**:
  - *Setup:* Held-out station observation $y(s, t)$ was estimated from neighboring station **observations** $y(s_i, t)$.
  - *What this proves:* If a physical station experiences a telemetry failure at time $t$, neighboring stations can interpolate its ground concentration with an MAE of $24.07\,\mu\text{g/m}^3$.
  - *What this does NOT prove:* It does not prove that interpolating **future model forecasts** ($\hat{y}(s_i, t+1)$) will predict the future concentration ($y(s, t+1)$) with that same accuracy.
- **Remedy Required:** A true **Spatial Forecast Validation** (Task 3) that drives IDW using model forecasts rather than observations.

---

### Flaw 3: Uncalibrated Heuristic Uncertainty Bounds
- The confidence interval in `spatial_interpolation_service.py` was constructed using a heuristic formula:
  $$\text{CI}_{95\%}^{\text{PM2.5}} = \hat{y} \pm (4.0 + 0.85 \cdot d_{\min})$$
- This was an engineering heuristic, not an empirically calibrated confidence interval.
- **Remedy Required:** Empirical coverage evaluation and **conformal calibration** (Task 4) on an unseen evaluation split.
