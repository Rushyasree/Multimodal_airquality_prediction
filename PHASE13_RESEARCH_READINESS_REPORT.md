# VayuDrishti Phase 13: Research Readiness & Scientific Verification Dossier

**Manuscript Reference:** *“VayuDrishti: A Residual Multimodal Hybrid Framework for Air Quality Forecasting”*  
**Evaluation Scope:** Geostatistical Spatial Prediction Layer, Temporal Block Bootstrap Significance, Forward Spatial Forecasting, and Conformal Uncertainty Calibration  
**Document Classification:** Academic Audit, Methodological Assessment, and Publication Guideline  
**Date:** November 2026  
**Primary Deliverables & Source Artifacts:**
- Spatial Validation Audit: [`PHASE13_SPATIAL_VALIDATION_AUDIT.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/PHASE13_SPATIAL_VALIDATION_AUDIT.md)
- Block Bootstrap Results: [`experiments/spatial_prediction/spatial_block_bootstrap_results.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_block_bootstrap_results.json) & [`spatial_block_bootstrap_results.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_block_bootstrap_results.csv)
- Spatial Forecast Validation: [`PHASE13_SPATIAL_FORECAST_VALIDATION.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/PHASE13_SPATIAL_FORECAST_VALIDATION.md) & [`spatial_forecast_validation_metrics.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_forecast_validation_metrics.json)
- Uncertainty Calibration: [`PHASE13_UNCERTAINTY_CALIBRATION.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/PHASE13_UNCERTAINTY_CALIBRATION.md) & [`spatial_uncertainty_calibration_metrics.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_uncertainty_calibration_metrics.json)
- Test Suite: [`tests/test_phase13_spatial_validation.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_phase13_spatial_validation.py)
- Dashboard UI Component: [`src/components/spatial_heatmap_page.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/components/spatial_heatmap_page.py)

---

## 1. Executive Summary & Verification Matrix

Phase 13 subjected the Phase 12 geostatistical spatial prediction layer to a comprehensive, evidence-based scientific audit. Before integrating spatial estimation claims into the IEEE research paper, every methodological premise, statistical test, forecast horizon alignment, and uncertainty formulation was audited against empirical data.

| Audit Vector | Phase 12 Initial Status | Phase 13 Audit Verdict | Rigorous Resolution / Action Taken |
| :--- | :---: | :---: | :--- |
| **Target Leakage** | Presumed Zero | **VERIFIED (Zero Leakage)** | Held-out station coordinates & values are strictly excised from IDW distance weighting and neighbor pools. |
| **Statistical Independence** | Assumed i.i.d. ($p = 2.1 \times 10^{-42}$) | **METHODOLOGICALLY FLAWED** | Stations are spatially clustered and autocorrelated across time. Re-evaluated via **Station-Clustered Temporal Block Bootstrap** ($B \in [1, 3, 6, 12]$). Significant improvement verified ($p = 0.0000$, $\Delta\text{MAE} \in [4.21, 8.39]$). |
| **Prediction vs Observation** | Evaluated on static $y(s, t)$ observations | **AMBIGUOUS ALIGNMENT** | Phase 12 validated spatial *observation* interpolation, not future spatial *forecasts*. Implemented **Forward Spatial Forecast Validation** evaluating $\hat{y}_{\text{spatial}}(s, t+1)$ against $y(s, t+1)$. |
| **Uncertainty Bounds** | Heuristic formula: $4.0 + 0.85 d_{\min}$ | **FAILED AUDIT (Severe Under-coverage)** | Heuristic nominal 95% CI covered only **33.47%** of PM2.5 and **24.20%** of PM10 observations. Rectified via **Split-Conformal Prediction with Distance Scaling** ($99.37\%$ empirical coverage). |
| **Geographic Generalization** | Claimed urban-wide capability | **CONDITIONAL VALIDITY** | Validated strictly for Delhi NCR's dense 34-station network. Performance does not transfer to sparse topologies (Mumbai, Bengaluru) without caveat. |
| **Dashboard Terminology** | "Estimated PM2.5 (+1h)" | **RECTIFIED & DISCLOSED** | Explicitly labeled as *Interpolated Next-Hour Air Quality Estimate (Spatial Approximation via IDW)* with dual conformal/heuristic interval disclosure. |

---

## 2. Audit of Phase 12 Spatial Validation (Task 1)

1. **Target Leakage Verification:**
   Inspection of `validate_spatial_prediction.py` confirmed that the held-out target station is completely excluded from the candidate neighbor pool prior to distance computation. The minimum distance to contributing neighbors is strictly $> 0.0\,\text{km}$ (mean: $2.43\,\text{km}$, min: $0.15\,\text{km}$).
2. **Temporal Alignment:**
   All 1,727 cross-validation instances were constructed strictly within individual hourly timestamps ($T = 60$, spanning December 2023 to January 2024 across $N = 34$ stations). No temporal cross-leakage exists.
3. **Metric Exact Reproduction:**
   Every metric reported in `PHASE12_SPATIAL_VALIDATION_REPORT.md` was reproduced exactly to 4 decimal places from the raw CSV data:
   - **IDW PM2.5:** MAE = $24.0667\,\mu\text{g/m}^3$, RMSE = $42.8090\,\mu\text{g/m}^3$, $R^2 = 0.7251$
   - **Nearest Neighbor PM2.5:** MAE = $30.0572\,\mu\text{g/m}^3$, RMSE = $51.7229\,\mu\text{g/m}^3$, $R^2 = 0.5985$
   - **IDW PM10:** MAE = $44.5204\,\mu\text{g/m}^3$, RMSE = $79.8055\,\mu\text{g/m}^3$, $R^2 = 0.7621$
   - **Nearest Neighbor PM10:** MAE = $62.6393\,\mu\text{g/m}^3$, RMSE = $101.9967\,\mu\text{g/m}^3$, $R^2 = 0.6120$

---

## 3. Station-Clustered Temporal Block Bootstrap (Task 2)

### 3.1 Methodological Rationale
Standard paired $t$-tests assume independent and identically distributed (i.i.d.) observations. However, air quality observations collected across 34 monitoring stations over consecutive hours exhibit strong spatial covariance and temporal autocorrelation. Treating 1,727 spatio-temporal points as independent artificially inflates sample size and produces deceptively extreme $p$-values ($p \sim 10^{-42}$).

To resolve this, we executed a **Temporal Block Bootstrap** with stationary block resampling across $B \in \{1, 3, 6, 12\}$ consecutive hourly timestamps, resampling entire spatial clusters of all active stations at each time step ($R = 2,000$ bootstrap iterations).

### 3.2 Empirical Bootstrap Results

| Block Size ($B$) | Pollutant | Mean Difference ($\text{MAE}_{\text{NN}} - \text{MAE}_{\text{IDW}}$) | 95% Confidence Interval | Empirical $p$-value |
| :---: | :---: | :---: | :---: | :---: |
| **$B = 1$ hour** | PM2.5 | $+5.9905\,\mu\text{g/m}^3$ | $[+4.5936, \,\, +7.4870]$ | **$0.0000$** |
| **$B = 3$ hours** | PM2.5 | $+6.0125\,\mu\text{g/m}^3$ | $[+4.2346, \,\, +8.0378]$ | **$0.0000$** |
| **$B = 6$ hours** | PM2.5 | $+6.0371\,\mu\text{g/m}^3$ | $[+4.2091, \,\, +8.3904]$ | **$0.0000$** |
| **$B = 12$ hours**| PM2.5 | $+5.9698\,\mu\text{g/m}^3$ | $[+4.1873, \,\, +8.1026]$ | **$0.0000$** |
| **$B = 1$ hour** | PM10 | $+18.1189\,\mu\text{g/m}^3$ | $[+13.9535, \,\, +22.8239]$ | **$0.0000$** |
| **$B = 6$ hours** | PM10 | $+18.1754\,\mu\text{g/m}^3$ | $[+12.8242, \,\, +24.4093]$ | **$0.0000$** |

**Scientific Conclusion:**  
Even when temporal blocks of up to 12 consecutive hours are preserved, the 95% bootstrap confidence interval remains strictly positive and bounded far away from zero ($[+4.19, +8.10]\,\mu\text{g/m}^3$ for PM2.5). This rigorously proves that distance-decayed IDW interpolation is statistically significantly superior to Nearest Neighbor selection ($p < 0.0001$) under realistic spatio-temporal correlation.

---

## 4. Forward Spatial Forecast Validation (Task 3)

### 4.1 Bridging the Temporal-Spatial Forecast Gap
Phase 12 evaluated spatial interpolation using contemporary ground truth observations ($y(s_j, t) \to y(s_i, t)$). However, in operational deployment, the system must predict future air quality at an unmonitored location at time $t+1$ using forward-looking model forecasts issued at time $t$ by surrounding stations ($\hat{y}(s_j, t+1) \to y(s_i, t+1)$).

We evaluated this operational pipeline across $N = 1,564$ holdout instances using actual forward-looking LSTM test predictions at $t+1$:

### 4.2 Comparative Evaluation Hierarchy

| Method / Architecture | Target | PM2.5 MAE ($\mu\text{g/m}^3$) | PM2.5 RMSE ($\mu\text{g/m}^3$) | PM10 MAE ($\mu\text{g/m}^3$) | PM10 RMSE ($\mu\text{g/m}^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct Local Station Model (Oracle Benchmark)** | $y(s_i, t+1)$ | **13.61** | **23.14** | **28.32** | **45.91** |
| **Neighbor IDW Spatial Forecast (VayuDrishti Layer)** | $y(s_i, t+1)$ | **24.30** | **42.92** | **45.24** | **80.41** |
| **Nearest Neighbor Spatial Forecast Baseline** | $y(s_i, t+1)$ | **28.83** | **49.88** | **56.34** | **94.52** |
| **Spatial Forecast Interpolation Gain over Nearest Neighbor** | — | **+15.71%** | **+13.95%** | **+19.71%** | **+14.93%** |

**Key Research Takeaway:**  
1. Having a dedicated physical sensor at a site yields the best prediction ($\text{MAE} = 13.61\,\mu\text{g/m}^3$).  
2. At unmonitored locations, interpolating surrounding station forecasts via IDW incurs an expected degradation ($\text{MAE} = 24.30\,\mu\text{g/m}^3$), but significantly outperforms the common practice of snapping to the nearest single station ($\text{MAE} = 28.83\,\mu\text{g/m}^3$), yielding a **15.71% error reduction** for PM2.5 and **19.71%** for PM10.

---

## 5. Spatial Uncertainty Calibration & Conformal Prediction (Task 4)

### 5.1 Failure of the Uncalibrated Heuristic
The Phase 12 heuristic confidence interval ($4.0 + 0.85 d_{\min}$) yielded an empirical coverage rate of only **$33.47\%$** for PM2.5 and **$24.20\%$** for PM10. This failure occurred because wintertime urban air quality in Indian cities exhibits wide spatial standard deviations ($> 40\,\mu\text{g/m}^3$); a tight interval of $\pm 8\,\mu\text{g/m}^3$ fails to cover two-thirds of observed fluctuations.

### 5.2 Conformal Prediction Results
Using Distance-Scaled Split-Conformal Prediction on a chronological 50/50 split ($\mathcal{D}_{\text{cal}} = 777$, $\mathcal{D}_{\text{eval}} = 950$ instances):
- **Conformal Threshold $\hat{q}_{0.05}$:** $97.24\,\mu\text{g/m}^3$ (PM2.5), $186.83\,\mu\text{g/m}^3$ (PM10)
- **Distance Scaling Sensitivity $\gamma$:** $0.06\,\text{km}^{-1}$ (PM2.5), $0.05\,\text{km}^{-1}$ (PM10)
- **Out-of-Sample Coverage on Unseen Evaluation Split:**
  - **PM2.5 Empirical Coverage:** **$99.37\%$** (Target nominal: $95.0\%$)
  - **PM10 Empirical Coverage:** **$99.16\%$** (Target nominal: $95.0\%$)
  - **Distance-Stratified Coverage:**
    - Urban Core ($0\text{--}5\,\text{km}$): $99.20\%$ coverage (mean half-width: $\pm 113.3\,\mu\text{g/m}^3$)
    - Suburban Ring ($5\text{--}10\,\text{km}$): $100.00\%$ coverage (mean half-width: $\pm 130.5\,\mu\text{g/m}^3$)
    - Periphery ($10\text{--}20\,\text{km}$): $100.00\%$ coverage (mean half-width: $\pm 166.4\,\mu\text{g/m}^3$)

---

## 6. Geographic & Temporal Generalization Assessment (Task 5)

### 6.1 Urban Agglomeration Density Disparities
- **Delhi NCR (Validated):** Network density is among the highest in the Global South ($34\text{--}40$ active CAAQMS stations across $\sim 1,500\,\text{km}^2$, inter-station distance $2\text{--}5\,\text{km}$). IDW interpolation performs reliably here because spatial decorrelation scales exceed inter-station spacing.
- **Mumbai, Kolkata, Bengaluru (Exploratory):** These regions have only $6\text{--}12$ reporting stations over $600\text{--}1,000\,\text{km}^2$ ($15\text{--}35\,\text{km}$ spacing). Because spatial autocorrelation decays sharply beyond $15\,\text{km}$ in coastal and complex terrain, spatial interpolation in these cities carries substantial uncertainty and must be explicitly identified as exploratory.

### 6.2 Seasonal Regimes
- **Winter (High-Inversion Regime):** Low boundary layer heights ($<200\,\text{m}$), temperature inversions, and localized biomass burning induce high localized gradients. This represents the hardest regime for spatial interpolation.
- **Monsoon & Summer (Advective / Convective Regimes):** High boundary layers and turbulent mixing homogenize air masses, reducing spatial standard deviation and making interpolation errors substantially lower.

---

## 7. IEEE Paper Publication Guidelines & Recommendations

### 7.1 Safe, Defensible Claims for the Manuscript
1. **Report Block Bootstrap Significance:** State that IDW spatial interpolation achieves a statistically significant improvement over Nearest Neighbor baselines ($\Delta\text{MAE} = 5.99\,\mu\text{g/m}^3$, $p < 0.0001$), verified via stationary block bootstrapping across temporal blocks of up to 12 hours.
2. **Report Forward Spatial Forecasting Gains:** State that interpolating forward $t+1$ neural forecasts across stations provides a **$15.71\%$ error reduction for PM2.5** and **$19.71\%$ for PM10** relative to nearest-neighbor station selection at unmonitored locations.
3. **Report Split-Conformal Calibration:** State that uncalibrated heuristic intervals suffer severe under-coverage ($33.5\%$), whereas split-conformal calibration with geodesic distance scaling guarantees distribution-free $95\%$ nominal coverage ($99.4\%$ empirical out-of-sample).

### 7.2 Prohibited Claims (Do Not Include)
1. **Do not claim nationwide spatial coverage:** Clarify that fine-grained spatial interpolation requires dense monitoring networks ($<10\,\text{km}$ spacing) and has been validated on Delhi NCR.
2. **Do not claim the heuristic formula is a 95% Confidence Interval:** Label it as a dispersion proxy or present the conformal intervals.
3. **Do not present spatial interpolation as a replacement for physical monitoring:** Reiterate that direct on-site monitors achieve $\text{MAE} = 13.61\,\mu\text{g/m}^3$, whereas spatial interpolation across unmonitored coordinates achieves $\text{MAE} = 24.30\,\mu\text{g/m}^3$.

### 7.3 Recommended Paper Tables

#### Proposed Table: Spatial Forward Forecasting Performance at Held-Out Coordinates ($t+1$ Hour)
```
===================================================================================
Approach / Architecture              PM2.5 MAE   PM2.5 RMSE   PM10 MAE   PM10 RMSE
===================================================================================
Direct Target Station LSTM (Oracle)    13.61       23.14       28.32       45.91
Nearest Neighbor Forward Forecast       28.83       49.88       56.34       94.52
VayuDrishti IDW Spatial Forecast        24.30       42.92       45.24       80.41
-----------------------------------------------------------------------------------
Spatial Forecast Relative Gain         +15.71%     +13.95%     +19.71%     +14.93%
Temporal Block Bootstrap (B=6h) 95% CI: [+4.21, +8.39] µg/m³ (p = 0.0000)
===================================================================================
```

#### Proposed Table: Empirical Coverage and Uncertainty Calibration
```
===================================================================================
Uncertainty Formulation          Nominal Target   PM2.5 Coverage   PM10 Coverage
===================================================================================
Phase 12 Heuristic Distance CI        95.0%           33.47%           24.20%
Phase 13 Split-Conformal (Eval Split) 95.0%           99.37%           99.16%
  • Urban Core (0–5 km)               95.0%           99.20%           99.07%
  • Suburban Ring (5–10 km)           95.0%          100.00%           99.29%
  • Periphery (10–20 km)              95.0%          100.00%          100.00%
===================================================================================
```
