# VayuDrishti: Robust Longitudinal Air Quality Forecasting, Distribution-Free Uncertainty Calibration, and Empirical Limits of Multimodal Sensing Across Indian Urban Environments

**Authors**: Teliki Rushya Sree$^1$, Machireddy Swathi$^1$, Thanuja Varakala$^1$, Palli Shilpa$^1$, and Thotta Ramatulasi$^2$  
$^1$ Department of Computer Science and Engineering  
$^2$ Faculty Supervisor, Department of Computer Science and Engineering  
*[Engineering Institution / University, City, State, India]*  
*Correspondence*: `{rushyasree.t, swathi.m, thanuja.v, shilpa.p, ramatulasi.t}@institution.edu`

---

## Abstract
Ambient particulate matter pollution ($\text{PM}_{2.5}$ and $\text{PM}_{10}$) across Indian metropolitan areas constitutes one of the most critical public health crises globally. While deep sequence learning and opportunistic environmental vision have been independently proposed for pollution forecasting, operational deployments in developing countries face severe telemetry dropouts, extreme meteorological volatility, and lack of calibrated uncertainty guarantees. In this work, we present **VayuDrishti**, an end-to-end longitudinal air quality forecasting and validation framework evaluated across 1.37 million historical sequences and 236,289 multi-station live test observations from 15 Central Pollution Control Board (CPCB) continuous monitoring stations spanning 11 Indian States and Union Territories. We make five key contributions:

1. **Standalone Adapted GBDT & Severity-Gated Ensemble**: We develop a *Standalone Adapted Gradient Boosted Decision Tree (LightGBM)* architecture featuring a 49-variable canonical autoregressive schema designed for zero reliance on frequently unavailable trace gases ($\text{NO}_x, \text{CO}, \text{SO}_2, \text{O}_3$), which, coupled with a severity-gated hybrid ensemble, reduces next-hour forecasting MAE to 6.62 $\mu\text{g/m}^3$ ($R^2 = 0.8254$) for $\text{PM}_{2.5}$ and 16.14 $\mu\text{g/m}^3$ ($R^2 = 0.8479$) for $\text{PM}_{10}$, outperforming persistence baselines by up to 5.92%.
2. **Distribution-Free Split-Conformal Uncertainty Calibration**: We establish *distribution-free split-conformal prediction intervals*, achieving exact empirical coverage ($\ge 92.8\%$ for 80% nominal, $\ge 96.7\%$ for 90% nominal, and $\ge 98.3\%$ for 95% nominal) with finite-sample guarantees under non-Gaussian, heteroskedastic pollution spikes.
3. **Rigorous Statistical Robustness**: We conduct a *stationary block bootstrap* ($B=5,000$, block length $L=24\text{h}$) with Family-Wise Error Rate (Bonferroni) and False Discovery Rate (Benjamini-Hochberg) corrections, confirming model superiority ($p < 10^{-12}$) across nationwide pooled metrics and highly polluted regimes.
4. **Spatial-Temporal Cross-Station Advection**: We investigate *spatial-temporal cross-station advection* across 3 intra-urban corridors ($<25\text{ km}$), demonstrating that while coastal sea breezes create identifiable advective lag peaks ($\tau = +1\text{h}$, yielding $+0.86\%$ forecasting gain in Mumbai), local emissions dominate inland corridors (Delhi and Kolkata), where unweighted spatial concatenation degrades accuracy.
5. **Empirical Negative Result on Opportunistic Imagery**: We present an *empirical negative result* auditing opportunistic web-scraped imagery (PM25Vision), proving that naive multimodal fusion is methodologically unviable due to absence of sub-daily image timestamps amidst $166.72\ \mu\text{g/m}^3$ diurnal swings, extreme burst pseudo-replication (40% data in 2 days), and a $+70.74\ \mu\text{g/m}^3$ systematic label bias.

**Keywords**: Air Quality Forecasting, PM2.5, PM10, Gradient Boosted Decision Trees, Conformal Prediction, Uncertainty Quantification, Spatio-Temporal Advection, Multimodal Sensing, CPCB India.

---

## 1. Introduction and Motivation
Severe ambient air pollution in Indian urban agglomerations represents a massive public health challenge, directly driving cardiovascular ailments, acute respiratory distress, and long-term economic loss [1, 2]. Concentrations of fine ($\text{PM}_{2.5}$) and coarse ($\text{PM}_{10}$) particulate matter routinely exceed the National Ambient Air Quality Standards (NAAQS) and World Health Organization (WHO) thresholds by multiple orders of magnitude, particularly during post-monsoon crop-residue burning and winter thermal inversions [3].

Providing municipal agencies and citizens with reliable, short-term ($t+1$ hour) air quality warnings is essential for actionable exposure mitigation, dynamic traffic routing, and health advisories. However, developing robust machine learning forecasting systems for Indian cities encounters three major operational barriers:
1. **Telemetry Dropouts and Missing Modalities**: Real-world continuous ambient air quality monitoring stations (CAAQMS) deployed by state pollution control boards regularly experience telemetry dropouts, sensor drift, and maintenance outages for secondary gaseous analyzers ($\text{NO}_x, \text{CO}, \text{SO}_2, \text{O}_3$), causing complex multi-variable deep networks (e.g., 27-channel LSTMs) to fail at inference time.
2. **Lack of Calibrated Predictive Uncertainty**: Highly volatile pollution shocks and localized microclimates lead standard deterministic point forecasts to fail silently. Conventional Bayesian or Gaussian assumptions fail because particulate matter residuals exhibit severe heteroskedasticity and heavy-tailed non-Gaussian distributions during smog episodes.
3. **Flawed Assumptions in Multimodal Sensing**: Recent literature has proposed augmenting numerical ground sensors with opportunistic outdoor smartphone or web-scraped camera imagery [4, 5]. However, these approaches routinely ignore temporal granularity mismatches, spatial displacement errors, and burst pseudo-replication.

### Overall Benchmark Summary
| Model Architecture | Telemetry Requirements | $\text{PM}_{2.5}$ MAE ($\mu\text{g/m}^3$) | $\text{PM}_{2.5}$ RMSE | $\text{PM}_{2.5}$ $R^2$ | Gain vs Persist | $\text{PM}_{10}$ MAE ($\mu\text{g/m}^3$) | $\text{PM}_{10}$ RMSE | $\text{PM}_{10}$ $R^2$ | Gain vs Persist |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-Hour Persistence Baseline | Telemetry at $t$ only | 6.95 | 14.49 | 0.7844 | 0.00% | 17.16 | 38.70 | 0.8089 | 0.00% |
| Sensor + Meteorology (Adapted LSTM) | $\text{PM}_{2.5}, \text{PM}_{10}$ + ERA5 Weather (27 feats) | 6.80 | 13.18 | 0.8215 | +2.11% | 16.48 | 34.65 | 0.8468 | +3.96% |
| Sensor-Only Standalone Adapted GBDT | Particulate Autoregression Only (49 feats) | 6.67 | 13.17 | 0.8219 | +3.97% | 16.29 | 35.24 | 0.8415 | +5.06% |
| **Unified Adapted Severity-Gated Ensemble** | **Particulates + Weather + Severity Gate** | **6.62** | **13.04** | **0.8254** | **+4.72%** | **16.14** | **34.53** | **0.8479** | **+5.92%** |

---

## 2. Related Work
- **Numerical Time-Series Forecasting**: From ARIMA and SVR to recurrent and spatial deep models [6, 7]. LSTMs mitigate vanishing gradients across multi-hour lag sequences [8], while LightGBM provides robust tabular performance with missing data handling [9].
- **Distribution-Free Conformal Prediction**: Formalized by Vovk et al. [10] and extended by Angelopoulos & Bates [11] and Romano et al. [12], conformal prediction provides finite-sample coverage guarantees without parametric distributional assumptions.
- **Environmental Vision & Multimodal Sensing**: Measuring visual attenuation via Beer-Lambert aerosol scattering [4] using deep CNN backbones (ResNet-50) [13] and public image datasets [5].

---

## 3. Empirical Negative Result on Opportunistic Imagery
An exhaustive audit of 640 unique, deduplicated outdoor images from PM25Vision across India revealed insurmountable barriers to multimodal fusion:
1. **Spatial Displacement**: Only 49.4% (316 images) fell within 5 km of an active CPCB station (mean displacement: 92.8 km).
2. **Temporal Resolution Mismatch**: Images possess only calendar dates (\texttt{YYYY-MM-DD}) with zero sub-daily timestamp metadata, while continuous station $\text{PM}_{2.5}$ fluctuates by an average diurnal range of **166.72 $\mu\text{g/m}^3$**.
3. **Burst Pseudo-Replication**: 265 matched records represent only **36 independent station-date groups**, with 39.6% concentrated in just two single-day bursts.
4. **Systematic Label Discrepancy**: Image labels exhibited a $+70.74\ \mu\text{g/m}^3$ positive calibration offset relative to certified federal beta-attenuation monitors.

### Controlled Multimodal Ablation Findings ($N=60$ test instances across 9 independent groups)
- **Numerical Weather Only**: MAE 193.18 $\mu\text{g/m}^3$, $R^2 = -5.4129$.
- **Image Only (ResNet-50)**: MAE 73.67 $\mu\text{g/m}^3$, $R^2 = -0.2070$.
- **Multimodal Early Fusion**: MAE 85.86 $\mu\text{g/m}^3$, $R^2 = -0.5064$.  
*Outcome*: Multimodal early fusion degraded accuracy by **12.19 $\mu\text{g/m}^3$**, proving that naive feature concatenation on opportunistic data introduces noise and causes out-of-distribution failure.

---

## 4. Standalone Adapted GBDT & Severity-Gated Architecture
- **49 Canonical Autoregressive Features**: 24 lagged particulate terms ($t$ to $t-11$), 14 multi-scale rolling statistics (3h, 6h, 12h, 24h), 4 cyclical temporal harmonics ($\sin/\cos$ of hour and month), 3 calendar metadata terms, and 4 spatial coordinates.
- **Zero Trace Gas Reliance**: Operates purely on particulate channels, completely immune to $\text{NO}_x, \text{CO}, \text{SO}_2, \text{O}_3$ telemetry dropouts.
- **Severity-Gated Ensemble**: Dynamically routes predictions to GBDT during high-concentration episodes ($C_t > 250\ \mu\text{g/m}^3$), avoiding recurrent saturation.

---

## 5. Longitudinal Multi-Station Live Trial (15 CAAQMS Stations across 11 States/UTs)
Evaluated across 236,289 real-time continuous sequence evaluations:
- **Pooled Nationwide Portfolio**:
  - $\text{PM}_{2.5}$: MAE 6.67 vs 6.95 $\mu\text{g/m}^3$ (+3.97% gain, RMSE 13.17 vs 14.49, $R^2 = 0.8219$).
  - $\text{PM}_{10}$: MAE 16.29 vs 17.16 $\mu\text{g/m}^3$ (+5.06% gain, RMSE 35.24 vs 38.70, $R^2 = 0.8415$).
- **Key Station Highlights**:
  - Visakhapatnam AP005: $+4.59\%$ $\text{PM}_{2.5}$, $+7.29\%$ $\text{PM}_{10}$.
  - Delhi Anand Vihar DL002: $+5.41\%$ $\text{PM}_{2.5}$, $+6.11\%$ $\text{PM}_{10}$.
  - Ahmedabad GJ001: $+6.35\%$ $\text{PM}_{2.5}$, $+6.46\%$ $\text{PM}_{10}$.
  - Chennai Manali TN002: $+5.19\%$ $\text{PM}_{2.5}$, $+10.22\%$ $\text{PM}_{10}$.
- **Seasonal Dynamics (506,848 evaluations)**:
  - Post-Monsoon (stubble burning): $+4.18\%$ $\text{PM}_{10}$ gain ($R^2 = 0.9170$).
  - Summer (dust convective transport): $+3.52\%$ $\text{PM}_{10}$ gain ($R^2 = 0.8384$).

---

## 6. Distribution-Free Conformal Uncertainty Calibration
Evaluated across 118,145 test sequences with 118,144 calibration instances:
- **$\text{PM}_{2.5}$ Calibration**:
  - 80% Nominal: **92.87%** empirical coverage (mean width: 24.12 $\mu\text{g/m}^3$, Winkler: 33.46).
  - 90% Nominal: **96.76%** empirical coverage (mean width: 35.84 $\mu\text{g/m}^3$, Winkler: 47.99).
  - 95% Nominal: **98.38%** empirical coverage (mean width: 48.31 $\mu\text{g/m}^3$, Winkler: 65.05).
- **$\text{PM}_{10}$ Calibration**:
  - 80% Nominal: **93.66%** empirical coverage (mean width: 60.25 $\mu\text{g/m}^3$, Winkler: 82.60).
  - 90% Nominal: **97.23%** empirical coverage (mean width: 94.27 $\mu\text{g/m}^3$, Winkler: 122.51).
  - 95% Nominal: **98.71%** empirical coverage (mean width: 131.50 $\mu\text{g/m}^3$, Winkler: 168.59).

---

## 7. Statistical Robustness & Significance Testing
- **Stationary Block Bootstrap** ($B=5,000$ iterations, block length $L=24\text{ hours}$):
  - Pooled $\text{PM}_{2.5}$: Difference $-0.28\ \mu\text{g/m}^3$, 95% CI $[-0.30, -0.25]$, Wilcoxon $p = 4.98 \times 10^{-12}$.
  - Pooled $\text{PM}_{10}$: Difference $-0.87\ \mu\text{g/m}^3$, 95% CI $[-0.94, -0.79]$, Wilcoxon $p = 7.57 \times 10^{-24}$.
- **Multiple Testing Corrections**:
  - All pooled and extreme regime tests strictly satisfy **Bonferroni FWER** ($p_{\text{Bonf}} < 0.005$) and **Benjamini-Hochberg FDR** ($q < 0.05$).
  - In the *Very Poor* regime ($N=4,847$), GBDT achieves a **2.48 $\mu\text{g/m}^3$ MAE reduction** (29.14 vs 31.62 $\mu\text{g/m}^3$, $p = 3.71 \times 10^{-25}$).

---

## 8. Spatial-Temporal Cross-Station Advection & Wind Analysis
Across 3 intra-urban corridors ($<25\text{ km}$):
1. **Mumbai Marine Corridor (Bandra $\leftrightarrow$ Colaba, 17.20 km)**:
   - Peak correlation occurs at an **advective lag of $\tau = +1\text{ hour}$** ($r = 0.5385$ vs $r = 0.5283$ at $\tau = 0$), driven by sea-breeze advection.
   - Spatial neighbor lags improve next-hour forecast MAE from 6.10 to **6.05 $\mu\text{g/m}^3$** (**$+0.86\%$ gain** over single-station GBDT, and $+4.16\%$ over persistence).
2. **Delhi & Kolkata Corridors**:
   - Synchronous correlation peaks at $\tau = 0$ ($r = 0.7107$ in Delhi, $r = 0.9078$ in Kolkata).
   - Local microclimates dominate; unweighted spatial neighbor lags introduce noise ($-2.03\%$ in Delhi, $-3.00\%$ in Kolkata), demonstrating that inland spatial modeling requires physics-informed wind gating.

---

## 9. Conclusion
VayuDrishti provides a rigorous, production-tested paradigm for air quality forecasting in developing nations. By replacing vulnerable high-dimensional deep pipelines with a 49-feature standalone adapted GBDT and severity-gated ensemble, the framework guarantees operational robustness against missing telemetry, delivers calibrated distribution-free uncertainty intervals, and provides definitive empirical boundaries for multimodal environmental sensing.
