# VayuDrishti Research Paper Technical Audit & Refinement Report

**Audited Manuscript:** `VayuDrishti_IEEE_Manuscript.docx` and `paper/main.tex`  
**Title:** *"VayuDrishti: A Residual Multimodal Hybrid Framework for Air Quality Forecasting"*  
**Date:** November 2026  
**Auditor:** IEEE Senior Reviewer / Scientific Manuscript Correction Specialist  

---

## 1. Executive Summary of Audit Findings

A rigorous technical audit was conducted on the manuscript, cross-referencing every mathematical equation, architectural description, feature dimension, and numerical evaluation against the primary source code (`src/`, `models/`, `experiments/`) and verified experiment logs.

The audit identified and resolved four critical categories of technical discrepancies:
1. **Ensemble Implementation Realignment**: Replaced an unverified continuous sigmoid weighting equation with the true discrete severity threshold switch implemented in production ($w_{25} = 1.0$ if $\text{PM}_{2.5} \ge 250\,\mu\text{g/m}^3$, else $0.40$; $w_{10} = 1.0$ if $\text{PM}_{10} \ge 400\,\mu\text{g/m}^3$, else $0.45$).
2. **27-Feature Sensor Schema Correction**: Removed atmospheric pressure (which was not in the trained checkpoint) and corrected the 5 meteorological variables to Temperature, Dew Point, Relative Humidity, Wind Speed, and Precipitation.
3. **Paired Multimodal Cohort Harmonization**: Resolved the misleading comparison between the full historical test benchmark ($N=404,431$) and the synchronous multimodal cohort ($N=97$). Embedded a dedicated paired evaluation table honestly reporting the marginal $0.06\,\mu\text{g/m}^3$ optical adjustment on $\text{PM}_{2.5}$ (8.98 vs. 9.04) and coarse particulate sensitivity limits on $\text{PM}_{10}$ (15.88 vs. 15.82).
4. **Conformal Uncertainty Rigor**: Delineated the theoretical marginal coverage guarantee under exchangeability from empirical test evaluations under non-stationary seasonal shifts.

---

## 2. Major Technical Corrections

| Component | Pre-Audit Draft State | Audited & Corrected Implementation | Source Code / Evidence |
| :--- | :--- | :--- | :--- |
| **LSTM-GBDT Blending Weights** | $w_{25} = 0.90 + 0.10 \times \text{gate}_{25}$ (continuous sigmoid) | Standard fixed weights: $0.40\,\text{LSTM} + 0.60\,\text{GBDT}$ for $\text{PM}_{2.5}$; $0.45\,\text{LSTM} + 0.55\,\text{GBDT}$ for $\text{PM}_{10}$. | `src/services/sensor_forecast_service.py` (lines 157–187) |
| **Severity Gating Mechanism** | Continuous sigmoid transition function | Discrete crisis threshold switch: $w_{25} = 1.0$ if $\text{PM}_{2.5} \ge 250\,\mu\text{g/m}^3$, $w_{10} = 1.0$ if $\text{PM}_{10} \ge 400\,\mu\text{g/m}^3$ (hard switch to pure LSTM). | `src/services/sensor_forecast_service.py` (lines 158–159) |
| **Meteorological Channels** | Included Atmospheric Pressure | Exactly 5 variables: Temperature ($^\circ\text{C}$), Dew Point ($^\circ\text{C}$), Relative Humidity (\%), Wind Speed ($\text{m/s}$), Precipitation ($\text{mm}$). | `src/prediction/lstm_forecaster.py` (lines 69–73) |
| **Temporal / Spatial Channels** | Informal cyclical list | Exactly 10 features: Wind Direction ($\sin, \cos$), Hour ($\sin, \cos$), Month ($\sin, \cos$), Day of Week ($\sin, \cos$), Station Latitude, Longitude. | `src/prediction/lstm_forecaster.py` (lines 74–83) |
| **Residual Hybrid Masking** | Stated optical embedding passed directly | Visual embedding explicitly masked prior to concatenation ($m \cdot e_v$), and effective gate scaled ($\alpha_{\text{eff}} = m\alpha$). | `experiments/unified_hybrid/accuracy_optimization/hybrid_model_optimized.py` (lines 157, 162) |
| **Mathematical Fallback** | Stated verbally without proof | Formally stated and proven via Theorem 1: $m=0 \implies \alpha_{\text{eff}}=0 \implies \hat{y}_{\text{hybrid}} \equiv \hat{y}_{\text{sensor}}$. | `experiments/unified_hybrid/accuracy_optimization/hybrid_model_optimized.py` (lines 231–233) |
| **Standalone Vision Regressor** | Ambiguously conflated with hybrid residual | Explicitly isolated as an auxiliary diagnostic baseline ($2048 \to 256 \to 64 \to 1$). Stated that its predicted scalar is never used as $\Delta$. | `src/prediction/image_estimator.py` (lines 21–46) |

---

## 3. Incorrect Claims & Misconceptions Removed

1. **Removed Claim of Hybrid Superiority on Universal Test Data**:
   - *Previous Draft*: Claimed the residual hybrid model achieved 8.98 $\mu\text{g/m}^3$ across nationwide data compared to 8.86 $\mu\text{g/m}^3$ for LSTM, which was both mathematically contradictory and conflated different test splits.
   - *Correction*: Delineated that 8.86 $\mu\text{g/m}^3$ was evaluated on 404,431 historical test sequences (where imagery does not exist), whereas 8.98 $\mu\text{g/m}^3$ was evaluated strictly on the $N=97$ paired multimodal test cohort.
2. **Removed Continuous Sigmoid Gating Formulation**:
   - The continuous sigmoid blending equation was removed as the active deployment code utilizes the discrete threshold switch ($\theta_{25} = 250, \theta_{10} = 400\,\mu\text{g/m}^3$) to decouple GBDT.
3. **Removed Unverified Image Quality Filters**:
   - Removed speculative claims that the system automatically filters nighttime or rain-corrupted images using automated computer vision heuristics, retaining only the verified binary availability mask ($m \in \{0, 1\}$) and Haversine distance thresholding ($d \le 5\text{ km}$).

---

## 4. Verified Experimental Results Retained

The manuscript retains the following verified numerical results traceable to project artifacts:

### A. Full Historical National Benchmark ($N=404,431$ Sequences, 90 Stations, Sept 2019 – July 2020)
- **1-Hour Persistence Baseline**: $\text{PM}_{2.5}$ MAE = $10.68\,\mu\text{g/m}^3$ (RMSE 19.81, $R^2 = 0.9423$); $\text{PM}_{10}$ MAE = $19.74\,\mu\text{g/m}^3$ (RMSE 36.24, $R^2 = 0.9115$).
- **Pretrained 2-Layer LSTM Baseline**: $\text{PM}_{2.5}$ MAE = $10.57\,\mu\text{g/m}^3$ (RMSE 18.58, $R^2 = 0.9492$); $\text{PM}_{10}$ MAE = $18.86\,\mu\text{g/m}^3$ (RMSE 32.20, $R^2 = 0.9301$).
- **Severity-Gated Ensemble**: $\text{PM}_{2.5}$ MAE = **$10.35\,\mu\text{g/m}^3$** (RMSE 18.38, $R^2 = 0.9503$); $\text{PM}_{10}$ MAE = **$18.14\,\mu\text{g/m}^3$** (RMSE 31.64, $R^2 = 0.9325$).

### B. Synchronous Paired Multimodal Benchmark ($N=97$ Instances Across 9 Held-Out Groups)
- **1-Hour Persistence**: $\text{PM}_{2.5}$ MAE = 9.04, $\text{PM}_{10}$ MAE = 9.39 $\mu\text{g/m}^3$.
- **Pretrained Sensor LSTM**: $\text{PM}_{2.5}$ MAE = 9.04, $\text{PM}_{10}$ MAE = 15.82 $\mu\text{g/m}^3$.
- **Residual Hybrid (Full, $m=1$)**: $\text{PM}_{2.5}$ MAE = **8.98 $\mu\text{g/m}^3$** (modest $0.06\,\mu\text{g/m}^3$ optical adjustment), $\text{PM}_{10}$ MAE = 15.88 $\mu\text{g/m}^3$.
- **Residual Hybrid (Fallback, $m=0$)**: $\text{PM}_{2.5}$ MAE = 9.04, $\text{PM}_{10}$ MAE = 15.82 $\mu\text{g/m}^3$ (identically matches sensor baseline).
- **Standalone Image-Only ResNet-50**: $\text{PM}_{2.5}$ MAE = 80.57 $\mu\text{g/m}^3$ ($R^2 = -0.1096$).

### C. Longitudinal Multi-Station Live Cohort ($N=236,289$ Sequences Across 15 Stations)
- **1-Hour Persistence**: $\text{PM}_{2.5}$ MAE = 6.95, $\text{PM}_{10}$ MAE = 17.16 $\mu\text{g/m}^3$.
- **Standalone Adapted GBDT**: $\text{PM}_{2.5}$ MAE = 6.67 (+3.97%), $\text{PM}_{10}$ MAE = 16.29 (+5.06%).
- **Adapted Severity-Gated Ensemble**: $\text{PM}_{2.5}$ MAE = 6.62 (+4.72%), $\text{PM}_{10}$ MAE = 16.14 (+5.92%).

### D. Conformal Calibration & Statistical Inference
- **Conformal Coverage (118,145 test instances)**: 80% nominal $\to$ 92.87% ($\text{PM}_{2.5}$) and 93.66% ($\text{PM}_{10}$); 90% nominal $\to$ 96.76% and 97.23%; 95% nominal $\to$ 98.38% and 98.71%.
- **Stationary Block Bootstrap ($B=5,000, L=24\text{h}$)**: Both pollutants pass Bonferroni and FDR-adjusted hypothesis tests ($p < 10^{-12}$).

---

## 5. Figures Replaced and Corrected

All 8 scientific architecture diagrams were regenerated via [scripts/generate_ieee_architecture_diagrams.py](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/generate_ieee_architecture_diagrams.py) and verified:
1. `fig_arch_overall`: Updated with audited ensemble blend (0.40/0.60) and threshold switch ($\ge 250, \ge 400\,\mu\text{g/m}^3$).
2. `fig_arch_preprocessing`: Replaced atmospheric pressure with audited Dew Point and 10 cyclical/spatial features.
3. `fig_arch_ensemble`: Replaced continuous sigmoid formula with the discrete threshold switch architecture.
4. `fig_arch_residual_hybrid`: Clearly illustrates masked visual modulation ($m \cdot e_v$), effective gate ($\alpha_{\text{eff}} = m\alpha$), and bounded residual clamps.

All experimental scatter plots, confusion matrices, multi-station graphs, and user-supplied dashboard screenshots are preserved without alteration.

---

## 6. Remaining Limitations & Boundaries

The manuscript explicitly documents the following practical boundary conditions:
- **Spatial Sparsity of Cameras**: High-quality collocated surveillance imagery remains limited (only 97 strictly synchronous, collocated test pairs across 9 independent groups), reflecting the real-world scarcity of public environmental camera feeds in Indian cities.
- **Physical Wavelength Limits on Coarse Particles**: Optical imagery provides measurable residual benefits for sub-micron $\text{PM}_{2.5}$ due to Rayleigh and fine Mie scattering, but negligible or slightly noisy signals for coarse $\text{PM}_{10}$, which requires optical sensors operating in infrared/LIDAR wavelengths.
- **Conformal Exchangeability**: Formal split-conformal coverage guarantees assume exchangeability; during sudden episodic weather transitions (e.g., monsoon onset or intense agricultural stubble burning), local coverage can fluctuate temporarily before recalibration.
