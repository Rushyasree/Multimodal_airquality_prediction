---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f8fafc
color: #0f172a
---

# **VayuDrishti**
### Robust Longitudinal Air Quality Forecasting, Distribution-Free Uncertainty Calibration, and Empirical Limits of Multimodal Sensing Across India

**Authors:** Teliki Rushya Sree, Machireddy Swathi, Thanuja Varakala, Palli Shilpa  
**Faculty Supervisor:** Prof. Thotta Ramatulasi  
*Department of Computer Science and Engineering*

---

## The Public Health Crisis in Urban India

- **Severe Particulate Burden**: $\text{PM}_{2.5}$ and $\text{PM}_{10}$ concentrations routinely exceed CPCB NAAQS ($60/100\ \mu\text{g/m}^3$) and WHO limits ($15/45\ \mu\text{g/m}^3$) by $5\times$ to $10\times$.
- **Extreme Episodic Volatility**: Post-monsoon agricultural stubble burning and winter thermal surface inversions trigger acute respiratory hazards.
- **Urgent Need for Short-Term Warnings**: 1-hour ahead ($t+1$) forecasts are vital for municipal traffic diversions, school closures, and personal exposure management.

---

## Three Critical Operational Barriers

1. **Telemetry Dropouts & Missing Gaseous Channels**
   - High-dimensional models (e.g. 27-channel LSTMs) depend on trace gases ($\text{NO}_x, \text{CO}, \text{SO}_2, \text{O}_3$).
   - In live CAAQMS networks, trace gas analyzers suffer frequent maintenance outages, crashing naive inference pipelines.
2. **Uncalibrated Risk in Heavy-Tailed Regimes**
   - Gaussian or homoskedastic error assumptions fail during severe pollution episodes.
3. **Flawed Multimodal Assumptions**
   - Recent literature advocates fusing opportunistic smartphone / web images with ground telemetry without checking spatio-temporal alignment.

---

## Datasets & Nationwide Scope

- **Continuous Ground Telemetry (CPCB)**
  - 1,369,779 cleaned hourly records (2015–2020) across 90 continuous monitoring stations in 23 States/UTs.
- **ERA5-Land Atmospheric Reanalysis**
  - Hourly meteorology ($T, T_d, RH, u, v, P$) bilinearly matched to all 90 CAAQMS coordinates.
- **Nationwide Longitudinal Live Portfolio**
  - 15 CAAQMS stations spanning 11 Indian States/UTs (236,289 live sequence evaluations; 506,848 seasonal evaluations).
- **Environmental Vision**
  - Curated outdoor imagery from PM25Vision across Indian coordinates.

---

## Historical Benchmark: 2-Layer LSTM Forecaster

Evaluated on **404,431 held-out test sequences** across 90 continuous stations:
- **Continuous Concentration Accuracy**:
  - $\text{PM}_{2.5}$: MAE **10.57 $\mu\text{g/m}^3$**, RMSE 18.58 $\mu\text{g/m}^3$, $R^2 = 0.9492$
  - $\text{PM}_{10}$: MAE **18.86 $\mu\text{g/m}^3$**, RMSE 32.20 $\mu\text{g/m}^3$, $R^2 = 0.9301$
  - Beats 1-hour persistence baseline by 6.2% ($\text{PM}_{2.5}$) and 11.1% ($\text{PM}_{10}$) in RMSE.
- **Categorical NAQI Health Severity Assessment**:
  - Exact category accuracy: **77.04%** ($\text{PM}_{2.5}$) and **80.88%** ($\text{PM}_{10}$)
  - Adjacent-category tolerance ($\pm 1$ bin): **$>99.3\%$ accuracy**

---

## Empirical Negative Result: Opportunistic Multimodal Sensing

**Rigorous feasibility audit of 640 unique outdoor images from PM25Vision:**
1. **Spatial Displacement**: 50.6% of images exceed a 5 km radius from the nearest monitor (mean: 92.8 km).
2. **Temporal Mismatch**: Images have only calendar dates (`YYYY-MM-DD`) with zero sub-daily timestamps, despite a **166.72 $\mu\text{g/m}^3$** average diurnal $\text{PM}_{2.5}$ range at collocated stations.
3. **Burst Clustering**: Matched data ($N=265$) contains only **36 independent groups** (40% from 2 single-day events).
4. **Calibration Bias**: Image labels exhibit a **$+70.74\ \mu\text{g/m}^3$** positive offset vs. regulatory monitors.
5. **Controlled Ablation Failure**: Multimodal early fusion degraded out-of-distribution MAE by **12.19 $\mu\text{g/m}^3$** (85.86 vs 73.67 $\mu\text{g/m}^3$, $R^2 = -0.5064$).

---

## Standalone Adapted GBDT Architecture

**Canonical 49-Variable Schema (Zero Trace Gas Reliance):**
- **24 Lagged Particulate Terms**: $\text{PM}_{2.5}(t-k)$ and $\text{PM}_{10}(t-k)$ for $k \in \{0, \dots, 11\}$.
- **14 Multi-Scale Rolling Statistics**: 3h, 6h, 12h, 24h rolling mean, std, min, max, and coarse/fine ratio.
- **4 Cyclical Harmonics**: $\sin/\cos$ of hour-of-day and month-of-year.
- **3 Calendar Terms**: Day of week, day of year, weekend indicator.
- **4 Spatial Metadata**: Latitude, longitude, elevation, target-encoded station ID.
- **Strictly Causal**: All inputs terminate at $\le t$. Completely immune to trace gas outages.

---

## Severity-Gated Adaptive Ensemble

Combines the long-term representation power of recurrent deep networks with the localized split robustness of LightGBM:

$$\hat{y}(t+1) = \alpha(C_t) \hat{y}_{\text{GBDT}}(t+1) + (1 - \alpha(C_t)) \hat{y}_{\text{LSTM}}(t+1)$$

- $C_t = \max(\text{PM}_{2.5}(t), \text{PM}_{10}(t))$ monitors instantaneous pollution severity.
- During hazardous smog episodes ($C_t > 250\ \mu\text{g/m}^3$), the gate dynamically routes weight to GBDT, avoiding LSTM activation saturation.

---

## Multi-Station Live Portfolio (236,289 Instances Across 15 Stations)

| Model Architecture | $\text{PM}_{2.5}$ MAE | $\text{PM}_{2.5}$ RMSE | $R^2$ | Gain | $\text{PM}_{10}$ MAE | $\text{PM}_{10}$ RMSE | $R^2$ | Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-Hour Persistence | 6.95 | 14.49 | 0.7844 | 0.0% | 17.16 | 38.70 | 0.8089 | 0.0% |
| Adapted LSTM (27 feats) | 6.80 | 13.18 | 0.8215 | +2.1% | 16.48 | 34.65 | 0.8468 | +4.0% |
| Standalone GBDT (49 feats) | 6.67 | 13.17 | 0.8219 | +4.0% | 16.29 | 35.24 | 0.8415 | +5.1% |
| **Unified Severity-Gated** | **6.62** | **13.04** | **0.8254** | **+4.7%** | **16.14** | **34.53** | **0.8479** | **+5.9%** |

- High-impact gains in industrial zones: Visakhapatnam (+7.3%), Delhi Anand Vihar (+6.1%), Ahmedabad (+6.5%), Chennai Manali (+10.2%).

---

## Distribution-Free Split-Conformal Calibration

Finite-sample coverage guarantees without parametric distributional assumptions across **118,145 test instances**:

| Parameter | Nominal Level | Empirical Coverage | Coverage Slack ($\Delta$) | Mean Interval Width |
| :--- | :---: | :---: | :---: | :---: |
| $\text{PM}_{2.5}$ | 80.0% | **92.87%** | +12.87% | 24.12 $\mu\text{g/m}^3$ |
| $\text{PM}_{2.5}$ | 90.0% | **96.76%** | +6.76% | 35.84 $\mu\text{g/m}^3$ |
| $\text{PM}_{2.5}$ | 95.0% | **98.38%** | +3.38% | 48.31 $\mu\text{g/m}^3$ |
| $\text{PM}_{10}$ | 80.0% | **93.66%** | +13.66% | 60.25 $\mu\text{g/m}^3$ |
| $\text{PM}_{10}$ | 90.0% | **97.23%** | +7.23% | 94.27 $\mu\text{g/m}^3$ |
| $\text{PM}_{10}$ | 95.0% | **98.71%** | +3.71% | 131.50 $\mu\text{g/m}^3$ |

Mathematically guarantees $\mathbb{P}(y_{t+1} \in \mathcal{C}_{1-\alpha}) \ge 1-\alpha$ under heteroskedastic smog shocks.

---

## Rigorous Statistical Robustness

- **Stationary Block Bootstrap** ($B=5,000$ iterations, block length $L=24\text{h}$):
  - Pooled $\text{PM}_{2.5}$: Mean diff $-0.28\ \mu\text{g/m}^3$, 95% CI $[-0.30, -0.25]$, Wilcoxon $p = 4.98 \times 10^{-12}$.
  - Pooled $\text{PM}_{10}$: Mean diff $-0.87\ \mu\text{g/m}^3$, 95% CI $[-0.94, -0.79]$, Wilcoxon $p = 7.57 \times 10^{-24}$.
- **Multiple Testing Corrections**:
  - Passes Bonferroni FWER ($p < 0.0028$) and Benjamini-Hochberg FDR ($q < 0.05$).
- **Hazardous Severity Regimes**:
  - In *Very Poor* air ($N=4,847$), GBDT reduces MAE by **2.48 $\mu\text{g/m}^3$** (29.14 vs 31.62 $\mu\text{g/m}^3$, $p = 3.71 \times 10^{-25}$).

---

## Spatial-Temporal Advection: Sea Breeze vs Inland Emissions

Evaluated across 3 intra-urban corridors ($<25\text{ km}$):
1. **Mumbai Marine Corridor (Bandra $\leftrightarrow$ Colaba, 17.20 km)**:
   - Peak cross-correlation occurs at an **advective lag of $\tau = +1\text{ hour}$** ($r = 0.5385$ vs $r = 0.5283$ at $\tau = 0$).
   - Spatial neighbor lags improve next-hour forecast MAE from 6.10 to **6.05 $\mu\text{g/m}^3$** (**$+0.86\%$ gain** over single GBDT, $+4.16\%$ over persistence).
2. **Inland Corridors (Delhi & Kolkata)**:
   - Synchronous correlation peaks at $\tau = 0$ ($r = 0.7107$ in Delhi, $r = 0.9078$ in Kolkata).
   - In-situ microclimate emissions dominate; unweighted neighbor concatenation degrades performance ($-2.03\%$ in Delhi, $-3.00\%$ in Kolkata), requiring physics-informed wind gating.

---

## Production Deployment & Interactive Platform

- **Operational Web Dashboard (`app.py`)**:
  - 8 modules: Overview, Live Telemetry, Mode A (Individual), Mode B (Hybrid), Comparison, 3D Map, Performance, and System Health.
  - Passes 100% of end-to-end smoke tests (`tests/test_eight_sections_smoke.py`).
- **Autonomous Hourly Collector Daemon**:
  - Background worker querying OpenAQ v3 with exponential backoff, composite key deduplication, and immutable retraining lock.
- **Production Containerization**:
  - `Dockerfile` & `docker-compose.yml` for 1-click cloud / server deployment.

---

## Conclusion & Research Impact

1. **Operational Robustness**: Proved that a 49-variable standalone GBDT eliminates reliance on failing trace gas telemetry while outperforming persistence by up to 5.92%.
2. **Mathematical Safety**: Delivered distribution-free split-conformal calibration with guaranteed finite-sample coverage ($\ge 92.8\%$).
3. **Physical Clarity on Advection**: Isolated coastal advective gains from inland local emission dominance.
4. **Methodological Honesty**: Established the definitive empirical negative result on opportunistic environmental imagery, defining strict prerequisites for future synchronous camera networks.

**Full Replication Suite**: `python scripts/reproduce_all_results.py` (100% verified)
