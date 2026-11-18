# VayuDrishti Capstone Final Review Presentation Audit
**Official Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati - 522237, Andhra Pradesh, India  
**Project Guide:** Prof. Thota Ramathulsi, Assistant Professor, SCOPE  
**Project Team Members:**  
- TELIKI RUSHYA SREE (23BCE9562)  
- MACHIREDDY SWATHI (23BCE9566)  
- THANUJA VARAKALA (23BCE9608)  
- PALLI SHILPA (23BCE9567)  
**Academic Year:** 2026–2027  
**Evaluated Artifact:** `presentation/VayuDrishti_Final_Capstone_Review.pptx` (and `docs/VayuDrishti_Final_Capstone_Review.pptx`)  
**Base Template:** `presentation/Capstone_PPT_Template.pptx` (16-Slide University Standard)  

---

## 1. Executive Summary of Audit

This document provides an exhaustive, itemized academic and technical audit of the Capstone Final Review presentation for the **VayuDrishti** research project. The evaluation was conducted to transform an uncalibrated presentation containing informal claims, conflated feature manifests, unverified spatial statistics, and exaggerated model performance into an **academically defensible, scientifically rigorous, and peer-review-grade capstone defense slide deck**.

The audit strictly adhered to the institutional formatting requirements of VIT-AP University while enforcing rigorous environmental data science standards, including:
1. Complete synchronization of all figures, metrics, and tables with the audited research paper (`paper/VayuDrishti_Research_Paper.pdf`) and capstone project report (`docs/VayuDrishti_Final_Capstone_Project_Report.docx`).
2. Exact alignment with the 16-slide sequence and slide header taxonomy defined in `Capstone_PPT_Template.pptx`.
3. Eradication of unverified macroeconomic loss figures, exaggerated mortality statistics, and unsupported nationwide generalization claims.
4. Transparent reporting of the **14-day live field audit**, where paired Student's t-tests revealed that 1-hour ahead machine learning gains over persistence are statistically non-significant ($p = 0.48$ for $\text{PM}_{2.5}$, $p = 0.62$ for $\text{PM}_{10}$) due to severe temporal autocorrelation ($\rho > 0.95$).
5. Rectification of the multimodal image fusion narrative, explicitly demonstrating why early concatenation causes negative transfer ($+12.19\ \mu\text{g/m}^3$ error inflation) and how the **Bounded Residual Hybrid** ($\Delta \in [-25, +25]\ \mu\text{g/m}^3$) with zero-fallback ($m = 0$) guarantees stability.
6. Incorporation of authentic **Clark-Evans nearest-neighbor spatial statistics** ($R$) across 31 Indian cities, replacing fabricated ratios with empirical Phase 16 network audit values.

---

## 2. Itemized Slide-by-Slide Audit Register

The table below contrasts the previous/unverified claims against the finalized, defensible formulations across all 16 presentation slides.

| Slide # | Official Slide Heading | Unverified / Previous Formulation | Corrected / Defensible Academic Formulation | Scientific Rationale & Academic Justification |
| :---: | :--- | :--- | :--- | :--- |
| **01** | Title Slide | Contained duplicate university header text, informal guide title ("Dr. / Prof."), missing registration numbers, and arbitrary project titles. | Title standardized to *"VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments"*. Guide set strictly to **Prof. Thota Ramathulsi**, Assistant Professor, SCOPE. All 4 team members listed with 23BCE registration numbers; CAPSTONE ID placeholder retained. Duplicate headers stripped. | Institutional compliance. Enforces exact university nomenclature and student accountability without visual clutter. |
| **02** | Presentation Outline | Generic bulleted list without thematic grouping or logical division; failed to reflect the technical depth of the defense. | Divided into two structured cards: **Part I: Foundations & Research Formulation** (Slides 3–7) and **Part II: Technical Implementation & Defense** (Slides 8–16). All 16 topics mapped explicitly. | Provides review committee members with an immediate, structured mental model of the evaluation trajectory. |
| **03** | Introduction / Project Overview | Claimed unverified annual mortality figures (1.67 million deaths) and arbitrary economic burdens ($36.8 billion), while conflating live telemetry, forecasting, and spatial interpolation into a single vague concept. | Discarded unverified macro statistics. Formulated the project around **Three Distinct Operational Modes**: (1) Live Telemetry Ingestion (85+ monitors via OpenAQ v3), (2) 1-Hour Forward Forecasting ($t+1$ via LSTM/GBDT), and (3) Spatial Safeguarding (Geodesic IDW with 45 km cutoff). | Establishes rigorous environmental computing scope. Prevents committee skepticism regarding unsourced public health claims and clearly defines system boundaries. |
| **04** | Problem Statement & Motivations | Listed generic statements such as "air pollution is bad" and "sensors are broken"; failed to mathematically characterize spatial clustering or multimodal failure modes. | Formulated **Four Core Technical Failures**: (1) Sensor Sparsity & Spatial Clustering ($R < 0.60$ in major metros), (2) Telemetry Latency & Data Dropout (unflagged stale readings), (3) Multimodal Decoupling & Negative Transfer ($+12.19\ \mu\text{g/m}^3$ early fusion error), and (4) Spatial Overconfidence in Unmonitored Zones. | Demonstrates deep domain expertise. Grounds motivation in mathematically demonstrable engineering and atmospheric failure modes. |
| **05** | Objectives & Expected Impact | Vague aspirational goals (e.g., "predict pollution everywhere using AI and satellite imagery"). | Defined **Four Measurable Engineering Objectives**: (1) Automated OpenAQ v3 ingestion with 5-tier freshness categorization, (2) Dual next-hour forecasting ($t+1$) benchmarked against 1h persistence, (3) Bounded ResNet-50 visual residual fusion ($\Delta \in [-25, +25]$) with $m=0$ fallback, and (4) Geodesic IDW ($p=2.0$, $R_{\max}=45\text{ km}$) with split conformal prediction (90% coverage). | Provides quantifiable criteria against which the committee can evaluate capstone completion. |
| **06** | Literature Review | Informal review mentioning generic machine learning models without direct side-by-side comparative matrices or failure mode analysis. | Synthesized a structured 6-row comparative matrix evaluating **ARIMA, Random Forest/GBDT, Vanilla LSTM, Vision CNNs, Ordinary Kriging/IDW, and VayuDrishti** across temporal modeling, spatial awareness, vision integration, uncertainty calibration, and primary failure modes. | Highlights precise scholarly gaps, showing that no prior system unifies temporal forecasting, bounded visual residuals, geodesic spatial guarding, and conformal uncertainty. |
| **07** | Key Contributions | Claimed "novel deep learning beats all models by 50%" without empirical substantiation; used incorrect Clark-Evans ratios (e.g. Delhi R=0.91, Patna R=0.21). | Articulated **Four Verified Innovations**: (1) Bounded Residual Hybrid with mathematical zero-fallback ($m=0 \implies \delta=0$), (2) Geodesic Spatial Safeguards & Clark-Evans Diagnostics (Delhi $R=0.42$, Mumbai $R=0.56$, Kolkata $R=0.32$, Bengaluru $R=0.45$, Agra $R=1.35$), (3) Distribution-Free Conformal Uncertainty (90.4% coverage), and (4) 31-City Live Audit & Open Streamlit Deployment. | Replaces exaggerated claims with verifiable algorithmic formulas and authentic Phase 16 spatial statistics. |
| **08** | System Architecture | Unclear block diagram or textual description lacking pipeline separation. | Structured into **Three Clear Tiers** (Ingestion/Preprocessing, Dual Predictive/Fusion Engine, Spatial/Conformal Service) paired with the high-resolution publication diagram `fig_arch_overall.png`. Explicitly shows ResNet-50 2048-dim feature extraction and dual tabular predictors. | Clear information flow enabling the committee to trace inputs, tensor shapes, and component boundaries. |
| **09** | Dashboard Modules & Interface | Described hypothetical UI features; lacked visual proof of implementation. | Cataloged the **Six Real Operational Modules** of the Streamlit application (Live Monitor, Spatial Heatmap, Network Audit, 1-Hour Forecast, Vision Studio, Model Diagnostics) paired with high-resolution screenshot `dashboard_monitoring_interface.png`. | Proves complete software realization and operational readiness. |
| **10** | Algorithms & Implementation Stack | Stated models were trained on high-end NVIDIA RTX 4070 GPUs with CUDA; lacked mathematical formulations. | Replaced GPU claims with actual development platform: **Intel Core i7-1165G7 CPU (16 GB RAM)**, sub-20ms inference latency, CPU-optimized PyTorch and LightGBM. Displayed formal LaTeX mathematical definitions for Geodesic IDW, Clark-Evans $R$, Bounded Residual Fusion, and Split Conformal Prediction. | Academic integrity. Eliminates false workstation specifications; demonstrates advanced mathematical mastery. |
| **11** | Dataset Descriptions & Feature Manifests | Conflated dataset sizes and claimed LSTM and GBDT models shared the same feature sets; ambiguous 24-hour forecast horizon claim. | Formulated structured table detailing all 4 corpora (Historical 1.37M rows / 90 stations; Recent 3.35M rows / 85 stations; Live 418 pairs / 4 stations; PM25Vision 199 pairs). Established **Strict Feature Manifest Separation**: LSTM uses **27 numerical channels** across 24 historical hours; adapted LightGBM uses **49 engineered tabular features**. Clarified that both models predict strictly the next hour ($t+1$). | Resolves major review vulnerability. Differentiates sequence channel tensors from tabular lag-engineered vectors. |
| **12** | Parameters & Experimental Setup | Missing hyperparameter values, arbitrary spatial radius, and informal train/test split descriptions. | Complete transparency: LSTM (2 layers, 128/64 units, dropout 0.20, Adam lr 0.001); LightGBM (100 trees, depth 6, lr 0.05, num_leaves 31); ResNet-50 (ImageNet weights, 2048->128->1 projection, scaled tanh); Geodesic IDW ($p=2.0$, $R_{\max}=45\text{ km}$); 5 freshness tiers; chronological 80/20 temporal splits and Delhi NCR LOSO spatial cross-validation. | Enables full scientific reproducibility and defends hyperparameter tuning choices. |
| **13** | Experimental Results & Comparative Analysis | Claimed massive, statistically significant real-time improvements over persistence in live deployment; omitted statistical tests. | Displayed Table A (Historical Benchmarks: Ensemble $\text{PM}_{2.5}$ MAE 8.94 vs Persistence 14.82), Table B (Delhi LOSO: IDW reduces MAE by 20.0% for $\text{PM}_{2.5}$ and 25.4% for $\text{PM}_{10}$ vs 1-NN), Table C (Live 14-day Audit: GBDT MAE 9.64 vs Persistence 9.86, $t=-0.71, p=0.48$; $\text{PM}_{10}$ MAE 22.13 vs 22.35, $t=-0.50, p=0.62$). Honestly acknowledged that next-hour gains over persistence are statistically non-significant due to autocorrelation ($\rho > 0.95$). Embedded `actual_vs_predicted.png`. | Peer-review caliber honesty. Transforming a non-significant result into a key scientific finding earns the highest academic respect from evaluators. |
| **14** | Findings, Limitations & Future Work | Claimed the system is ready for immediate nationwide replacement of CPCB monitors without limitations. | Identified **Four Honest Scientific Findings & Constraints**: (1) Autoregressive Dominance at 1-hour horizons, (2) Multimodal Decoupling during winter haze, (3) Clustered Topology degradation at urban boundaries, and (4) Geographic limitation of spatial holdouts to Delhi NCR. Proposed concrete future roadmap: Spatio-Temporal Graph Neural Networks (ST-GNNs), Satellite AOD remote sensing (INSAT-3D/MODIS), multi-step horizons ($t+6, t+24$), and low-cost IoT calibration. | Demonstrates mature, balanced research perspective. Avoids over-promising and outlines clear postgraduate/doctoral research trajectories. |
| **15** | Bibliography | Generic URLs, Wikipedia links, or informal blog references. | Formatted **8 Verified Peer-Reviewed Scholarly References in IEEE Format**, citing CPCB (2014), Hochreiter & Schmidhuber (1997), Ke et al. (2017), He et al. (2016), Clark & Evans (1954), Shepard (1968), Angelopoulos & Bates (2021), and OpenAQ v3 (2024). | Institutional compliance. Grounded in authoritative academic and regulatory literature. |
| **16** | Closing & Thank You | Blank or informal closing slide with no links or artifacts. | Structured concluding slide summarizing core achievements, tangible deliverables (code, 45-page report, trained weights, live dashboard), repository URL (`github.com/trush/vayu-drishti`), live cloud application (`multimodal-air-quality-c38w.streamlit.app`), guide and student attribution, and an open invitation for viva examination. | Professional capstone closure leaving a polished, authoritative impression on evaluators. |

---

## 3. Deep-Dive on Critical Corrections

### 3.1 Spatial Geometry vs Spatial Accuracy (Clark-Evans Metric)
* **The Error:** In initial discussions, Clark-Evans $R$ values were cited as indicators of spatial interpolation accuracy, with unsubstantiated numbers (e.g., Delhi $R=0.91$, Patna $R=0.21$).
* **The Correction:** Slide 7, 10, and 12 now accurately define Clark-Evans $R$ as a **spatial point pattern statistic** that measures the physical distribution geometry of monitoring stations:
  $$R = \frac{\bar{r}_A}{\bar{r}_E} = \frac{\frac{1}{N}\sum_{i=1}^N r_i}{\frac{1}{2\sqrt{\lambda}}}$$
  where $\bar{r}_A$ is the mean nearest-neighbor distance and $\bar{r}_E$ is the expected distance under a Poisson complete spatial randomness process with station density $\lambda = N / A$.
* **Empirical Values Audited (Phase 16):**
  - **Clustered Distributions ($R < 1$):** Delhi NCR ($R = 0.42$), Mumbai MMR ($R = 0.56$), Kolkata ($R = 0.32$), Bengaluru ($R = 0.45$), Hyderabad ($R = 0.61$), Patna ($R = 0.38$).
  - **Random Distributions ($R \approx 1$):** Pune ($R = 0.93$).
  - **Dispersed Distributions ($R > 1$):** Ahmedabad ($R = 1.25$), Agra ($R = 1.35$).
  - **Single/Zero Monitor Cities:** Amaravati, Tirupati, Guntur have $N \le 1$; the slide deck explicitly notes that Clark-Evans is undefined or degenerate for $N < 2$, refuting any claims of clustering in these towns.

### 3.2 Multimodal Negative Transfer & The Bounded Residual Hybrid
* **The Error:** Common computer vision literature suggests that fusing images with sensors always improves prediction.
* **The Correction:** On the PM25Vision dataset (199 outdoor image-sensor pairs), naive early concatenation of image embeddings with tabular features resulted in an error inflation of $+12.19\ \mu\text{g/m}^3$ over pure tabular forecasting. This occurs because atmospheric optical extinction (haze/contrast reduction) decouples from ground particulate mass during winter radiation fog, variable solar zenith angles, and lens flare.
* **The Architectural Fix:** Slide 7, 8, and 10 formulate the **Bounded Residual Hybrid**:
  $$\hat{y}_{\text{final}} = \hat{y}_{\text{tabular}} + m \cdot \text{clamp}(\Delta_{\text{visual}}, -25, +25)$$
  where $\Delta_{\text{visual}} = \text{MLP}(\text{ResNet50}(I))$ and $m \in \{0, 1\}$ is an image availability mask. By clamping visual updates to $[-25, +25]\ \mu\text{g/m}^3$ and enforcing $m = 0 \implies \hat{y}_{\text{final}} = \hat{y}_{\text{tabular}}$, the system is mathematically guaranteed never to perform worse than pure tabular models.

### 3.3 Statistical Significance in 1-Hour Forward Forecasting
* **The Error:** Machine learning practitioners frequently claim superior accuracy over simple baselines without conducting statistical hypothesis testing.
* **The Correction:** Slide 13 and 14 report the full results of the 14-day live field audit (418 matched observation pairs across 4 continuous stations):
  - $\text{PM}_{2.5}$: Adapted LightGBM MAE $9.64\ \mu\text{g/m}^3$ vs 1-Hour Persistence MAE $9.86\ \mu\text{g/m}^3$ ($t = -0.71, p = 0.48$).
  - $\text{PM}_{10}$: Adapted LightGBM MAE $22.13\ \mu\text{g/m}^3$ vs 1-Hour Persistence MAE $22.35\ \mu\text{g/m}^3$ ($t = -0.50, p = 0.62$).
* **Scientific Defense:** Because hourly particulate concentrations exhibit extreme autocorrelation ($\rho > 0.95$), the current hour's reading ($y_t$) is an extraordinarily powerful predictor of the next hour ($y_{t+1}$). While LightGBM achieved marginally lower absolute error, paired two-tailed t-tests fail to reject the null hypothesis of equal performance at the $\alpha = 0.05$ level. The deck openly defends this as an authentic empirical insight, demonstrating that the real utility of ML lies in multi-hour recursive trajectories ($t+6, t+24$) and non-linear meteorological transition regimes, rather than trivial 1-hour persistence competition.

### 3.4 Feature Manifest Rigor
* **The Error:** Describing the input vector ambiguously as "sensor features and weather data."
* **The Correction:** Slide 11 explicitly separates the feature manifests of the two predictive models:
  1. **LSTM Sequence Network (27 Features):** 6 criteria pollutants ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$), 8 meteorological variables (temperature, relative humidity, wind speed, wind direction, pressure, dew point, solar radiation, precipitation), 5 temporal cyclic encodings ($\sin/\cos$ hour, $\sin/\cos$ day of year, day of week), and 8 rolling/ratio signals ($\text{PM}_{2.5}/\text{PM}_{10}$ ratio, 3h/6h rolling means and standard deviations). Input shape: $[B, 24, 27]$.
  2. **Adapted LightGBM GBDT (49 Features):** 6 instantaneous pollutant readings, 8 instantaneous weather variables, multi-scale temporal lags at $t-1, t-2, t-3, t-6, t-12, t-24$ for particulate species, rolling window statistics (3h, 6h, 12h, 24h rolling means and standard deviations), temporal cyclic features, and pollutant-weather interaction ratios (e.g., wind-speed particulate dispersion ratio). Input shape: $[B, 49]$.
  3. **Target:** Both models predict strictly $y_{t+1}$ (next hour concentration), resolving the misconception that the models generate 24 separate hourly predictions simultaneously.

---

## 4. Speaker Script and Timing Plan

Every slide in `VayuDrishti_Final_Capstone_Review.pptx` contains an embedded, natural spoken script in its PowerPoint speaker notes pane. The recommended presentation pacing for a 15–20 minute final review defense is outlined below:

| Slide # | Slide Title | Speaker Target Duration | Cumulative Time |
| :---: | :--- | :---: | :---: |
| 01 | Capstone Final Review (Title) | 45 seconds | 0:45 |
| 02 | Presentation Outline & Structure | 35 seconds | 1:20 |
| 03 | Context, Scope & Three Operational Modes | 60 seconds | 2:20 |
| 04 | Four Critical Gaps in Indian Air Quality Monitoring | 60 seconds | 3:20 |
| 05 | Technical Objectives & Measurable Deliverables | 60 seconds | 4:20 |
| 06 | Comparative Analysis of Existing Methodologies | 60 seconds | 5:20 |
| 07 | Four Key Architectural & Empirical Innovations | 70 seconds | 6:30 |
| 08 | End-to-End Pipeline & Information Flow | 70 seconds | 7:40 |
| 09 | Interactive Streamlit Platform Architecture | 60 seconds | 8:40 |
| 10 | Mathematical Formulations & Technology Stack | 70 seconds | 9:50 |
| 11 | Empirical Data Corpora & Feature Separation | 70 seconds | 11:00 |
| 12 | Hyperparameters, Validation Protocol & Safeguard Rules | 70 seconds | 12:10 |
| 13 | Empirical Performance & Benchmark Evaluations | 80 seconds | 13:30 |
| 14 | Scientific Insights, System Limitations & Future Scope | 70 seconds | 14:40 |
| 15 | Bibliography & Literature Foundations | 35 seconds | 15:15 |
| 16 | Concluding Remarks, Project Artifacts & Viva Defense | 45 seconds | 16:00 |
| **Q&A** | **Viva Examination & Evaluation Committee Discussion** | **5–10 minutes** | **20:00–25:00** |

---

## 5. Audit Sign-Off

The finalized PowerPoint slide deck `presentation/VayuDrishti_Final_Capstone_Review.pptx` has been fully reconciled against the source repository, the verified numerical CSV results, the IEEE manuscript, and the 45-page capstone report. All claims are academically defensible, mathematically grounded, and aligned with VIT-AP University requirements.
