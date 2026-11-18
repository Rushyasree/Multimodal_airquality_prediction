# VayuDrishti: Capstone Final Review Presentation Guide & Oral Script

**Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**University:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati  
**Degree:** B.Tech Computer Science and Engineering  
**Supervised by:** Prof. Thota Ramathulsi, Department of SCOPE  
**SDP ID:** [Insert SDP ID]  
**Project Team Members:**  
1. TELIKI RUSHYA SREE (23BCE9562)  
2. MACHIREDDY SWATHI (23BCE9566)  
3. THANUJA VARAKALA (23BCE9608)  
4. PALLI SHILPA (23BCE9567)  

---

## Slide 1

### Slide Visual Content

```
Capstone Final Review
VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments
B.Tech Computer Science and Engineering | Capstone Project Final Evaluation
```

```
Supervised By: Prof. Thota Ramathulsi
Department of Software and Systems Engineering, School of Computer Science and Engineering (SCOPE)
```

```
SDP ID: [Insert SDP ID]
Name of the Student(s) with Regd. No.:






Project Team Members (SCOPE, CSE):
• TELIKI RUSHYA SREE (23BCE9562)
• MACHIREDDY SWATHI (23BCE9566)
• THANUJA VARAKALA (23BCE9608)
• PALLI SHILPA (23BCE9567)
```

```
School of Computer Science and Engineering (SCOPE)
VIT-AP University,
Amravati, India
VIT-AP University, Amaravati
Andhra Pradesh – 522237, IndiaAcademic Year: 2025–2026
```

```
1
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Good morning respected panel members and faculty guide Prof. Thota Ramathulsi. We are here to present our B.Tech Capstone Project titled 'VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments'. Our team comprises Teliki Rushya Sree, Machireddy Swathi, Thanuja Varakala, and Palli Shilpa from the School of Computer Science and Engineering.

ANTICIPATED VIVA QUESTIONS & DEFENSE:
Q: What is the primary scope and goal of VayuDrishti?
A: To create a research-grade, scientifically sound system for next-hour (t+1) particulate matter (PM2.5 and PM10) forecasting and urban spatial interpolation across Indian cities, integrating live telemetry, historical logs, and distance-scaled conformal uncertainty bounds without fabricating unsupported claims.

---

## Slide 2

### Slide Visual Content

```
Presentation Outline
```

```
Roadmap of Capstone Final Review
```

```
2
```

```
Part I: Foundations, Problem & Architecture
1. Introduction & Overview: Air pollution context, PM2.5/PM10 health impact, monitoring dilemma.
2. Problem Statement: Spatial sparsity, non-linear dynamics, lack of anticipatory warnings.
3. Objectives & Impact: 7 core research goals, scientific provenance, risk-aware alerts.
4. Background & Related Work: ARIMA/SVR, LSTM/GRU, GBDT, Koschmieder optical models, gaps.
5. Improvements Over Existing: Bounded residual fusion, IDW vs 1-NN, 5-state freshness, Clark-Evans.
6. Proposed Architecture: 5-tier end-to-end data pipeline from multi-source ingestion to UI.
```

```
Part II: Engineering, Empirical Evaluation & Impact
7. Key Dashboard Modules: 6 interactive views (Telemetry, Forecaster, Hybrid, Comparison, Heatmaps).
8. Algorithms & Tech Stack: PyTorch, LightGBM, Haversine IDW, Conformal bounds, Streamlit.
9. Datasets & Feature Pipeline: Historical (1.37M rows), Recent (3.3M rows), Live OpenAQ, 27 features.
10. Parameters & Setup: 24h lookback predicting t+1, IDW p=2.0, R_max=45km, Freshness tiers.
11. Experimental Outcomes: LSTM test (R2>0.93), Recent ensemble, Delhi IDW (+20-25%), Live audit.
12. Summary & Future Scope: Key takeaways, honest scientific limitations, future ST-GNN & satellite work.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Our presentation is systematically divided into two halves. The first half establishes the scientific foundation, the research problem, comparative literature review, and the system architecture. The second half walks through the engineering implementation, the extensive empirical benchmarks across 4.67 million hourly records, the live sensor audits, and honest research limitations.

ANTICIPATED VIVA QUESTIONS:
Q: Why is the presentation divided between historical validation and live evaluation?
A: Because historical cross-validation benchmarks temporal sequence learning over multi-year distributions, whereas live evaluation evaluates operational reliability under streaming latency and missing telemetry.

---

## Slide 3

### Slide Visual Content

```
Introduction
```

```
Project Overview: Urban Air Quality Crisis & VayuDrishti Framework
```

```
3
```

```
The Urban Pollution Crisis
Particulate Hazard: Severe airborne health hazard across Indian metropolitan regions.
Health Impact: PM2.5 and PM10 penetrate deep into bronchi and bloodstream, driving cardiopulmonary mortality.
WHO Guidelines: Indo-Gangetic plain and major metros regularly exceed WHO safety guidelines by 500% to 1000%.
Urgency: Demands continuous, high-resolution temporal tracking.
```

```
The Monitoring Dilemma
CAAQMS Telemetry: Continuous Ambient Air Quality Monitoring Stations provide gold-standard reference readings.
Capital Intensity: Extreme capital cost (~₹1.2–1.5 Crore/station) restricts deployment to selected government sites.
Spatial Sparsity: Only ~85–90 physical monitors across major multi-million citizen agglomerations.
Blindspots: Extensive urban, peri-urban, and residential zones remain complete blindspots.
```

```
The VayuDrishti Framework
Unified Platform: An end-to-end multimodal spatio-temporal AI platform for Indian cities.
Multimodal Core: Combines 24-hour sequence learning, vectorized spatial IDW, and live OpenAQ ingestion.
Actionable Horizon: Provides strictly next-hour (t+1) forecasts with distance-scaled conformal prediction bounds.
Cloud Delivery: Interactive research prototype deployed on Streamlit Cloud.
```

```
Crucial Scientific Distinction: Physical Telemetry vs. Forecast vs. Spatial Estimate
• Direct CAAQMS Telemetry: Ground-truth physical sensor measurements (observed empirical fact).
• AI Sequence Forecast: Inferred next-hour (t+1) predictive trajectory (temporal mathematical forecast).
• Spatial Heatmap Surface: Continuous Inverse Distance Weighted (p=2) interpolation (spatial approximation with uncertainty).
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Air pollution in Indian cities is an acute public health crisis. While the Central Pollution Control Board (CPCB) operates high-grade reference stations, these continuous monitors are capital-intensive and geographically sparse. VayuDrishti bridges this gap through an end-to-end framework combining temporal deep learning and spatial interpolation. Most importantly, we maintain strict scientific provenance: we never represent an interpolated heatmap value or a model prediction as a physical sensor measurement.

ANTICIPATED VIVA QUESTIONS:
Q: Why cannot low-cost sensors simply replace expensive CAAQMS stations?
A: Low-cost sensors suffer from severe relative humidity drift, baseline degradation, and photochemical cross-sensitivity, requiring frequent co-location recalibration against reference CAAQMS stations.

---

## Slide 4

### Slide Visual Content

```
Introduction
```

```
Problem Statement & Research Motivations
```

```
4
```

```
1. Spatial Sparsity & Geographic Blindspots
Asymmetric Coverage: Physical monitors are clustered in government or industrial hubs, leaving suburban corridors unmonitored.
Resolution Deficit: Inter-station distances frequently exceed 15–30 km, hiding localized micro-climate hotspots.
Public Exposure: Residents between stations have no access to ground-truth localized particulate readings.
```

```
2. Extreme Temporal Non-Linearity & Drifts
Diurnal Swings: Particulate concentrations swing drastically across diurnal rush hours and nocturnal boundary-layer trapping.
Episodic Inversions: Winter temperature inversions and post-monsoon crop residue burning trigger severe episodic spikes (>400 µg/m³).
Model Breakdown: Classical linear statistical models (ARIMA) fail to capture abrupt meteorological regime changes.
```

```
3. Absence of Anticipatory Warning Signals
Retrospective Lag: Standard government portals publish retrospective 24-hour rolling averages (historical AQI).
Actionability Void: A 24-hour retrospective average lags behind acute real-time particulate influxes by up to a day.
Proactive Foresight: Public health and traffic management require proactive, calibrated next-hour (t+1) warning signals.
```

```
4. Multimodal Fusion Pitfalls & Spatial Uncertainty
Negative Transfer: Naive early concatenation of outdoor webcam images degrades tabular sequence accuracy (+12.19 µg/m³ error).
Unquantified Risk: Standard spatial interpolations (e.g. 1-NN) present false confidence without distance-based bounds.
Research Gap: Need for bounded residual fusion architectures and conformal prediction uncertainty bands.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Here we articulate the core research challenges. Traditional air-quality reporting is fundamentally retrospective—it tells citizens what they breathed yesterday, not what they will face during the next rush hour. Furthermore, spatial sparsity creates massive geographic blindspots. In machine learning, naively fusing camera images with tabular sequences often results in negative transfer. VayuDrishti addresses these challenges through a bounded residual hybrid architecture and distance-scaled conformal prediction.

ANTICIPATED VIVA QUESTIONS:
Q: Why does early multimodal fusion fail on air quality time-series?
A: High-capacity CNN backbones can overfit to lighting, cloud cover, and camera orientation artifacts, injecting high-frequency variance that degrades the smooth, physically constrained numerical sequence learned by recurrent networks.

---

## Slide 5

### Slide Visual Content

```
Introduction
```

```
Research Objectives & Expected Societal Impact
```

```
5
```

```
Core Research & Engineering Objectives
Obj 1: Next-Hour Forecasting: Formulate next-hour (t+1) PM2.5 and PM10 predictive models from 24h lag sequences.
Obj 2: Architectural Exploration: Benchmark Bi-LSTM, LSTM, LightGBM, and Persistence across multi-year data.
Obj 3: Multimodal Residual Fusion: Design a bounded residual architecture with guaranteed vision fallback (m=0).
Obj 4: Recent Multi-Year Adaptation: Adapt models using 2020–2025 CAAQMS logs (3.3M records) reflecting post-COVID shifts.
Obj 5: Vectorized Spatial Interpolation: Implement continuous Inverse Distance Weighting (IDW) to estimate inter-station surfaces.
Obj 6: Comparative Baselines: Empirically evaluate spatial IDW against nearest-neighbor (1-NN) holdouts.
Obj 7: Governance & Conformal Bounds: Deploy 5-state telemetry freshness governance and distance-scaled conformal bounds.
```

```
Expected Academic & Public Health Impact
Scientific Rigor: Strict provenance standards distinguish direct physical sensor measurement from mathematical estimates, preventing misinformation.
Preventive Health: Provides commuters, schools, and vulnerable respiratory patients with proactive next-hour risk awareness.
Infrastructure Planning: Identifies severe urban sensor clustering (Clark-Evans R < 0.5) to guide municipal sensor placement.
Open Science: An open-source, fully reproducible research artifact featuring 31 Indian cities deployed online.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
We formulated seven specific, measurable research objectives spanning temporal modeling, multimodal fusion, spatial interpolation, and data governance. Rather than promising broad nationwide predictions, our focus is rigorous, scientifically validated next-hour (t+1) forecasting and calibrated spatial surfaces. The expected impact is both academic—through open-science reproducibility—and societal, empowering citizens with proactive health foresight.

ANTICIPATED VIVA QUESTIONS:
Q: Why is the forecast horizon specifically next-hour (t+1) instead of 24 hours ahead?
A: High-frequency particulate dispersion in urban canyons is governed by turbulent atmospheric diffusion and immediate traffic pulses. While 24h ahead forecasts have high uncertainty, t+1 provides high operational accuracy for actionable decisions.

---

## Slide 6

### Slide Visual Content

```
Background & Related Work
```

```
Related Work, Methodological Approaches & Known Limitations
```

```
6
```

```
Statistical Models(ARIMA / SARIMAX / SVR)
Strengths: Mathematically tractable; effective on smooth linear diurnal periodicity.
Limitations: Fails during non-linear meteorological shocks, inversion trapping, and high-dimensional multi-sensor lag inputs.
```

```
Deep Recurrent Nets(LSTM / Bi-LSTM / GRU)
Strengths: Models long-range sequential dependencies across 24h lagged multivariate time steps.
Limitations: High compute footprint; prone to distribution drift across seasonal transitions; requires extensive hyperparameter tuning.
```

```
Gradient Boosted Trees(LightGBM / XGBoost)
Strengths: Fast histogram split finding; natively handles missing tabular features; superior tabular efficiency.
Limitations: Cannot natively learn sequential temporal topologies without explicit manual lag feature engineering.
```

```
Computer Vision & 1-NN(Koschmieder / ResNet / 1-NN)
Strengths: Koschmieder optical extinction extracts haze; 1-NN provides simple spatial assignment.
Limitations: Vision sensitive to fog, camera angle, and nighttime; 1-NN produces artificial discontinuous Voronoi step boundaries.
```

```
Summary of Critical Literature Gaps Addressed by VayuDrishti
• Ignored Telemetry Latency: Existing platforms treat stale 12h-old readings as live, corrupting spatial surfaces.
• Naive Early Multimodal Fusion: Concatenating raw images with tabular sequences inflates MAE by +12.19 µg/m³.
• Omission of Spatial Conformal Bounds: Most models provide point predictions without distance-scaled confidence bands.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
In our literature survey, we analyzed classical statistical models, recurrent neural networks, gradient boosted trees, and computer vision approaches. We identified three major gaps in the literature: first, existing systems rarely account for sensor latency and treat stale data as live. Second, naive multimodal fusion frequently causes negative transfer. Third, existing spatial models output point estimates without conformal uncertainty bounds. VayuDrishti addresses all three.

ANTICIPATED VIVA QUESTIONS:
Q: Why did LightGBM perform so competitively against deep LSTM models?
A: LightGBM's histogram-based binning efficiently splits non-linear decision boundaries on rolling statistical aggregations (mean, std, min, max), making it exceptionally robust against tabular sensor noise and missing values.

---

## Slide 7

### Slide Visual Content

```
Background & Related Work
```

```
Methodological Innovations & Guardrails Over Existing Solutions
```

```
7
```

```
1. Bounded Residual Multimodal Architecture
Catastrophic Fix: Solves naive multimodal degradation (+12.19 µg/m³ error inflation in joint concatenation models).
Residual Bounding: Constrains optical correction to a bounded residual scale (Δ ∈ [-M, +M]), anchoring to numerical baseline.
Automatic Fallback: Guarantees graceful fallback (m=0) when outdoor camera imagery is noisy, cloudy, or unavailable.
```

```
2. Vectorized IDW vs. 1-NN Spatial Baseline
Continuous Blending: Replaces discrete nearest-station step functions with continuous Inverse Distance Weighting (p=2.0).
Empirical Proof: Reduces Delhi spatial holdout error by 20.0% (PM2.5: 24.07 vs 30.06 µg/m³) and 25.4% (PM10: 52.76 vs 70.68 µg/m³).
Boundary Safeguards: Enforces minimum 3-station support and 45 km radius cutoffs to prevent distant spurious extrapolation.
```

```
3. Operational 5-State Telemetry Governance
Age Audit: Categorizes all stations into Fresh (≤2h), Delayed (2–6h), Stale (6–24h), Offline (>24h), or Unknown.
Surface Purity: Prevents inactive or stale monitors from corrupting real-time spatial interpolation surfaces.
Transparency: Explicitly communicates telemetry latency directly to the end-user on dashboard cards.
```

```
4. 31-City Clark-Evans Spatial Geometry Audit
Spatial Rigor: Applies spatial point pattern analysis using the Clark-Evans nearest-neighbor statistic (R).
Clustering Reality: Discloses severe clustering in cities like Patna (R=0.21) and Ranchi (R=0.34) versus distributed Delhi (R=0.91).
Interpolation Guard: Guards against unconstrained spatial interpolation in cities lacking geometric dispersion.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Slide 7 showcases the core innovations of VayuDrishti. To fix multimodal degradation, we developed a bounded residual hybrid where visual features can only refine predictions within a constrained residual envelope [-M, +M], falling back to pure tabular prediction when imagery is uninformative. In spatial estimation, our vectorized IDW demonstrably outperforms 1-NN by 20% to 25%. Furthermore, our 5-state telemetry freshness audit and 31-city Clark-Evans analysis prevent misleading spatial surfaces.

ANTICIPATED VIVA QUESTIONS:
Q: What is the Clark-Evans R statistic and why does it matter here?
A: Clark-Evans R measures the ratio of observed mean nearest-neighbor distance to expected distance in a random Poisson process. R < 1 indicates clustering, R = 1 is random, and R > 1 is dispersed. A low R indicates monitors are clumped together, leaving vast urban areas unmonitored.

---

## Slide 8

### Slide Visual Content

```
Proposed Solution
```

```
End-to-End System Architecture & Data Pipeline
```

```
8
```

```
5-Tier Architectural Flow
Tier 1: Data Ingestion Layer Ingests 1.37M historical CPCB records, 3.3M recent records, live OpenAQ REST telemetry, and PM25Vision images.
Tier 2: Preprocessing & Features Rolling 24-hr sequences, cyclical hour/month sin-cos encodings, min-max scaling, and Haversine spatial indexing.
Tier 3: Predictive & Fusion Core 2-layer LSTM temporal network, LightGBM tree regressor, and Bounded Residual ResNet-18 visual corrector.
Tier 4: Spatial & Uncertainty Core Vectorized IDW surface interpolator with minimum 3-station support and distance-scaled conformal prediction bounds.
Tier 5: Operational UI Delivery Streamlit Cloud interactive dashboard exposing 6 modular views with sub-second responsive updates.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
This diagram illustrates the five distinct tiers of VayuDrishti. In Tier 1, heterogeneous data sources—including historical archives, recent CAAQMS logs, live OpenAQ REST streams, and outdoor images—are ingested. Tier 2 performs sliding-window sequencing and cyclical feature engineering. Tier 3 houses our parallel predictive engines: LSTM, LightGBM, and the residual vision branch. Tier 4 applies vectorized IDW and conformal calibration, feeding Tier 5's Streamlit Cloud dashboard.

ANTICIPATED VIVA QUESTIONS:
Q: How do you handle missing values in the live telemetry stream?
A: For short gaps (≤3 hours), we apply forward-fill and linear interpolation. For extended dropouts (>3 hours), the station is flagged as Delayed or Stale and excluded from the spatial interpolation kernel.

---

## Slide 9

### Slide Visual Content

```
Proposed Solution
```

```
Key Functional Components & Dashboard Modules
```

```
9
```

```
Six Operational Research Modules
1. Home Dashboard: Real-time atmospheric summary, active city selector across 31 Indian cities, and operational health telemetry.
2. Station Telemetry: Live CAAQMS telemetry (PM2.5, PM10, NO2, SO2, CO, O3) with explicit freshness badges and 24h historical logs.
3. AI Sequence Forecaster: 24-hour temporal sliding window generating deterministic next-hour (t+1) forecasts with feature attribution.
4. Multimodal Hybrid: Experimental bounded residual fusion demonstration comparing numerical baseline vs optical haze refinement.
5. Model Comparison: Side-by-side empirical performance metrics (MAE, RMSE, R²) evaluating LSTM, LightGBM, and Persistence.
6. Spatial Heatmap: High-resolution 2D continuous urban particulate grid with live station overlays and uncertainty toggles.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Here we show the six operational modules of the VayuDrishti dashboard, as deployed live on Streamlit Cloud. The interface allows faculty and users to switch seamlessly between direct station telemetry, next-hour sequence forecasting, empirical model comparison, and 2D spatial heatmaps. Every page is built with strict provenance safeguards—ensuring that users are always clear whether they are viewing raw sensor data, an AI forecast, or a spatial interpolation.

ANTICIPATED VIVA QUESTIONS:
Q: Why are these six views separated rather than merged into a single dashboard screen?
A: To maintain cognitive clarity and scientific distinction. Placing spatial approximations and direct sensor observations on separate tabs prevents users from confusing interpolated estimates with measured ground truth.

---

## Slide 10

### Slide Visual Content

```
Proposed Solution
```

```
Algorithms, Mathematical Formulations & Technology Stack
```

```
10
```

```
Core Algorithmic Foundations
Long Short-Term Memory (LSTM): 2-layer recurrent architecture with 64 hidden units and input/forget/output gates modeling long-range lag dependencies.
LightGBM Regressor: Fast histogram-based gradient boosted decision trees for sub-millisecond next-hour inference over 27 lag features.
Inverse Distance Weighting (IDW): Vectorized 2D spatial interpolation: ẑ(x) = (∑ d_i⁻² · z_i) / (∑ d_i⁻²), where d_i is Haversine great-circle distance.
Conformal Uncertainty Bands: Generates distribution-free prediction intervals dynamically scaled by distance to nearest monitor: ŷ ± q_(1-α)(1 + γ·d_min).
Clark-Evans Point Pattern R: Measures spatial aggregation: R = r̄_obs / r̄_exp to detect monitoring network clustering across 31 cities.
```

```
Technologies, Libraries & Tooling Stack
Deep Learning & ML: PyTorch 2.x, Torchvision, Scikit-learn, LightGBM.
Scientific Computing & GIS: NumPy, Pandas, SciPy (Spatial k-d tree & Clark-Evans R calculation).
Visualization Frameworks: Plotly Interactive Suite, Folium GIS Mapping, Matplotlib, Seaborn.
Application & Ingestion: Streamlit Cloud, OpenAQ v2/v3 REST APIs, Python 3.10+ Runtime.
DevOps & Source Control: Git, GitHub CI/CD, Automated Verification Test Suite.
Research Documentation: LaTeX (IEEE Transactions style) & Pandoc automated pipeline.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Slide 10 details our mathematical formulations and software stack. The temporal engine utilizes an LSTM network paired with a LightGBM regressor. Spatial interpolation is computed via vectorized IDW using Haversine distances. Uncertainty is quantified using split conformal prediction, where the interval width expands as distance from the nearest active sensor increases. The entire software stack is built in Python 3.10 using PyTorch, LightGBM, and Streamlit.

ANTICIPATED VIVA QUESTIONS:
Q: Why use Haversine distance rather than Euclidean distance for IDW?
A: Euclidean distance on latitude and longitude coordinates assumes a planar surface, introducing significant geometric distortion at Indian latitudes (~8°N to 32°N). Haversine computes true great-circle surface distance across the spherical earth.

---

## Slide 11

### Slide Visual Content

```
Implementation and Results
```

```
Dataset Descriptions & Input-Output Pipeline Specifications
```

```
11
```

```
Historical CPCB Dataset
Volume: 1,373,439 hourly records.
Stations: 90 CPCB CAAQMS stations.
Time Horizon: 2015 to July 2020.
Geographic Reach: 24 Indian cities.
Role: Establishes long-term baseline temporal sequence representations.
```

```
Recent CAAQMS Dataset
Volume: 3,300,000+ cleaned records.
Stations: 85 mapped stations.
Time Horizon: July 2020 to September 2025.
Target Metros: Delhi, Mumbai, Kolkata, Bengaluru.
Role: Captures post-COVID shifts in industrial & vehicular baselines.
```

```
Live & Experimental Vision
Live Telemetry: OpenAQ REST API live stream.
Live Audit: 4 operational stations; 418 matched prediction pairs.
Vision Dataset: PM25Vision ground-level photos (1,235 samples).
Role: Rigorous live audit + bounded residual vision branch.
```

```
Input Feature Vector (27 Features) & Strict Next-Hour Target (t+1)
• Input Features (27): 7 Pollutants (PM2.5, PM10, NO2, NH3, SO2, CO, O3) + 5 Weather (Temp, RH, Wind Speed, Direction, Solar) + Cyclical Time (sin/cos hour, month) + 6h/12h/24h rolling aggregations.
• Target Formulation: Strictly next-hour particulate concentration (t+1) in µg/m³ for PM2.5 and PM10.
• Scientific Guardrail: A 24-hour input lookback sequence predicts the NEXT 1 HOUR (t+1) — NOT a 24-hour-ahead forecast!
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Our experimental pipeline evaluates an unprecedented volume of data: 1.37 million historical CPCB records and over 3.3 million recent records spanning 2020 to 2025, plus live streaming OpenAQ telemetry. Our input vector consists of 27 engineered features including pollutant lags, meteorological variables, cyclical temporal embeddings, and rolling statistical windows. Most critically, we clarify that our 24-hour input sequence predicts strictly the NEXT ONE HOUR (t+1), never a 24-hour-ahead forecast.

ANTICIPATED VIVA QUESTIONS:
Q: Why did you separate the 2015-2020 historical dataset from the 2020-2025 recent dataset?
A: During 2020, nationwide pandemic lockdowns caused an abrupt structural break in urban emission baselines. Evaluating both eras independently allows us to measure model adaptation and prevent distribution shift from biasing results.

---

## Slide 12

### Slide Visual Content

```
Implementation and Results
```

```
Hyperparameter Setup & Operational Governance Thresholds
```

```
12
```

```
Temporal Hyperparameters
Lookback Window: 24 hourly steps (t-23 to t).
Forecast Horizon: Strictly next hour (t+1).
LSTM Architecture: 2 Layers, Hidden Dim = 64, Dropout = 0.20.
Optimizer: Adam (LR = 0.001, Batch = 64).
Training Setup: 50 Epochs, Early Stopping patience = 7.
LightGBM Config: 100 trees, Depth = 6, Leaves = 31, LR = 0.05.
Normalization: Min-Max [-1, 1] fitted on train fold.
```

```
Spatial Hyperparameters
Weighting Power: Inverse Square Distance (p = 2.0).
Distance Metric: Haversine formula (R_earth = 6371.0 km).
Min Neighbors: k_min = 3 active reporting monitors.
Search Radius: R_max = 45.0 km experimental limit.
Grid Resolution: 50 × 50 uniform mesh (~1.2 km cell pitch).
Physical Bounds: Strict non-negative clamping: max(0, ẑ).
Conformal Scaling: Scaled by min distance: ŷ ± q(1 + γ·d_min).
```

```
Freshness Governance Tiers
Fresh (Green): Latency ≤ 2 hours. Fully verified for live prediction & spatial interpolation.
Delayed (Amber): Latency 2–6 hours. Flagged with cautionary amber badge on dashboard.
Stale (Orange): Latency 6–24 hours. Excluded from spatial surface computation.
Offline (Red): Latency > 24 hours. Marked inactive and strictly quarantined.
Unknown (Gray): Historical or unverified latency logs.
Governance Goal: Precludes corrupting live maps with stale cached observations.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Slide 12 defines our operational hyperparameters. On the temporal side, we utilize a 24-step sequence to predict t+1, training our 2-layer 64-hidden-unit LSTM with Adam and early stopping. On the spatial side, we configure IDW with power p=2.0 and Haversine metric, requiring at least 3 active stations within 45 km. Our data governance protocol defines five explicit freshness tiers—Fresh, Delayed, Stale, Offline, and Unknown—ensuring that spatial maps are never corrupted by stale sensor telemetry.

ANTICIPATED VIVA QUESTIONS:
Q: Why did you choose p=2.0 for the IDW power parameter?
A: An IDW power of 2.0 represents the physical inverse-square law of atmospheric dispersion. A power of 1 produces overly diffuse surfaces influenced by distant monitors, while p=3 creates localized steep peaks ('bullseyes').

---

## Slide 13

### Slide Visual Content

```
Implementation and Results
```

```
Empirical Evaluation: Historical, Adapted, Spatial & Live Benchmarks
```

```
13
```

```
| Pollutant | MAE (µg/m³) | RMSE (µg/m³) | R² Score |
| --- | --- | --- | --- |
| PM2.5 | 10.5719 | 18.5789 | 0.9492 |
| PM10 | 18.8643 | 32.2027 | 0.9301 |
```

```
| Pollutant | Adapted MAE | Adapted RMSE | Reduction |
| --- | --- | --- | --- |
| PM2.5 | 6.6216 | 13.0285 | -37.4% |
| PM10 | 16.1660 | 34.6875 | -14.3% |
```

```
| Pollutant | IDW MAE | 1-NN Baseline | Relative Gain |
| --- | --- | --- | --- |
| PM2.5 | 24.07 | 30.06 | +20.0% |
| PM10 | 52.76 | 70.68 | +25.4% |
```

```
Live Operational Audit & Conformal Bounds
• Live Evaluation (4 Stations, 418 Pairs): GBDT PM2.5 MAE = 9.64 vs Persistence 9.86 (p=0.48). Short-term gains are statistically non-significant.
• Conformal Calibration: ~99% empirical test coverage achieved; intervals widen appropriately during severe winter smog episodes.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Slide 13 provides our empirical results across four distinct evaluation regimens. Table A shows historical LSTM performance achieving an R² exceeding 0.93. Table B confirms that domain adaptation on recent 2020–2025 data significantly lowers test error. Table C presents our Delhi spatial holdout experiment: IDW outperforms the nearest-neighbor baseline by 20.0% on PM2.5 and 25.4% on PM10. Finally, our 14-day live audit reveals that next-hour persistence is an extremely strong baseline; GBDT slightly outperforms persistence, though the difference is statistically non-significant given the 418-sample size.

ANTICIPATED VIVA QUESTIONS:
Q: Why was the live forecasting improvement over persistence not statistically significant?
A: Over a 1-hour horizon (t+1), air pollution displays exceptionally high temporal autocorrelation (ρ > 0.95). In the absence of abrupt meteorological fronts, the current reading is already an outstanding predictor of the next hour.

---

## Slide 14

### Slide Visual Content

```
Summary
```

```
Key Findings, Honest Scientific Limitations & Future Directions
```

```
14
```

```
Key Research Findings
Strong Temporal Fit: Deep LSTM and GBDT models deliver robust next-hour predictive fidelity on multi-year historical logs (R² > 0.93).
Adapted Generalization: Domain adaptation on recent post-COVID data substantially lowers generalization error (PM2.5 MAE 6.62 µg/m³).
Spatial Superiority: Vectorized IDW provides superior spatial continuity over discrete 1-NN, reducing holdout error by 20–25%.
Governance Efficacy: 5-tier freshness governance successfully shields live spatial surfaces from stale sensor telemetry.
```

```
Honest Scientific Limitations
Live Sample Scope: Live evaluation is currently restricted to 4 operational stations over a 14-day audit (418 paired points).
Network Asymmetry: Non-metro cities display severe sensor clustering (Patna R=0.21, Ranchi R=0.34), precluding unconstrained IDW.
Vision Synchronization: Unsynchronized outdoor camera imagery causes negative transfer in early fusion; requires micro-climate sync.
Smog Uncertainty: Conformal intervals widen substantially during extreme winter episodes (>300 µg/m³).
```

```
Promising Future Directions
Spatio-Temporal GNNs: Formulating Spatio-Temporal Graph Neural Networks (ST-GNNs) incorporating wind-vector advection graphs.
Satellite AOD Fusion: Fusing satellite Aerosol Optical Depth (AOD from INSAT-3D & Sentinel-5P) to bridge Tier-2/Tier-3 coverage voids.
Pan-India Scaling: Conducting multi-seasonal operational validation across the full 500+ Pan-India CPCB monitoring network.
Physics-Informed ML: Automating physics-informed neural network (PINN) priors for boundary-layer dispersion.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
In summary, VayuDrishti demonstrates that unifying recurrent deep learning, vectorized spatial interpolation, and data freshness governance produces a robust, research-grade air quality intelligence platform. However, scientific integrity requires acknowledging our limitations: our live evaluation covers 4 stations, non-metro networks suffer from severe clustering, and outdoor vision requires synchronized micro-meteorological inputs. In future work, we plan to explore Spatio-Temporal Graph Neural Networks and satellite Aerosol Optical Depth fusion.

ANTICIPATED VIVA QUESTIONS:
Q: How would you extend this system to an unmonitored Tier-3 Indian city with zero physical CAAQMS stations?
A: By integrating satellite Aerosol Optical Depth (AOD) products from INSAT-3D or Sentinel-5P combined with boundary-layer meteorological reanalysis (ERA5), providing a top-down physical estimate calibrated against regional CAAQMS stations.

---

## Slide 15

### Slide Visual Content

```
Bibliography
```

```
Verified Academic References (IEEE Style Citation Format)
[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[2] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 3146–3154.
[3] D. Shepard, "A two-dimensional interpolation function for irregularly-spaced data," in Proc. 23rd ACM National Conf., 1968, pp. 517–524.
[4] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv preprint arXiv:2107.07511, 2021.
[5] Central Pollution Control Board (CPCB), "National Ambient Air Quality Standards," Ministry of Environment, Forest and Climate Change, Govt. of India, Notification B-29016/20/90/PCI-L, 2009.
[6] OpenAQ Community, "OpenAQ: A global open air quality data platform," Environmental Science & Technology, vol. 56, no. 1, pp. 12–15, 2022.
[7] P. J. Clark and F. C. Evans, "Distance to nearest neighbor as a measure of spatial relationships in populations," Ecology, vol. 35, no. 4, pp. 445–453, 1954.
[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
Here we list our foundational academic references, formatted in standard IEEE citation style. These include Hochreiter and Schmidhuber for LSTM foundations, Ke et al. for LightGBM, Donald Shepard for Inverse Distance Weighting, Angelopoulos and Bates for conformal prediction uncertainty, CPCB statutory guidelines, OpenAQ platform documentation, Clark and Evans for spatial point-pattern analysis, and He et al. for deep residual architectures.

ANTICIPATED VIVA QUESTIONS:
Q: Did you verify every reference included in this bibliography?
A: Yes, every citation corresponds to a peer-reviewed paper or official regulatory publication directly implemented and utilized within our project codebase and manuscript.

---

## Slide 16

### Slide Visual Content

```
Capstone Final Review — VayuDrishti
Thank You! Questions, Discussion & Faculty Feedback
```

```
16
```

```
Project Summary, Provenance & Repository Links
• Guide: Prof. Thota Ramathulsi | Department of SCOPE, VIT-AP University, Amaravati
• Team: Teliki Rushya Sree (23BCE9562), Machireddy Swathi (23BCE9566), Thanuja Varakala (23BCE9608), Palli Shilpa (23BCE9567)
• Public Streamlit Cloud Application: https://vayudrishti-airquality.streamlit.app/
• GitHub Research Repository: https://github.com/rushya9/multimodal_air_quality_prediction
```

### Speaker Notes & Faculty Viva Preparation

SPEAKER SCRIPT:
We thank our supervisor Prof. Thota Ramathulsi, our department faculty, and the examination panel for their invaluable guidance throughout this capstone project. Our interactive prototype is live on Streamlit Cloud, and all source code, reproducible notebooks, model weights, and IEEE manuscript artifacts are archived on GitHub. We are now open for viva questions and faculty evaluation.

---

