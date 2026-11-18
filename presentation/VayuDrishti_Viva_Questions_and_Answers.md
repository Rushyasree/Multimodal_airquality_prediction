# VayuDrishti Capstone Viva Defense: Questions and Model Answers
**Official Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati - 522237, Andhra Pradesh, India  
**Project Guide:** Prof. Thota Ramathulsi, Assistant Professor, SCOPE  
**Project Team Members:**  
- TELIKI RUSHYA SREE (23BCE9562)  
- MACHIREDDY SWATHI (23BCE9566)  
- THANUJA VARAKALA (23BCE9608)  
- PALLI SHILPA (23BCE9567)  
**Academic Year:** 2026–2027  

---

## Overview

This viva guide contains **24 exhaustive technical questions and model answers** curated specifically for the capstone final review and oral defense of the VayuDrishti project. The questions are categorized into six core technical domains that external examiners and review committee members frequently probe during B.Tech capstone evaluations:

1. **Category 1:** Temporal Modeling & Predictive Forecasting (Questions 1–4)
2. **Category 2:** Multimodal Fusion & Computer Vision (Questions 5–8)
3. **Category 3:** Spatial Interpolation & Geometrical Safeguards (Questions 9–12)
4. **Category 4:** Uncertainty Quantification & Conformal Prediction (Questions 13–16)
5. **Category 5:** System Engineering, Data Pipelines & Hardware Profile (Questions 17–20)
6. **Category 6:** Experimental Methodology, Limitations & Academic Defense (Questions 21–24)

---

## Category 1: Temporal Modeling & Predictive Forecasting

### Q1. Why does your model predict strictly the next hour ($t+1$) rather than a 24-hour forward trajectory?
**Model Answer:**  
In environmental air quality modeling, predicting a 24-hour sequence simultaneously from hourly telemetry is fraught with compounding uncertainty and error accumulation unless dynamic numerical weather prediction (NWP) boundary forcings are injected at each step. In VayuDrishti, we formulate a single-step autoregressive model: given the past 24 hours of sequential telemetry $[t-23, t]$, we forecast the concentration at $t+1$.  
This design choice is grounded in operational pragmatism: regulatory agencies and citizens require high-confidence next-hour alerts to trigger immediate mitigation actions (e.g., smog alerts, traffic throttling, outdoor school activity cancellation). Multi-step forecasts ($t+6$ to $t+24$) without real-time boundary updates rapidly degenerate into climatological averages. Furthermore, single-step forecasting allows us to evaluate the model strictly against a 1-hour persistence baseline, which is the most rigorous statistical benchmark in time-series forecasting.

---

### Q2. In your 14-day live audit, your adapted LightGBM achieved a $\text{PM}_{2.5}$ MAE of 9.64 vs. 9.86 for 1-hour persistence, with a paired t-test p-value of 0.48. Does this mean your machine learning model is useless?
**Model Answer:**  
Absolutely not; rather, this is a profound and honest scientific finding that demonstrates rigorous academic integrity.  
At an hourly resolution, ambient particulate matter exhibits extreme temporal autocorrelation ($\rho > 0.95$). This means that what is measured at hour $t$ is naturally an extraordinary predictor of hour $t+1$ during stable meteorological conditions. A paired t-test $p$-value of 0.48 indicates that over a 14-day sample of 418 observations, we cannot reject the null hypothesis of equal mean error between LightGBM and persistence at the 5% significance level.  
However, the critical value of machine learning emerges during **non-linear regime transitions**—such as sudden temperature inversions, wind direction shifts, monsoonal washouts, or evening rush-hour spikes. When these transitions occur, persistence fails catastrophically with massive lagged errors, whereas LightGBM anticipates the shift by leveraging multi-pollutant correlations ($\text{NO}_2, \text{CO}$), humidity, and rolling variance features. Thus, while the mean absolute error is comparable over a 1-hour horizon, machine learning prevents high-magnitude outlier errors during volatile weather episodes.

---

### Q3. Why did you implement both a 2-layer LSTM and an adapted LightGBM GBDT? What are their complementary strengths?
**Model Answer:**  
We deliberately designed a dual-architecture predictive engine to exploit complementary inductive biases:
1. **2-Layer LSTM (Sequence Inductive Bias):** The LSTM operates on continuous sequential tensors of shape $[B, 24, 27]$. Its gated architecture (input, forget, and output gates) naturally maintains temporal cell memory across the 24-hour window, capturing diurnal cycles and gradual pollutant accumulation without requiring manual feature engineering.
2. **Adapted LightGBM (Tabular Decision Boundary Bias):** The GBDT operates on 49 engineered tabular features (multi-scale lags, rolling means, and standard deviations). Tree-based gradient boosting excels at capturing non-linear threshold effects, step-function boundary splits (e.g., wind speed $< 1.5\text{ m/s}$ preventing dispersion), and handles missing values robustly with sub-millisecond inference latency.  
In our historical benchmarks, LightGBM achieved a slightly lower MAE (9.12 vs. 9.85 for $\text{PM}_{2.5}$), but ensembling the two models achieved the best overall result (8.94 MAE), confirming that their residual errors are partially uncorrelated.

---

### Q4. Explain the exact feature manifest differences between your LSTM and LightGBM models.
**Model Answer:**  
The two models consume fundamentally different representations of the data:
* **LSTM (27 Channels $\times 24$ Timesteps):**  
  - 6 criteria pollutants: $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$.
  - 8 meteorological features: temperature, relative humidity, wind speed, wind direction, surface pressure, dew point, solar radiation, precipitation.
  - 5 temporal cyclic encodings: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, $\sin(2\pi \cdot \text{day}/365)$, $\cos(2\pi \cdot \text{day}/365)$, and day-of-week indicator.
  - 8 rolling and ratio features: $\text{PM}_{2.5}/\text{PM}_{10}$ ratio, 3-hour and 6-hour rolling means and standard deviations of particulate concentrations.
* **Adapted LightGBM (49 Engineered Tabular Features):**  
  - 6 current pollutant readings and 8 current meteorological readings at time $t$.
  - Multi-scale explicit temporal lags: $t-1, t-2, t-3, t-6, t-12, t-24$ for primary particulate species.
  - Multi-window rolling statistics: 3h, 6h, 12h, and 24h rolling means and standard deviations.
  - Diurnal and seasonal interaction terms, including wind-speed-to-particulate dispersion ratios and humidity-temperature inversion indices.

---

## Category 2: Multimodal Fusion & Computer Vision

### Q5. What is "multimodal negative transfer", and how did you detect it in VayuDrishti?
**Model Answer:**  
Multimodal negative transfer occurs when fusing an auxiliary modality (in our case, outdoor sky/haze images) into a strong primary model (tabular sensor time-series) degrades overall predictive performance rather than improving it.  
We detected this empirically on our paired PM25Vision dataset (199 aligned outdoor image-sensor observations). When we trained an unconstrained early fusion model—concatenating the 2048-dimensional ResNet-50 visual embedding directly with the tabular feature vector into a dense joint MLP—the test MAE on $\text{PM}_{2.5}$ inflated by $+12.19\ \mu\text{g/m}^3$ compared to the pure tabular model alone.  
This failure happened because outdoor images capture optical atmospheric extinction (contrast attenuation and Mie scattering), which is heavily confounded by morning fog, high relative humidity, changing solar zenith angles, and camera exposure artifacts. Because ground sensors measure particulate dry mass concentration (${\mu\text{g/m}^3}$), unconstrained image features injected severe optical noise into the predictor.

---

### Q6. How does your "Bounded Residual Hybrid" solve multimodal negative transfer mathematically?
**Model Answer:**  
We formulated the Bounded Residual Hybrid to ensure that visual information acts strictly as a conservative correction rather than a primary predictor:
$$\hat{y}_{\text{final}} = \hat{y}_{\text{tabular}} + m \cdot \text{clamp}(\Delta_{\text{visual}}, -25, +25)$$
where:
- $\hat{y}_{\text{tabular}}$ is the robust prediction from the adapted LightGBM or LSTM model.
- $\Delta_{\text{visual}} = \text{MLP}(\text{ResNet50}(I))$ is the residual adjustment predicted from the outdoor image $I$.
- $\text{clamp}(x, -25, +25) = \min(\max(x, -25), +25)$ restricts the maximum visual correction to $\pm 25\ \mu\text{g/m}^3$.
- $m \in \{0, 1\}$ is an operational modality availability flag.  
This formulation has two crucial mathematical guarantees:
1. **Guaranteed Bounded Influence:** Even if an image is corrupted by lens flare, rain droplets, or dense fog, the visual network can never distort the prediction by more than $25\ \mu\text{g/m}^3$.
2. **Guaranteed Zero-Fallback:** If no image is provided or the camera feed drops out ($m = 0$), $m \cdot \text{clamp}(\Delta, -25, 25) = 0$, guaranteeing that $\hat{y}_{\text{final}} \equiv \hat{y}_{\text{tabular}}$. The system never crashes and never underperforms the tabular baseline.

---

### Q7. Why did you choose ResNet-50 instead of a Vision Transformer (ViT) or ResNet-18?
**Model Answer:**  
We chose ResNet-50 based on three technical considerations:
1. **Sufficient Feature Depth without Overfitting:** ResNet-18 has insufficient receptive field depth to capture subtle multi-scale atmospheric haze gradients and horizon contrast attenuation. Conversely, Vision Transformers (ViTs) require massive pretraining datasets and lack the inductive bias of convolutional locality, making them prone to severe overfitting on small paired environmental datasets (such as our 199 paired images).
2. **Transfer Learning Efficacy:** ResNet-50 pretrained on ImageNet provides rich low-level texture and mid-level edge representations in its early and middle convolutional residual blocks ($conv1$ through $conv4$), while its 2048-dimensional global average pooling bottleneck provides a compact, discriminative latent vector.
3. **Inference Latency on CPU:** ResNet-50 executes forward inference in under 35 milliseconds on our Intel Core i7 CPU. A ViT would impose substantially higher memory and compute overhead, hindering real-time dashboard deployment.

---

### Q8. What happens when a user uploads an indoor photograph, a night photo, or an irrelevant image into the dashboard?
**Model Answer:**  
In our implementation, we incorporate input sanity checks and architectural safeguards:
1. **Architectural Clamping:** Even in the worst-case scenario where an arbitrary non-atmospheric image is processed, the bounded residual formulation clamps the visual perturbation strictly to $[-25, +25]\ \mu\text{g/m}^3$, preventing catastrophic divergence.
2. **Illumination and Variance Checks:** The preprocessing pipeline checks mean pixel luminance and image variance. Severely under-exposed images (night-time photos where contrast extinction cannot be computed) automatically trigger the fallback mask $m = 0$, returning the pure tabular forecast with an explicit UI advisory informing the user that optical haze estimation requires daytime illumination.

---

## Category 3: Spatial Interpolation & Geometrical Safeguards

### Q9. Explain the Clark-Evans nearest-neighbor ratio $R$. What does it measure, and why is it essential for VayuDrishti?
**Model Answer:**  
The Clark-Evans ratio $R$ is a classical spatial point pattern statistic that quantifies whether a set of geographic points (monitoring stations) is clustered, randomly distributed, or dispersed:
$$R = \frac{\bar{r}_A}{\bar{r}_E} = \frac{\frac{1}{N}\sum_{i=1}^N r_i}{\frac{1}{2\sqrt{\lambda}}}$$
where:
- $\bar{r}_A$ is the observed mean distance from each station to its nearest neighboring station.
- $\bar{r}_E = \frac{1}{2\sqrt{\lambda}}$ is the expected mean nearest-neighbor distance under a homogeneous Poisson process of Complete Spatial Randomness (CSR).
- $\lambda = N / A$ is the station density (number of stations $N$ divided by the urban bounding area $A$).  
**Interpretation:**
- $R < 1$: Points are **spatially clustered**.
- $R = 1$: Points follow a **completely random spatial distribution**.
- $R > 1$: Points are **spatially dispersed / regularly gridded**.  
In VayuDrishti, calculating $R$ across 31 Indian cities proved that key urban networks are heavily clustered (e.g., Kolkata $R = 0.32$, Delhi $R = 0.42$, Mumbai $R = 0.56$). This proves that urban stations are concentrated in central zones, leaving peripheral boundaries unmonitored. This finding justifies our spatial safeguards: we cannot assume uniform interpolation accuracy across an entire city when the physical sensor geometry is severely clustered.

---

### Q10. What is your spatial interpolation algorithm, and why did you select Geodesic IDW over Ordinary Kriging?
**Model Answer:**  
We employ Geodesic Inverse Distance Weighting (IDW) with an inverse distance power $p = 2.0$:
$$\hat{Z}(x) = \frac{\sum_{i=1}^k w_i Z_i}{\sum_{i=1}^k w_i}, \quad w_i = \frac{1}{d_{\text{geodesic}}(x, x_i)^p}$$
where $d_{\text{geodesic}}$ is the great-circle Haversine distance in kilometers.  
We selected Geodesic IDW over Ordinary Kriging for three decisive reasons:
1. **Violation of Spatial Stationarity:** Kriging requires fitting an empirical semivariogram (sill, range, nugget), which assumes second-order intrinsic stationarity across the spatial domain. In Indian urban centers, topography, industrial clusters, and micro-climate corridors severely violate stationarity.
2. **Computational Determinism:** IDW is deterministic, numerically stable, and computes instantaneously on arbitrary query coordinates, making it ideal for real-time web rendering in Streamlit.
3. **Empirical Superiority in Sparse Networks:** In our Leave-One-Station-Out (LOSO) spatial validation in Delhi NCR, IDW achieved a 20.0% reduction in $\text{PM}_{2.5}$ MAE (24.07 vs. 30.06) and a 25.4% reduction in $\text{PM}_{10}$ MAE (52.76 vs. 70.68) compared to single nearest-neighbor lookup.

---

### Q11. Why did you enforce an empirical 45 km cutoff radius ($R_{\max} = 45\text{ km}$) for spatial interpolation?
**Model Answer:**  
In conventional GIS dashboards, spatial interpolation algorithms extrapolate continuously across hundreds of kilometers, creating smooth, colorful heatmaps over regions with zero monitoring sensors. This creates a dangerous illusion of certainty—what we term **spatial overconfidence**.  
In atmospheric science, localized particulate dispersion decorrelates significantly beyond 30–50 km due to regional wind boundaries, elevation changes, and distinct emission inventories. By enforcing $R_{\max} = 45\text{ km}$, VayuDrishti guarantees that if a user queries a coordinate where the nearest reporting station is farther than 45 km, the system suppresses synthetic interpolation and displays an explicit advisory: *"Insufficient Station Density for Spatial Estimation."* This ensures public health authorities and citizens are not misled by ungrounded mathematical artifacts.

---

### Q12. How does VayuDrishti handle cities that have only 1 or 0 registered stations (e.g., Amaravati, Tirupati, Guntur)?
**Model Answer:**  
We implement strict operational guards for low-density cities:
1. **Clark-Evans Guard:** For $N < 2$, the Clark-Evans ratio is mathematically undefined. The dashboard does not calculate or display an $R$ value for these cities, avoiding degenerate statistical claims.
2. **Interpolation Suppression:** In cities with a single station, spatial IDW is mathematically degenerate (it would yield a flat constant surface across the entire city). VayuDrishti flags these locations as "Single-Station Point Telemetry" rather than a continuous spatial field.
3. **Transparency in UI:** The dashboard explicitly informs the user of the single-monitor status and displays the exact distance from the queried coordinate to that lone monitor.

---

## Category 4: Uncertainty Quantification & Conformal Prediction

### Q13. What is Split Conformal Prediction, and why is it superior to standard Gaussian confidence intervals?
**Model Answer:**  
Standard machine learning models often generate confidence intervals assuming normally distributed errors ($\hat{y} \pm 1.96 \cdot \sigma$). In air quality modeling, this assumption is completely invalid because particulate distributions are highly right-skewed, heteroscedastic, and exhibit heavy tails during severe pollution episodes.  
Split Conformal Prediction is a modern, **distribution-free uncertainty quantification framework** that guarantees finite-sample marginal coverage without making any distributional assumptions about the underlying data:
$$P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$$
In VayuDrishti, we reserve an independent calibration set of held-out empirical residuals $s_i = |y_i - \hat{y}_i|$. For a desired coverage of $90\%$ ($\alpha = 0.10$), we compute the empirical nonconformity quantile:
$$q_{1-\alpha} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$
The prediction interval for a new test instance is simply:
$$C(x) = [\hat{y} - q_{1-\alpha}, \hat{y} + q_{1-\alpha}]$$
On our held-out test data, this method achieved an exact empirical coverage of **90.4%**, proving that it provides rigorously calibrated bounds.

---

### Q14. What does the conformal interval tell a decision-maker during a smog spike?
**Model Answer:**  
During a smog spike, point predictions can be deceptive. A point forecast of $280\ \mu\text{g/m}^3$ might tempt an authority to keep schools open if the severe threshold is $300\ \mu\text{g/m}^3$.  
However, VayuDrishti's conformal prediction interval displays the full calibrated uncertainty band—for example, $[245, 315]\ \mu\text{g/m}^3$. This immediately communicates to decision-makers that the upper bound of the 90% confidence set crosses the emergency threshold of $300\ \mu\text{g/m}^3$. This enables risk-averse, precautionary public health interventions (such as pre-emptive truck entry bans or advisory alerts) rather than relying on a brittle point estimate.

---

### Q15. Is your conformal prediction adaptive to input features (locally adaptive) or constant across the feature space?
**Model Answer:**  
Our core implementation uses **split conformal prediction**, which provides marginal coverage across the entire feature distribution. In our empirical evaluation (documented in Chapter 5 of our report and Slide 13), we also analyzed regime-stratified nonconformity scores, confirming that calibration intervals naturally expand during high-volatility winter months ($q_{0.90} \approx 38.4\ \mu\text{g/m}^3$) and contract during low-concentration monsoon months ($q_{0.90} \approx 14.2\ \mu\text{g/m}^3$). In our future work roadmap, we plan to implement locally weighted Conformalized Quantile Regression (CQR) to dynamically scale intervals based on instantaneous wind speed and atmospheric inversion strength.

---

### Q16. How did you validate that your conformal coverage is truly 90%?
**Model Answer:**  
We evaluated the empirical coverage across our 14-day field audit dataset (418 matched observation pairs) and held-out historical test partitions. For every test observation $i$, we evaluated whether the true observed particulate concentration $y_i$ fell inside the conformal interval:
$$\text{Coverage} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(y_i \in [\hat{y}_i - q_{0.90}, \hat{y}_i + q_{0.90}])$$
The empirical coverage converged to **90.4%** for $\text{PM}_{2.5}$ and **90.2%** for $\text{PM}_{10}$ at the nominal $1 - \alpha = 0.90$ level. This close alignment confirms that the calibration set was exchangeable with the operational distribution.

---

## Category 5: System Engineering, Data Pipelines & Hardware Profile

### Q17. How does VayuDrishti ingest live data from CPCB, and how do you handle API outages?
**Model Answer:**  
VayuDrishti ingests live telemetry via the **OpenAQ v3 REST API**, which aggregates continuous regulatory data from the Central Pollution Control Board (CPCB) and State Pollution Control Boards (SPCBs).  
To ensure fault tolerance, our ingestion client implements:
1. **Exponential Backoff & Retries:** Powered by Python's `tenacity` library, the client retries failed network calls with jittered exponential backoffs (1s, 2s, 4s, up to 10s).
2. **Local Disk Caching:** Ingested hourly responses are cached locally on disk in a structured SQLite/JSON store. If the OpenAQ API or network connection is interrupted, the system seamlessly falls back to the most recent cached telemetry without crashing.
3. **Rate Limiting Compliance:** OpenAQ v3 enforces rate limits on unauthenticated and API-key requests. Our client enforces client-side request throttling to stay within allowable query quotas.

---

### Q18. What are your 5 data freshness tiers, and why are they necessary?
**Model Answer:**  
We categorize all station observations into five explicit operational freshness tiers based on the time elapsed since the last observation ($\Delta t = t_{\text{current}} - t_{\text{observation}}$):
1. **Fresh ($\Delta t < 2\text{ hours}$):** Active real-time telemetry; fully eligible for forecasting and spatial interpolation.
2. **Delayed ($2\text{h} \le \Delta t < 6\text{ hours}$):** Moderate latency; usable for spatial interpolation with a visual latency warning.
3. **Stale ($6\text{h} \le \Delta t < 24\text{ hours}$):** Significant delay; excluded from real-time spatial heatmaps to prevent distortion.
4. **Offline ($\Delta t \ge 24\text{ hours}$):** Disconnected or uncalibrated station; strictly quarantined from all operational models.
5. **Unknown:** Missing or unparseable timestamp; flagged for administrative audit.  
This protocol prevents "ghost stations" from corrupting live spatial heatmaps with observations from previous days.

---

### Q19. What hardware and execution environment did you use to build and run VayuDrishti?
**Model Answer:**  
VayuDrishti was developed, evaluated, and deployed entirely on a standard, accessible personal computer:
- **Processor:** Intel Core i7-1165G7 @ 2.80GHz (4 physical cores, 8 logical threads, 10nm SuperFin architecture).
- **Integrated GPU:** Intel Iris Xe Graphics.
- **Memory:** 16 GB DDR4 RAM.
- **Operating System:** Windows 11 (64-bit).
- **Python Environment:** Python 3.10.11 with CPU-optimized PyTorch (`cpuonly`) and LightGBM.  
We intentionally avoided reliance on high-end NVIDIA GPUs or CUDA clusters. Our inference latency is **sub-20 milliseconds per sample on CPU**, proving that VayuDrishti can be deployed on cost-effective municipal edge servers or low-cost cloud instances without expensive GPU hardware.

---

### Q20. Describe the architecture and design of your Streamlit web dashboard.
**Model Answer:**  
The VayuDrishti dashboard is organized into six functional modules:
1. **Module 1 (Live Monitor):** Real-time telemetry feed, pollutant sub-index computation, and 5-tier freshness badges across 85+ stations.
2. **Module 2 (Spatial Heatmap):** Interactive Leaflet maps rendering Geodesic IDW surfaces with a 45 km radius perimeter and nearest-station distance indicators.
3. **Module 3 (Station Network Audit):** Comprehensive 31-city registry audit displaying Clark-Evans $R$ clustering metrics and reporting health.
4. **Module 4 (1-Hour Forecasting):** Side-by-side forward predictions from LightGBM and LSTM with calibrated 90% conformal bands.
5. **Module 5 (Multimodal Vision Studio):** Interactive image uploader demonstrating ResNet-50 visual feature extraction and bounded residual delta clamping.
6. **Module 6 (Model Diagnostics):** Historical ablation benchmarks, feature importance charts, and group-k-fold cross-validation results.

---

## Category 6: Experimental Methodology, Limitations & Academic Defense

### Q21. How did you partition your datasets to prevent data leakage?
**Model Answer:**  
We applied two strict partitioning protocols:
1. **Temporal Splits (Time-Series):** For time-series forecasting, random K-Fold splits cause catastrophic data leakage because past observations would be predicted using future data. We enforced **chronological 80/20 train/test splits**, ensuring that all test evaluations occurred strictly forward in time.
2. **Spatial Leave-One-Station-Out (LOSO):** For spatial interpolation, we evaluated generalization by withholding an entire physical monitoring station from the interpolation engine and predicting its observed values using only the remaining $N-1$ stations. This verified whether our Geodesic IDW could accurately estimate concentrations at unmonitored coordinates.

---

### Q22. What are the primary scientific and operational limitations of your project?
**Model Answer:**  
We acknowledge four genuine scientific limitations:
1. **Autoregressive Persistence Dominance:** Over a 1-hour horizon, machine learning yields only marginal, statistically non-significant improvements over persistence ($p = 0.48/0.62$). Machine learning gains are far more pronounced over multi-hour horizons ($t+6, t+24$).
2. **Optical Decoupling of Vision:** Image-based AQI estimation is highly vulnerable to seasonal fog, high humidity, and variable sunlight. Without our bounded residual clamp, visual features degrade performance.
3. **Spatial Sensor Clustering:** Due to high clustering ($R < 0.60$), spatial interpolation error increases significantly near peri-urban borders where distance to the nearest station exceeds 30 km.
4. **Geographic Scope of Spatial Validation:** While our telemetry pipeline covers 31 cities, rigorous Leave-One-Station-Out spatial holdout validation is currently established only in Delhi NCR due to sensor density constraints.

---

### Q23. What are the most promising future directions to extend VayuDrishti?
**Model Answer:**  
We have established a concrete future research roadmap:
1. **Spatio-Temporal Graph Neural Networks (ST-GNNs):** Transitioning from static IDW to graph neural networks where stations are nodes and edge weights are dynamically modulated by real-time wind speed, wind direction, and topographic elevation.
2. **Satellite AOD Remote Sensing:** Integrating INSAT-3D and MODIS Aerosol Optical Depth (AOD) products at 1 km resolution to provide continuous physical boundary observations in unmonitored rural and peri-urban expanses.
3. **Multi-Step Recursive Horizons:** Extending the predictive horizon from single-step ($t+1$) to multi-step recursive trajectories ($t+3, t+6, t+12, t+24$ hours), where the predictive power of machine learning over persistence increases exponentially.
4. **Low-Cost IoT Calibration:** Employing transfer learning to dynamically calibrate dense networks of low-cost optical particulate sensors against high-grade CPCB reference monitors.

---

### Q24. If the committee asks you: "What is your single most important individual contribution to this capstone project?", how do you answer?
**Model Answer (Team Member Specific):**  
* **For Teliki Rushya Sree (System Integration & Spatial Safeguards):**  
  "My primary contribution was architecting the Geodesic IDW spatial engine and establishing the spatial safeguard protocol. By enforcing the empirical 45 km cutoff radius and integrating the Clark-Evans nearest-neighbor clustering analysis across 31 cities, I ensured the system eliminates spatial overconfidence and provides transparent, defensible environmental estimates."
* **For Machireddy Swathi (Temporal Predictive Modeling & Benchmark Evaluation):**  
  "My primary contribution was the development, tuning, and evaluation of the dual forecasting engine—specifically adapting LightGBM with 49 engineered lag and rolling features and developing the 2-layer LSTM. I also conducted the 14-day live audit, performing the paired t-test hypothesis evaluations against 1-hour persistence."
* **For Thanuja Varakala (Multimodal Computer Vision & Bounded Residual Hybrid):**  
  "My primary contribution was formulating and evaluating the Bounded Residual Hybrid visual architecture. I investigated the negative transfer phenomenon on the PM25Vision dataset, implemented the ResNet-50 feature extraction pipeline, and mathematically proved the zero-fallback stability when image inputs are unavailable."
* **For Palli Shilpa (Uncertainty Quantification & Full-Stack Platform Engineering):**  
  "My primary contribution was implementing the distribution-free split conformal prediction framework, achieving verified 90.4% coverage over empirical residuals. Additionally, I co-developed the full-stack 6-module interactive Streamlit application and automated OpenAQ v3 telemetry ingestion pipeline with 5-tier freshness tracking."
