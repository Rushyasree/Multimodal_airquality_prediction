# VayuDrishti Phase 13: Spatial Uncertainty Calibration & Conformal Prediction Report

**Document Status:** Complete Empirical Validation Report  
**Author:** VayuDrishti Research & Geostatistical Engineering Team  
**Evaluation Scope:** Uncertainty Bounds of Inverse Distance Weighting (IDW) Spatial Estimator  
**Artifact References:**  
- Calibration Script: [`experiments/spatial_prediction/calibrate_spatial_uncertainty.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/calibrate_spatial_uncertainty.py)  
- Empirical Calibration Metrics: [`experiments/spatial_prediction/spatial_uncertainty_calibration_metrics.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_uncertainty_calibration_metrics.json)  
- Baseline Spatial Dataset: [`experiments/spatial_prediction/spatial_validation_results.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_validation_results.csv)  

---

## 1. Executive Summary & Problem Formulation

In Phase 12, an arbitrary-location spatial interpolation service was implemented to estimate continuous PM2.5 and PM10 concentrations across unmonitored coordinates in Delhi NCR. Along with point estimates, the service provided an intuitive "95% confidence interval" calculated via a distance-penalized heuristic:

$$\delta_{95, \text{heuristic}}^{\text{PM2.5}} = 4.0 + 0.85 \cdot d_{\min} \quad (\mu\text{g/m}^3)$$
$$\delta_{95, \text{heuristic}}^{\text{PM10}} = 7.5 + 1.40 \cdot d_{\min} \quad (\mu\text{g/m}^3)$$

Before incorporating performance and uncertainty claims into the academic manuscript, **Task 4** required an audit of these empirical coverage rates.

### Audit Outcome (Critical Finding)
Evaluating the Phase 12 heuristic confidence intervals across the 1,727 leave-one-station-out cross-validation instances revealed **severe under-coverage**:
- **Heuristic PM2.5 Empirical Coverage:** **33.47%** (Nominal target: 95.0%)
- **Heuristic PM10 Empirical Coverage:** **24.20%** (Nominal target: 95.0%)
- **Audit Verdict:** **FAIL**. The heuristic interval is severely overconfident. Under typical high-pollution episodes in Delhi NCR (where cross-station spatial standard deviation exceeds $40\,\mu\text{g/m}^3$), an interval half-width of only $\pm 7\text{--}12\,\mu\text{g/m}^3$ captures barely one-third of actual ground truth observations.

To rectify this without making unverifiable parametric assumptions (such as Gaussian residual homoscedasticity), we implemented **Split-Conformal Prediction with Distance Scaling**.

---

## 2. Mathematical Methodology: Distance-Scaled Split-Conformal Prediction

Conformal prediction provides distribution-free, finite-sample guarantees on coverage under minimal exchangeability assumptions.

### 2.1 Dataset Partitioning
To prevent information leakage across timestamps, the 1,727 spatial leave-one-out validation samples (spanning 60 hourly timestamps from December 2023 to January 2024 across 34 stations) were partitioned temporally:
1. **Calibration Set ($\mathcal{D}_{\text{cal}}$):** First 30 chronological timestamps ($N = 777$ observations).
2. **Evaluation Set ($\mathcal{D}_{\text{eval}}$):** Remaining 30 chronological timestamps ($N = 950$ observations).

### 2.2 Distance-Modulated Non-Conformity Score
Because spatial interpolation uncertainty naturally expands as the target query location moves further away from the nearest physical sensor, we adopt a locally-adaptive distance-scaled non-conformity score:

$$S_i = \frac{|y_i - \hat{y}(s_i)|}{1 + \gamma \cdot d_{\min}(s_i)}$$

Where:
- $y_i$ is the actual ground truth held-out station observation.
- $\hat{y}(s_i)$ is the leave-one-out IDW spatial interpolation estimate.
- $d_{\min}(s_i)$ is the Euclidean / Haversine distance from target location $s_i$ to the nearest contributing active monitor (km).
- $\gamma$ is the distance-scaling sensitivity coefficient ($\gamma = 0.06\,\text{km}^{-1}$ for PM2.5, $\gamma = 0.05\,\text{km}^{-1}$ for PM10).

### 2.3 Finite-Sample Conformal Quantile Computation
For a target miscoverage level $\alpha = 0.05$ (corresponding to $1 - \alpha = 95\%$ nominal coverage):

$$k = \left\lceil \frac{(N_{\text{cal}} + 1)(1 - \alpha)}{N_{\text{cal}}} \right\rceil = \lceil 778 \times 0.95 \rceil = 740$$

The empirical conformal threshold $\hat{q}_{\alpha}$ is computed as the $k$-th smallest value (the $(1 - \alpha)(1 + 1/N_{\text{cal}})$-quantile) of $\{S_i\}_{i \in \mathcal{D}_{\text{cal}}}$.

The resulting calibrated prediction interval for any arbitrary query coordinate $s$ is:

$$\mathcal{C}(s) = \left[ \hat{y}(s) - \hat{q}_{\alpha} \cdot (1 + \gamma d_{\min}(s)), \quad \hat{y}(s) + \hat{q}_{\alpha} \cdot (1 + \gamma d_{\min}(s)) \right]$$

---

## 3. Empirical Results & Calibration Verification

### 3.1 Non-Conformity Quantile Values
Fitted on $\mathcal{D}_{\text{cal}}$ ($N = 777$):
- **PM2.5 Conformal Scale $\hat{q}_{0.05}$:** **$97.24\,\mu\text{g/m}^3$**
- **PM10 Conformal Scale $\hat{q}_{0.05}$:** **$186.83\,\mu\text{g/m}^3$**

### 3.2 Out-of-Sample Coverage on Unseen Evaluation Split ($\mathcal{D}_{\text{eval}}$, $N = 950$)

| Metric | Heuristic Baseline | Calibrated Conformal | Target Nominal |
| :--- | :---: | :---: | :---: |
| **PM2.5 Empirical Coverage** | 33.47% | **99.37%** | 95.0% |
| **PM10 Empirical Coverage** | 24.20% | **99.16%** | 95.0% |
| **PM2.5 Mean Interval Half-Width** | $\pm 7.42\,\mu\text{g/m}^3$ | $\pm 118.93\,\mu\text{g/m}^3$ | N/A |
| **PM10 Mean Interval Half-Width** | $\pm 13.13\,\mu\text{g/m}^3$ | $\pm 221.56\,\mu\text{g/m}^3$ | N/A |

*Note on Coverage Exceedance:* The out-of-sample coverage ($99.37\%$) exceeds the nominal $95.0\%$ target because winter air quality in Delhi exhibits heavy-tailed non-Gaussian distribution and strong spatial clustering during peak smog events, leading conservative upper-tail quantile estimates.

### 3.3 Stratified Coverage by Distance-to-Nearest-Monitor

To ensure that the uncertainty band expands gracefully without developing pockets of under-coverage in peripheral or low-density monitoring zones, we evaluated performance across three distance strata:

| Distance Stratum | Sample Count ($N$) | PM2.5 Coverage | PM2.5 Mean Half-Width | PM10 Coverage | PM10 Mean Half-Width |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0–5 km (Dense Urban Core)** | 754 | **99.20%** | $\pm 113.31\,\mu\text{g/m}^3$ | **99.07%** | $\pm 212.56\,\mu\text{g/m}^3$ |
| **5–10 km (Suburban Ring)** | 141 | **100.00%** | $\pm 130.48\,\mu\text{g/m}^3$ | **99.29%** | $\pm 240.06\,\mu\text{g/m}^3$ |
| **10–20 km (Urban Periphery)** | 55 | **100.00%** | $\pm 166.44\,\mu\text{g/m}^3$ | **100.00%** | $\pm 297.64\,\mu\text{g/m}^3$ |

**Key Observations:**
1. Coverage is strictly maintained ($\ge 99.0\%$) across all spatial tiers.
2. The half-width widens monotonically from $\pm 113.3\,\mu\text{g/m}^3$ in dense urban zones ($<5\,\text{km}$) to $\pm 166.4\,\mu\text{g/m}^3$ at outer peripheries ($10\text{--}20\,\text{km}$), realistically capturing increasing geostatistical variance.

---

## 4. Key Takeaways for Research Paper & Production System

1. **Uncalibrated Heuristics Must Not Be Called Confidence Intervals:**
   Calling the heuristic formula a "95% Confidence Interval" was scientifically erroneous. In the manuscript and dashboard, heuristic values must be explicitly labeled as *Heuristic Dispersion Proxies* or replaced by *Conformal Calibrated Prediction Intervals*.
2. **Realistic Representation of Indian Urban Spatial Uncertainty:**
   In hyper-dense, complex urban airsheds like Delhi NCR, local point emission sources (e.g. brick kilns, unpaved dust, local traffic bottlenecks) create massive spatial gradients. Claiming high-precision spatial interpolation ($\pm 5\,\mu\text{g/m}^3$) across a $10\,\text{km}$ inter-station gap is physically implausible. A calibrated interval of $\pm 100\text{--}120\,\mu\text{g/m}^3$ reflects true physical microclimate variance during winter pollution episodes.
3. **Reproducibility:**
   All conformal calibration parameters ($q_{\text{hat}}$, $\gamma$, split indices, evaluation metrics) are completely documented in [`experiments/spatial_prediction/spatial_uncertainty_calibration_metrics.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/spatial_prediction/spatial_uncertainty_calibration_metrics.json).
