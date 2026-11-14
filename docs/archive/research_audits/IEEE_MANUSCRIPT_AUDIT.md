# VayuDrishti Research Integrity Audit: IEEE Manuscript Verification

**Audit Phase:** Phase 16 — Stage 10  
**Target Manuscript:** `paper/main.tex`  
**Bibliography:** `paper/references.bib`  
**Tables:** `paper/tables/*.tex` (7 tables)  
**Figures:** `paper/figures/*.png` (7 figures)  

---

## 1. Executive Summary

A comprehensive line-by-line audit of the IEEE conference manuscript was conducted to verify formatting compliance, mathematical correctness, data fidelity, table/figure reconciliation, and ethical research transparency.

### Overall Manuscript Audit Status: **READY FOR IEEE SUBMISSION**

---

## 2. Structural & Formatting Verification

- [x] **Document Class:** Standard `\documentclass[conference]{IEEEtran}` with `\IEEEoverridecommandlockouts`.
- [x] **Title & Author Block:**
  - Title: *"VayuDrishti: Robust Longitudinal Air Quality Forecasting, Distribution-Free Uncertainty Calibration, and Empirical Limits of Multimodal Sensing Across Indian Urban Environments"*
  - Authors: Teliki Rushya Sree, Machireddy Swathi, Thanuja Varakala, Palli Shilpa, and Professor Thota Ramathulsi (Guided by: Professor Thota Ramathulsi) (Faculty Supervisor).
  - Department and correspondence emails correctly structured.
- [x] **Abstract & Keywords:**
  - Concise, highly quantitative 300-word abstract detailing sample size ($N=236,289$), 15 stations, 11 States/UTs, GBDT gains ($+3.97\%$ and $+5.06\%$), split-conformal coverage, and empirical negative result.
  - Comprehensive IEEE keywords included.
- [x] **Section Progression:**
  - Section I: Introduction & Motivation
  - Section II: Related Work & Prior Art
  - Section III: Empirical Negative Result on Opportunistic Imagery
  - Section IV: Adapted GBDT and Severity-Gated Architecture
  - Section V: Longitudinal Multi-Station Live Trial
  - Section VI: Distribution-Free Conformal Uncertainty Calibration
  - Section VII: Statistical Robustness and Significance Testing
  - Section VIII: Spatial-Temporal Cross-Station Advection Analysis
  - Section IX: Deployment, Governance, and Threats to Validity
  - Section X: Conclusion and Future Work

---

## 3. Table and Figure Verification Trace

| Entity | Description | File Path | Data Source | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Table I** | Overall Model Benchmark ($N=236,289$) | `tables/table_overall_benchmarks.tex` | `experiments/phase12/model_comparison.csv` | **100% MATCH** |
| **Table II** | Multimodal Ablation Negative Result | `tables/table_negative_result_ablation.tex` | `experiments/unified_hybrid/accuracy_optimization/ablation_results.json` | **100% MATCH** |
| **Table III** | Multi-Station Breakdown (15 stations) | `tables/table_multistation_performance.tex` | `experiments/phase12/station_metrics.csv` | **100% MATCH** |
| **Table IV** | Seasonal Dynamics (4 IMD seasons) | `tables/table_seasonal_metrics.tex` | `experiments/phase12/seasonal_metrics.csv` | **100% MATCH** |
| **Table V** | Conformal Calibration Quantiles | `tables/table_conformal_calibration.tex` | `experiments/phase12/uncertainty_metrics.csv` | **100% MATCH** |
| **Table VI** | Statistical Significance & Bootstrap | `tables/table_statistical_tests.tex` | `experiments/phase12/statistical_tests.csv` | **100% MATCH** |
| **Table VII**| Cross-Station Spatial Advection | `tables/table_spatial_advection.tex` | `experiments/phase13/spatial_forecast_comparison.csv` | **100% MATCH** |
| **Fig. 1** | PM25Vision Alignment Diagnostics | `figures/alignment_discrepancy_distribution.png` | `experiments/unified_hybrid/DATA_ALIGNMENT_AUDIT.md` | **EXISTS & CITED** |
| **Fig. 2** | Multi-Station Performance Bars | `figures/fig3_multistation_performance.png` | `experiments/phase12/figures/` | **EXISTS & CITED** |
| **Fig. 3** | Seasonal Performance Comparison | `figures/fig2_seasonal_model_comparison.png` | `experiments/phase12/figures/` | **EXISTS & CITED** |
| **Fig. 4** | Conformal Calibration Curves | `figures/fig1_conformal_calibration.png` | `experiments/phase12/figures/` | **EXISTS & CITED** |
| **Fig. 5** | Severity Regime Error Curves | `figures/fig4_regime_error_curve.png` | `experiments/phase12/figures/` | **EXISTS & CITED** |
| **Fig. 6** | Cross-Correlation Lag Functions | `figures/fig_spatial_cross_correlation.png` | `experiments/phase13/figures/` | **EXISTS & CITED** |
| **Fig. 7** | Spatial Forecast Comparison | `figures/fig_spatial_forecast_comparison.png` | `experiments/phase13/figures/` | **EXISTS & CITED** |

---

## 4. Mathematics and Notation Audit

1. **Particulate Concentrations:** Consistently rendered as $\text{PM}_{2.5}$ and $\text{PM}_{10}$ in text and math mode.
2. **Evaluation Metrics:** Consistently defined:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Split-Conformal Formulations:** Finite-sample quantile formula correctly states:
   $$\hat{q}_{1-\alpha} = \text{Quantile}\left(\{s_i\}_{i=1}^n; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$
   and guarantees:
   $$\mathbb{P}\left(y_{n+1} \in \mathcal{C}_{1-\alpha}(\mathbf{x}_{n+1})\right) \ge 1 - \alpha$$
4. **Normalized Cross-Correlation:** Strictly standard definition for time-series signal processing.

---

## 5. Ethical Research Transparency and Claim Calibration

The manuscript maintains the highest standards of scientific honesty:
- **No Exaggerated Multimodal Claims:** Rather than claiming vision revolutionizes AQI forecasting, the paper publishes an **empirical negative result** exposing why opportunistic web photography cannot be naively spliced into continuous ground monitors.
- **Physical Constraints on Spatial Modeling:** Explicitly reports that in landlocked urban corridors (Delhi, Kolkata), spatial neighbor concatenation degrades accuracy by $-2.03\%$ to $-3.00\%$ due to local source dominance, rather than falsely claiming universal spatial transfer.
- **Pilot Trial Sample Size:** The paper notes the pilot nature of the live shadow deployment ($N=418$) and avoids asserting premature statistical significance where confidence intervals cross zero.

---

## 6. Pre-Submission Checklist

- [x] All 7 `.tex` table files compile without unescaped characters.
- [x] All 7 figure files are present at 300+ DPI in `paper/figures/`.
- [x] `paper/references.bib` contains 20 complete entries with proper citations.
- [x] No `TODO`, `TBD`, or placeholder text exists in `paper/main.tex`.
- [x] Author affiliations and department names are complete.

**Conclusion:** The manuscript is thoroughly verified, evidence-backed, and **READY FOR SUBMISSION**.
