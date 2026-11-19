# VayuDrishti Capstone Project Report – Figure Quality & Visual Layout Audit

**Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, India  
**Date:** November 19, 2026  
**Associated Documents:**  
- `docs/VayuDrishti_Final_Capstone_Report_Repaired.docx`  
- `docs/VayuDrishti_Final_Capstone_Report_Repaired.pdf`  

---

## 1. Overview of Figure Asset Verification

A complete visual audit was performed across all **17 figures** embedded in the VayuDrishti Capstone Project Report. The inspection verified:
- High rendering resolution (300+ DPI raster, vector PDF/SVG equivalents).
- Legible typography using standard clean sans-serif/serif faces at or above 9 pt.
- Generous internal padding and bounding boxes preventing any text clipping or bounding box collisions.
- Clear, unambiguous arrow flowcharts with explicit offset labels.
- Standard academic figure placement: figure centered, caption placed strictly **BELOW** the figure with `keep_with_next` formatting on the image paragraph to prevent orphaned captions across page breaks.

---

## 2. Complete Inventory & Status of Figures

| Fig # | Figure Description | Source File | Dimensions / Format | DPI / Quality | Status |
|---|---|---|---|---|---|
| **Fig. 2.1** | End-to-end VayuDrishti Tri-Modal Architecture | `paper/figures/fig_arch_overall.png` | $2700 \times 1350$ px | 300 DPI / Vector PDF/SVG | **Verified** |
| **Fig. 2.2** | Automated Multimodal Preprocessing Pipeline | `paper/figures/fig_arch_preprocessing.png` | $2700 \times 1050$ px | 300 DPI / Vector PDF/SVG | **Verified** |
| **Fig. 2.3** | Deep Computer Vision Pipeline (ResNet-50) | `paper/figures/fig_arch_resnet.png` | $2700 \times 1050$ px | 300 DPI / Vector PDF/SVG | **Verified** |
| **Fig. 2.4** | Deep 2-Layer Stacked LSTM Sequence Model | `paper/figures/fig_arch_lstm.png` | $2700 \times 1050$ px | 300 DPI / Vector PDF/SVG | **Verified** |
| **Fig. 2.5** | Multi-Stream Cross-Modal Attention Fusion | `paper/figures/fig_arch_fusion.png` | $2700 \times 1200$ px | 300 DPI / Vector PDF/SVG | **Verified** |
| **Fig. 2.6** | Residual-Corrected Blending Architecture | `paper/figures/fig_arch_residual_hybrid.png` | $2700 \times 1200$ px | 300 DPI / Vector PDF/SVG | **Verified** |
| **Fig. 2.7** | Spatial Interpolation Engine (IDW) | `paper/figures/fig_arch_idw_spatial.png` | $3000 \times 1200$ px | 300 DPI / Vector PDF/SVG | **REDESIGNED & VERIFIED** |
| **Fig. 2.8** | Data Freshness & Sensor Reliability Gate | `paper/figures/fig_arch_reliability.png` | $3000 \times 1200$ px | 300 DPI / Vector PDF/SVG | **REDESIGNED & VERIFIED** |
| **Fig. 2.9** | Uncertainty Quantification (Conformal Prediction) | `paper/figures/fig_arch_conformal.png` | $3000 \times 1200$ px | 300 DPI / Vector PDF/SVG | **REDESIGNED & VERIFIED** |
| **Fig. 4.1** | Multimodal vs. Unimodal Error Distribution | `paper/figures/fig_results_error_dist.png` | $2400 \times 1600$ px | 300 DPI High-Res | **Verified** |
| **Fig. 4.2** | 24-Hour Forecast Horizon Error Trajectory | `paper/figures/fig_results_horizon_rmse.png` | $2400 \times 1500$ px | 300 DPI High-Res | **Verified** |
| **Fig. 4.3** | Cross-Modal Attention Weight Map Across Hours | `paper/figures/fig_results_attention_heatmap.png`| $2400 \times 1600$ px | 300 DPI High-Res | **Verified** |
| **Fig. 4.4** | Empirical Conformal Prediction Intervals | `paper/figures/fig_results_conformal_series.png` | $2700 \times 1500$ px | 300 DPI High-Res | **Verified** |
| **Fig. 4.5** | Global SHAP Feature Importance Summary | `paper/figures/fig_results_shap_beeswarm.png` | $2400 \times 1800$ px | 300 DPI High-Res | **Verified** |
| **Fig. 4.6** | Live Operational Station Map & Spatial Contours | `paper/figures/fig_results_spatial_map.png` | $2700 \times 1650$ px | 300 DPI High-Res | **Verified** |
| **Fig. 4.7** | VayuDrishti Live Streamlit Production UI | `docs/images/dashboard_screenshot.png` | $2400 \times 1350$ px | 300 DPI Uncompressed | **Verified** |
| **Fig. A.1** | Live API Data Ingestion & Microservice Schema | `paper/figures/fig_arch_pipeline_flow.png` | $2700 \times 1200$ px | 300 DPI / Vector PDF/SVG | **Verified** |

---

## 3. Detailed Redesign Analysis for Challenging Diagrams

In previous drafts of the report, Figures 2.7, 2.8, and 2.9 suffered from narrow card dimensions, label collisions, and overlapping connector arrows. The visual generator script (`scripts/generate_ieee_architecture_diagrams.py`) was completely overhauled to resolve these issues.

### 3.1 Figure 2.7: Spatial Inverse Distance Weighting (IDW) Workflow
- **Defects in Previous Version:**
  - The Euclidean search radius ($R_{\max} = 45\text{ km}$) collided with the station node icons.
  - The power decay equation $w_i = d_i^{-p}$ was squeezed inside an undersized card.
  - The fallback branch ("If Active Stations < 3") was ambiguous and lacked a clear terminal state.
- **Redesign Actions Taken:**
  - Canvas extended to $10.0 \times 4.0$ inches at 300 DPI ($3000 \times 1200$ px).
  - Four distinct, clearly demarcated workflow cards:
    1. **Target Query Location:** Displays $(x_0, y_0)$ coordinates and active telemetry radius.
    2. **Radial Distance Filtering:** Displays explicit condition $d_i \le 45\text{ km}$, identifying active neighborhood $\mathcal{N}(x_0)$.
    3. **Weight Computation:** Displays power-decay weighting $w_i = d_i^{-2.0} / \sum d_j^{-2.0}$.
    4. **Interpolated Surface:** Displays spatial estimate $\hat{y}_{\text{spatial}}(x_0) = \sum w_i y_i$ with fallback indicator.
  - Connector arrows now have generous horizontal gaps (0.75 in), with floating offset text labels that do not collide with card borders.

### 3.2 Figure 2.8: Telemetry Ingestion, Freshness & Sensor Reliability Gate
- **Defects in Previous Version:**
  - The circular buffer diagram overlapped with the time-to-live (TTL) staleness condition ($t_{\text{diff}} > 3\text{ hours}$).
  - Reliability scoring math was partially clipped at the bottom margin.
- **Redesign Actions Taken:**
  - Expanded into a linear 4-stage pipeline with dedicated color-coded health badges:
    1. **Raw Ingestion Stream:** Incoming CPCB API / live scraper JSON payloads.
    2. **Freshness Verification:** Strict TTL threshold check ($t_{\text{current}} - t_{\text{sensor}} \le 180\text{ min}$).
    3. **Reliability Scoring Engine:** Combined formula card integrating uptime ratio, missingness penalty, and physical range sanity checks ($0 \le \text{PM}_{2.5} \le 1000\ \mu\text{g/m}^3$).
    4. **Operational Station Dispatch:** Dynamic gate bifurcating stations into *Active Weights* ($R_i \ge 0.70$) vs. *Suppressed / Quarantine*.
  - Every arrow connection utilizes `FancyArrowPatch` with curve-free linear paths and high-contrast labeling.

### 3.3 Figure 2.9: Split Conformal Prediction Uncertainty Calibration
- **Defects in Previous Version:**
  - Nonconformity score histogram was cramped and superimposed over the calibration set splitting text.
  - Quantile formula $\hat{q}_{\alpha}$ overlapped with the output prediction interval bracket.
- **Redesign Actions Taken:**
  - Restructured into a clear three-step inductive inference pipeline:
    1. **Model Residual Scoring (Calibration Set $\mathcal{D}_{\text{cal}}$):** Explicit definition of normalized nonconformity score $s_i = |y_i - \hat{\mu}_i| / \hat{\sigma}_i$.
    2. **Empirical Quantile Computation:** Clear mathematical representation of the finite-sample adjusted conformal quantile:
       $$\hat{q}_{1-\alpha} = \text{Quantile}\left(1-\alpha; \left\{s_i\right\}_{i=1}^n \times \frac{n+1}{n}\right)$$
    3. **Guaranteed Prediction Interval Construction:** Crisp visualization of the symmetric coverage bounds:
       $$C_{1-\alpha}(x_{\text{test}}) = \left[\hat{\mu}(x) - \hat{q}_{1-\alpha} \cdot \hat{\sigma}(x),\ \hat{\mu}(x) + \hat{q}_{1-\alpha} \cdot \hat{\sigma}(x)\right]$$
  - Color palette updated to soft slate blue and emerald accents, ensuring maximum clarity in monochrome or color printing.

---

## 4. Empirical Result Visualizations & Dashboard Capture

- **Fig. 4.1 to Fig. 4.6:** Generated directly from verified test set evaluations and SHAP TreeExplainer runs. All axis labels ($PM_{2.5}$ concentration in $\mu\text{g/m}^3$, Forecast Horizon in Hours, SHAP value impact) are rendered at 10 pt bold with clear gridlines and high-contrast colorblind-friendly palettes.
- **Fig. 4.7 (Live Production UI):** Embedded with native aspect ratio ($16:9$, unscaled width 6.0 in) showing live telemetry gauges, the interactive Leaflet air quality heat map, and conformal uncertainty band overlays.

---

## 5. Verification Conclusion

All 17 figures in `VayuDrishti_Final_Capstone_Report_Repaired.docx` and `VayuDrishti_Final_Capstone_Report_Repaired.pdf` satisfy high-standard academic and IEEE publication requirements:
- Zero overlapping labels or colliding arrows.
- High visual contrast and clarity.
- Unified typographic hierarchy.
- Flawless document layout with zero orphaned captions.
