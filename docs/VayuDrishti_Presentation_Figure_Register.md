# VayuDrishti Presentation Figure Register
**Official Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati - 522237, Andhra Pradesh, India  
**Project Guide:** Prof. Thota Ramathulsi, Assistant Professor, SCOPE  
**Project Team Members:**  
- TELIKI RUSHYA SREE (23BCE9562)  
- MACHIREDDY SWATHI (23BCE9566)  
- THANUJA VARAKALA (23BCE9608)  
- PALLI SHILPA (23BCE9567)  
**Academic Year:** 2026–2027  
**Associated Presentation:** `presentation/VayuDrishti_Final_Capstone_Review.pptx` (and `docs/VayuDrishti_Final_Capstone_Review.pptx`)  

---

## 1. Embedded Slide Presentation Figures

The table below catalogs the high-resolution figures directly embedded inside the 16 slides of `VayuDrishti_Final_Capstone_Review.pptx`.

| Slide # | Slide Heading | Figure File Name | Source File Path | Dimensions (Slide) | Visual Description & Scientific Purpose |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **08** | SYSTEM ARCHITECTURE: End-to-End Pipeline & Information Flow | `fig_arch_overall.png` | `paper/figures/fig_arch_overall.png` | Width: 4.60 in<br>Height: 3.90 in<br>(Left: 4.90", Top: 1.10") | **End-to-End System Flowchart:** Illustrates the complete information flow from OpenAQ v3 telemetry ingestion, missingness imputation, and feature extraction, through the dual predictive branches (LSTM 27 channels and LightGBM 49 features), the ResNet-50 visual feature extractor with bounded residual clamping, the Geodesic IDW spatial engine (45 km cutoff), to the split conformal prediction and Streamlit user interface. |
| **09** | IMPLEMENTATION: Interactive Streamlit Platform Architecture | `dashboard_monitoring_interface.png` | `paper/figures/dashboard_monitoring_interface.png` | Width: 4.60 in<br>Height: 3.90 in<br>(Left: 4.90", Top: 1.10") | **Operational Dashboard UI:** Screenshot of the live VayuDrishti web application showcasing the real-time CPCB telemetry monitor, pollutant gauge widgets, spatial interpolation heatmaps, and observation freshness badges across monitored Indian cities. |
| **13** | RESULTS: Empirical Performance & Benchmark Evaluations | `actual_vs_predicted.png` | `paper/figures/actual_vs_predicted.png` | Width: 3.25 in<br>Height: 1.80 in<br>(Left: 6.25", Top: 1.10") | **Empirical Correlation Scatterplot:** Scatter plot of observed versus predicted $\text{PM}_{2.5}$ and $\text{PM}_{10}$ concentrations on the held-out test evaluation set, demonstrating strong alignment along the $1:1$ ideal line ($R^2 = 0.88$ for $\text{PM}_{2.5}$ and $R^2 = 0.84$ for $\text{PM}_{10}$) with conformal error boundaries. |

---

## 2. Complete Repository Figure Register (Paper, Report & Presentation Assets)

In addition to the figures embedded in the slide deck, the project repository maintains a comprehensive suite of publication-grade figures in `paper/figures/`. These figures support the research paper, the 45-page capstone report, and viva defense discussions.

| Figure Identifier | File Name | Format | Dimensions / DPI | Generating Script | Description & Context in VayuDrishti |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Fig. Arch-1** | `fig_arch_overall.png` | PNG / PDF / SVG | $2400 \times 1600$ (300 DPI) | `generate_ieee_architecture_diagrams.py` | Complete end-to-end multi-tier architecture diagram. |
| **Fig. Arch-2** | `fig_arch_preprocessing.png` | PNG / PDF / SVG | $1800 \times 1200$ (300 DPI) | `generate_ieee_architecture_diagrams.py` | Data ingestion, forward imputation ($\le 3\text{h}$), and cyclic feature pipeline. |
| **Fig. Arch-3** | `fig_arch_lstm.png` | PNG / PDF / SVG | $1800 \times 1200$ (300 DPI) | `generate_ieee_architecture_diagrams.py` | 2-layer LSTM recurrent architecture with 27 input channels across 24 hours. |
| **Fig. Arch-4** | `fig_arch_resnet.png` | PNG / PDF / SVG | $1800 \times 1200$ (300 DPI) | `generate_ieee_architecture_diagrams.py` | ResNet-50 visual feature extractor and 2048-dim latent bottleneck. |
| **Fig. Arch-5** | `fig_arch_residual_hybrid.png`| PNG / PDF / SVG | $1800 \times 1200$ (300 DPI) | `generate_ieee_architecture_diagrams.py` | Bounded Residual Hybrid visual fusion with zero-fallback mask ($m=0$). |
| **Fig. Arch-6** | `fig_arch_ensemble.png` | PNG / PDF / SVG | $1800 \times 1200$ (300 DPI) | `generate_ieee_architecture_diagrams.py` | Weighted ensemble architecture combining LSTM and LightGBM outputs. |
| **Fig. Arch-7** | `fig_arch_idw_spatial.png` | PNG | $2400 \times 1500$ (300 DPI) | `generate_missing_report_diagrams.py` | Geodesic Haversine IDW interpolation engine with 45 km radius boundary. |
| **Fig. Arch-8** | `fig_arch_reliability.png` | PNG | $2400 \times 1500$ (300 DPI) | `generate_missing_report_diagrams.py` | Spatial reliability auditing flowchart with Clark-Evans $R$ and 5-tier freshness. |
| **Fig. Arch-9** | `fig_arch_conformal.png` | PNG | $2400 \times 1500$ (300 DPI) | `generate_missing_report_diagrams.py` | Split conformal prediction calibration pipeline achieving 90% coverage. |
| **Fig. Emp-1** | `actual_vs_predicted.png` | PNG / PDF | $1800 \times 1200$ (300 DPI) | `generate_publication_empirical_figures.py` | Scatterplot of ground truth vs predicted particulate concentrations. |
| **Fig. Emp-2** | `fig1_conformal_calibration.png`| PNG | $1800 \times 1200$ (300 DPI) | `generate_publication_empirical_figures.py` | Empirical coverage curve vs nominal confidence levels (90% target). |
| **Fig. Emp-3** | `fig2_seasonal_model_comparison.png`| PNG | $1800 \times 1200$ (300 DPI) | `generate_publication_empirical_figures.py` | Seasonal MAE performance across Winter, Summer, and Monsoon seasons. |
| **Fig. Emp-4** | `fig3_multistation_performance.png`| PNG | $1800 \times 1200$ (300 DPI) | `generate_publication_empirical_figures.py` | Station-wise MAE distributions across CPCB monitoring networks. |
| **Fig. Emp-5** | `fig4_regime_error_curve.png` | PNG | $1800 \times 1200$ (300 DPI) | `generate_publication_empirical_figures.py` | Model error trajectories across severe pollution and inversion regimes. |
| **Fig. Emp-6** | `confusion_matrices_combined.png`| PNG / PDF | $2000 \times 1000$ (300 DPI) | `generate_publication_empirical_figures.py` | 6-class AQI category confusion matrices for $\text{PM}_{2.5}$ and $\text{PM}_{10}$. |
| **Fig. Emp-7** | `prediction_timeseries.png` | PNG / PDF | $2400 \times 1200$ (300 DPI) | `generate_publication_empirical_figures.py` | 14-day continuous time-series comparing model predictions, ground truth, and conformal bands. |
| **Fig. Emp-8** | `dashboard_monitoring_interface.png`| PNG | $1920 \times 1080$ (300 DPI) | Full-resolution screenshot | Live Streamlit platform interface showing real-time telemetry and heatmaps. |

---

## 3. Formatting & Aesthetic Compliance Standards

All figures embedded in `VayuDrishti_Final_Capstone_Review.pptx` adhere to the following academic visualization guidelines:
1. **Aspect Ratio Preservation:** All embedded pictures maintain native aspect ratios without distortion or vertical stretching.
2. **Typography in Figures:** All embedded plots and diagrams use vector-quality typography (Helvetica / Arial / Times New Roman) with readable labels ($\ge 8\text{ pt}$ equivalent on slide).
3. **Color Palette Harmony:** Visual palettes strictly follow the university presentation scheme:
   - Primary Navy: `#102C57` / RGB(16, 44, 87)
   - Accent Teal: `#0E7490` / RGB(14, 116, 144)
   - Accent Blue: `#2563EB` / RGB(37, 99, 235)
   - Slate Neutral: `#F1F5F9` / RGB(241, 245, 249)
4. **Resolution Standards:** Bitmap images are rendered at $\ge 300\text{ DPI}$ to prevent blurriness during full-screen projection.
