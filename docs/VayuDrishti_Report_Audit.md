# VayuDrishti Capstone Project Report: Comprehensive Academic & Technical Audit

**Document Audited:** `docs/VayuDrishti_Capstone_Project_Report.docx`  
**Corrected Output:** `docs/VayuDrishti_Final_Capstone_Project_Report.docx`  
**Audit Date:** November 18, 2026  
**Auditor:** Senior Academic Research Editor & Environmental Data Scientist  
**Institutional Affiliation:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati  

---

## 1. Executive Summary of Revisions & Corrections

This audit was conducted to transform the existing capstone project report into an academically defensible, technically accurate, and professionally formatted document suitable for submission to VIT-AP University and rigorous viva examination. 

The audit identified and resolved several critical areas:
1. **Title Alignment:** The report title was updated from the outdated *"A Residual Multimodal Hybrid Framework for Air Quality Forecasting"* to the comprehensive official title: **"VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments"** across all sections, headers, and certificates.
2. **Elimination of Fabricated Hardware Claims:** Removed unverified claims regarding dedicated NVIDIA GeForce RTX 4070 Laptop GPUs (12 GB VRAM) and Intel Core i7-13700H workstations. Documented the authentic local development environment: **Intel Core i7-1165G7 CPU @ 2.80 GHz with integrated Intel Iris Xe graphics, 16 GB RAM, and CPU-compatible PyTorch/LightGBM execution**.
3. **Multimodal Negative Transfer Qualification:** Eliminated unsupported claims of "proven nationwide multimodal forecasting superiority" or "guaranteed physical accuracy improvements from visual fusion." Accurately reported that opportunistic image matching on the PM25Vision dataset (199 aligned pairs across 18 stations) resulted in **catastrophic negative transfer (+12.19 $\mu\text{g/m}^3$ error inflation in naive early concatenation)**, thereby justifying the bounded residual hybrid architecture with automatic fallback ($m=0 \implies \delta=0$).
4. **Live Telemetry Evaluation Integrity:** Accurately characterized the 14-day prospective live evaluation across four stations (418 matched pairs). Acknowledged that the GBDT model's slight gain over 1-hour persistence ($\text{PM}_{2.5}$ MAE 9.64 vs. 9.86, $p=0.48$; $\text{PM}_{10}$ MAE 22.13 vs. 22.35, $p=0.62$) was **statistically non-significant**, explaining the high temporal autocorrelation of urban particulate dispersion over short 1-hour horizons.
5. **Spatial Generalization Boundaries:** Clarified that the verified 20.0% to 25.4% error reduction of geodesic Inverse Distance Weighting (IDW) over nearest-neighbor baselines was established in a **Leave-One-Station-Out holdout experiment concentrated in Delhi NCR**, refraining from extrapolating this claim nationwide.
6. **Separation of Distinct Dataset Cohorts:** Maintained strict methodological separation between the historical 2015–2020 CPCB benchmark (1,369,779 records, 90 stations, test MAE 10.57 $\mu\text{g/m}^3$) and the recent 2020–2025 multi-year adaptation cohort (3.3M records, 85 stations, test MAE 6.62 $\mu\text{g/m}^3$), noting that post-COVID structural baseline shifts prevent treating these as an apple-to-apple comparison.
7. **Forecast Horizon Clarification:** Explicitly distinguished the **24-hour input lookback sequence** from the strictly **next-hour ($t+1$) forecast horizon**, eliminating any ambiguity that the model was forecasting 24 hours into the future.
8. **Conformal Coverage vs. Precision:** Documented that high empirical split conformal coverage (~99%) is achieved through distance-scaled interval expansion, which produces wide intervals (up to 48.3 $\mu\text{g/m}^3$ for $\text{PM}_{2.5}$) during severe episodic winter smog.

---

## 2. Technical Claims Verification Matrix

| # | Technical Dimension | Status in Original Draft | Audited & Corrected Status | Underlying Source Artifact |
|---|---|---|---|---|
| **1** | **Project Title** | "A Residual Multimodal Hybrid Framework..." | Updated to "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments" | Master Project Specification |
| **2** | **Project Guide** | "Dr. Thota Ramathulsi" | Corrected to **"Prof. Thota Ramathulsi"** (*strictly Prof., not Dr.*) | Institutional Faculty Registry |
| **3** | **Historical Dataset** | 1.37M records (ends July 2020) | Verified: **1,369,779 cleaned records, 90 stations, 2015–July 1, 2020** | `data/cpcb_historical_clean.csv` |
| **4** | **Historical LSTM Results** | $\text{PM}_{2.5}$ MAE 10.5719, $R^2$ 0.9492; $\text{PM}_{10}$ MAE 18.8643, $R^2$ 0.9301 | **Verified and Retained**: Exact historical test benchmark (404,431 sequences) | `results/historical_lstm_eval.json` |
| **5** | **Historical Severity-Gated** | Reported in some sections, absent in others | **Verified and Documented**: $\text{PM}_{2.5}$ MAE 10.3508, $\text{PM}_{10}$ MAE 18.1369 | `results/severity_gated_eval.json` |
| **6** | **Recent Dataset Adaptation** | Described vaguely as "recent data" | **Verified**: 3.3M cleaned records, 85 stations, July 2020–Sept 2025; Adapted MAE: $\text{PM}_{2.5}$ 6.6216, $\text{PM}_{10}$ 16.1660 | `results/adapted_ensemble_eval.json` |
| **7** | **Multimodal Vision Cohort** | Claimed large-scale visual fusion | **Corrected**: 199 verified aligned image-sensor pairs across 18 stations; early fusion degraded by +12.19 $\mu\text{g/m}^3$; residual hybrid fallback verified | `results/multimodal_alignment_audit.json` |
| **8** | **Live Telemetry Audit** | Implied significant real-world superiority | **Corrected**: 4 stations, 418 matched pairs; GBDT MAE 9.64 vs Persistence 9.86 ($p=0.48$, non-significant) | `results/live_telemetry_audit_14day.json` |
| **9** | **Delhi Spatial Interpolation** | Generalized as nationwide spatial accuracy | **Corrected**: Geodesic IDW ($p=2.0$) vs Nearest-Neighbor: $\text{PM}_{2.5}$ MAE 24.07 vs 30.06 (-20.0%); $\text{PM}_{10}$ 52.76 vs 70.68 (-25.4%) in Delhi holdout | `results/delhi_loso_spatial_eval.json` |
| **10** | **31-City Spatial Audit** | Mixed up active vs registered monitors | **Corrected**: 586 catalogued stations nationwide, 349 within defined boundaries; Clark-Evans $R$ index; 5 operational reliability tiers | `results/phase16_station_reliability_audit.json` |
| **11** | **Conformal Calibration** | Claimed absolute guaranteed precision | **Corrected**: Empirical coverage 99.37% ($\text{PM}_{2.5}$) and 99.16% ($\text{PM}_{10}$) with distance-scaled interval expansion | `results/conformal_calibration_metrics.json` |
| **12** | **Development Hardware** | Claimed NVIDIA RTX 4070 / i7-13700H | **Corrected**: Intel Core i7-1165G7 CPU, Intel Iris Xe graphics, 16 GB RAM, CPU-compatible execution | Local System Diagnostics |
| **13** | **Deployment Expenditure** | Stated arbitrary hardware costs | **Corrected**: Open-source stack, public APIs (OpenAQ v3), free Streamlit Cloud hosting (Direct Expenditure: INR 0.00) | Development Accounting Audit |

---

## 3. Detailed Audit of Claims Removed or Qualified

### A. Hardware & Compute Environment
- **Removed Claim:** *"The model was trained on a dedicated high-performance deep learning workstation featuring an Intel Core i7-13700H processor and an NVIDIA GeForce RTX 4070 Laptop GPU with 12 GB of GDDR6 VRAM running CUDA 12.1."*
- **Audit Finding:** The local physical machine is an **Intel Core i7-1165G7 laptop with integrated Intel Iris Xe graphics and 16 GB RAM**. No dedicated NVIDIA GPU or CUDA driver was active.
- **Correction Applied:** Documented the genuine CPU execution environment in Section 2.4.2 and Section 3.1. Emphasized that all models are engineered for CPU compatibility and low-latency inference on commodity hardware.

### B. Multimodal Image Fusion
- **Removed Claim:** *"Experimental results demonstrate that visual feature integration from outdoor cameras provides statistically significant accuracy gains for operational next-hour forecasting across Indian cities."*
- **Audit Finding:** In reality, the PM25Vision image dataset exhibits severe timestamp and coordinate misalignment with physical CAAQMS stations. In an unconstrained joint concatenation model, early fusion resulted in **catastrophic negative transfer**, increasing $\text{PM}_{2.5}$ MAE by **+12.19 $\mu\text{g/m}^3$** (from 10.57 to 22.76 $\mu\text{g/m}^3$).
- **Correction Applied:** Re-positioned multimodal learning in Section 2.7 and Section 4.3 as an **exploratory research branch**. Clearly explained the necessity of the Bounded Residual Hybrid architecture, where visual corrections are restricted to a bounded envelope $\Delta \in [-25, +25]\ \mu\text{g/m}^3$ and automatically fall back to pure sensor predictions ($m=0 \implies \delta=0$) when image quality or alignment is compromised.

### C. Prospective Live Telemetry Audit
- **Removed Claim:** *"Live field deployment validates the definitive superiority of the GBDT forecasting engine over operational baselines in real-world metropolitan air quality monitoring."*
- **Audit Finding:** The prospective 14-day live audit on four Delhi stations yielded 418 matched prediction-observation pairs. The GBDT model achieved an MAE of 9.64 $\mu\text{g/m}^3$ compared to 9.86 $\mu\text{g/m}^3$ for 1-hour persistence. A paired t-test produced $p = 0.48$ for $\text{PM}_{2.5}$ and $p = 0.62$ for $\text{PM}_{10}$.
- **Correction Applied:** In Section 4.4, explicitly acknowledged that this difference is **statistically non-significant**. Explained the physical rationale: over a short 1-hour horizon ($t+1$), urban air pollution displays extreme temporal autocorrelation ($\rho > 0.95$), making persistence an exceptionally competitive benchmark. Emphasized that establishing true operational superiority requires multi-season longitudinal validation.

### D. Nationwide Spatial Interpolation
- **Removed Claim:** *"Inverse Distance Weighting establishes high-precision spatial air quality estimation across all 31 Indian urban locations."*
- **Audit Finding:** Spatial holdout validation was conducted exclusively in **Delhi NCR** due to its relatively dense monitoring network (98 catalogued stations, 61 with complete 24h histories). Non-metro cities (e.g., Patna, Ranchi, Amaravati, Visakhapatnam) suffer from severe sensor clustering or have fewer than 3 active monitors.
- **Correction Applied:** Clarified in Section 2.8 and Section 4.5 that IDW validation is geographically concentrated in Delhi NCR. Integrated the 31-city Clark-Evans spatial geometry audit in Section 4.6, explicitly categorizing cities into 5 reliability tiers and enforcing sparse-network safeguards (suppressing continuous surfaces when fewer than 3 monitors are available within 45 km).

### E. Conformal Calibration Guarantees
- **Removed Claim:** *"Conformal prediction guarantees narrow, highly precise prediction intervals across all geographic zones."*
- **Audit Finding:** Conformal prediction guarantees *marginal coverage*, not narrow interval width. While empirical coverage reached 99.37% on $\text{PM}_{2.5}$, the average interval width expanded to 48.3 $\mu\text{g/m}^3$ (and up to 131.5 $\mu\text{g/m}^3$ for $\text{PM}_{10}$) during severe winter pollution episodes.
- **Correction Applied:** Section 2.10 and Section 4.7 now explicitly discuss the fundamental tradeoff between coverage and precision, clarifying that wide intervals during severe inversion episodes accurately reflect genuine atmospheric unpredictability.

---

## 4. Remaining Author Confirmations

The following minor items are flagged for final author verification prior to binding and submission:
1. **CAPSTONE ID Confirmation:** The SDP identifier placeholder `F5815A89` appears on the Title Page and Certificate. The student team should insert their assigned university SDP project ID.
2. **Compact Disk (CD) Attachment:** In accordance with the mandatory institutional note on the Biodata page, verify that the bound hard copy contains the digital submission CD affixed inside the back cover envelope, containing all Python source codes, model weight checkpoints, presentation slides, and the 5–10 minute dashboard demonstration video.
3. **Streamlit Cloud Active Deployment:** Verify that the public deployment URL (`https://vayudrishti-airquality.streamlit.app/`) remains awake and accessible during the scheduled examination turn.

---

## 5. Auditor Conclusion

The revised report, **`docs/VayuDrishti_Final_Capstone_Project_Report.docx`**, is fully compliant with the highest standards of scientific integrity, IEEE technical writing style, and VIT-AP University academic regulations. All mathematical formulations, dataset volumes, empirical benchmarks, and architectural workflows are fully verified against the project repository and are completely defensible during faculty viva examination.
