# VayuDrishti Capstone Project Report – Comprehensive Audit & Repair Report

**Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, India  
**Degree:** Bachelor of Technology in Computer Science and Engineering  
**Project Guide:** Prof. Thota Ramathulsi  
**Capstone ID:** F5815A89  
**Student Team:**  
- TELIKI RUSHYA SREE (23BCE9562)  
- MACHIREDDY SWATHI (23BCE9566)  
- THANUJA VARAKALA (23BCE9608)  
- PALLI SHILPA (23BCE9567)  

**Date:** November 19, 2026  
**Audited & Repaired Document:** `docs/VayuDrishti_Final_Capstone_Report_Repaired.docx`  
**Compiled PDF Document:** `docs/VayuDrishti_Final_Capstone_Report_Repaired.pdf`  

---

## 1. Executive Summary of Audit & Repairs

This audit documents the complete technical, structural, visual, and typographical repair of the VayuDrishti Capstone Project Report. The original manuscript exhibited minor formatting inconsistencies, diagram collisions, and structural deviations from standard academic submission requirements.

The finalized, repaired document adheres strictly to the following standards:
1. **Classical Academic Typography & Theme:** 100% white page background, high-contrast black typography, Times New Roman throughout, standard 1.0-inch margins (left, right, top, bottom), 1.15 line spacing, and 0.35-inch first-line paragraph indentation. All non-academic presentation elements (colored banner cards, dark backgrounds, decorative callout boxes) were eliminated.
2. **Table of Contents Redesign:** Rebuilt to match the university reference specification (`Screenshot_20261119_113702_Word.jpg`). All entries feature exact right-aligned page numbers (at 6.50 inches / 9360 dxa) joined by continuous dotted tab leaders (`w:leader="dot"`).
3. **Dual Pagination System:**
   - **Section 1 (Front Matter):** Numbered in lower-case Roman numerals (`ii` through `x`), with the title page correctly set to unnumbered ("Different First Page").
   - **Section 2 (Main Report Body):** Separated by an XML continuous section break, restarting at Arabic page `1` and spanning sequentially through page `35`.
4. **Figure Redesign & High-Resolution Vector Assets:** Addressed critical visual flaws in diagrammatic workflows. Figures 2.7 (Spatial IDW), 2.8 (Freshness & Telemetry Reliability), and 2.9 (Split Conformal Prediction) were completely redesigned and regenerated with generous spacing, zero text collisions, and clear flowcharts.
5. **Rigorous Scientific & Empirical Fact-Checking:** Synchronized all numerical quantities, feature counts, dataset rows, statistical p-values, spatial search bounds, and conformal coverage metrics across all chapters, tables, and appendices.

---

## 2. Document Hierarchy & Page Structure Audit

The repaired document contains exactly 45 physical pages structured into two distinct Word XML sections:

### Section 1: Preliminary Matter (Roman Numerals ii–x)
| Preliminary Item | Physical Page | Displayed Page No. | Verification Status |
|---|---|---|---|
| Title Page / Cover Sheet | Page 1 | Unnumbered (Different First Page) | Verified (Times New Roman, Capstone ID F5815A89) |
| Certificate of Originality | Page 2 | ii | Verified (Guide & Review Committee Signatures) |
| Acknowledgements | Page 3 | iii | Verified (Formal Guide & Faculty Appreciation) |
| Abstract | Page 4–5 | iv–v | Verified (2-page detailed technical abstract) |
| Table of Contents | Page 6–7 | vi–vii | Verified (Exact match to reference screenshot) |
| List of Tables | Page 8 | viii | Verified (10 formal academic tables listed) |
| List of Figures | Page 9 | ix | Verified (17 formal figures listed) |
| List of Abbreviations | Page 10 | x | Verified (Alphabetical acronym definitions) |

### Section 2: Main Chapters, Appendices & References (Arabic Numerals 1–35)
| Chapter / Component | Physical Pages | Arabic Page Range | Exact Sections Covered |
|---|---|---|---|
| **Chapter 1: Introduction** | Pages 11–14 | pp. 1–4 | 1.1 Background & Motivation<br>1.2 Problem Statement<br>1.3 Objectives of the Project<br>1.4 Project Scope and Boundaries<br>1.5 Key Contributions<br>1.6 Organization of the Report |
| **Chapter 2: System Architecture & Methodology** | Pages 15–22 | pp. 5–12 | 2.1 Multimodal Ingestion Pipeline<br>2.2 Computer Vision Pipeline (ResNet-50)<br>2.3 Deep Sequence Modeling (Bi-LSTM)<br>2.4 Tabular Gradient Boosting (LightGBM)<br>2.5 Residual-Corrected Blending Architecture<br>2.6 Spatial Interpolation (IDW Engine)<br>2.7 Data Freshness & Sensor Reliability Gate<br>2.8 Uncertainty Quantification (Conformal Prediction) |
| **Chapter 3: Dataset Curation & Preprocessing** | Pages 23–26 | pp. 13–16 | 3.1 Historical Telemetry Ingestion (CPCB)<br>3.2 Live Telemetry Scraping & Ingestion<br>3.3 Ground-Level Imagery Dataset (PM25Vision)<br>3.4 Synchronous Cross-Modal Dataset Assembly<br>3.5 Feature Engineering Pipeline<br>3.6 Data Cleansing & Missing Value Imputation |
| **Chapter 4: Experimental Results & Analysis** | Pages 27–37 | pp. 17–27 | 4.1 Evaluation Setup & Metrics<br>4.2 Benchmark Model Comparisons<br>4.3 Ablation Studies (Sensor Modalities)<br>4.4 Spatial Interpolation Validation<br>4.5 Conformal Prediction Coverage & Uncertainty<br>4.6 Interpretability & Feature Attribution (SHAP)<br>4.7 Live Deployment & Operational Latency |
| **Chapter 5: Conclusion & Future Scope** | Pages 38–39 | pp. 28–29 | 5.1 Concluding Remarks<br>5.2 Summary of Major Achievements<br>5.3 Operational Limitations & Failure Modes<br>5.4 Future Research Directions<br>5.5 Final Project Contribution Statement |
| **Appendix A: Supplementary Documentation** | Pages 40–42 | pp. 30–32 | A.1 Complete Feature Dictionary & Transformations<br>A.2 Hyperparameter Configurations<br>A.3 Live API Endpoint Specifications & Schema |
| **References** | Pages 43–44 | pp. 33–34 | 25 Formal IEEE Peer-Reviewed Citations |
| **Biodata of Authors** | Page 45 | p. 35 | Author profiles, registration numbers & roles |

---

## 3. Detailed Scientific Consistency & Fact-Checking Audit

Every empirical claim and numerical constant in the report was audited against the codebase and empirical logs:

### 3.1 Dataset Sizes and Sample Counts
- **CPCB Historical Telemetry:** Exactly **1,369,779 hourly observations** across major Indian urban centers spanning 2015–2024.
- **Recent Telemetry Corpus:** >3.3 million rows ingested from the CPCB open data portal and CPCB live scrapers.
- **PM25Vision Ground-Level Camera Dataset:** Exactly **1,235 validated skyward images** matched with ground truth $PM_{2.5}$ monitor telemetry.
- **Synchronous Cross-Modal Validation Dataset:** Exactly **199 synchronous image-telemetry pairs** spanning 18 distinct monitoring stations across Delhi-NCR and northern India.
- **Live Deployment Paired Test Corpus:** Exactly **418 live evaluation pairs** harvested across 4 active operational stations over 7 consecutive days.

### 3.2 Feature Dimensions & Model Inputs
- **Bi-LSTM Sequence Model:** Ingests a sliding temporal window of 24 time steps $\times$ **27 normalized numerical features** (pollutant concentrations, meteorological variables, temporal sine/cosine harmonics, and rolling trends).
- **LightGBM Tabular Booster:** Ingests **49 engineered features**, comprising 27 base features plus 22 multi-scale rolling statistics (1h, 3h, 6h, 12h, 24h rolling means, standard deviations, min/max spreads, and rate-of-change ratios).

### 3.3 Live Deployment Statistical Audit
- **Telemetry-only RMSE:** $18.42\ \mu\text{g/m}^3$.
- **Multimodal RMSE:** $17.86\ \mu\text{g/m}^3$ (a 3.04% marginal error reduction).
- **Paired Two-Tailed t-test:** $p = 0.48$.
- **Wilcoxon Signed-Rank Test:** $p = 0.62$.
- **Academic Defense / Explanation:** The difference in live evaluation does not achieve statistical significance ($p > 0.05$). This is honestly documented in Chapter 4 and Chapter 5 as an expected consequence of sample size ($N = 418$ across 4 stations) and weather stability during the test window. The report emphasizes this transparently as evidence of academic integrity.

### 3.4 Spatial Bounds & Interpolation Parameters
- **Maximum Spatial Cutoff Radius ($R_{\max}$):** Exactly **45.0 km**.
- **Inverse Distance Weighting Power Parameter ($p$):** Exactly **2.0**.
- **Minimum Operational Anchor Requirement:** At least **3 reporting stations** within 45 km to compute an IDW surface; otherwise falls back to nearest-station observation with an expanded uncertainty flag.

### 3.5 Uncertainty Quantification & Conformal Coverage
- **Calibration Framework:** Inductive Split Conformal Prediction using normalized nonconformity scores:
  $$s_i = \frac{|y_i - \hat{\mu}(x_i)|}{\hat{\sigma}(x_i)}$$
- **Nominal Coverage vs. Empirical Realization:**
  - Nominal 80% confidence level: Empirical coverage = **87.2%**
  - Nominal 90% confidence level: Empirical coverage = **94.8%**
  - Nominal 95% confidence level: Empirical coverage = **98.2%**
  - Live deployment empirical coverage: **99.04%** (reflecting conservative calibration quantile estimation on non-stationary peak episodes).

### 3.6 Hardware Environment & Production Latency
- **Inference Hardware:** CPU-only deployment on Intel Core i7-11800H @ 2.30 GHz (8 cores, 16 threads, 16 GB DDR4 RAM), without discrete GPU acceleration.
- **Inference Latency Profile:**
  - Computer Vision (ResNet-50 feature extraction on $224 \times 224$ image): **42.1 ms**
  - Sequence Modeling (Bi-LSTM on $24 \times 27$ tensor): **8.6 ms**
  - Tabular Gradient Boosting (LightGBM on 49 features): **3.2 ms**
  - Blending & Split Conformal Bounds: **1.8 ms**
  - End-to-End Pipeline Latency: **55.7 ms** per query (well under the 200 ms SLA).

---

## 4. Typography & Layout Specifications

| Element | Font Face | Size | Styling | Alignment | Spacing / Indentation |
|---|---|---|---|---|---|
| Document Title | Times New Roman | 24 pt | Bold, Uppercase | Centered | 18 pt after |
| Chapter Headings | Times New Roman | 16 pt | Bold | Centered | 12 pt before, 12 pt after |
| Section Headings (X.Y) | Times New Roman | 14 pt | Bold | Left-aligned | 12 pt before, 6 pt after |
| Subsection Headings (X.Y.Z) | Times New Roman | 12 pt | Bold | Left-aligned | 6 pt before, 4 pt after |
| Body Text Paragraphs | Times New Roman | 12 pt | Regular | Justified | 1.15 line spacing, 0.35 in indent, 4 pt after |
| Table Captions | Times New Roman | 11 pt | Bold | Centered | Placed **ABOVE** table, 6 pt before, 4 pt after |
| Figure Captions | Times New Roman | 10.5 pt | Bold Label + Regular text | Centered | Placed **BELOW** figure, 4 pt before, 12 pt after |
| Table Cells (Header) | Times New Roman | 10 pt | Bold | Centered / Left | Shading `#E2E8F0`, padding 6 pt top/bottom |
| Table Cells (Body) | Times New Roman | 9.5 pt | Regular | Centered / Left | Borders `#CCCCCC`, padding 4 pt top/bottom |

---

## 5. Summary of Deliverables

1. **Repaired Microsoft Word Document:**  
   [VayuDrishti_Final_Capstone_Report_Repaired.docx](file:///e:/Documents/multimodal_air_quality_vscode_starter/docs/VayuDrishti_Final_Capstone_Report_Repaired.docx)  
   Complete 45-page editable document with embedded vector/high-res figures, XML dotted-leader TOC, dual-section pagination, and standard academic typography.

2. **Repaired Publication PDF:**  
   [VayuDrishti_Final_Capstone_Report_Repaired.pdf](file:///e:/Documents/multimodal_air_quality_vscode_starter/docs/VayuDrishti_Final_Capstone_Report_Repaired.pdf)  
   Direct COM export from Microsoft Word ensuring 100% fidelity, exact page numbers, vector line art, and print-ready quality.

3. **Figure Audit Document:**  
   [VayuDrishti_Figure_Quality_Audit.md](file:///e:/Documents/multimodal_air_quality_vscode_starter/docs/VayuDrishti_Figure_Quality_Audit.md)  
   Inspection details for all 17 figures.
