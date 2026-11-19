# VayuDrishti Capstone Review Presentation: Light-Theme Design & Verification Audit

**Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**Subtitle:** Air Quality Forecasting and Location-Based Pollution Mapping Using Multiple Data Sources  
**Capstone Project ID:** `F5815A89`  
**Academic Year:** 2025–2026  
**Institution:** School of Computer Science & Engineering (SCOPE), VIT-AP University, Amaravati  
**Project Supervisor:** Prof. Thota Ramathulsi, Assistant Professor, SCOPE  
**Project Team:**
- Teliki Rushya Sree (Reg. No: 23BCE9562)
- Machireddy Swathi (Reg. No: 23BCE9566)
- Thanuja Varakala (Reg. No: 23BCE9608)
- Palli Shilpa (Reg. No: 23BCE9567)

---

## 1. Executive Summary & Design Transformation

The VayuDrishti final capstone presentation has undergone a complete transformation from the previous dark-themed version (`VayuDrishti_Final_Review_Layout_Fixed.pptx`) to an intentional, academic, high-readability **Light Theme Presentation** (`VayuDrishti_Final_Review_Light_Theme.pptx`).

The redesigned presentation has been engineered from the ground up to satisfy rigorous academic standards suitable for a B.Tech CSE Final Capstone Viva Defense. Rather than simply inverting colors, the entire visual hierarchy, typography, card layout, contrast, and figure presentation were redesigned using a structured mathematical design system in `python-pptx` with two-layer container architecture.

---

## 2. Academic Light-Theme Design System Specifications

| Element | Specification | Hex / RGB Value | Purpose & Rationale |
|---|---|---|---|
| **Primary Background** | Pure Crisp White | `#FFFFFF` (255, 255, 255) | Maximum contrast, traditional academic thesis aesthetic, optimal projector clarity. |
| **Alternate Section Fill** | Soft Ice Blue | `#F3F7FC` (243, 247, 252) | Subtle visual resting zones for secondary cards and alternating table rows. |
| **Card Background** | Off-White Slate Tint | `#F8FAFD` (248, 250, 253) | Delicately separates structural cards from the white slide canvas without heavy borders. |
| **Card Borders** | Refined Subtle Grey | `#DCE3EA` (220, 227, 234) | 1.0 pt clean boundary framing each logical grouping. |
| **Primary Headings** | Deep Academic Navy | `#15345B` (21, 52, 91) | High visual authority, matches university branding and professional research journals. |
| **Secondary Headings** | Executive Cobalt Blue | `#2463A6` (36, 99, 166) | Sub-headings, section callouts, and key bold concept prefixes. |
| **Accent Highlights** | Environmental Teal | `#159A9C` (21, 154, 156) | Category trackers, live API badges, and technical parameter callouts. |
| **Pill / Badge Fill** | Soft Sky Blue | `#DCEBFA` (220, 235, 250) | Review badge, slide number indicators, and metric highlight cards. |
| **Body Typography** | Dark Charcoal | `#263445` (38, 52, 69) | Superior readability compared to pure black; eliminates harsh optical glare on projectors. |
| **Secondary Text** | Muted Slate Grey | `#667085` (102, 112, 133) | Subtitles, formal citations, and parenthetical metadata. |
| **Success Indicators** | Muted Sage Green | `#31856A` (49, 133, 106) | Conformal coverage milestones and operational validation highlights. |
| **Warning / Hazards** | Warm Amber | `#D49A36` (212, 154, 54) | Urban bottlenecks, sensor downtime risks, and hardware blindspots. |

### Typography Hierarchy
- **Slide Action Titles:** 16.0 pt Calibri Bold (`#15345B`), single-line, zero wrapping, strictly left-anchored at `X = 0.60 in` with safe margin before the slide badge at `X = 7.45 in`.
- **Category Trackers:** 8.5 pt Calibri Bold uppercase (`#159A9C`) tracking the thesis section.
- **Card Titles:** 10.5–12.0 pt Calibri Bold (`#15345B`).
- **Body & Bullet Text:** 7.4–8.5 pt Calibri Regular (`#263445`) with bold prefixes (`#2463A6`) and 1.10–1.12 line spacing.
- **Table Cells:** 7.2–7.8 pt Calibri with tight vertical padding (`0.02–0.04 in`) preventing vertical expansion.

---

## 3. Slide-by-Slide Transformation & Quality Verification

| Slide | Title & Category | Previous Dark Theme Status | Redesigned Light Theme Status | Visual Quality Verification |
|---|---|---|---|---|
| **01** | **Title Slide** (CSE Capstone Final Review) | Dark navy/black background, white logo, small font contrast. | Pure white canvas, Deep Navy converted VIT-AP University logo, Soft Sky Blue review badge, distinct Team & Supervisor cards, framed Amaravati geographic map snippet. | Clean academic presentation title page, Capstone ID `F5815A89` prominently displayed, zero text clipping. |
| **02** | **Presentation Outline & Research Trajectory** (01 / Roadmap) | Dark cards with cramped bullets extending off bottom. | Two structured light off-white cards (Part I: Research Foundation & Formulation; Part II: Engineering, Empirical Results & Viva). | 8.0 pt text with 3.0 pt spacing, perfectly contained with 0.4 in bottom padding. |
| **03** | **Three Distinct Operational Working Modes** (02 / Overview) | Cards overlapped bottom architectural principle banner. | 3 distinct vertical cards (Mode 1: Live Telemetry; Mode 2: 1-Hour Forecast; Mode 3: Location Mapping) ending at `Y = 4.50 in`, cleanly separated from the footer principle banner at `Y = 4.62 in`. | Strict adherence to core principle: physical sensors, ML forecasts, and spatial interpolations are kept completely separate. |
| **04** | **Four Bottlenecks in Urban Air Quality Intelligence** (03 / Problem Statement) | Dark cards, bottom-right card text touched border. | 2x2 grid of balanced light cards with amber warning badges (Sparsity, Downtime, Vision vulnerabilities, Spatial artifacts). | Balanced spacing, dark charcoal text, all bullets contained with ample internal breathing room. |
| **05** | **Four Core Engineering Goals & Deliverables** (04 / Objectives) | Inconsistent card heights and dark background. | 4 sleek vertical cards with color-coded Goal IDs (GOAL 01: Telemetry; GOAL 02: Primary Forecaster; GOAL 03: Safeguards; GOAL 04: Dashboard). | Color-coded tech stack badges (`#159A9C`), cleanly aligned across the full widescreen layout. |
| **06** | **Comparative Paradigm Analysis: Prior Works vs VayuDrishti** (05 / Literature Review) | Dark table with low border contrast and long wrapped title touching slide badge. | High-contrast 7x6 comparison table: Deep Navy header with white text, alternating white and light-blue rows, highlighted VayuDrishti row in soft sky blue (`#DCEBFA`). | Title fits cleanly on 1 line before the badge; table cell margins set to 0.04 in, zero overflow. |
| **07** | **Four Methodological & Engineering Contributions** (06 / Key Contributions) | Dark cards with dense paragraphs. | 2x2 grid highlighting VayuDrishti innovations: Bounded Vision Residual Gating ($\pm 25\ \mu\text{g/m}^3$), Spatial 45 km Cutoff, 90% Conformal Prediction, Full-Stack Web Platform. | Crisp typography, dark charcoal body text on off-white cards, professional visual balance. |
| **08** | **End-to-End System Pipeline & Multimodal Information Flow** (07 / System Architecture) | Title overlapped badge; right card text conflicted with embedded flowchart. | Left card: 3 structured computational tiers (Ingestion, Dual Forecasters, Spatial/Conformal Delivery). Right card: High-resolution Figure 1 architecture flowchart framed with top title and bottom bullet takeaways. | Flowchart aspect ratio (2.15) perfectly preserved without vertical distortion; caption and bullets cleanly framed. |
| **09** | **Interactive Web Dashboard: Six Operational Modules** (08 / User Interface) | Dark theme screenshot blended into slide background; text ran off card. | Left card: Complete descriptions of Modules 1 through 6 with Streamlit Cloud URL. Right card: High-resolution Figure 2 production dashboard screenshot neatly framed inside a card with caption. | UI screenshot neatly framed at 3.90 in width, caption sits comfortably inside card boundary. |
| **10** | **Mathematical Formulations & Production Technology Stack** (09 / Algorithms & Stack) | Long wrapped title touching badge; dark cards. | 2x2 grid of clean mathematical cards: IDW formula ($p=2.0$), Clark-Evans Index ($R$), Conformal Intervals, Residual Hybrid Vision Gate, plus a bottom Production Software Stack banner. | Single-line title, clear mathematical typography, bottom banner highlights CPU efficiency (zero GPU needed). |
| **11** | **Dataset Scale & Dual Feature Engineering Architectures** (10 / Datasets & Features) | 49-feature list ran past bottom card border. | Top: 4 metric cards (1.37M CPCB records, 85+ live stations, 418 live pairs, PM25Vision dataset). Bottom: 2 large cards contrasting LSTM 27 channels vs LightGBM 49 features. | All 7 LightGBM feature categories and 5 LSTM channel groups fit completely inside card boundaries. |
| **12** | **Training Setup, Hyperparameters & Operational Safeguards** (11 / Experimental Setup) | Title collided with badge; center card text touched border. | 3 vertical cards: Model Hyperparameters (unidirectional LSTM, LightGBM, ResNet-50), Data Protocol & Splits (chronological splits, recent holdout, LOSO, live audit), Operational Safeguards (45 km, $\pm 25$ clamp, 5 freshness tiers, severe gating). | Single-line header, 7.4 pt bullets with 3.0 pt spacing, flawless containment. |
| **13** | **Comprehensive Empirical Benchmarks & 14-Day Live Audit** (12 / Results) | Table overlapped bottom card; scatter plot was squished. | Restructured 3-panel layout: Top Left = 7-row Benchmark Table with separated cohorts; Bottom Left = 2 side-by-side cards (Delhi NCR LOSO Test and 14-Day Live Test); Right = Full-height Parity Scatter card (Figure 3) showing both PM2.5 and PM10 hexbins. | Zero overlap between table and cards; both hexbin subplots (a) and (b) are fully legible with high fidelity. |
| **14** | **Key Empirical Findings & Future Research Roadmap** (13 / Conclusions) | Dark cards with dense text. | Two large structured columns: "What We Learned" (persistence baseline strength, vision clamp necessity, spatial anchors, conformal value) vs "Future Research Roadmap" (ST-GNNs, Satellite AOD, multi-step transformers, IoT calibration). | Generous padding, balanced columns, easy to present during viva defense. |
| **15** | **Foundational Academic Literature & Regulatory Standards** (14 / References) | Low-contrast dark citations. | Two structured columns containing formal citations across Deep Learning, Computer Vision, Spatial GIS, Gradient Boosting, Conformal Prediction, and Regulatory Standards. | Clear IEEE-style academic citations, dark charcoal text on off-white cards. |
| **16** | **VayuDrishti: Summary & Open Discussion** (15 / Conclusion & Viva Defense) | Dark closing slide. | Top card: Formal project recap with Capstone ID `F5815A89`, live Streamlit link, GitHub link, team member names and supervisor. Bottom banner: Deep Navy "Thank You!" banner with university and department subtitle. | Professional, elegant conclusion slide designed for viva panel examination and defense. |

---

## 4. Technical Rigor & Scientific Accuracy Audit

The presentation strictly adheres to all project technical constraints and verified empirical facts:

1. **Primary Forecasting Architecture:**
   - The primary forecasting system is clearly documented as a **Combined 2-layer stacked unidirectional LSTM** ($H = 64$, 24-hour sequence $\times$ 27 continuous channels) + **LightGBM/GBDT** (49 canonical tabular features).
   - Nominal blending weights: 40% LSTM + 60% GBDT for $\text{PM}_{2.5}$; 45% LSTM + 55% GBDT for $\text{PM}_{10}$.
   - Severity-gated switching disengages GBDT to 100% LSTM during severe episodes ($\text{PM}_{2.5} \ge 250\ \mu\text{g/m}^3$ and $\text{PM}_{10} \ge 400\ \mu\text{g/m}^3$).
   - ResNet-50 is clearly presented as an auxiliary optical estimator strictly bounded to $\pm 25\ \mu\text{g/m}^3$ with guaranteed zero fallback ($m = 0$) when sky images are absent.

2. **Spatial Interpolation Safeguards:**
   - Spatial Inverse Distance Weighting (IDW, $p = 2.0$) strictly enforces a hard **45 km maximum correlation cutoff**.
   - Stations exceeding 45 km return "Out of Operational Range" to prevent unsupported continuous extrapolation.
   - Spatial geometry is audited using the Clark-Evans Nearest Neighbor index ($R$).

3. **Separation of Evaluation Cohorts:**
   - Historical CPCB Holdout (2015–2020, 404K sequences): Hybrid $\text{PM}_{2.5}$ MAE 10.35, RMSE 18.38; $\text{PM}_{10}$ MAE 18.14, RMSE 31.64.
   - Recent 2025 Holdout (Apr–Aug 2025, 236K sequences across 75 stations): Adapted Hybrid $\text{PM}_{2.5}$ MAE 6.62, RMSE 13.04; $\text{PM}_{10}$ MAE 16.14, RMSE 34.53.
   - 14-Day Live Operational Audit (418 matched prediction-observation pairs): GBDT $\text{PM}_{2.5}$ MAE 9.64 vs Persistence MAE 9.86 ($t = -0.71, p = 0.48$, non-significant), honestly demonstrating high autocorrelation and the vital role of conformal prediction intervals (90.4% live coverage).

4. **Official Identity & Deliverables:**
   - Title: "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments"
   - Capstone ID: `F5815A89`
   - Supervisor: Prof. Thota Ramathulsi, Assistant Professor, SCOPE, VIT-AP University
   - Team Members: Teliki Rushya Sree (23BCE9562), Machireddy Swathi (23BCE9566), Thanuja Varakala (23BCE9608), Palli Shilpa (23BCE9567)
   - Public Deployment: `https://vayudrishti-airquality.streamlit.app`
   - Repository: `https://github.com/Rushyasree/Multimodal_airquality_prediction`

---

## 5. Artifact & File Locations

- **Presentation Master (PPTX):** [presentation/VayuDrishti_Final_Review_Light_Theme.pptx](file:///e:/Documents/multimodal_air_quality_vscode_starter/presentation/VayuDrishti_Final_Review_Light_Theme.pptx)
- **Docs Copy (PPTX):** [docs/VayuDrishti_Final_Review_Light_Theme.pptx](file:///e:/Documents/multimodal_air_quality_vscode_starter/docs/VayuDrishti_Final_Review_Light_Theme.pptx)
- **Workspace Root Copy (PPTX):** [VayuDrishti_Final_Review_Light_Theme.pptx](file:///e:/Documents/multimodal_air_quality_vscode_starter/VayuDrishti_Final_Review_Light_Theme.pptx)
- **Generation Script:** [scripts/generate_light_theme_pptx.py](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/generate_light_theme_pptx.py)
- **Exported Slide Images (1920x1080 PNG):** [scratch/light_theme_slides/](file:///e:/Documents/multimodal_air_quality_vscode_starter/scratch/light_theme_slides/) (`slide_01.png` through `slide_16.png`)
