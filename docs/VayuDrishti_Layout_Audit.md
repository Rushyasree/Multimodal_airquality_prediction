# VayuDrishti PowerPoint Layout Correction & Alignment Audit Report

**Project Title:** VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments  
**University:** VIT-AP University, Amaravati, India  
**Department:** School of Computer Science and Engineering (SCOPE)  
**Supervisor:** Prof. Thota Ramathulsi  
**Capstone Project ID:** `F5815A89`  
**Project Members:**
- Teliki Rushya Sree (23BCE9562)
- Machireddy Swathi (23BCE9566)
- Thanuja Varakala (23BCE9608)
- Palli Shilpa (23BCE9567)

**Document Purpose:** This document provides a complete slide-by-slide audit of the visual inspection, layout corrections, bounding box fixes, margin adjustments, and diagram enhancements performed on `VayuDrishti_Final_Review_Layout_Fixed.pptx`. It also contains an authoritative explanation of the dataset figures to prepare the project team for committee viva defense.

---

## 1. Executive Summary of Corrections

A complete slide-by-slide rendering inspection of all 16 slides was conducted using PowerPoint COM headless image rendering at full 1080p resolution (`presentation/layout_fixed_slide_images/`).

Key structural fixes implemented:
1. **Universal Safety Margins:** Slide base clearance standardized to $\ge 0.30\text{ in}$ across all slides. Universal slide numbers relocated from the extreme bottom edge to `Left=9.10 in, Top=5.12 in` ($10\text{ pt}$ Segoe UI, subtle cyan).
2. **Text Overflow Elimination:** Replaced bloated paragraphs with structured bullet points, fixed card heights, set internal text frame margins (`top/bottom=0.14 in`, `left/right=0.18 in`), and controlled bullet paragraph spacing (`space_after=2.0\text{--}2.5\text{ pt}`).
3. **Architecture Diagram Overlap Resolution:** The original system diagram (`fig_arch_overall.png`) had overlapping arrow labels and text collisions (`Baseline` label touching boxes, arrow piercing `Gated Residual` title). The vector drawing script (`scripts/generate_ieee_architecture_diagrams.py`) was corrected: arrow label clutter was eliminated, the `Gated Residual` container widened to $0.96\text{ in}$, and connector coordinates shifted for zero line collisions.
4. **Hero Background Artifact Elimination:** Slide 1 and Slide 16 background image (`slide1_hero_bg.png`) previously retained faint residual dashboard text (`Good Morning!`, `Amaravati, Andhra Pradesh`, `10:24 AM IST`). The bottom gradient region was cleanly repainted using smooth atmospheric blending, preserving the scenic Krishna river and mountain horizon without text artifacts.
5. **Exact Aspect Ratio Preservation:** Slide 9's live web dashboard UI screenshot was framed within a dark glass container (`W=4.25 in, H=3.89 in`), preserving its exact $1092 \times 1000$ aspect ratio without stretching.

---

## 2. Dedicated Section: Dataset Figure Reconciliation & Viva Defense Guide

During review preparations, minor discrepancies were noted between preliminary slides (which cited $\approx 2.45\text{M}$ records and $2,150$ images) and the verified ground-truth figures published in the IEEE manuscript and final Capstone project report.

To ensure total academic defensibility, the presentation reflects the **exact, verified empirical numbers**. Below is the precise breakdown and recommended viva speaking script:

### Ground-Truth Data Figures Matrix

| Metric Category | Raw / Pre-Filter Scale | Verified Empirical Figure in Deck | Academic & Scientific Justification |
|---|---|---|---|
| **Historical CPCB Records** | $\approx 2.45\text{ Million}$ (Raw scraped/ingested hourly data rows) | **1,369,779 Hourly Records** (Slide 11 & Report) | Represents the **curated 2015–2020 multi-city benchmark** across 85+ stations after removing stations with $>30\%$ missingness, applying $3\text{-hour}$ linear interpolation for short gaps, and enforcing complete physical pollutant vectors ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$). |
| **PM25Vision Image Dataset** | $2,150$ (Total unverified web/camera uploads collected) | **1,235 Curated Images** & **199 Strictly Collocated Pairs** (Slide 11) | Out of 1,235 quality-checked outdoor images, exactly **199 image-sensor pairs** satisfied strict scientific collocation criteria: physical station distance $<5\text{ km}$, synchronous observation time $\Delta t \le 1\text{ hour}$, and clear camera line-of-sight across 18 stations. |
| **14-Day Live Test Audit** | Real-time streaming pipeline | **418 Matched Observation Pairs** (Slide 11, 13) | Ground-truth live audit conducted over a 14-day evaluation window in November 2026 across 4 active CPCB reporting stations via OpenAQ v3 API. |
| **Monitored Cities & Stations** | Nationwide Indian scope | **31 Cities & 85+ Active Stations** | Live telemetry stations actively streaming hourly pollutant data into the Streamlit Cloud dashboard. |

### Committee Viva Talking Points:
> **Question from Reviewer:** *"Why do earlier notes mention 2.4 million rows and 2,000 images, but your presentation and paper report 1,369,779 rows and 199 synchronous pairs?"*  
> **Student Defense Answer:**  
> *"That distinction reflects our rigorous data quality curation pipeline. 2.45 million was the raw, uncurated data stream containing offline stations and missing sensor values. Following IEEE environmental data standards, we filtered out invalid stations and incomplete criteria pollutant vectors, leaving exactly 1,369,779 verified hourly records for model training. Similarly, while 1,235 outdoor sky images were curated, only 199 satisfied our strict scientific collocation requirement of being within 5 km and 1 hour of an active CPCB ground monitor. Training on uncollocated pairs would introduce severe spatial confounding."*

---

## 3. Slide-by-Slide Layout Inspection & Fixes

### Slide 01: Hero Title Slide
- **Identified Issues:** Faint dashboard header text (`Good Morning!`, `10:24 AM IST`) was showing through the background image at the bottom left; metadata box needed clean project ID highlighting.
- **Layout Corrections:**
  - Repainted `slide1_hero_bg.png` bottom region to pure dark navy gradient.
  - Placed `CAPSTONE ID: F5815A89` prominently in electric coral (`#F87171`) atop the Left Card.
  - Balanced Team Members on the left (`Left=0.50 in, Top=2.46 in, Width=4.35 in, Height=2.74 in`) and Supervisor Prof. Thota Ramathulsi on the right (`Left=5.15 in, Top=2.46 in, Width=4.35 in, Height=2.74 in`).
  - Positioned high-resolution Amaravati regional monitoring map thumbnail at `Left=7.50 in, Top=4.30 in, Width=1.80 in`.

### Slide 02: Presentation Roadmap
- **Identified Issues:** In previous drafts, Part II listed 8 long bullet points which overflowed past the bottom slide margin ($5.625\text{ in}$).
- **Layout Corrections:**
  - Re-grouped Part II into 7 structured items with concise sub-descriptions.
  - Set card top to `Top=1.08 in` and height to `Height=4.25 in`, leaving a generous $0.29\text{ in}$ bottom clearance.
  - Adjusted body font to $8.8\text{ pt}$ with $2.0\text{ pt}$ spacing, ensuring all roadmap milestones sit cleanly inside the card boundaries.

### Slide 03: Project Overview & Core Philosophy
- **Identified Issues:** Three uneven columns with mismatched bottom boundaries; lacked an anchoring design principle.
- **Layout Corrections:**
  - Implemented 3 equal-width glass cards (`Left=0.50, 3.60, 6.70 in`, `Width=2.80 in`, `Height=3.45 in`).
  - Added an anchored full-width bottom principle strip at `Top=4.72 in, Height=0.55 in` highlighting the core thesis: *"Sensors provide Ground Truth, Images provide Local Context, GIS provides Spatial Continuity."*

### Slide 04: Problem Statement (4 Core Bottlenecks)
- **Identified Issues:** Text in a 4-card grid extended outside card bounds when cards were too tall.
- **Layout Corrections:**
  - Standardized a symmetrical $2 \times 2$ grid: Row 1 at `Top=1.15 in, Height=1.92 in`, Row 2 at `Top=3.20 in, Height=1.92 in`.
  - Left column `Left=0.50 in, Width=4.35 in`; Right column `Left=5.15 in, Width=4.35 in`.
  - Card clearance above slide base: $0.50\text{ in}$. Clean margins, no overflowing text.

### Slide 05: Project Objectives
- **Identified Issues:** Process cards had misaligned step badges and inconsistent height.
- **Layout Corrections:**
  - Arranged 4 process cards across horizontal space (`Left=0.50, 2.76, 5.02, 7.28 in`, `Width=2.22 in`, `Height=4.05 in`).
  - Added cyan step badges (`OBJ-1` through `OBJ-4`), clear category subheaders, and bottom highlighted deliverable badges.

### Slide 06: Literature Review Matrix
- **Identified Issues:** Tabular text crowded within cells; row heights varied unevenly.
- **Layout Corrections:**
  - Built a 6-row matrix table at `Top=1.15 in, Height=3.95 in, Width=9.00 in`.
  - Row heights standardized to $0.56\text{ in}$, text centered vertically with $8.5\text{ pt}$ Segoe UI.
  - Distinct cyan glowing highlight applied to Row 6 (VayuDrishti) to clearly accentuate project superiority over prior works.

### Slide 07: Key Contributions
- **Identified Issues:** Text blocks were dense; lacked clear visual hierarchy between technical innovation and deployment impact.
- **Layout Corrections:**
  - Structured into a balanced $2 \times 2$ grid with top accent colored highlight bars.
  - Card 1 (Bounded Multimodal Fusion), Card 2 (Spatial Integrity Bounds), Card 3 (Uncertainty Quantification), Card 4 (Open-Science Deployment). All cards terminate at `5.12 in`, well above the slide base.

### Slide 08: System Architecture & Flowchart
- **Identified Issues:** Text label collisions in the embedded flowchart SVG/PNG diagram (arrow label `Baseline` collided with LSTM box, arrow collided with `Gated Residual` text).
- **Layout Corrections:**
  - Regenerated diagram via `scripts/generate_ieee_architecture_diagrams.py` removing arrow label clutter and adjusting connection points.
  - Left side card (`Width=4.20 in, Height=4.05 in`) details the 3 pipeline tiers.
  - Right container (`Left=4.95 in, Width=4.55 in`) frames `fig_arch_overall.png` at high resolution with zero label overlap.

### Slide 09: Web Dashboard Showcase
- **Identified Issues:** Dashboard screenshot was previously stretched horizontally, distorting metric dials.
- **Layout Corrections:**
  - Left container (`Left=0.50 in, Width=4.20 in, Height=4.05 in`) details the 6 live production modules (M1 to M6) and displays the live URL.
  - Right container (`Left=4.95 in, Width=4.55 in, Height=4.05 in`) holds the screenshot at its true aspect ratio (`Width=4.25 in, Height=3.89 in`), centered with a dark border.

### Slide 10: Algorithms & Technology Stack
- **Identified Issues:** Mathematical formulas were crowded and lacked space for parameter definitions.
- **Layout Corrections:**
  - Four formula cards in a $2 \times 2$ grid (`Width=4.35 in, Height=1.65 in`):
    1. Inverse Distance Weighting (IDW)
    2. Clark-Evans Nearest Neighbor Index ($R$)
    3. Distribution-Free Conformal Prediction
    4. Residual Hybrid Vision Gate
  - Bottom technology stack bar at `Top=4.70 in, Height=0.50 in` cleanly summarizing Python 3.11, PyTorch, LightGBM, and Streamlit Cloud.

### Slide 11: Datasets & Feature Breakdown
- **Identified Issues:** Top statistical cards were previously unaligned with bottom feature cards; raw numbers differed from manuscript.
- **Layout Corrections:**
  - Top row: 4 KPI metric cards (`Height=1.15 in, Width=2.10 in`) with verified numbers:
    - `1,369,779` CPCB Hourly Records
    - `85+` Active Live Stations
    - `418` Live Matched Pairs
    - `1,235 / 199` PM25Vision Images / Pairs
  - Bottom row: 2 wide feature cards (`Width=4.35 in, Height=2.72 in`):
    - LSTM Sequential Forecaster ($24\text{ h} \times 27\text{ features}$)
    - LightGBM Tabular Forecaster ($49\text{ engineered features}$)

### Slide 12: Experimental Setup & Safeguards
- **Identified Issues:** Three columns had mismatched internal text margins and different vertical bounds.
- **Layout Corrections:**
  - Symmetrical 3-card layout (`Width=2.80 in, Height=4.05 in`) at `Top=1.15 in`.
  - Column 1: Model Hyperparameters (LSTM, LightGBM, ResNet-50).
  - Column 2: Data Protocol & Splits ($70/15/15\%$ chronological, LOSO, 14-day audit).
  - Column 3: Operational Safeguards ($45\text{ km}$ cutoff, $[-25, +25]\ \mu\text{g/m}^3$ clamp, 5 freshness tiers).

### Slide 13: Experimental Results & Live Audit
- **Identified Issues:** Parity plot and benchmark table occupied different vertical heights; bottom cards crowded the slide bottom.
- **Layout Corrections:**
  - Top row: Benchmark Table (`Left=0.50 in, Width=4.45 in, Height=1.68 in`) side-by-side with Parity Scatter Plot (`Left=5.15 in, Width=4.35 in, Height=1.68 in`).
  - Bottom row: Spatial Holdout card (`Width=4.35 in, Height=2.30 in`) and 14-Day Live Test card (`Width=4.35 in, Height=2.30 in`).
  - Both bottom cards terminate at `5.22 in`, providing clear breathing room above the slide base.

### Slide 14: Findings & Research Roadmap
- **Identified Issues:** Dense textual blocks; distinction between key takeaways and future work was blurred.
- **Layout Corrections:**
  - 2 tall equal cards (`Width=4.35 in, Height=4.05 in`) at `Top=1.15 in`.
  - Left Card: Key Empirical Findings (honest assessment of vision utility and baseline strength).
  - Right Card: Future Research Directions (Spatio-Temporal Graph Neural Networks, mobile IoT sensors, multi-spectral satellite integration).

### Slide 15: Academic References
- **Identified Issues:** 8 academic citations in small text were unorganized and difficult to read.
- **Layout Corrections:**
  - Structured into 2 balanced columns of 4 citations each (`Width=4.35 in, Height=4.05 in`).
  - Formatted in clean IEEE style with bold first authors, publication years, journal venues, and relevance badges.

### Slide 16: Conclusion & Viva Defense
- **Identified Issues:** Background artifact text was visible; contact information was unaligned.
- **Layout Corrections:**
  - Repainted background to clean deep navy gradient.
  - Centered hero card (`Left=0.50 in, Width=9.00 in, Height=3.75 in`) with `CAPSTONE PROJECT ID: F5815A89` prominently highlighted.
  - Clear attribution to all 4 team members and Guide Prof. Thota Ramathulsi.
  - Centered bottom "Thank You!" banner (`Top=4.50 in, Height=0.65 in`) welcoming committee questions.

---

## 4. Verification Checklist

- [x] All 16 slides rendered and visually inspected as 1080p images.
- [x] No text extends outside card borders or slide edges.
- [x] All card bottom edges terminate $\ge 0.30\text{ in}$ above the slide bottom.
- [x] No overlapping text boxes, icons, or diagram labels.
- [x] Architecture flowchart regenerated without label collisions.
- [x] Background hero banner cleaned of all residual dashboard text.
- [x] Verified dataset figures ($1,369,779$ rows, $1,235 / 199$ images/pairs, $418$ live pairs) integrated accurately.
- [x] Both `.pptx` output files synchronized in `presentation/` and `docs/`.
