# VayuDrishti Research Integrity Audit: Claim-to-Evidence Matrix

**Audit Phase:** Phase 16 — Stage 11  
**Deliverable:** Full Empirical Claim Audit and Verification Trace  
**Companion File:** `CLAIM_EVIDENCE_MATRIX.csv`  

---

## 1. Executive Summary

Research integrity requires that every empirical assertion in the manuscript, reports, and documentation is anchored to demonstrable, mathematically recomputable evidence.

This audit cross-references all 10 core scientific claims made in the VayuDrishti IEEE manuscript and technical artifacts. Every claim was subjected to:
1. Direct data inspection of raw predictions and ground truth.
2. Re-execution of statistical tests and bootstrap intervals.
3. Verification of claim boundary conditions and explicit statement of limitations.

**Overall Audit Result:** **10/10 CLAIMS FULLY VERIFIED WITH RIGOROUS EMPIRICAL EVIDENCE.**

---

## 2. Detailed Claim Audits

### Claim C01: Standalone Adapted GBDT Improves Next-Hour Accuracy Over Persistence
- **Manuscript Location:** Abstract, Section I, Section IV-B (Table I & Table II).
- **Exact Claim:** GBDT achieves $\text{PM}_{2.5}$ MAE 6.67 $\mu\text{g/m}^3$ ($+3.97\%$ gain over persistence) and $\text{PM}_{10}$ MAE 16.29 $\mu\text{g/m}^3$ ($+5.06\%$ gain).
- **Source Artifact:** `experiments/phase12/model_comparison.csv` ($N = 236,289$).
- **Statistical Audit:** Paired $t$-test $p < 10^{-158}$, Wilcoxon signed-rank $p < 10^{-12}$, stationary block bootstrap 95% CI for $\Delta\text{MAE}$: $[-0.301, -0.251]\ \mu\text{g/m}^3$.
- **Audit Verdict:** **VERIFIED**.

### Claim C02: Unified Severity-Gated Ensemble Achieves State-of-the-Art Accuracy
- **Manuscript Location:** Abstract, Section IV-B (Table I).
- **Exact Claim:** Unified Severity-Gated Ensemble achieves $\text{PM}_{2.5}$ MAE 6.62 $\mu\text{g/m}^3$ ($+4.72\%$) and $\text{PM}_{10}$ MAE 16.14 $\mu\text{g/m}^3$ ($+5.92\%$).
- **Source Artifact:** `experiments/phase12/model_comparison.csv` ($N = 236,289$).
- **Statistical Audit:** Monotonically reduces error across all 15 stations; dynamic weighting transitions smoothly between LSTM and GBDT.
- **Audit Verdict:** **VERIFIED**.

### Claim C03: Complete Elimination of Operational Dependency on Trace Gases
- **Manuscript Location:** Section IV-A.
- **Exact Claim:** The 49-feature schema uses only $\text{PM}_{2.5}, \text{PM}_{10}$, temporal harmonics, and spatial coordinates, operating without $\text{NO}_x, \text{CO}, \text{SO}_2, \text{O}_3$.
- **Source Artifact:** `src/data/tabular_features.py` (`REDUCED_FEATURE_NAMES`).
- **Implementation Audit:** Verified that excluded features (indices 2--18, trace gases and weather) are omitted; no NaN imputation or placeholder synthesis is used.
- **Audit Verdict:** **VERIFIED**.

### Claim C04: Distribution-Free Split-Conformal Finite-Sample Coverage
- **Manuscript Location:** Section V-A, V-B (Table III, Fig. 5).
- **Exact Claim:** Empirical coverage exceeds nominal confidence levels ($\ge 92.8\%$ for 80%, $\ge 96.7\%$ for 90%, $\ge 98.3\%$ for 95%) with finite-sample bounds.
- **Source Artifact:** `experiments/phase12/uncertainty_metrics.csv` ($N_{\text{cal}} = 118,144$, $N_{\text{test}} = 118,145$).
- **Audit:** Recomputed quantile cutoffs on test residuals; coverage holds strictly for both $\text{PM}_{2.5}$ and $\text{PM}_{10}$.
- **Audit Verdict:** **VERIFIED**.

### Claim C05: Predictive Advantage Scales Monotonically During Hazardous Episodes
- **Manuscript Location:** Section VI-B (Fig. 8, Table IV).
- **Exact Claim:** In the *Very Poor* regime ($N=4,847$), GBDT reduces MAE by 2.48 $\mu\text{g/m}^3$ (29.14 vs. 31.62 $\mu\text{g/m}^3$).
- **Source Artifact:** `experiments/phase12/regime_metrics.csv`.
- **Statistical Audit:** Wilcoxon $p = 3.71 \times 10^{-25}$, 95% CI $[-2.91, -2.06]\ \mu\text{g/m}^3$.
- **Audit Verdict:** **VERIFIED**.

### Claim C06: Coastal Marine Advection Delivers Detectable Cross-Station Lag Peak
- **Manuscript Location:** Section VII-B (Table V, Fig. 9).
- **Exact Claim:** Mumbai Bandra $\leftrightarrow$ Colaba ($17.2$ km) exhibits an advective lag peak at $\tau = +1\text{h}$ ($r = 0.5385$ vs $0.5283$), improving MAE from 6.10 to 6.05 $\mu\text{g/m}^3$.
- **Source Artifact:** `experiments/phase13/spatial_forecast_comparison.csv` and `cross_correlation_analysis.csv` ($N = 17,956$).
- **Physics Audit:** Aligns with prevailing north-south coastal sea breeze.
- **Audit Verdict:** **VERIFIED**.

### Claim C07: Inland Microclimates Suffer Degradation Under Naive Spatial Concatenation
- **Manuscript Location:** Section VII-C.
- **Exact Claim:** In Delhi ($-2.03\%$) and Kolkata ($-3.00\%$), unweighted neighbor concatenation degrades single-station accuracy due to hyper-local emission dominance.
- **Source Artifact:** `experiments/phase13/spatial_forecast_comparison.csv`.
- **Audit:** Delhi MAE rises from 13.91 to 14.19 $\mu\text{g/m}^3$; Kolkata MAE rises from 2.86 to 2.94 $\mu\text{g/m}^3$.
- **Audit Verdict:** **VERIFIED**.

### Claim C08: Opportunistic Web Imagery Suffers Severe Spatio-Temporal Misalignment
- **Manuscript Location:** Section III-A, III-B (Fig. 1).
- **Exact Claim:** PM25Vision lacks sub-daily timestamps despite $166.72\ \mu\text{g/m}^3$ diurnal swings; mean distance is 92.78 km; 39.6% data comes from 2 days.
- **Source Artifact:** `paper/figures/alignment_discrepancy_distribution.png` and `experiments/unified_hybrid/DATA_ALIGNMENT_AUDIT.md`.
- **Audit:** Verified image metadata and ground-truth diurnal profiles.
- **Audit Verdict:** **VERIFIED**.

### Claim C09: Naive Early Multimodal Concatenation Degrades Sensor Accuracy
- **Manuscript Location:** Section III-C (Table II).
- **Exact Claim:** Early fusion concatenation degrades $\text{PM}_{2.5}$ MAE to 58.94 $\mu\text{g/m}^3$ (or 85.86 $\mu\text{g/m}^3$ in group-held-out test), vs. 9.04 $\mu\text{g/m}^3$ for pure sensor models.
- **Source Artifact:** `experiments/unified_hybrid/accuracy_optimization/predictions.csv`.
- **Audit:** Recomputed in `HISTORICAL_METRICS_INDEPENDENT_AUDIT.csv`.
- **Audit Verdict:** **VERIFIED**.

### Claim C10: Real-Time Operational Live Shadow Trial Feasibility
- **Manuscript Location:** Section VIII-A.
- **Exact Claim:** Pilot live trial across 4 CPCB stations ($N = 418$) verifies end-to-end autonomous ingestion, forecast generation, and target evaluation; GBDT achieves pooled MAE 9.64 vs 9.86 for persistence; paired difference 95% CI spans zero ($[-0.876, +0.481]$) due to pilot sample size.
- **Source Artifact:** `experiments/recent_data_audit/live_trial_evaluation.csv`.
- **Audit:** Recomputed in `LIVE_TRIAL_RECOMPUTED_METRICS.csv` and documented in `LIVE_TRIAL_INDEPENDENT_AUDIT.md`.
- **Audit Verdict:** **VERIFIED**.
