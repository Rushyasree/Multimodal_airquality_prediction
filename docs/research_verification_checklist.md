# VayuDrishti Research Verification Checklist

This document provides a comprehensive technical audit and verification record for the IEEE manuscript:  
**"VayuDrishti: A Residual Multimodal Hybrid Framework for Air Quality Forecasting"**

---

## 1. Architectural & Codebase Truths

| Architectural Parameter | Code Source of Truth | Verified Implementation Value |
| :--- | :--- | :--- |
| **Input Sequence Dimension** | `src/prediction/lstm_forecaster.py` | $B \times 24 \times 27$ (24 lookback hours, 27 continuous telemetry channels) |
| **Telemetry Channel Breakdown** | `experiments/accuracy_optimization_v2/` | 12 pollutants, 5 meteorology, 6 temporal harmonics ($\sin/\cos$), 2 wind cyclical, 2 spatial coordinates |
| **LSTM Sensor Encoder** | `models/lstm_baseline/best_lstm_model.pt` | 2-layer stacked LSTM, hidden dimension 64, dropout 0.2, final hidden state $h_s \in \mathbb{R}^{64}$ |
| **Visual Feature Extractor** | `src/prediction/image_estimator.py` | ResNet-50 ImageNet backbone (frozen), Global Average Pooling producing $v_{\text{raw}} \in \mathbb{R}^{2048}$ |
| **Low-Rank Visual Projector** | `experiments/unified_hybrid/accuracy_optimization/hybrid_model_optimized.py` | Linear($2048 \to 64$) $\to$ LN $\to$ GELU $\to$ Dropout(0.2) $\to$ Linear($64 \to 32$) $\to$ LN $\to$ GELU ($e_v \in \mathbb{R}^{32}$) |
| **Spatial Distance Embedding** | `hybrid_model_optimized.py` | Linear($1 \to 8$) $\to$ ReLU ($e_d \in \mathbb{R}^8$) from Haversine distance $d$ (km) |
| **Fused Vector Dimension** | `hybrid_model_optimized.py` | $\mathbf{F} = [h_s \,\|\, (m \cdot e_v) \,\|\, e_d] \in \mathbb{R}^{104}$ ($64 + 32 + 8 = 104$) |
| **Learned Quality Gate** | `hybrid_model_optimized.py` | $\alpha_{\text{raw}} = \sigma(\text{MLP}(\mathbf{F})) \in [0, 1]$, effective gate $\alpha_{\text{eff}} = m \cdot \alpha_{\text{raw}}$ |
| **Residual Correction Bounds** | `hybrid_model_optimized.py` | $\Delta_{2.5} = 25 \tanh(R_{2.5}(\mathbf{F})) \in [-25, 25]$, $\Delta_{10} = 40 \tanh(R_{10}(\mathbf{F})) \in [-40, 40]$ |
| **Residual Initialization** | `hybrid_model_optimized.py` | Output layer weights $\mathbf{W}_{r2} = \mathbf{0}$, bias $\mathbf{b}_{r2} = \mathbf{0}$ (initial $\boldsymbol{\Delta} = \mathbf{0}$) |
| **Masked Fallback Guarantee** | `hybrid_model_optimized.py` | If $m = 0 \implies \alpha_{\text{eff}} = 0 \implies \hat{\mathbf{y}}_{\text{hybrid}} \equiv \hat{\mathbf{y}}_{\text{sensor}}$ identically |
| **Physical Bound Clamping** | `hybrid_model_optimized.py` | $\hat{\mathbf{y}}_{\text{hybrid}} = \max(\mathbf{0}, \hat{\mathbf{y}}_{\text{sensor}} + \alpha_{\text{eff}} \boldsymbol{\Delta})$ |

---

## 2. Strict Pathway Independence Verification

The codebase maintains strict separation between four distinct models/pathways. They are not conflated:

1. **Pathway 1 (Sensor-Only LSTM Baseline)**:
   - Module: `src/prediction/lstm_forecaster.py`
   - Inputs: 24h $\times$ 27 telemetry channels.
   - Outputs: $[\hat{y}_{\text{PM2.5}}, \hat{y}_{\text{PM10}}]$ in $\mu\text{g/m}^3$.
   - Test Benchmark: MAE 8.86 ($\text{PM}_{2.5}$), 15.54 ($\text{PM}_{10}$) across 404,431 test sequences.
2. **Pathway 2 (Severity-Gated LSTM–GBDT Ensemble)**:
   - Module: `src/services/sensor_forecast_service.py`
   - Inputs: LSTM output + 49 canonical tabular lag features.
   - Blending: $w_{25} = 0.90 + 0.10 \cdot \sigma((\text{PM}_{2.5} - 250)/25)$.
   - Decoupling Thresholds: $\ge 250\,\mu\text{g/m}^3$ ($\text{PM}_{2.5}$), $\ge 400\,\mu\text{g/m}^3$ ($\text{PM}_{10}$).
3. **Pathway 3 (Residual Multimodal Hybrid Model)**:
   - Module: `src/services/hybrid_prediction_service.py` & `hybrid_model_optimized.py`
   - Inputs: Frozen LSTM hidden state ($h_s$), ResNet-50 visual features ($e_v$), distance ($e_d$), availability mask ($m$).
   - Outputs: Quality-gated bounded residual adjustment with exact fallback.
4. **Pathway 4 (Auxiliary Image-Only Baseline)**:
   - Module: `src/prediction/image_estimator.py`
   - Inputs: Single $224 \times 224$ RGB image.
   - Head: Dense regressor ($2048 \to 256 \to 64 \to 1$).
   - Performance: MAE 66.21 $\mu\text{g/m}^3$, $R^2 = 0.1524$ (confirms vision cannot serve as an autonomous sensor).

---

## 3. Empirical Test Metrics & Evaluation Truths

| Evaluation Metric | Dataset / Cohort | Verified Value | Grounded Reference |
| :--- | :--- | :--- | :--- |
| **LSTM $\text{PM}_{2.5}$ MAE** | 404,431 Benchmark Sequences | **8.86 $\mu\text{g/m}^3$** ($R^2 = 0.9492$) | `models/lstm_baseline/test_metrics.json` |
| **LSTM $\text{PM}_{10}$ MAE** | 404,431 Benchmark Sequences | **15.54 $\mu\text{g/m}^3$** ($R^2 = 0.9817$) | `models/lstm_baseline/test_metrics.json` |
| **Persistence $\text{PM}_{2.5}$ MAE** | 404,431 Benchmark Sequences | 10.96 $\mu\text{g/m}^3$ ($R^2 = 0.8878$) | `paper/tables/table_overall_benchmarks.tex` |
| **Persistence $\text{PM}_{10}$ MAE** | 404,431 Benchmark Sequences | 19.49 $\mu\text{g/m}^3$ ($R^2 = 0.9416$) | `paper/tables/table_overall_benchmarks.tex` |
| **Adapted GBDT $\text{PM}_{2.5}$ MAE** | 236,289 Multi-Station Cohort | **6.67 $\mu\text{g/m}^3$** ($R^2 = 0.8219$) | `paper/tables/table_multistation_performance.tex` |
| **Adapted GBDT $\text{PM}_{10}$ MAE** | 236,289 Multi-Station Cohort | **16.29 $\mu\text{g/m}^3$** ($R^2 = 0.8415$) | `paper/tables/table_multistation_performance.tex` |
| **Residual Hybrid $\text{PM}_{2.5}$ MAE** | Paired Multi-Station Test Set | **8.98 $\mu\text{g/m}^3$** ($R^2 = 0.9628$) | `experiments/unified_hybrid/accuracy_optimization/metrics.json` |
| **Residual Hybrid Fallback MAE** | Paired Test Set ($m=0$) | **9.04 $\mu\text{g/m}^3$** ($R^2 = 0.9625$) | `metrics.json` |
| **Split-Conformal 80% Coverage** | 118,145 Calibration/Test Split | **92.87%** ($\text{PM}_{2.5}$), **93.66%** ($\text{PM}_{10}$) | `paper/tables/table_conformal_calibration.tex` |
| **Split-Conformal 90% Coverage** | 118,145 Calibration/Test Split | **96.76%** ($\text{PM}_{2.5}$), **97.23%** ($\text{PM}_{10}$) | `paper/tables/table_conformal_calibration.tex` |
| **Split-Conformal 95% Coverage** | 118,145 Calibration/Test Split | **98.38%** ($\text{PM}_{2.5}$), **98.71%** ($\text{PM}_{10}$) | `paper/tables/table_conformal_calibration.tex` |
| **Stationary Block Bootstrap** | $B=5,000$, $L=24\text{h}$ | $p_{\text{paired-t}} = 5.7 \times 10^{-158}$, $p_{\text{Wilcoxon}} < 10^{-11}$ | `paper/tables/table_statistical_tests.tex` |

---

## 4. Empirical Negative Result on Opportunistic Imagery

Audited against 640 unique outdoor images from the PM25Vision dataset:
1. **Spatial Displacement**: Mean Haversine distance to nearest CAAQMS is 92.78 km; only 49.4% within 5 km.
2. **Temporal Mismatch**: Images have only calendar dates (`YYYY-MM-DD`) with zero sub-daily hours, while continuous station $\text{PM}_{2.5}$ exhibits a $166.72\,\mu\text{g/m}^3$ average diurnal swing.
3. **Burst Pseudo-Replication**: 265 matched records represent only 36 independent station-date groups (39.6% from just two single-day bursts).
4. **Label Divergence**: Systematic $+70.74\,\mu\text{g/m}^3$ positive bias relative to certified federal beta-attenuation monitors.
5. **Controlled Ablation Degradation**: Naive early feature concatenation degraded test MAE by **$+12.19\,\mu\text{g/m}^3$** (85.86 vs. 73.67 $\mu\text{g/m}^3$, $R^2 = -0.5064$).

---

## 5. Verification Checklist Sign-Off

- [x] Title matches exact prompt specifications: *"VayuDrishti: A Residual Multimodal Hybrid Framework for Air Quality Forecasting"*
- [x] All 8 publication-grade vector architecture figures generated in `paper/figures/` (.png, .svg, .pdf).
- [x] User-supplied actual-vs-predicted scatter graph (`actual_vs_predicted.png`) and dashboard interface (`dashboard_monitoring_interface.png`) integrated.
- [x] LaTeX source (`paper/main.tex`) fully verified: all 7 tables, 19 figures, and 21 citations verified by `scripts/validate_paper.py`.
- [x] Full editable IEEE DOCX manuscript (`paper/VayuDrishti_IEEE_Manuscript.docx`) generated with all sections, formulas, and figures embedded.
- [x] Markdown manuscript (`paper/manuscript.md`) synchronized.
- [x] No unverified performance metrics or invented baseline numbers.
