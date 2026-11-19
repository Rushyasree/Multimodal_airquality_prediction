# VayuDrishti — Deployment Readiness Checklist & Production Governance Protocol

**Project Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Audit Date:** October 25, 2026  
**Auditor:** Senior ML Research Engineer / Time-Series Specialist  
**Evaluation Scope:** Transition from Historical Models to Post-2020 Adapted Ensemble  
**Status:** **CONDITIONALLY APPROVED FOR PRODUCTION DEPLOYMENT**  

---

## 1. Checkpoint & Artifact Inventory

All models and supporting artifacts have been verified, with exact paths, serialization formats, and sizes recorded:

| Component | Path | Format | Size | Checkpoint Role |
| :--- | :--- | :---: | :---: | :--- |
| **Historical LSTM (Baseline)** | [`models/lstm_baseline/best_lstm_model.pt`](file:///e:/Documents/multimodal_air_quality_vscode_starter/models/lstm_baseline/best_lstm_model.pt) | PyTorch State Dict | ~250 KB | **CURRENT PRODUCTION DEFAULT (Mode A)** |
| **Historical GBDT (PM2.5)** | [`experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib) | Scikit-Learn Joblib | ~350 KB | **CURRENT PRODUCTION DEFAULT (Ensemble)** |
| **Historical GBDT (PM10)** | [`experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm10.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm10.joblib) | Scikit-Learn Joblib | ~400 KB | **CURRENT PRODUCTION DEFAULT (Ensemble)** |
| **Fine-Tuned LSTM (Adapted)** | [`experiments/recent_data_audit/checkpoints/best_finetuned_lstm.pt`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/best_finetuned_lstm.pt) | PyTorch State Dict | 250 KB | **VERIFIED ADAPTED CANDIDATE** |
| **Adapted Reduced GBDT (PM2.5)**| [`experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib) | Scikit-Learn Joblib | 366 KB | **VERIFIED ADAPTED CANDIDATE (49 feats)** |
| **Adapted Reduced GBDT (PM10)** | [`experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib) | Scikit-Learn Joblib | 410 KB | **VERIFIED ADAPTED CANDIDATE (49 feats)** |
| **Canonical 2025 Predictions** | [`experiments/recent_data_audit/final_predictions.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/final_predictions.csv) | CSV File | 64.2 MB | 236,289 ground truth & model predictions |
| **Paired Bootstrap Metrics** | [`experiments/recent_data_audit/paired_recent_bootstrap.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/paired_recent_bootstrap.json) | JSON File | 2.8 KB | 95% CIs and p-values across 3 methods |
| **Validation Audit Report** | [`experiments/recent_data_audit/INDEPENDENT_VALIDATION_REPORT.md`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/INDEPENDENT_VALIDATION_REPORT.md) | Markdown | ~20 KB | Exhaustive independent reproduction report |

---

## 2. Independent Verification & Reproducibility Sign-Off

- [x] **Exact Metric Reproduction (100%):** All 8 model architectures reproduced their reported 2025 holdout MAE values down to $0.0000$ difference on identical target rows ($N = 236,289$, 75 stations).
  - 1-Hour Persistence: $\text{PM}_{2.5}$ 6.9476, $\text{PM}_{10}$ 17.1584
  - Fine-Tuned LSTM: $\text{PM}_{2.5}$ 6.8011, $\text{PM}_{10}$ 16.4791
  - Adapted Reduced GBDT: $\text{PM}_{2.5}$ 6.6716, $\text{PM}_{10}$ 16.2906
  - Adapted Ensemble: $\text{PM}_{2.5}$ 6.6216, $\text{PM}_{10}$ 16.1660
  - Adapted Severity-Gated Ensemble: $\text{PM}_{2.5}$ 6.6199, $\text{PM}_{10}$ 16.1425
- [x] **Causality & Zero-Leakage Sign-Off:**
  - Prediction target $y_{t+1}$ corresponds strictly to hour $24$; all input features strictly belong to steps $\le 23$.
  - Sequences partitioned strictly by station; zero cross-station boundary leaking.
  - Training (2021–2023), Validation (H1 2024), and Final Holdout (Apr–Aug 2025) are mutually disjoint.
  - Validation-selected blend weights ($w_{\text{PM2.5}} = 0.40, w_{\text{PM10}} = 0.45$) were frozen before holdout evaluation.
- [x] **Reduced-Feature Audit (49 Features):**
  - All 49 features used by the adapted GBDT are strictly computable from observed station telemetry, time, and coordinates.
  - All 17 unobserved weather and trace gas channels were cleanly excluded.

---

## 3. Statistical Uncertainty & Robustness Sign-Off

- [x] **Statistical Significance Confirmed:**
  - $\text{PM}_{2.5}$ Adapted Ensemble vs. Persistence: $\Delta\text{MAE} = -0.3277\ \mu\text{g/m}^3$, 95% CI $[-0.4283, -0.2270]$, $p < 0.0001$ (Method B).
  - $\text{PM}_{10}$ Adapted Ensemble vs. Persistence: $\Delta\text{MAE} = -1.0160\ \mu\text{g/m}^3$, 95% CI $[-1.2861, -0.7664]$, $p < 0.0001$ (Method B).
  - Fine-Tuned LSTM vs. Frozen LSTM: Highly significant neural gain on both pollutants ($p < 0.0001$).
- [x] **Monthly Robustness:**
  - The Adapted Severity-Gated Ensemble beats 1-Hour Persistence across **all 5 calendar months** (April, May, June, July, August 2025).
- [x] **Station-Level Distribution:**
  - Outperforms persistence at **53 of 75 stations ($70.7\%$)** on $\text{PM}_{2.5}$ and **55 of 75 stations ($73.3\%$)** on $\text{PM}_{10}$.

---

## 4. System Health & Performance Sign-Off

- [x] **CPU Inference Feasibility:**
  - LSTM forward pass: **$3.12$ ms** per sequence.
  - Reduced GBDT inference: **$24.78$ ms** per sequence.
  - Combined Ensemble latency: **$27.89$ ms** (< 30 ms on CPU host).
- [x] **Dashboard Smoke Tests:**
  - [`tests/test_eight_sections_smoke.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_eight_sections_smoke.py): **100% PASS with 0 ERRORS** across all 8 dashboard sections.
- [x] **Regression Unit Tests:**
  - [`tests/test_accuracy_optimization_v2.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_accuracy_optimization_v2.py): **5/5 PASS**.

---

## 5. Separation of Concerns & Scientific Integrity

- [x] **Sensor-Only vs. Vision Modality Separation:**
  - Adapted models strictly operate on numerical CAAQMS sensor telemetry.
  - The photographic image-estimation branch (`ResNet50Regressor`, Method 3) remains independent.
  - No claim is made that photographic images improve 1-hour sensor forecasting.
- [x] **Current Production Default Preserved:**
  - The operational backend service [`src/services/sensor_forecast_service.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/services/sensor_forecast_service.py) continues to serve the canonical Historical Severity-Gated Ensemble.
  - No production behavior or user-facing defaults were altered during this audit.

---

## 6. Phased Deployment Plan & Promotion Protocol

```
+---------------------------------------------------------------------------------------------------------+
|                                    PHASED DEPLOYMENT PROTOCOL                                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ STAGE 1: SHADOW DEPLOYMENT & OPT-IN FLAG ]  (IMMEDIATE)                                              |
|  ├── Expose adapted model in sensor_forecast_service.py via model_mode="adapted_severity_gated"         |
|  ├── Keep production default as model_mode="severity_gated" (Historical model)                         |
|  └── Run parallel shadow telemetry inference to monitor live station latency and memory                 |
|                                                                                                         |
|  [ STAGE 2: WINTER INVERSION BENCHMARK ]  (Q1 2027)                                                     |
|  ├── Retrieve CPCB CAAQMS telemetry for November 2025 to January 2026 (Winter episode)                  |
|  ├── Validate severe NAQI tier underprediction on peak smog events (> 250 ug/m3)                         |
|  └── Confirm that severity gating continues to eliminate tree-saturation drag                           |
|                                                                                                         |
|  [ STAGE 3: DEFAULT PROMOTION & SIGN-OFF ]  (POST-WINTER)                                               |
|  ├── Faculty & stakeholder review of full-year 2025-2026 continuous holdout results                     |
|  ├── Switch default model_mode in sensor_forecast_service.py to "adapted_severity_gated"                 |
|  └── Retain historical models as archived benchmarks in the dashboard Model Performance section        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## Final Sign-Off Summary

| Requirement | Evaluation Result | Compliance Status |
| :--- | :--- | :---: |
| **No Model Retraining** | Frozen checkpoints inspected; 0 epochs trained during Phase 4 | **COMPLIANT** |
| **Checkpoint Immutability** | Read-only loading; file sizes and timestamps preserved | **COMPLIANT** |
| **Zero Production Default Alteration**| `sensor_forecast_service.py` defaults untouched | **COMPLIANT** |
| **Numerical Reproduction** | 100% exact match across all 8 models down to 4 decimal places | **VERIFIED** |
| **Bootstrap Significance** | Statistically significant ($p < 0.05$) across 3 methodologies | **VERIFIED** |
| **Application Smoke Tests** | 8/8 dashboard sections pass smoke tests with 0 errors | **VERIFIED** |
| **Deployment Recommendation** | **CONDITIONALLY READY (Stage 1 Shadow Mode Approved)** | **SIGNED OFF** |
