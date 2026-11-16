# VayuDrishti — Phase 14 Final Scientific UI & Provenance Audit Report

**Date:** November 16, 2026  
**Project:** VayuDrishti Multimodal Air Quality Platform  
**Guidance & Supervision:** Professor Thota Ramathulsi, Department of SCOPE  
**Status:** Completed & Validated (All 30/30 Tests Passing)

---

## 1. Executive Summary & Scope

Phase 14 delivers a comprehensive, research-grade scientific UI, data provenance, and visual consistency audit of the VayuDrishti Streamlit platform. The primary objective is to align all dashboard presentations, visualizations, telemetry disclosures, and forecast intervals with empirical peer-reviewed research standards—**without retraining models, modifying validated checkpoints, altering historical evaluation metrics, changing forecasting horizons ($t+1$), or modifying spatial interpolation algorithms**.

All fabricated heuristics, unsupported multi-day forecast cards, unvalidated medical advisories, misleading atmospheric dispersion claims, and uncalibrated "high confidence" badges have been systematically eliminated and replaced with verifiable data provenance and calibrated uncertainty intervals.

---

## 2. Audit of Unsupported Dashboard Claims & Removals

| Section / Component | Prior Implementation (Flagged) | Phase 14 Scientific Correction | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Overview Row 3 (Card 1)** | Fabricated "5-Day Trend Forecast" applying heuristic multipliers (`[1.0, 0.95, 1.05, 0.90, 0.85]`). | **Removed completely.** Replaced with **Card 6: Operational Forecast Target ($t+1$) & 24h Window Telemetry**. | The model architecture is strictly validated for short-term single-step ($t+1$) forecasting. Multi-day heuristic extrapolation is unscientific and misleading. |
| **Overview Row 3 (Card 2)** | Unverified Medical/Health Advisories ("Avoid outdoor exercise", "Wear N95"). | **Removed completely.** Replaced with **Card 7: Telemetry Provenance & CAAQMS Authority Attribution**. | Health advisories must follow clinical/epidemiological authority guidelines, not unvalidated AI heuristics. Replaced with strict criteria pollutant disclosures. |
| **Overview Row 3 (Card 3)** | Fabricated statistics: `↓ 15% PM2.5 below recent averages`, `📈 Stable`, `⛅ Clear dispersion`. | **Removed completely.** Replaced with **Card 8: Scientific & Regulatory Scope Notice**. | Percentage drops and atmospheric dispersion interpretations were fabricated without meteorological reanalysis backing. |
| **Live AQI Fallback** | Fallback to `"Satisfactory"` (`#3B82F6`) when no criteria pollutants were detected. | Replaced with `"Insufficient Data"` (`#94A3B8`) with explicit null handling. | Prevents false-negative clean-air reporting when sensor feeds are offline or missing criteria measurements. |
| **Hybrid Forecast Insights** | Generic advice ("Mask recommended for sensitive groups"). | Replaced with **Indian NAAQS Threshold Reference** ($60\,\mu\text{g/m}^3$ 24-hr standard). | Contextualizes numerical predictions against statutory ambient air quality standards rather than providing medical prescriptions. |
| **Spatial Confidence Badges** | Blanket `"High Confidence Spatial Interpolation"` badge across all coordinates. | Dynamic tiering: `Conformal Calibrated (Delhi NCR ≤ 5 km)`, `Distance-Scaled (≤ 15 km)`, `Exploratory Regional Estimate (Non-NCR)`. | Conformal coverage is only empirically guaranteed on the dense Delhi NCR network; outside this region, spatial estimates are exploratory. |

---

## 3. Separation of Live vs. Historical Predictions

A fundamental research requirement is the absolute separation of historical benchmark demonstrations from real-time live sensor observations and inference.

1. **Model Comparison Service (`src/services/model_comparison_service.py`):**
   - **Historical Mode (`live_mode=False`):** Row 0 is explicitly designated:
     - Model Name: `Historical Ground Truth (CAAQMS)`
     - Result Type: `HISTORICAL BENCHMARK GROUND TRUTH`
     - Timestamp: Exact timestamp of the benchmark record (e.g., `2023-11-15 14:00`).
   - **Live Mode (`live_mode=True`):** Row 0 displays:
     - Model Name: `Live Ground Station Monitor (CAAQMS)`
     - Result Type: `LATEST PHYSICAL GROUND OBSERVATION`
     - Timestamp: Authentic telemetry observation timestamp $T$.
   - **Persistence Baseline:** In live mode, the baseline persistence prediction $\hat{y}_{t+1} = y_t$ is strictly computed from the latest physical ground observation, preventing cross-temporal contamination.
   - **Historical Benchmark Integrity:** Historical evaluation benchmarks in table views strictly draw from saved test-set evaluation artifacts (`mae`, `rmse`, `r2`) rather than on-the-fly live telemetry.

2. **Unified Hybrid Forecast (`src/components/hybrid_dashboard_pages.py`):**
   - Side-by-side comparison chart dynamically labels the reference benchmark line as either `"Live Ground Monitor (T)"` or `"Historical Ground Truth (CAAQMS)"` depending on session execution mode.

---

## 4. Forecast Timestamp & Temporal Horizon Integrity

To prevent misleading users regarding the forecasting horizon, all forecast presentations adhere to the following temporal schema:

- **Forecast Horizon:** Strictly $t+1$ (next 1 hour).
- **Antecedent Input Sequence:** Explicitly designated as a 24-hour historical window: $[T-23\text{h}, T]$ where $T$ is the forecast issue timestamp.
- **Target Prediction Time:** Explicitly designated as $T+1\text{h}$.
- **Sequence Verification:** Displays explicit verification: `"24 / 24 Timesteps Verified"`; partial or zero-imputed buffers are rejected by the multi-gate provenance validator.
- **Model Version & Provenance:** Every card discloses model architecture (`VayuDrishti Stacking Regressor v2.0` / `LightGBM-Ridge-RF Fusion`).

---

## 5. Image-Based & Multimodal Module Scientific Disclosures

The visual PM2.5 estimation module has been audited to reflect its true empirical characteristics:

1. **Experimental Status Disclosure:**
   - Explicit scientific disclosure rendered in Method 3 and Hybrid views:
     > *"Image-Based PM2.5 Estimator is an experimental feature ($R^2 \approx 0.15$). It is NOT a physical sensor or standalone calibrated forecaster."*
2. **Visual Fallback Guarantee ($\alpha \le 0.001$):**
   - When environmental conditions (low lighting, haze, rain) cause visual degradation or when no valid outdoor image is provided:
     > *"Visual Gate Weight: 0.0% (Zero Visual Contribution • Pure Sensor Baseline Guaranteed)"*
   - Informs users that the dynamic gating network dynamically collapses to the pure sensor model, preventing degradation of prediction accuracy.
3. **Suppression of Zero-Valued Diagnostic Cards:**
   - When image features are missing or zero-imputed, the dashboard suppresses deceptive "0.00" optical depth cards and displays an informative fallback notice.

---

## 6. Spatial Heatmap & Conformal Uncertainty Audit

The spatial interpolation page (`src/components/spatial_heatmap_page.py`) implements Inverse Distance Weighting (IDW) with calibrated conformal prediction intervals:

1. **Interpolation Provenance Display:**
   - **Interpolated Estimate:** $\hat{y}_{\text{IDW}}(x_0) = \frac{\sum_i w_i y_i}{\sum_i w_i}$ with power parameter $p=2.0$.
   - **Contributing Stations:** Active count of CAAQMS stations within the influence radius.
   - **Nearest Station & Distance:** Explicit reporting of the closest physical monitoring station and Euclidean distance in kilometers.
   - **Network Density Classification:** Dynamic classification into Dense ($\le 5\,\text{km}$), Moderate ($5\text{--}15\,\text{km}$), or Sparse ($> 15\,\text{km}$).
2. **Conformal Uncertainty & Interval Precision:**
   - Reports both empirical coverage percentage (**99.4%**) AND interval width / precision ($\pm \text{half-width}$ in $\mu\text{g/m}^3$).
   - Explicitly contrasts the Split-Conformal Calibrated Interval ($\pm 21.05\,\mu\text{g/m}^3$, 99.4% coverage) against heuristic distance-scaled proxies ($\pm 10.74\,\mu\text{g/m}^3$, under-coverage 53.6%), warning users against uncalibrated heuristics.
3. **Exploratory Regional Warning:**
   - Prominent disclaimer added for non-Delhi regions (Amaravati, Visakhapatnam, Vijayawada, Mumbai, Bengaluru, Kolkata):
     > *"Regional Notice: Formal conformal prediction intervals were calibrated on the dense Delhi NCR CAAQMS network (68 stations). Spatial interpolation in other regions is exploratory and subject to monitoring sparsity."*

---

## 7. Visual & Architectural Cleanup

1. **Theme & Navigation Retention:**
   - Dark theme maintained throughout with `#0F172A` background, `#111827` card containers, and `#1F293D` borders.
   - Exact 6 active views maintained in topbar and sidebar navigation:
     1. `Overview`
     2. `Live Air Quality`
     3. `Individual Forecasting Methods`
     4. `Unified Hybrid Forecast`
     5. `Side-by-Side Comparison`
     6. `Spatial Prediction & Heatmap`
2. **Typography & Emojis:**
   - Cleaned redundant, decorative emojis and emojis in headers.
   - Professionalized cards with clean SVGs and crisp typography.
3. **Supervisor Attribution:**
   - Clean academic attribution placed in the global footer:
     > *"VayuDrishti Research Platform • Guided by: Professor Thota Ramathulsi • Department of SCOPE"*

---

## 8. Model Checkpoint & Functionality Preservation

- **Zero Retraining:** All model checkpoints (`models/checkpoints/` and `src/models/`): LightGBM, Random Forest, Ridge, HistGradientBoosting, GRU, and ResNet-50 remain untouched and loadable.
- **Zero Parameter Drift:** IDW power $p=2.0$, distance thresholds, and conformal non-conformity scores ($q_{\text{val}} = 21.0475\,\mu\text{g/m}^3$) are strictly preserved.
- **Evaluation Artifacts Intact:** All benchmark evaluation JSON files and test artifacts remain unaltered.

---

## 9. Verification & Test Suite Execution

The verification test suite was executed across the entire repository. All 30 tests passed with 100% success rate:

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\Documents\multimodal_air_quality_vscode_starter

tests/test_dashboard_navigation.py::test_dashboard_navigation_and_cleanup PASSED [100%]
tests/test_phase13_spatial_validation.py::test_held_out_station_exclusion_zero_leakage PASSED
tests/test_phase13_spatial_validation.py::test_same_timestamp_temporal_integrity PASSED
tests/test_phase13_spatial_validation.py::test_conformal_calibrated_intervals_present PASSED
tests/test_phase13_spatial_validation.py::test_out_of_bounds_extrapolation_safety PASSED
tests/test_phase13_spatial_validation.py::test_phase13_saved_metrics_artifacts_exist PASSED
tests/test_phase13_spatial_validation.py::test_conformal_calibration_metrics_integrity PASSED
tests/test_phase13_spatial_validation.py::test_nearest_station_comparison_metrics PASSED
tests/test_phase13_spatial_validation.py::test_dashboard_sidebar_route_connectivity PASSED
tests/test_model_input_tensor_alignment.py::TestModelInputTensorAlignment::test_tensor_identity_on_demo_scenarios PASSED
tests/test_model_input_tensor_alignment.py::TestModelInputTensorAlignment::test_model_comparison_service_reports_tensor_parity PASSED
tests/test_model_input_tensor_alignment.py::TestModelInputTensorAlignment::test_tensor_mismatch_detection_on_tampered_input PASSED
tests/test_live_data_provenance_and_validation.py::TestLiveSequenceBuilderAndBuffer::test_24h_contiguous_sequence_construction PASSED
tests/test_live_data_provenance_and_validation.py::TestLiveSequenceBuilderAndBuffer::test_incomplete_buffer_rejection_without_synthetic_substitution PASSED
tests/test_live_data_provenance_and_validation.py::TestLiveSequenceBuilderAndBuffer::test_sub_hourly_aggregation_and_deduplication PASSED
tests/test_live_data_provenance_and_validation.py::TestLiveSequenceBuilderAndBuffer::test_physical_bounds_filtering PASSED
tests/test_live_data_provenance_and_validation.py::TestMultiGateLiveValidation::test_multi_gate_validation_passes_on_authentic_fresh_sequence PASSED
tests/test_live_data_provenance_and_validation.py::TestMultiGateLiveValidation::test_multi_gate_validation_rejects_stale_sequence PASSED
tests/test_live_data_provenance_and_validation.py::TestMultiGateLiveValidation::test_multi_gate_validation_rejects_non_authentic_sources PASSED
tests/test_live_data_provenance_and_validation.py::TestEndToEndLiveForecastingServices::test_sensor_forecast_service_live_pathway PASSED
tests/test_live_data_provenance_and_validation.py::TestEndToEndLiveForecastingServices::test_sensor_forecast_service_rejects_stale_in_live_mode PASSED
tests/test_live_data_provenance_and_validation.py::TestEndToEndLiveForecastingServices::test_hybrid_prediction_service_live_pathway PASSED
tests/test_live_data_provenance_and_validation.py::TestModelComparisonLiveParityAndRejection::test_live_synchronization_and_tensor_digest_parity PASSED
tests/test_live_data_provenance_and_validation.py::TestModelComparisonLiveParityAndRejection::test_incomplete_buffer_rejects_live_synchronization PASSED
tests/test_live_data_provenance_and_validation.py::TestModelComparisonLiveParityAndRejection::test_persistence_baseline_correctness PASSED
tests/test_accuracy_optimization_v2.py::TestAccuracyOptimizationV2::test_01_feature_extraction_dimension PASSED
tests/test_accuracy_optimization_v2.py::TestAccuracyOptimizationV2::test_02_temporal_leakage_prevention PASSED
tests/test_accuracy_optimization_v2.py::TestAccuracyOptimizationV2::test_03_checkpoints_exist_and_loadable PASSED
tests/test_accuracy_optimization_v2.py::TestAccuracyOptimizationV2::test_04_deterministic_inference PASSED
tests/test_accuracy_optimization_v2.py::TestAccuracyOptimizationV2::test_05_physical_constraints PASSED

======================== 30 passed in 152.51s (0:02:32) ========================
```

---

## 10. Summary of Modified Files

1. [`app.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/app.py)
   - Replaced unvalidated 5-day forecast, unverified medical advice, and fabricated dispersion percentages with 3 research-grade operational telemetry cards.
   - Fixed criteria pollutant fallback to `"Insufficient Data"`.
   - Updated global footer with academic supervision attribution.
2. [`src/components/hybrid_dashboard_pages.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/components/hybrid_dashboard_pages.py)
   - Replaced prescriptive health advisories with statutory Indian NAAQS concentration standards.
   - Added experimental disclosure for visual PM2.5 estimator and explicit visual fallback guarantee ($\alpha=0$).
   - Suppressed misleading zero-valued optical diagnostic cards.
   - Dynamically designated comparison charts between live observation and historical benchmark.
3. [`src/components/spatial_heatmap_page.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/components/spatial_heatmap_page.py)
   - Replaced static "High Confidence" badge with calibrated uncertainty tiers.
   - Displayed conformal prediction interval width/precision alongside coverage.
   - Added exploratory regional disclaimers for non-Delhi regions.
4. [`src/services/model_comparison_service.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/services/model_comparison_service.py)
   - Disallowed mixing live telemetry with historical benchmark records.
   - Explicitly labeled Row 0 as `Historical Ground Truth (CAAQMS)` in historical mode and `Live Ground Station Monitor (CAAQMS)` in live mode.
5. [`tests/test_dashboard_navigation.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/tests/test_dashboard_navigation.py)
   - Increased AppTest execution timeout to accommodate live multi-station telemetry requests during end-to-end integration testing.

---

## 11. Remaining Scientific Limitations

1. **Spatial Conformal Coverage Calibration Boundary:** Conformal prediction intervals are strictly calibrated against the dense 68-station Delhi NCR network ($q_{\text{val}} = 21.05\,\mu\text{g/m}^3$). Regional interpolation in cities with sparse networks (e.g., Amaravati, Vijayawada, Visakhapatnam) is exploratory and lacks empirical conformal coverage guarantees.
2. **Visual PM2.5 Correlation:** The ResNet-50 visual PM2.5 estimator achieves $R^2 \approx 0.15$ and serves solely as a secondary supplementary modality. In low-visibility or non-calibrated lighting conditions, the gating mechanism appropriately zeroes out visual weight ($\alpha=0$) to default to the sensor-only baseline.
3. **Single-Step Horizon ($t+1$):** Forecasts are strictly valid for the next immediate hour. Multi-step autoregressive rollouts or diurnal forecasts require future dynamical atmospheric dispersion coupling.
