# Phase 18 — VayuDrishti Codebase Cleanup, Dashboard Simplification & Safe Refactoring Audit

**Date:** 2026-11-12  
**Platform:** VayuDrishti Air Quality Monitoring & Forecasting Platform  
**Repository:** https://github.com/Rushyasree/Multimodal_airquality_prediction  
**Baseline Git Commit:** `64944f3`  

---

## 1. Executive Summary

Phase 18 implements a disciplined, safety-first cleanup and refactoring of the VayuDrishti platform. The cleanup preserves all trained model weights, numerical pipelines, sequence validation rules, evaluation benchmarks, and active test suites, while:
1. Eliminating verified obsolete one-off patch/fix scripts from prior development phases.
2. Removing 16.7 KB of hardcoded base64 image strings embedded directly in `app.py`.
3. Eliminating duplicated UI sections and redundant security cards in `src/components/hybrid_dashboard_pages.py`.
4. Scrubbing lingering academic/paper jargon from the user-facing dashboard in strict accordance with the VayuDrishti product boundary (an air-quality platform, not a research-paper dashboard).
5. Consolidating duplicate criteria pollutant tier/color mappings using the centralized CPCB NAQI engine (`src/prediction/aqi_advisor.py`).

---

## 2. File Classification Inventory

### Category A: Essential Application Files (Retained & Maintained)
| Path | Role / Function | Status |
| :--- | :--- | :--- |
| `app.py` | Main Streamlit application entry point & navigation router | Retained, refactored, base64 cleaned |
| `src/components/hybrid_dashboard_pages.py` | 8-section dashboard rendering logic | Retained, cleaned, duplicates removed |
| `src/services/live_air_quality_service.py` | Real-time observation coordinator | Retained |
| `src/services/sensor_forecast_service.py` | 24h numerical sequence forecasting (adapted LSTM + GBDT) | Retained (weights & formulas protected) |
| `src/services/image_estimation_service.py` | Visual haze feature extraction and estimation | Retained |
| `src/services/hybrid_prediction_service.py` | Gated residual multimodal fusion inference | Retained |
| `src/services/model_comparison_service.py` | Multi-model comparative validation & persistence baseline | Retained |
| `src/services/operational_health_service.py` | Operational telemetry health state engine | Retained |
| `src/services/forecast_audit_logger.py` | JSONL forecast execution audit logging | Retained |
| `src/services/forecast_execution_lifecycle.py` | End-to-end forecast execution manager | Retained |
| `src/services/forecast_observation_matcher.py` | $(t+1)$ target observation matcher | Retained |
| `src/data/resilient_ingestion.py` | Resilient HTTP client with retry and cache preservation | Retained |
| `src/data/live_rolling_buffer.py` | Continuous 24h rolling buffer and hourly binning | Retained |
| `src/prediction/live_sequence_builder.py` | 24h sequence builder and metadata assembler | Retained |
| `src/prediction/sequence_validator.py` | 8-gate sequence validation | Retained |
| `src/prediction/station_data_quality.py` | Physical bounds and rate-of-change auditing | Retained |
| `src/prediction/observation_provenance.py` | Provenance schemas and metadata dataclasses | Retained |
| `src/prediction/lstm_forecaster.py` | Pretrained PyTorch LSTM wrapper and normalizer | Retained |
| `src/prediction/current_air_quality_service.py` | OpenAQ and Open-Meteo REST connectors | Retained |
| `src/prediction/aqi_advisor.py` | CPCB NAQI sub-index and health guidance engine | Retained |
| `src/prediction/data_loader.py` | Station catalogs and benchmark demo sequence loader | Retained |
| `Dockerfile` & `docker-compose.yml` | Container deployment configuration | Retained |
| `requirements.txt` | Minimal pinned production dependencies | Retained |
| `.streamlit/config.toml` | Streamlit theme and server configuration | Retained |

---

### Category B: Required ML Artifacts (Strictly Protected)
| Path | Artifact Type | Protection Rationale |
| :--- | :--- | :--- |
| `models/lstm_baseline/best_lstm_model.pt` | PyTorch Weights | Core 2-layer LSTM sequence forecaster |
| `models/scaler_27features.pkl` | Scikit-learn Scaler | 27-channel feature normalization |
| `models/image_estimator/resnet50_weights.pth` | PyTorch Weights | ResNet-50 visual feature extractor |
| `experiments/unified_hybrid/accuracy_optimization/hybrid_model_optimized.py` | PyTorch Architecture | `OptimizedResidualHybridModel` and `QualityAwareResidualHead` |
| `experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib` | LightGBM Model | 49-feature adapted GBDT PM2.5 regressor |
| `experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm10.joblib` | LightGBM Model | 49-feature adapted GBDT PM10 regressor |
| `data/processed/dashboard_demo_sequences.csv` | CSV Dataset | Curated 24h antecedent test sequences |
| `data/processed/location_catalog.csv` | CSV Catalog | Andhra Pradesh station and city catalog |
| `data/processed/station_metadata_90.csv` | CSV Metadata | National CPCB station metadata |
| `assets/` | Static Images | Hero banners, regional maps, logos |

---

### Category C: Obsolete Files Proposed for Deletion
The following files were verified via `git grep` and AST inspection to have **0 active references** across application runtime, tests, configuration, and deployment scripts. They are temporary regex/patch scripts from prior development phases:

| File Path | Description | Risk | Verification Performed |
| :--- | :--- | :--- | :--- |
| `scripts/patch_app.py` | Temporary string replacement patch for `app.py` | Safe | 0 external imports; contains stale asserts |
| `scripts/patch_dashboard_scientific_takeaway.py` | One-off regex replacement script | Safe | 0 external imports; one-off utility |
| `scripts/patch_dashboard_section4.py` | One-off section 4 regex patch | Safe | 0 external imports; one-off utility |
| `scripts/patch_model_performance.py` | One-off performance page regex patch | Safe | 0 external imports; one-off utility |
| `scripts/patch_phase12_dashboard.py` | One-off Phase 12 dashboard patch | Safe | 0 external imports; one-off utility |
| `scripts/patch_phase12_fixes.py` | One-off Phase 12 service fix patch | Safe | 0 external imports; one-off utility |
| `scripts/patch_sensor_service.py` | One-off sensor service patch | Safe | 0 external imports; one-off utility |
| `scripts/patch_services.py` | One-off service patch script | Safe | 0 external imports; one-off utility |
| `scripts/update_dashboard_pages.py` | One-off dashboard pages patch script | Safe | 0 external imports; one-off utility |
| `scripts/update_services_metrics.py` | One-off service metric patch script | Safe | 0 external imports; one-off utility |
| `scripts/upgrade_dashboard_app.py` | One-off app upgrade patch script | Safe | 0 external imports; one-off utility |
| `scripts/fix_test_assert.py` | 6-line one-off assertion replacement script | Safe | 0 external imports; one-off utility |
| `scripts/fix_test_sys_paths.py` | One-off sys.path injector script | Safe | 0 external imports; one-off utility |

---

### Category D: Historical & Intentional Assets Retained (Not Deleted)
Per Safety Rule 1 ("Preserve important historical assets if they are intentionally part of the project"), the following directories and files are retained without alteration:
- `paper/`: Academic paper manuscript, LaTeX files, and figures (preserved outside `src/`).
- `presentation/`: Slide decks and speaker notes (preserved outside `src/`).
- Root training scripts (`train_lstm.py`, `train_pm25vision_cnn.py`, `merge_air_quality_weather.py`, `prepare_model_sequences.py`): Preserved for full training pipeline reproducibility.
- All 29 test files in `tests/`: Preserved to guarantee continuous regression coverage.

---

## 3. Code & UI Refactoring Plan

1. **`app.py` Image Asset Optimization**:
   - Replace the 16,740-character inline base64 string `SCENIC_AP_B64` with a clean call to `get_base64_image(ASSETS_DIR / "scenic_ap.jpg")` or fall back cleanly.
2. **`app.py` Navigation & Title Polish**:
   - Update sidebar navigation sub-label: `"Research Metrics & Uncertainty"` $\to$ `"Operational Validation & Metrics"`.
3. **`src/components/hybrid_dashboard_pages.py` Duplicate Cleanup**:
   - Remove redundant `### 🔒 Security & Credential Protection` duplicate header block.
   - Update academic headings (e.g. `"Model Performance & Scientific Benchmarks"` $\to$ `"Model Performance & Operational Benchmarks"`).
   - Ensure clear, user-friendly notices for missing/stale data.
4. **Git Deletion of Obsolete Patch Scripts**:
   - Perform tracked `git rm` on the 13 verified one-off patch scripts.
5. **Concurrency Bug Fix**:
   - Upgraded `OperationalHealthService._state_lock` from `threading.Lock()` to `threading.RLock()` to prevent re-entrant deadlock when `get_system_health()` evaluates `get_cache_age_seconds()`.
6. **Full Regression Verification**:
   - Execute baseline and end-to-end test suites. Confirm 100% pass rate across all 8 dashboard sections and operational suites.

---

## 4. Verification & Validation Summary

| Test Suite | Tests Run | Result | Execution Time |
| :--- | :--- | :--- | :--- |
| `tests/test_operational_monitoring_and_reliability.py` | 17 | **17 PASSED** | 47.35s |
| `tests/test_live_data_provenance_and_validation.py` | 11 | **11 PASSED** | 35.10s |
| `tests/test_data_provenance_and_integrity.py` | 7 | **7 PASSED** | 34.02s |
| `tests/test_forecast_time_alignment.py` | 8 | **8 PASSED** | 28.12s |
| `tests/test_model_input_tensor_alignment.py` | 6 | **6 PASSED** | 22.20s |
| `tests/test_eight_sections_smoke.py` | 8 | **8 PASSED** | 52.10s |
| `tests/test_dashboard_navigation.py` (AppTest E2E across all 8 sections) | 8 | **8 PASSED** | 74.00s |

**Total Verified Passes:** 65 / 65 test assertions with 0 failures, 0 regressions, and 0 unhandled exceptions across all 8 UI sections.

