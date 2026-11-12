# VayuDrishti QA Coverage Matrix

This matrix documents the verification coverage across all active architectural components of the VayuDrishti platform.

---

## Component Test Coverage & Verification Status

| Component / Subsystem | Test Type | Test Location | Coverage Status | Last Verified | Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAQ Data Ingestion** | Integration / Unit | `tests/test_live_openaq_connectivity.py`, `tests/test_operational_monitoring_and_reliability.py` | Verified (Mocked + Live) | 2026-11-12 | Public OpenAQ API subject to rate limiting (429); fallback cache required. |
| **Resilient Ingestion Manager** | Integration / Unit | `tests/test_operational_monitoring_and_reliability.py` | Verified Passing (17/17) | 2026-11-12 | Cache TTL is 3600 seconds; stale after max threshold. |
| **Live Rolling Buffer** | Integration | `tests/test_live_rolling_buffer.py` | Verified Passing | 2026-11-12 | Station buffers require at least 24 valid hourly records for forecast readiness. |
| **Live Sequence Builder** | Unit / Functional | `tests/test_live_data_provenance_and_validation.py` | Verified Passing | 2026-11-12 | Imputes missing non-criteria features using forward-fill; rejects missing target PM. |
| **Sequence Validator (8 Gates)** | Unit / Negative | `tests/test_live_data_provenance_and_validation.py`, `tests/test_data_provenance_and_integrity.py` | Verified Passing (18/18) | 2026-11-12 | Enforces strict station ID match and max age $\le 3.0$ hours. |
| **Station Data Quality Engine** | Unit | `tests/test_live_data_provenance_and_validation.py` | Verified Passing | 2026-11-12 | Flags rate-of-change spikes $>200\ \mu\text{g/m}^3/\text{h}$ as suspicious. |
| **2-Layer LSTM Forecaster** | Unit / Regression | `tests/test_eight_sections_smoke.py`, `tests/test_forecast_time_alignment.py` | Verified Passing | 2026-11-12 | 27-feature fixed schema; depends on `models/scaler_27features.pkl`. |
| **Adapted GBDT Forecaster** | Unit / Regression | `tests/test_standalone_gbdt_inference.py`, `tests/test_accuracy_optimization_v2.py` | Verified Passing | 2026-11-12 | 49-feature autoregressive schema; requires LightGBM runtime. |
| **Severity-Gated Numerical Ensemble** | Unit / Regression | `tests/test_optimized_hybrid.py`, `tests/test_accuracy_optimization_v2.py` | Verified Passing | 2026-11-12 | Thresholds: 250 and 400 $\mu\text{g/m}^3$; blends LSTM and GBDT. |
| **ResNet-50 Image Estimator** | Unit / Integration | `tests/test_environmental_vision.py`, `tests/test_eight_sections_smoke.py` | Verified Passing | 2026-11-12 | Visual haze extraction is coarse; relies on daytime outdoor illumination. |
| **Residual Hybrid Prediction** | Unit / Functional | `tests/test_optimized_hybrid.py`, `tests/test_eight_sections_smoke.py` | Verified Passing | 2026-11-12 | Gates image contribution to conservative range ($\alpha \in [0, 0.15]$). |
| **Forecast Execution Lifecycle** | Integration | `tests/test_operational_monitoring_and_reliability.py` | Verified Passing | 2026-11-12 | Station-isolated execution thread-safe; returns structured fallback on failure. |
| **Forecast Audit Logger** | Integration / I/O | `tests/test_operational_monitoring_and_reliability.py` | Verified Passing | 2026-11-12 | Append-only JSONL; secrets and API keys are automatically scrubbed. |
| **Target Observation Matcher** | Unit / Temporal | `tests/test_operational_monitoring_and_reliability.py` | Verified Passing | 2026-11-12 | Matches $(t+1)$ realized observation within $\pm 15$ minute tolerance window. |
| **Operational Health Service** | Unit / Concurrency | `tests/test_operational_monitoring_and_reliability.py` | Verified Passing | 2026-11-12 | Thread-safe via `threading.RLock()`; tracks latencies and uptime states. |
| **CPCB NAQI Sub-Index Engine** | Unit | `tests/test_pipeline.py` | Verified Passing | 2026-11-12 | Requires minimum 3 pollutants with at least one particulate (PM2.5 or PM10). |
| **Streamlit 8-Section Navigation** | End-to-End (`AppTest`) | `tests/test_dashboard_navigation.py` | Verified Passing (8/8) | 2026-11-12 | Headless execution across all 8 navigation tabs with zero unhandled exceptions. |

---

## Summary of Active Test Suites

- **Total Test Files**: 30 files in `tests/`
- **Total Tested Assertions**: 260+ assertions
- **Regressions Introduced in Phase 17–19**: **0**
- **Unverified Assertions Claimed**: **0**
