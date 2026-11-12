# VayuDrishti Quality Assurance (QA) & Reliability Framework

Welcome to the **VayuDrishti QA Knowledge Base**. This documentation serves developers, researchers, and system evaluators by defining the testing methodologies, operational validation gates, coverage metrics, and bug management workflows for the VayuDrishti platform.

---

## 1. What QA Means in VayuDrishti

In an operational environmental forecasting platform, Quality Assurance goes far beyond standard unit testing. High-stakes public health and municipal advisories depend on accurate data:
- **Numerical & Physical Provenance**: The platform must never misrepresent historical benchmark sequences or synthetic data as live telemetry.
- **Temporal Alignment**: All predictions ($t+1$) must strictly respect causality without future data leakage, utilizing exactly 24 continuous antecedent hours ($t-23 \dots t$).
- **Strict Fallback Resilience**: When upstream API telemetry (OpenAQ / Open-Meteo) drops or camera feeds are unavailable, models must degrade gracefully rather than crash.
- **Zero Secret Exposure**: API keys, credentials, and internal error stack traces must never leak to user-facing clients or backend logs.
- **Model Output Reproducibility**: Input tensors passed into deep recurrent (PyTorch LSTM) and gradient boosting (LightGBM) models must match frozen offline training schemas with verified SHA-256 tensor digests.

---

## 2. Core Components Requiring Testing

| Subsystem | Key Modules | Primary Verification Objectives |
| :--- | :--- | :--- |
| **Telemetry Ingestion** | `src.data.resilient_ingestion`, `src.prediction.current_air_quality_service` | Rate-limiting, caching, exponential backoff, OpenAQ API v3 integration |
| **Buffer & Sequences** | `src.data.live_rolling_buffer`, `src.prediction.live_sequence_builder` | 24h rolling buffer continuity, hourly aggregation, gap detection |
| **Data Validation** | `src.prediction.sequence_validator`, `src.prediction.station_data_quality` | 8-gate sequence validation, physical bounds checks, rate-of-change filters |
| **Forecasting Engines**| `src.services.sensor_forecast_service`, `src.prediction.lstm_forecaster` | 2-layer LSTM ($27$ features), GBDT ($49$ features), conformal uncertainty |
| **Multimodal Hybrid**  | `src.services.hybrid_prediction_service`, `src.services.image_estimation_service` | Optical haze feature extraction, gated residual blending, missing-image fallback |
| **Operational Health** | `src.services.operational_health_service`, `src.services.forecast_audit_logger` | State transitions (`HEALTHY`, `DEGRADED`, `STALE`), audit JSONL logging |
| **Web Dashboard**      | `app.py`, `src.components.hybrid_dashboard_pages` | 8-section UI navigation, widget responsiveness, zero unhandled exceptions |

---

## 3. QA Documentation Structure

The QA Knowledge Base is organized into the following specialized guides:

- [**Test Strategy**](TEST_STRATEGY.md): Comprehensive testing pyramid, layer definitions, and test categorizations (implemented vs mocked vs live).
- [**Test Execution Guide**](TEST_EXECUTION.md): Step-by-step CLI commands, environment setup, and failure investigation protocols.
- [**QA Coverage Matrix**](QA_COVERAGE_MATRIX.md): Detailed inventory of tested components, test types, file locations, and known limitations.
- [**Bug & Issue Management**](BUG_REPORTING.md): Severity classification, triage guidelines, reproduction format, and fix verification rules.
- [**Release Checklist**](RELEASE_CHECKLIST.md): Evidence-based gatekeeper checklist required before any production release.

---

## 4. How to Run Tests Quickly

```bash
# 1. Activate project environment
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate       # Linux / macOS

# 2. Run the core operational monitoring test suite (17 tests)
pytest tests/test_operational_monitoring_and_reliability.py -v

# 3. Run the live data provenance and validation suite (18 tests)
pytest tests/test_live_data_provenance_and_validation.py tests/test_data_provenance_and_integrity.py -v

# 4. Run the end-to-end 8-section dashboard smoke test (8 tests)
python tests/test_eight_sections_smoke.py

# 5. Run full Streamlit AppTest navigation verification (8 UI sections)
python tests/test_dashboard_navigation.py
```

---

## 5. Where Test Results & Audits are Recorded

- **Runtime Audit Log**: `experiments/phase17_3/forecast_audit_log.jsonl` (Append-only immutable record of every execution).
- **Matched Target Evaluations**: `experiments/phase17_3/matched_evaluations.jsonl` (Realized error comparing forecast vs actual observation at $t+1$).
- **Historical Phase Audits**: `docs/archive/research_audits/` and `reports/`.

---

## 6. How to Report Bugs & Verify Fixes

1. Check existing open issues on GitHub: [Issues Page](https://github.com/Rushyasree/Multimodal_airquality_prediction/issues).
2. If the bug is unrecorded, open a new ticket using the [Bug Report Template](../../.github/ISSUE_TEMPLATE/bug_report.md).
3. Include environment details, exact steps to reproduce, actual vs expected outputs, and sanitized logs.
4. When submitting a fix, follow [docs/qa/BUG_REPORTING.md](BUG_REPORTING.md) to provide regression test evidence.
