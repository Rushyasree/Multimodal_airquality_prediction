# VayuDrishti QA Release & Deployment Checklist

This evidence-based checklist defines the verification gates that must be satisfied before deploying a release of the VayuDrishti platform to production or publishing a version tag.

---

## 1. Mandatory Release Gates (Blockers)

All checks in this section **must pass (100%)** before release approval.

### A. Test Suite & Code Quality
- [ ] **Operational Health Suite**: `pytest tests/test_operational_monitoring_and_reliability.py` passes with 0 failures.
- [ ] **Provenance & Validation Suite**: `pytest tests/test_live_data_provenance_and_validation.py tests/test_data_provenance_and_integrity.py` passes with 0 failures.
- [ ] **Temporal & Tensor Alignment Suite**: `pytest tests/test_forecast_time_alignment.py tests/test_model_input_tensor_alignment.py` passes with 0 failures.
- [ ] **Smoke Suite (8 Sections)**: `python tests/test_eight_sections_smoke.py` passes with all 8 sections verified.
- [ ] **Headless AppTest Navigation**: `python tests/test_dashboard_navigation.py` executes without any unhandled exceptions across all 8 navigation tabs.

### B. Machine Learning Checkpoint Integrity
- [ ] **PyTorch LSTM Weights**: `models/lstm_baseline/best_lstm_model.pt` exists and loads cleanly.
- [ ] **ResNet-50 Vision Weights**: `models/image_estimator/resnet50_weights.pth` exists and initializes cleanly.
- [ ] **LightGBM Checkpoints**: `experiments/accuracy_optimization_v2/checkpoints/best_gbdt_pm25.joblib` and `best_gbdt_pm10.joblib` exist and are loadable via `joblib`.
- [ ] **Feature Scaler**: `models/scaler_27features.pkl` exists and matches the 27-channel LSTM feature schema.

### C. Live Telemetry & Ingestion Protection
- [ ] **Fallback Grace Period**: When external APIs (OpenAQ / Open-Meteo) are unreachable, the application falls back gracefully without unhandled exceptions.
- [ ] **Sequence Freshness**: Telemetry older than 3.0 hours is explicitly flagged as `STALE` and rejected for live forecasting.
- [ ] **Rate Limiting**: In-memory caching prevents duplicate requests to OpenAQ within the 3600-second TTL window.

### D. Security & Credential Protection
- [ ] **Zero API Token Leakage**: No `.env` files, API keys, or JWT tokens are committed to git or exposed in frontend HTML/markdown.
- [ ] **Sanitized Audit Logs**: All error messages written to `experiments/phase17_3/forecast_audit_log.jsonl` are scrubbed of sensitive strings.

### E. Dashboard User Experience & Separation
- [ ] **Product Focus**: Dashboard contains **zero** research manuscript drafts, IEEE submission tracking, or paper review checklists.
- [ ] **Responsive Navigation**: Top navbar and sidebar radio buttons switch sections cleanly.

---

## 2. Recommended & Operational Checks (Non-Blocking)

- [ ] **Container Build**: `docker build -t vayudrishti:latest .` compiles cleanly.
- [ ] **Dependencies Check**: `pip check` reports zero broken package requirements.
- [ ] **Streamlit Cloud Status**: Verify live application at `https://vayudrishti-airquality.streamlit.app/` connects to OpenAQ.
- [ ] **Documentation Links**: All internal markdown links resolve to valid files.

---

## 3. Release Sign-Off Template

```text
Release Version: vX.Y.Z
Commit Hash: [SHA]
Date: YYYY-MM-DD
Evaluator: [Name / Role]

Mandatory Gates: [PASS / FAIL]
Blocking Issues: [None / List]
Recommendation: [APPROVED FOR DEPLOYMENT / BLOCKED]
```
