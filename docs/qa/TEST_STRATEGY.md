# VayuDrishti QA Test Strategy

This document details the multi-tiered testing strategy employed across the VayuDrishti platform to guarantee system correctness, statistical reliability, data provenance, and crash resilience.

---

## 1. Testing Pyramid & Verification Levels

```text
               ▲
              / \
             /E2E\             Streamlit AppTest & UI Navigation
            /-----\
           / Integ \           Live Ingestion, Buffering, Audit Logging
          /---------\
         / Validatn  \         8-Gate Sequence Validation, Temporal Alignments
        /-------------\
       /     Unit      \       Algorithms, Math Helpers, AQI Sub-indices, Scalers
      /-----------------\
```

### Level 1: Unit Testing (Isolated Components)
- **Scope**: Mathematical transformations, CPCB NAQI formula calculations, date/time parsing, feature normalization scalers, and metric tracking.
- **Dependencies**: Pure Python, NumPy, Pandas, Scikit-learn. No network access or disk writes required.

### Level 2: Sequence Validation & Data Quality
- **Scope**: Rigorous 8-gate input validation before data reaches ML models:
  1. Structural row count (strictly 24 antecedent hourly steps).
  2. Chronological sequence order (strictly ascending).
  3. Strict 1-hour interval frequency ($3600 \pm 60$ seconds).
  4. Duplicate timestamp rejection.
  5. Feature schema integrity ($27$ features for LSTM, $49$ for GBDT).
  6. Station identity preservation (no cross-station contamination).
  7. Sequence freshness threshold ($\le 3.0$ hours from issue time).
  8. Numerical sanity bounds (PM2.5 $\in [0, 1000]$, PM10 $\in [0, 1500]$, no NaNs).
- **Dependencies**: `src.prediction.sequence_validator`, `src.prediction.station_data_quality`.

### Level 3: Integration & Ingestion Testing
- **Scope**: Ingestion resiliency, exponential backoff, station-isolated thread locks, 24-hour continuous rolling buffers, hourly time binning, and audit record generation.
- **Dependencies**: `src.data.resilient_ingestion`, `src.data.live_rolling_buffer`, `src.services.forecast_audit_logger`.

### Level 4: Model Inference & Multimodal Fusion Testing
- **Scope**: PyTorch LSTM sequence inference, LightGBM adapted GBDT inference, split-conformal uncertainty intervals, ResNet-50 visual feature extraction, and gated residual multimodal fusion ($y_{hybrid} = y_{sensor} + \alpha \cdot \text{clip}(\Delta_{img})$).
- **Dependencies**: Trained PyTorch checkpoints (`models/lstm_baseline/best_lstm_model.pt`), LightGBM models (`experiments/accuracy_optimization_v2/checkpoints/`), Scikit-learn scalers.

### Level 5: Dashboard & End-to-End Testing
- **Scope**: Full navigation across all 8 dashboard sections, widget interaction, state caching, session management, and Streamlit `AppTest` headless execution.
- **Dependencies**: `streamlit.testing.v1.AppTest`, `app.py`.

---

## 2. Test Implementation Classification

To provide complete transparency to developers and auditors, tests are categorized by their execution requirements and live status:

### Category A: Fully Implemented & Currently Passing (Offline / Mocked)
These tests run in any standard CI/CD or local development environment without API keys or internet access:
- `tests/test_operational_monitoring_and_reliability.py` (17 tests) — Verified passing.
- `tests/test_live_data_provenance_and_validation.py` (11 tests) — Verified passing.
- `tests/test_data_provenance_and_integrity.py` (7 tests) — Verified passing.
- `tests/test_forecast_time_alignment.py` (8 tests) — Verified passing.
- `tests/test_model_input_tensor_alignment.py` (6 tests) — Verified passing.
- `tests/test_eight_sections_smoke.py` (8 tests) — Verified passing.
- `tests/test_dashboard_navigation.py` (8 sections) — Verified passing.
- `tests/test_accuracy_optimization_v2.py` — Verified passing.
- `tests/test_standalone_gbdt_inference.py` — Verified passing.

### Category B: Live Network Tests (Requiring External APIs)
These tests make live HTTPS calls to upstream providers and require an active internet connection:
- `tests/test_live_openaq_connectivity.py`: Queries `https://api.openaq.org/v3/locations`.
  - *Status*: Passes when network is reachable. Automatically falls back to cached/mocked responses when offline.
- `tests/test_live_rolling_buffer.py`: Tests live telemetry polling.
  - *Requirement*: Subject to OpenAQ public rate limits (free tier).

### Category C: Tests Requiring Dedicated Hardware or Long Runtimes
- Full multi-seed model retraining scripts (`train_lstm.py`, `train_ablation_multiseed.py`).
  - *Status*: Validated offline; not run on continuous integration to conserve compute.

### Category D: Unimplemented Tests (Explicitly Documented)
The following tests are **not currently implemented** and must not be claimed as active:
- Automated mobile webview responsive touch gesture testing.
- Automated multi-browser cross-compatibility testing (Safari, Edge, Firefox headless).
- Automated stress testing exceeding 10,000 concurrent Streamlit WebSocket sessions.

---

## 3. Regression Testing & Safe Refactoring Rules

1. **Model Weights Protection**: No test or refactoring script may modify or overwrite `.pt`, `.pth`, `.joblib`, or `.pkl` files in `models/`.
2. **Formula Invariance**: The LSTM-GBDT blending ratios (40/60, 45/55), severity gating thresholds (250 / 400 $\mu\text{g/m}^3$), and conformal quantiles must remain invariant.
3. **Audit Verification**: Every forecast execution must append an entry to `forecast_audit_log.jsonl` with an immutable `run_id`.
