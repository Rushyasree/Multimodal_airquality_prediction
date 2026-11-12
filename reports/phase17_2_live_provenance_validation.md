# VayuDrishti — Phase 17.2: Live Data Provenance & End-to-End Forecast Validation Report

**Repository:** `https://github.com/Rushyasree/Multimodal_airquality_prediction`  
**Execution Timestamp:** November 12, 2026  
**Baseline Verified:** Phase 17.1 (Strict sequence validation, forecast time alignment, tensor digest verification)  
**Status:** **Fully Implemented & Verified (32/32 Tests Passing)**

---

## 1. Executive Summary

Phase 17.2 establishes an authentic, scientifically traceable live forecasting pathway for the VayuDrishti platform without modifying trained neural network weights (`models/lstm_baseline/best_lstm_model.pt`, `models/multimodal/best_model.pt`), historical benchmark evaluation metrics, or Phase 8/9 empirical trial results.

In Phase 17.1, we enforced strict sequence integrity on curated benchmark sequences, verified forecast issue time alignment ($t \to t+1$), and proved identical SHA-256 tensor digests across LSTM and Hybrid models. In Phase 17.2, we solved the foundational challenge of **authentic live data ingestion**: connecting real physical monitoring observations into the forecasting pipeline while guaranteeing that unmeasured estimates, regional reanalyses, and synthetic fallbacks can **never** be misrepresented as live ground telemetry.

### Core Guarantees Delivered
1. **Zero Silent Synthetic Replacement:** When fewer than 24 contiguous physical hourly measurements are present in a station's live buffer, the forecasting pipeline explicitly returns `status="unavailable"` and `LIVE INPUT UNAVAILABLE` with `pm25=None, pm10=None`. Synthetic or demo sequences are never substituted without explicit opt-in disclosure.
2. **Multi-Gate Live Validation:** Implemented `validate_live_forecast_sequence` with 7 independent, verifiable boolean gates (`sequence_structural_validity`, `station_identity_validity`, `source_authenticity`, `measurement_completeness`, `freshness_validity`, `scaler_compatibility`, and `live_forecast_eligible`).
3. **Traceable Provenance Badges:** Standardized explicit provenance classifications across backend and UI (`LIVE MEASURED OBSERVATION`, `LIVE FORECAST — VALIDATED INPUT`, `HISTORICAL DEMO FORECAST`, `REGIONAL MODEL ESTIMATE`, `IMAGE-BASED EXPERIMENTAL ESTIMATE`, `SYNTHETIC DEMONSTRATION`, and `LIVE INPUT UNAVAILABLE`).
4. **Deterministic Sub-Hourly Aggregation:** Raw sub-hourly monitor readings within clock-hour intervals $[H:00, H:59]$ UTC are aggregated via arithmetic mean with sample count tracking and deduplication.
5. **Synchronized Model Comparison & Persistence Baseline:** Added live physical telemetry mode to `ModelComparisonService`. When live telemetry is complete, SHA-256 tensor parity is verified between LSTM and Hybrid pipelines; when incomplete, synchronization claims are strictly rejected. Naïve 1-hour persistence baseline ($y_{t+1} = y_t$) is displayed alongside all parameterized models.

---

## 2. Upstream Data Source Architecture & Audit

| Data Stream | Upstream Provider / Inlet | Acquisition Cadence | Reporting Latency | Measurement Status |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Air Quality** | CPCB CAAQMS via OpenAQ v3 REST API | Sub-hourly (15m to 60m irregular) | 1 to 4 hours post-sampling | `LIVE_MEASURED` (Direct physical instrument) |
| **Atmospheric Weather** | Open-Meteo European Weather Ensemble | Hourly synced | ~15 minutes | `MODELED_REANALYSIS` (Numerical Weather Prediction) |
| **Historical Benchmark** | CPCB 2015–2020 Curated Holdout (404k sequences) | Static benchmark file | Zero (Archival) | `HISTORICAL_DEMO` (Held-out physical measurements) |
| **Simulated Demonstration** | Synthetic fallback generator | On-demand procedural | Zero | `SYNTHETIC_FALLBACK` (Non-physical demonstration) |

### Real-World Upstream Characteristics
- **Timestamp Irregularities:** Physical CPCB stations report on non-deterministic clock offsets (e.g. 10:14, 10:28).
- **Network Outages:** Stations frequently experience intermittent telemetry dropouts, resulting in non-contiguous intervals.
- **Reporting Latency:** Data from municipal CAAQMS monitors typically reaches central servers with a 1-to-3 hour physical telemetry lag.

---

## 3. Architecture & Implementation Details

### A. Explicit Data Provenance Schema (`src/prediction/observation_provenance.py`)
- **`ObservationProvenanceCategory`**: Enum defining mutually exclusive statuses: `LIVE_MEASURED`, `LIVE_PARTIAL`, `HISTORICAL_DEMO`, `REGIONAL_PROXY`, `MODELED_REANALYSIS`, `SYNTHETIC_FALLBACK`, `INVALID`.
- **`ObservationRecord`**: Immutable dataclass recording single-reading provenance (station ID, parameter, value, unit, UTC measurement timestamp, source provider, retrieval timestamp, sample count, and raw value).
- **`SequenceProvenanceMeta`**: Sequence audit container reporting 24h window completeness percentage, strictly measured flag, aggregation method, freshness, and status badge.

### B. Strict Live Sequence Builder (`src/prediction/live_sequence_builder.py`)
- Manages station-isolated rolling buffers (`LiveRollingBufferRegistry`).
- Aggregates sub-hourly measurements into deterministic clock hours $[H:00, H:59]$ UTC.
- Enforces physical plausibility bounds (rejecting negative concentrations or extreme artifacts like PM2.5 > 1000 $\mu\text{g/m}^3$).
- If fewer than 24 contiguous measured hours exist, returns `(None, meta)` with `meta.provenance_category = LIVE_PARTIAL` and `meta.status_badge = "LIVE INPUT UNAVAILABLE"`. Zero forward filling, zero median substitution, and zero synthetic interpolation are performed.

### C. Multi-Gate Live Validation (`src/prediction/sequence_validator.py`)
- `validate_live_forecast_sequence(...)` evaluates 7 independent gates:
  1. `sequence_structural_validity`: Exactly 24 rows, strictly ascending chronological order, exact 1-hour interval cadence ($\Delta t = 1.0\,\text{h}$), zero duplicate timestamps.
  2. `station_identity_validity`: `StationId` strictly matches requested station ID; sequence cannot mix multiple stations.
  3. `source_authenticity`: Must originate from physical monitor inlets (`LIVE_MEASURED`). Synthetic, historical demo, or regional proxy data fail this gate.
  4. `measurement_completeness`: Zero missing or null values across required criteria pollutants (PM2.5 and PM10).
  5. `freshness_validity`: Sequence endpoint age relative to reference system time must satisfy $0 \le \text{age} \le \text{max\_age\_hours}$ (default 3.0h).
  6. `scaler_compatibility`: Scaler metadata must contain all 27 canonical LSTM features with non-zero variances.
  7. `live_forecast_eligible`: Conjunction of all 6 preceding gates.

### D. End-to-End Service Integration
- **`SensorForecastService` (`src/services/sensor_forecast_service.py`)**:
  - Added `live_mode=True` and `reference_time` parameters.
  - In live mode, runs live sequence acquisition and `validate_live_forecast_sequence`.
  - If validation fails or buffer is incomplete, returns `status="unavailable"`, `pm25=None`, `pm10=None`, and human-auditable rejection reasons.
- **`HybridPredictionService` (`src/services/hybrid_prediction_service.py`)**:
  - In live mode, verifies the 24h antecedent sequence.
  - Eliminated silent `torch.zeros((1, 24, 27))` fallbacks. If sensor input is rejected, returns `status="unavailable"` with `pm25=None, pm10=None`.
- **`ModelComparisonService` (`src/services/model_comparison_service.py`)**:
  - In live mode, ingests from `LiveSequenceBuilder`.
  - When buffer is incomplete, both LSTM and Hybrid are marked `LIVE INPUT UNAVAILABLE`, `tensor_matched=False`, and synchronization status reports `"Live Synchronization Rejected (Incomplete Measured History)"`.
  - When buffer is complete, asserts SHA-256 tensor equality and reports `"Strictly Synchronized (Authentic Live Telemetry • Identical Tensor Digest)"`.
  - Includes naïve 1-hour **Persistence Baseline** row ($y_{t+1} = y_t$).

### E. Dashboard User Experience (`src/components/hybrid_dashboard_pages.py`)
- Added **Data Ingestion Horizon** selector to Section 4 (Side-by-Side Model Comparison):
  - `Historical Benchmark Holdout (Reproducible Demo)`
  - `Live Physical Telemetry (Authentic Ingestion)`
- Displays clear visual alerts when live telemetry is unavailable due to buffer incompleteness.
- Gracefully handles `None` pollutant concentrations in comparison tables and Plotly bar charts without numerical crashes.

---

## 4. Verification & Test Suite Results

All 32 automated verification and regression tests executed and passed cleanly:

```bash
pytest tests/test_live_data_provenance_and_validation.py tests/test_forecast_time_alignment.py tests/test_model_input_tensor_alignment.py tests/test_data_provenance_and_integrity.py -v
```

### Test Suite Summary:
1. `tests/test_live_data_provenance_and_validation.py` (13 tests — Phase 17.2 Core):
   - `test_24h_contiguous_sequence_construction`: **PASSED**
   - `test_incomplete_buffer_rejection_without_synthetic_substitution`: **PASSED**
   - `test_sub_hourly_aggregation_and_deduplication`: **PASSED**
   - `test_physical_bounds_filtering`: **PASSED**
   - `test_multi_gate_validation_passes_on_authentic_fresh_sequence`: **PASSED**
   - `test_multi_gate_validation_rejects_stale_sequence`: **PASSED**
   - `test_multi_gate_validation_rejects_non_authentic_sources`: **PASSED**
   - `test_sensor_forecast_service_live_pathway`: **PASSED**
   - `test_sensor_forecast_service_rejects_stale_in_live_mode`: **PASSED**
   - `test_hybrid_prediction_service_live_pathway`: **PASSED**
   - `test_live_synchronization_and_tensor_digest_parity`: **PASSED**
   - `test_incomplete_buffer_rejects_live_synchronization`: **PASSED**
   - `test_persistence_baseline_correctness`: **PASSED**

2. `tests/test_forecast_time_alignment.py` (11 tests — Phase 17.1 Temporal Horizon):
   - All 11 tests **PASSED** (24 rows, chronological monotonicity, 1h cadence, station isolation, $t \to t+1$ target alignment).

3. `tests/test_model_input_tensor_alignment.py` (3 tests — Phase 17.1 Tensor Parity):
   - All 3 tests **PASSED** (tensor identity on scenarios, SHA-256 parity reporting, mismatch detection on tampered input).

4. `tests/test_data_provenance_and_integrity.py` (5 tests — Transparency & Baselines):
   - All 5 tests **PASSED** (calibrated baseline never labeled live, Open-Meteo labeled reanalysis, sequence retrieval, zero random drift, synchronized sequence).

5. Smoke & Navigation Suites (`tests/test_eight_sections_smoke.py`, `tests/test_dashboard_navigation.py`):
   - All 9 dashboard sections and navigation callbacks **PASSED**.

---

## 5. Photographic Modality Disclosure

In strict compliance with user instructions:
- **Calendar-Date Precision:** Public camera datasets (e.g. PM25Vision) record capture dates without verified sub-daily clock hours.
- **Asynchronous Optical Haze Prior:** The fine-tuned ResNet-50 visual estimator operates strictly as an auxiliary optical haze feature extractor ($R^2 \approx 0.15$ standalone). It is **never** presented as an autonomous, high-frequency camera sensor.
- When outdoor imagery is unselected or missing, the gated multimodal fusion layer automatically drives $\alpha \to 0$ and executes verified sensor-telemetry fallback with zero numerical degradation.

---

## 6. Real-World Telemetry Limitations & Engineering Boundaries

1. **Station Uptime & Maintenance:** Real-world CAAQMS monitors undergo routine inlet filter changes, sensor calibrations, and telemetry downtime. An authentic live pipeline will legitimately reject live forecasting during these windows. This is scientifically correct behavior.
2. **Upstream API Quotas:** The OpenAQ v3 public tier enforces rate limits. Production deployments must maintain an asynchronous worker buffer (such as `AutonomousLiveCollector`) to pre-aggregate hourly records rather than issuing high-frequency burst queries during user page renders.
3. **No Imputation Rule:** Unlike research environments that utilize spline or forward-fill imputation to achieve 100% data coverage, VayuDrishti Phase 17.2 adheres to a strict zero-silent-imputation standard for live forecasts.

---

## 7. Conclusion

Phase 17.2 successfully delivers an authentic, auditable, and scientifically defensible end-to-end live forecasting system for VayuDrishti. Model predictions remain reproducible, time horizons remain mathematically aligned ($t \to t+1$), tensor parity is cryptographically verified, and data provenance is transparently disclosed across all dashboard interfaces.
