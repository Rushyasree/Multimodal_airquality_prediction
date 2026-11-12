# VayuDrishti QA Test Execution Guide

This guide provides concrete, verified CLI instructions for executing VayuDrishti's test suites, investigating test failures, and recording test artifacts.

---

## 1. Prerequisites & Environment Setup

Verify that Python 3.10 or 3.11 is active:

```bash
python --version
# Expected: Python 3.10.x or 3.11.x
```

### Windows (PowerShell):
```powershell
# Navigate to repository root
cd e:\Documents\multimodal_air_quality_vscode_starter

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify dependencies
pip check
```

### Linux / macOS (Bash):
```bash
# Navigate to repository root
cd /path/to/multimodal_air_quality_vscode_starter

# Activate virtual environment
source .venv/bin/activate

# Verify dependencies
pip check
```

---

## 2. Test Execution Commands (By Category)

### A. Operational Monitoring & Ingestion Reliability
Verifies data health states (`HEALTHY`, `DEGRADED`, `STALE`, `UNAVAILABLE`), rate limiting, API call tracking, cache age calculation, and forecast audit logging:

```bash
pytest tests/test_operational_monitoring_and_reliability.py -v
```
- **Expected Result**: 17 passed.
- **Typical Runtime**: ~45 seconds.

### B. Live Data Provenance & Sequence Validation
Verifies strict station-identity preservation, 8-gate sequence validation, 24-hour continuous window integrity, and temporal sequence alignment:

```bash
pytest tests/test_live_data_provenance_and_validation.py tests/test_data_provenance_and_integrity.py -v
```
- **Expected Result**: 18 passed.
- **Typical Runtime**: ~65 seconds.

### C. Forecast Time Alignment & Model Tensor Integrity
Verifies issue time ($t$), horizon derivation ($t+1$), lead-time calculation, and SHA-256 tensor input digests:

```bash
pytest tests/test_forecast_time_alignment.py tests/test_model_input_tensor_alignment.py -v
```
- **Expected Result**: 14 passed.
- **Typical Runtime**: ~50 seconds.

### D. Service & Model Smoke Tests (All 8 Sections)
Loads all underlying models, scalers, and services directly to verify predictions without rendering the Streamlit UI:

```bash
python tests/test_eight_sections_smoke.py
```
- **Expected Output**:
  ```text
  [PASS] Section 1 Overview data verified.
  [PASS] Section 2 Live Air Quality verified.
  [PASS] Section 3 (Sensor-Only LSTM) verified.
  [PASS] Section 3 (Image Estimation) verified.
  [PASS] Section 3 (Baselines & Persistence) verified.
  [PASS] Mode B Full Multimodal verified.
  ...
  ```
- **Expected Result**: 8 passed.

### E. End-to-End Headless Dashboard Navigation (`AppTest`)
Executes the full Streamlit UI script across all 8 navigation tabs in a simulated browser session:

```bash
python tests/test_dashboard_navigation.py
```
- **Expected Output**:
  ```text
  [OK] Section 1: Overview rendered successfully
  [OK] Section 'Live Air Quality' rendered successfully with 0 exceptions
  [OK] Section 'Individual Forecasting Methods' rendered successfully with 0 exceptions
  [OK] Section 'Unified Hybrid Forecast' rendered successfully with 0 exceptions
  [OK] Section 'Side-by-Side Comparison' rendered successfully with 0 exceptions
  [OK] Section 'Station Map' rendered successfully with 0 exceptions
  [OK] Section 'Model Performance' rendered successfully with 0 exceptions
  [OK] Section 'Data Sources and System Health' rendered successfully with 0 exceptions
  ALL 8 DASHBOARD SECTIONS VERIFIED AND FUNCTIONAL!
  ```
- **Typical Runtime**: ~70–80 seconds.

---

## 3. How to Interpret Test Failures

### 1. `AssertionError: Rejection reason expected ... got None`
- **Cause**: Sequence validator let an invalid or incomplete 24h sequence pass.
- **Investigation**: Check `src/prediction/sequence_validator.py`. Inspect the sequence DataFrame length, timestamp frequency, and station ID matching.

### 2. `pyarrow.lib.ArrowInvalid: Could not convert '...' with type str`
- **Cause**: Streamlit table or dataframe received mixed data types in a numeric column (e.g., `"N/A (Weather Only)"` in a float column).
- **Investigation**: Ensure categorical placeholders are converted or stringified uniformly before passing to `st.dataframe()` or `st.table()`.

### 3. `OperationalHealthState` Deadlock or Hang
- **Cause**: A non-reentrant lock `threading.Lock()` was acquired twice on the same thread.
- **Investigation**: Confirm `OperationalHealthService._state_lock` uses `threading.RLock()` to permit reentrant status checks.

### 4. `HTTPSConnectionPool` / DNS Resolution Failure
- **Cause**: Upstream OpenAQ API unreachable due to network outage or sandboxed environment.
- **Investigation**: Ensure the tested component activates `ResilientDataIngestionManager` fallback without propagating an unhandled exception.

---

## 4. Recording Test Evidence

When submitting changes, record test execution logs for auditability:

```bash
# Save pytest output with timestamp
pytest tests/test_operational_monitoring_and_reliability.py -v > tests_operational_output.txt

# Inspect recent audit logs
tail -n 10 experiments/phase17_3/forecast_audit_log.jsonl
```
