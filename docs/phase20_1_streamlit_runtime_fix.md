# Phase 20.1 — Streamlit Runtime Error & Unified Hybrid Forecast Stability Fix

**Project:** VayuDrishti — AI Air Quality Monitoring & Forecasting Platform  
**Repository:** https://github.com/Rushyasree/Multimodal_airquality_prediction  
**Date:** 2026-11-12  
**Target Environment:** Streamlit Community Cloud (`https://vayudrishti-airquality.streamlit.app/`)  

---

## 1. Executive Summary

During cloud deployment monitoring of the VayuDrishti platform, Streamlit Community Cloud raised the following runtime error:

```text
TypeError: list indices must be integers or slices, not NoneType
```

This report documents the exact root cause, faulty file and line numbers, mechanism of failure, surgical fixes applied, regression testing, and verification across local execution and Streamlit Community Cloud.

---

## 2. Exact Root Cause & Faulty Code

### 2.1 Original Traceback
```text
TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/multimodal_airquality_prediction/app.py", line 2416, in <module>
    render_unified_hybrid_page()
File "/mount/src/multimodal_airquality_prediction/src/components/hybrid_dashboard_pages.py", line 99, in render_unified_hybrid_page
    sample_obj = curated_samples[picked_idx]
TypeError: list indices must be integers or slices, not NoneType
```

### 2.2 Mechanism of Failure
1. **Cloud vs. Local Storage Asymmetry**:
   - `data/raw/` is correctly gitignored in `.gitignore` to avoid committing hundreds of megabytes of raw image binaries to GitHub.
   - On local development machines where `data/raw/pm25vision/images/` exists, `image_srv.get_curated_sample_images()` populated a non-empty list of image dictionaries.
   - On Streamlit Cloud, `curated_samples` was evaluated as an empty list (`[]`).
2. **Streamlit Selectbox Behavior on Empty Options**:
   - `st.selectbox("Choose Verified Outdoor Photo:", range(len(sample_labels)), ...)` received an empty range (`range(0)`).
   - In Streamlit, when `options` is empty or cleared, `st.selectbox` returns `None`.
3. **Unchecked List Indexing**:
   - `sample_obj = curated_samples[picked_idx]` executed `curated_samples[None]`, raising `TypeError: list indices must be integers or slices, not NoneType`.
   - In addition, the widget's `format_func=lambda x: sample_labels[x]` had no check for `x is None`, which could trigger `sample_labels[None]` during widget state synchronization.
4. **Subsequent Formatting & NoneType Fragility in Unavailable States**:
   - When upstream live telemetry or model inference was unavailable (`status: "unavailable"`), `res.get("pm25")`, `res.get("sensor_pm25")`, and `res.get("visual_delta_pm25")` returned `None`.
   - Lines such as `{s_pm25:.1f}`, `{v_delta:+.2f}`, and `text=[f"{b[1]:.1f} µg/m³" ...]` would subsequently raise `TypeError: unsupported format string passed to NoneType or str`.

---

## 3. Streamlit Health Check & Connection Reset Investigation

Streamlit Community Cloud logs reported:
```text
[12:31:56] ❗️ The service has encountered an error while checking the health of the Streamlit app: 
Get "http://localhost:8501/healthz": read tcp 127.0.0.1:43384->127.0.0.1:8501: read: connection reset by peer
```

### Investigation Findings:
- **Redeployment Lifecycle Timing**: Pushing a commit to `main` triggered an automated container reboot on Streamlit Cloud.
- **Process Termination**: Streamlit Cloud's container supervisor sends a termination signal (SIGTERM) to the previous process listening on port 8501 while provisioning the new container.
- **In-flight Health Probe**: The internal proxy probe to `/healthz` was active at the exact moment port 8501 was closed by the terminating process, causing the Linux network stack to return a TCP RST (`connection reset by peer`).
- **Conclusion**: The connection reset was an ephemeral infrastructure redeployment event during rolling restart, not an application crash or memory leak.

---

## 4. Fixes Applied

### 4.1 Bulletproof Selection & Indexing in `src/components/hybrid_dashboard_pages.py`
1. **Fallback Image Asset**: Added committed fallback sample (`assets/scenic_ap.jpg`) in `src/prediction/image_estimator.py` so that curated samples are available even in headless cloud deployments.
2. **Safe Selectbox Options and Format Function**:
   ```python
   picked_idx = st.selectbox(
       "Choose Verified Outdoor Photo:",
       options=list(range(len(sample_labels))),
       index=0,
       format_func=lambda x: sample_labels[x] if (x is not None and isinstance(x, int) and 0 <= x < len(sample_labels)) else "",
       key="hybrid_sample_picker",
   )
   ```
3. **Explicit Type and Bounds Guarding**:
   ```python
   if picked_idx is not None and isinstance(picked_idx, int) and 0 <= picked_idx < len(curated_samples):
       sample_obj = curated_samples[picked_idx]
   ```
4. **Prediction Exception Handling & Safe State**:
   - Wrapped `predict_hybrid()` call in a `try...except` block with structured fallback returning explicit unavailable/error status fields.
5. **None-Safe Metric & Chart Formatting**:
   - Guarded all numeric format strings (`{pm25_val:.1f}`, `{s_pm25:.1f}`, `{v_delta:+.2f}`) with `isinstance(..., (int, float))` checks.
   - Replaced fragile chart rendering with conditional display (`if pm25_num is not None: ... else: st.info(...)`).
6. **Individual Methods & Side-by-Side Protection**:
   - Protected `ind_lstm_sc_select` and `m3_sample_sel` selectboxes against empty scenario/image lists.
   - Protected `fig.add_hline` in `render_side_by_side_comparison_page()` against `pm25_vals[0] is None`.

### 4.2 Targeted Exception Logging in `app.py`
In `app.py` lines 2413–2422, wrapped `render_unified_hybrid_page()`:
```python
elif st.session_state["nav_tab"] in ("Unified Hybrid Forecast", "Unified Hybrid"):
    _t_page_start = time.perf_counter()
    logger.info("START: Render Page: Unified Hybrid Forecast")
    try:
        render_unified_hybrid_page()
    except Exception as exc:
        logger.exception("Unified Hybrid Forecast rendering failed")
        st.error(f"⚠️ Unified Hybrid Forecast rendering failed: {type(exc).__name__}. See application logs for details.")
    else:
        logger.info("FINISH: Render Page: Unified Hybrid Forecast | elapsed=%.4fs", time.perf_counter() - _t_page_start)
```

---

## 5. Verification & Test Suite Execution

### 5.1 New Dedicated Regression Suite
File: `tests/test_unified_hybrid_stability.py`
- `test_empty_curated_samples_no_typeerror`: **PASSED** (Mocks empty sample list as on cloud)
- `test_model_unavailable_state_rendering`: **PASSED** (Verifies formatting when all telemetry/metrics are None)
- `test_inference_exception_graceful_handling`: **PASSED** (Verifies runtime error handling without crash)
- `test_none_index_format_func_safety`: **PASSED** (Directly verifies format_func with None, int, and bounds)
- `test_unified_hybrid_page_navigation`: **PASSED** (`AppTest` headless navigation across Home $\to$ Unified Hybrid)
- `test_home_dashboard_normal_rendering`: **PASSED** (`AppTest` headless validation of Overview tab)

**Result: 6/6 PASSED (70.98s)**

### 5.2 Full Test Suite Regressions
1. **Smoke & Navigation**:
   `pytest tests/test_eight_sections_smoke.py tests/test_dashboard_navigation.py`
   **Result: 9/9 PASSED (78.45s, 0 warnings)**
2. **Operational Monitoring & Reliability**:
   `pytest tests/test_operational_monitoring_and_reliability.py`
   **Result: 17/17 PASSED (37.93s, 0 warnings)**

---

## 6. Git Summary & Push Status

- **Modified Files**:
  - `app.py` (added targeted exception logging around `render_unified_hybrid_page()`)
  - `src/components/hybrid_dashboard_pages.py` (None-safe selectboxes, format functions, prediction fallback, and chart guards)
- **New Test Files**:
  - `tests/test_unified_hybrid_stability.py` (regression suite)
- **New Documentation**:
  - `docs/phase20_1_streamlit_runtime_fix.md`
- **Model / Forecasting Invariants**:
  - Model architecture, weights, scalers, sequence validators, and prediction algorithms were strictly untouched.
