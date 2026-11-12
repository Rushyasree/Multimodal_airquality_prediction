# VayuDrishti — Final Production Run Guide

This guide provides tested, reproducible, copy-pasteable commands to run, verify, and reproduce every component of the **VayuDrishti** Multimodal Air Quality Forecasting and Spatial Intelligence platform.

---

## 1. Environment Activation & Verification

Ensure your Python virtual environment (Python 3.10+) is active in the project root:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Verify your Python runtime and core libraries:
```bash
python -c "import torch, sklearn, streamlit; print(f'PyTorch: {torch.__version__} | Scikit-Learn: {sklearn.__version__} | Streamlit: {streamlit.__version__}')"
```

---

## 2. Launching the Interactive Streamlit Dashboard

To start the full interactive 8-section research dashboard with sidecar integration, real-time map visualization, and multimodal inference:

```bash
streamlit run app.py
```

### Dashboard Access:
- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://<your-ip>:8501`

### Exploring Dashboard Sections:
1. **Overview**: Executive summary, NAQI categorical distributions, and platform capabilities.
2. **Live Air Quality**: Real-time CPCB telemetry via OpenAQ API v3 with automatic caching and graceful offline fallback.
3. **Individual Forecasting Methods**: Direct evaluation of Historical LSTM, Standalone Adapted GBDT, and Vision ResNet-50.
4. **Unified Hybrid Forecast**: Severity-gated multimodal fusion with adaptive weighting ($\alpha$) and sensory degradation fallbacks.
5. **Side-by-Side Comparison**: Multi-model consensus matrix and cross-paradigm error metrics.
6. **Station Map**: PyDeck 3D geospatial visualization across AP CAAQMS stations.
7. **Model Performance**: Multi-horizon evaluation, residual histograms, and conformal calibration curves.
8. **Data Sources and System Health**: Real-time API latency, buffer state, telemetry freshness, and sensor status.

---

## 3. Running Autonomous Hourly Data Ingestion

The autonomous live telemetry pipeline ingests hourly observations from OpenAQ API v3, updates rolling buffers, and computes next-hour forecasts without target leakage.

### Single Ingestion Cycle (Dry Run / Test):
```bash
python run_phase12_autonomous_collector.py --once
```

### Continuous Daemon Mode (Autonomous Hourly Operation):
```bash
python run_phase12_autonomous_collector.py --interval-seconds 3600
```

Logs are written to `experiments/phase12_autonomous_live/collector.log`.

---

## 4. Running Model Loading & Forecast Verification

To run a controlled inference verification across all 4 machine learning paradigms:

```bash
python scripts/verify_models_controlled.py
```

### Verified Paradigms:
1. **Historical 2-Layer LSTM Forecaster**: 24-step contiguous sequence input, 27 physical/temporal features, 80% conformal uncertainty intervals.
2. **Standalone Adapted GBDT Forecaster**: 49 engineered tabular lag/rolling features, HistGradientBoostingRegressor checkpoints.
3. **Severity-Gated Multimodal Hybrid**: Adaptive $\alpha$ weighting, automatic visual fallback when camera imagery is unavailable.
4. **ResNet-50 Environmental Vision Estimator**: Optical scatter and transmission PM2.5 estimation with explicit date-only scientific disclosure.

---

## 5. Master Research Replication Engine

To reproduce all scientific benchmark tables, conformal prediction bounds, block bootstrap intervals, and paper LaTeX tables in a single command:

```bash
python scripts/reproduce_all_results.py
```

Outputs verified:
- Checkpoint integrity & SHA-256 hashes
- Split-conformal prediction intervals (80% and 90% coverage)
- Station-stratified block bootstrap paired differences ($N = 10{,}000$ iterations)
- 2D advection-diffusion spatial wind-transport metrics
- Paper LaTeX source syntax (`paper/sections/tables.tex`)
- Headless smoke verification of all 8 Streamlit dashboard sections

---

## 6. Running the Complete Automated Test Suite

To run all 24 unit, integration, and regression test suites with structured CSV logging:

```bash
python scripts/run_full_audit_tests.py
```

To run a specific test suite via pytest:
```bash
pytest -p no:langsmith tests/test_eight_sections_smoke.py -v
pytest -p no:langsmith tests/test_accuracy_optimization_v2.py -v
pytest -p no:langsmith tests/test_phase13_spatial_advection.py -v
```

---

## 7. Docker Deployment (Optional / Containerized)

If Docker is available on your host environment:

### Build and Launch via Docker Compose:
```bash
docker compose up -d --build
```

### Inspect Services:
```bash
docker compose ps
docker compose logs -f dashboard
```

### Stop Services:
```bash
docker compose down
```
