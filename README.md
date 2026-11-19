# VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vayudrishti-airquality.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcpu-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-passing%20(315%2F321)-success.svg)](tests/)

> **VayuDrishti** (*Sanskrit: "Vision of Air"*) is an operational, multimodal environmental intelligence and forecasting platform designed for Indian urban ecosystems. It integrates continuous ground-station telemetry, meteorological reanalysis, and opportunistic environmental vision to deliver calibrated next-hour particulate matter ($PM_{2.5}$ and $PM_{10}$) forecasts with empirical uncertainty quantification.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. System Architecture](#2-system-architecture)
- [3. Forecasting Methodology](#3-forecasting-methodology)
- [4. Key Platform Features](#4-key-platform-features)
- [5. Repository Directory Structure](#5-repository-directory-structure)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. Configuration & Environment Variables](#7-configuration--environment-variables)
- [8. Running the Application](#8-running-the-application)
- [9. Automated Testing & Verification](#9-automated-testing--verification)
- [10. Model Checkpoints & Reproducibility](#10-model-checkpoints--reproducibility)
- [11. Data Provenance & Citations](#11-data-provenance--citations)
- [12. Cloud Deployment Notes](#12-cloud-deployment-notes)
- [13. Responsible AI & Safety Disclaimer](#13-responsible-ai--safety-disclaimer)
- [14. Authors & Academic Affiliation](#14-authors--academic-affiliation)
- [15. License & Acknowledgements](#15-license--acknowledgements)

---

## 1. Project Overview

Severe atmospheric particulate pollution poses severe public health challenges across rapidly urbanizing Indian metropolitan corridors. Complex microclimatic boundary layers, episodic biomass burning, traffic density, and seasonal meteorological inversions induce sharp non-linear pollution spikes that conventional deterministic chemical transport models struggle to predict at high temporal resolution.

**VayuDrishti** addresses these challenges through a unified software and machine learning platform tailored for Indian urban conditions:
- **Geographic Coverage:** Multi-station spatial monitoring covering urban clusters across Delhi NCR, Hyderabad, Amaravati, and surrounding municipal zones.
- **Operational Cadence:** Ingests live telemetry from the Central Pollution Control Board (CPCB) via OpenAQ API v3, synchronizes surface meteorology, and computes calibrated next-hour predictions.
- **Decision Support:** Provides municipal administrators, environmental researchers, and citizens with actionable, uncertainty-bounded forecasts rather than opaque point predictions.

---

## 2. System Architecture

The VayuDrishti platform consists of six decoupled architectural layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                            │
│   • CPCB Continuous Ambient Air Quality Monitoring Stations (CAAQMS)   │
│   • Live Telemetry via OpenAQ API v3 (REST endpoints + fallback cache) │
│   • Hourly Surface Meteorology (Temperature, RH, Wind Speed/Direction) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      FEATURE ENGINEERING PIPELINE                      │
│   • 24-Hour Rolling Temporal Sequence (27 standardized features)       │
│   • Tabular Feature Manifest (49 features: lags, rolling means, ratios)│
│   • Strict Leakage Prevention & Validated Scaler Transforms            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       PRIMARY FORECASTING ENGINE                       │
│   ┌───────────────────────────┐    ┌─────────────────────────────────┐ │
│   │ Stacked 2-Layer Unidir    │    │ LightGBM Gradient Boosted Trees │ │
│   │ LSTM (Hidden: 128, 27-in) │    │ (49 Tabular Engineered Features)│ │
│   └─────────────┬─────────────┘    └────────────────┬────────────────┘ │
│                 │                                   │                  │
│                 └───────────────┬───────────────────┘                  │
│                                 ▼                                      │
│                 Dynamic Severity-Gated Ensemble Hybrid                 │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────┐
│                  AUXILIARY & UNCERTAINTY QUANTIFICATION                │
│   • Conformal Prediction Intervals (Empirical Coverage: 80% & 90%)     │
│   • Interpretability Engine: SHAP TreeExplainer Local Attributions     │
│   • Auxiliary Vision: ResNet-50 Visual Haze Estimation (Opportunistic) │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────┐
│                    INTERACTIVE USER INTERFACE LAYER                    │
│   • Streamlit Multi-Page Scientific Monitoring Application             │
│   • Real-Time AQI Cards, Spatial Heatmap (IDW), Model Comparison View  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Forecasting Methodology

### 3.1 Primary Hybrid Model (LSTM + LightGBM)
The primary next-hour predictor in VayuDrishti is a genuine **stacked 2-layer unidirectional LSTM + LightGBM/GBDT hybrid model**. Neither model operates as an isolated black box; both execute over synchronized, leakage-free representations of the same underlying atmospheric state:

1. **Temporal Sequence Path (LSTM):**
   - **Architecture:** Stacked 2-layer unidirectional LSTM (`input_size=27`, `hidden_size=128`, `dropout=0.2`).
   - **Input:** 24-step chronological window ($T-23$ to $T_0$) across 27 standardized environmental parameters (criteria pollutants, meteorological metrics, cyclical diurnal/annual encodings).
   - **Role:** Captures persistent temporal momentum, diurnal cyclical trends, and multi-hour atmospheric accumulation patterns.

2. **Tabular Feature Path (LightGBM / GBDT):**
   - **Architecture:** HistGradientBoosting / LightGBM regressor with optimized tree depth and regularization.
   - **Input:** 49 engineered tabular features capturing instantaneous gradients, multi-scale rolling statistics ($3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$ means, standard deviations, min/max ratios), and interaction terms.
   - **Role:** Captures sharp non-linear thresholds, sudden wind shifts, and rapid local source injections.

3. **Severity-Gated Ensemble Synthesis:**
   - Predictions are synthesized using research-validated base weights and dynamic severity gating:
     $$\hat{y}_{\text{hybrid}} = w_{\text{LSTM}} \cdot \hat{y}_{\text{LSTM}} + w_{\text{GBDT}} \cdot \hat{y}_{\text{GBDT}}$$
   - **Standard Regime ($PM_{2.5} \le 120\,\mu\text{g/m}^3$, $PM_{10} \le 250\,\mu\text{g/m}^3$):**
     - $PM_{2.5}$: $w_{\text{LSTM}} = 0.50$, $w_{\text{GBDT}} = 0.50$
     - $PM_{10}$: $w_{\text{LSTM}} = 0.45$, $w_{\text{GBDT}} = 0.55$
   - **Severe Event Regime ($PM_{2.5} > 120\,\mu\text{g/m}^3$ or $PM_{10} > 250\,\mu\text{g/m}^3$):**
     - $w_{\text{GBDT}}$ shifts dynamically to $0.70$ ($w_{\text{LSTM}} = 0.30$) to suppress autoregressive inertia and respond immediately to extreme pollution episodes.

### 3.2 Auxiliary Environmental Vision (ResNet-50)
The computer vision component utilizes a fine-tuned ResNet-50 convolutional neural network to estimate ambient $PM_{2.5}$ density from daytime outdoor horizon and sky images. 
- **Important Design Principle:** Image-based estimation is strictly an **auxiliary experimental tool** intended for opportunistic citizen sensing when physical CAAQMS telemetry is unavailable. It **never** overrides calibrated sensor-based LSTM + GBDT forecasts in official monitoring workflows.

---

## 4. Key Platform Features

- **Real-Time Monitoring:** Live AQI cards, Indian National Air Quality Index (NAQI) sub-index calculation, and sub-pollutant breakdowns.
- **Synchronized Model Comparison:** Direct side-by-side evaluation of **Hybrid**, **LSTM**, **GBDT**, and **1-Hour Persistence** models operating on verified, identical input tensors (guaranteed via SHA-256 tensor digest verification).
- **Spatial Heatmap & Urban Interpolation:** Multi-city continuous spatial pollution maps generated via inverse-distance weighting (IDW) interpolation across regional sensor nodes.
- **Uncertainty Bounds (Conformal Prediction):** Statistically valid, distribution-free prediction intervals at $80\%$ and $90\%$ confidence levels.
- **Explainable AI (SHAP):** Local TreeSHAP attributions revealing the exact meteorological and historical factors driving each prediction.
- **Robust Fallback & Graceful Degradation:** Automatic failover to synthetic demo sequences and historical station medians if live API connections experience upstream network failure.

---

## 5. Repository Directory Structure

```
multimodal_air_quality_prediction/
├── app.py                      # Main Streamlit application entrypoint
├── streamlit_app.py            # Streamlit Cloud deployment entrypoint
├── config.yaml                 # Central application & model configuration
├── requirements.txt            # Pinned Python package dependencies (CPU wheels)
├── packages.txt                # System Debian libraries for Streamlit Cloud
├── uv.toml                     # Fast resolver index strategy configuration
├── Dockerfile                  # Containerized deployment definition
├── docker-compose.yml          # Multi-container local orchestration
├── LICENSE                     # MIT Open-Source License
├── assets/                     # Application visual assets, logos, and UI banners
├── data/
│   └── processed/              # Runtime datasets, station metadata, demo buffers
│       ├── location_catalog.csv
│       ├── station_coordinates_90.csv
│       ├── station_metadata_90.csv
│       └── dashboard_demo_sequences.csv
├── docs/                       # Technical architecture & deployment guides
│   ├── DEPLOYMENT_AND_VERIFICATION_GUIDE.md
│   ├── ENVIRONMENT_SETUP.md
│   └── TROUBLESHOOTING.md
├── experiments/
│   └── recent_data_audit/
│       └── checkpoints/        # Adapted runtime GBDT & LSTM checkpoints
├── models/
│   ├── lstm_baseline/          # Baseline 2-layer LSTM checkpoint
│   └── pm25vision_cnn/         # Auxiliary ResNet-50 image regression checkpoint
├── src/                        # Core Python application packages
│   ├── components/             # Reusable Streamlit UI cards, charts, and metrics
│   ├── config/                 # Pydantic schema validation & setting loaders
│   ├── models/                 # PyTorch model architectures (LSTM, CNN)
│   ├── prediction/             # Hybrid orchestrator, GBDT engine, sequence validator
│   ├── services/               # Live API ingestion, spatial IDW, SHAP service
│   └── utils/                  # Mathematical metrics, NAQI calculators, loggers
└── tests/                      # Automated test suite (Pytest)
```

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (64-bit)
- Git

### Standard Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/rushya9/multimodal_air_quality_prediction.git
cd multimodal_air_quality_prediction

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Configuration & Environment Variables

Copy the example environment template:
```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAQ_API_KEY` | Optional | `""` (Empty) | OpenAQ API v3 key for live CAAQMS telemetry. If omitted, the system operates gracefully using cached and demo sequences. |
| `STREAMLIT_SERVER_PORT` | Optional | `8501` | Local development port. |
| `LOG_LEVEL` | Optional | `INFO` | Application logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 8. Running the Application

### Launch Local Streamlit Dashboard
```bash
streamlit run app.py --server.port 8501
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 9. Automated Testing & Verification

The repository includes a comprehensive unit and integration test suite covering pipeline integrity, model synchronicity, and UI components:

```bash
# Run all automated tests
python -m pytest tests/

# Run the primary hybrid model test suite
python -m pytest tests/test_lstm_gbdt_hybrid.py -v

# Run the environmental vision test suite
python -m pytest tests/test_environmental_vision.py -v

# Run UI smoke tests
python -m pytest tests/test_dashboard_navigation.py -v
```

---

## 10. Model Checkpoints & Reproducibility

All production checkpoints utilized during inference are version-controlled and verified:

| Component | Architecture | Input Dimensions | Checkpoint Location |
|---|---|---|---|
| **Temporal Forecaster** | 2-Layer Unidirectional LSTM | `(batch, 24, 27)` | `experiments/recent_data_audit/checkpoints/best_finetuned_lstm.pt` |
| **Tabular Forecaster** | LightGBM / GBDT Regressor | `(batch, 49)` | `experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib` |
| **Tabular Forecaster** | LightGBM / GBDT Regressor | `(batch, 49)` | `experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib` |
| **Auxiliary Vision** | ResNet-50 Feature Extractor | `(batch, 3, 224, 224)` | `models/pm25vision_cnn/best_model.pt` |

---

## 11. Data Provenance & Citations

1. **Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India:** Continuous Ambient Air Quality Monitoring Station (CAAQMS) official telemetry records.
2. **OpenAQ Platform:** Global ambient air quality API v3 ([openaq.org](https://openaq.org)).
3. **Open-Meteo & ERA5 Reanalysis:** Historical and numerical weather prediction feeds for relative humidity, ambient temperature, wind vectors, and surface pressure.

---

## 12. Cloud Deployment Notes

The platform is optimized for **Streamlit Community Cloud** and standard container runtimes:
- **PyTorch CPU Wheel Pinning:** `requirements.txt` specifies `torch==2.5.1+cpu` and `torchvision==0.20.1+cpu` via index URL `https://download.pytorch.org/whl/cpu` to avoid downloading multi-gigabyte CUDA binaries on CPU-only instances.
- **System Dependencies:** `packages.txt` provides required headless graphics and OpenMP libraries (`libgl1`, `libglib2.0-0`, `libgomp1`).
- **Resolver Flags:** `uv.toml` configures `index-strategy = "unsafe-best-match"` to ensure reproducible resolution between PyPI and the PyTorch CPU index.

---

## 13. Responsible AI & Safety Disclaimer

> **IMPORTANT NOTICE:** VayuDrishti is an academic and experimental decision-support research platform. Air quality forecasts, conformal uncertainty intervals, and visual estimations produced by this system are indicative machine-learning approximations. They do not constitute official regulatory data and must not supersede binding public health directives, statutory air quality alerts, or emergency guidelines issued by municipal authorities, state pollution control boards, or the Central Pollution Control Board (CPCB).

---

## 14. Authors & Academic Affiliation

**Capstone Project Team (B.Tech Computer Science and Engineering):**
- **Teliki Rushya Sree** (Reg. No: 23BCE9562)
- **Machireddy Swathi** (Reg. No: 23BCE9566)
- **Thanuja Varakala** (Reg. No: 23BCE9608)
- **Palli Shilpa** (Reg. No: 23BCE9586)

**Institution:**  
School of Computer Science and Engineering (SCOPE)  
**VIT-AP University**, Amaravati, Andhra Pradesh – 522237, India.

---

## 15. License & Acknowledgements

- **License:** Distributed under the [MIT License](LICENSE).
- **Acknowledgements:** We gratefully acknowledge the open data initiatives provided by the Central Pollution Control Board (CPCB), OpenAQ, and the open-source machine learning communities behind PyTorch, LightGBM, and Streamlit.
