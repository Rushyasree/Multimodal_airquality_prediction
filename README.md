# VayuDrishti — Breathe Smarter | Air Quality Intelligence for India

**A Robust Spatio-Temporal Machine Learning Framework for Longitudinal Air Quality Forecasting, Distribution-Free Conformal Calibration, and Empirical Multimodal Evaluation Across Indian Urban Environments**

**Authors:**
- Teliki Rushya Sree
- Machireddy Swathi
- Thanuja Varakala
- Palli Shilpa

**Faculty Supervisor:**
- Prof. Thotta Ramatulasi

*Department of Computer Science and Engineering, India*

---

## 1. Project Overview & Operational Capabilities

**VayuDrishti** (*Breathe Smarter — Air Quality Intelligence for India*) is an operational environmental monitoring, forecasting, and uncertainty quantification platform. Built for high-stakes municipal and public health decision-making in developing countries, VayuDrishti bridges atmospheric deep learning with real-world sensor resilience.

### Core Datasets & Scope:
- **Historical Regulatory Telemetry:** 1,369,779 cleaned hourly records from 90 Central Pollution Control Board (CPCB) continuous ambient air quality monitoring stations (CAAQMS) spanning 2015–2020 across 23 States and Union Territories.
- **Meteorological Reanalysis:** ECMWF ERA5-Land hourly atmospheric variables ($T$, $T_d$, $RH$, $u$, $v$, $P$) bilinearly interpolated to all 90 CAAQMS coordinates.
- **Nationwide Longitudinal Live Portfolio:** 15 continuous CAAQMS stations spanning 11 Indian States/UTs with 236,289 live and holdout test sequences and 506,848 seasonal evaluations.
- **Environmental Vision:** Curated outdoor photographs from the PM25Vision dataset across Indian coordinates.
- **CPCB NAQI Standard:** Official Indian National Air Quality Index engine with piecewise linear sub-indices, dominant pollutant identification, 3-pollutant minimum requirement, and targeted health advisories.

---

## 2. Nationwide Multi-Station & Andhra Pradesh Portfolio

VayuDrishti features nationwide coverage across major climatic, industrial, and coastal regimes:

| Station ID | City, State / UT | Station Name | Climatic / Regional Zone | Sample Count ($N$) |
|:---:|:---|:---|:---|:---:|
| `AP001` | Amaravati, Andhra Pradesh | Secretariat, Amaravati - APPCB | Coastal Plain / Capital Region | 149 (Live) |
| `AP005` | Visakhapatnam, Andhra Pradesh | GVM Corporation - APPCB | Coastal Marine / Port | 3,550 |
| `DL002` | Delhi, National Capital Territory | Anand Vihar - DPCC | Indo-Gangetic / Transport Hub | 3,357 |
| `DL008` | Delhi, National Capital Territory | DTU, Delhi - CPCB | Indo-Gangetic / Institutional | 2,674 |
| `MH005` | Mumbai, Maharashtra | Bandra, Mumbai - MPCB | West Coast Marine | 2,828 |
| `MH008` | Mumbai, Maharashtra | Colaba, Mumbai - MPCB | Coastal Peninsula | 2,226 |
| `GJ001` | Ahmedabad, Gujarat | Maninagar - GPCB | Western Semi-Arid / Urban | 3,059 |
| `WB008` | Kolkata, West Bengal | Bidhannagar - WBPCB | Lower Gangetic Basin | 3,455 |
| `WB010` | Kolkata, West Bengal | Jadavpur - WBPCB | Lower Gangetic Basin | 3,562 |
| `BR008` | Patna, Bihar | Muradpur, Patna - BSPCB | Middle Gangetic Basin | 3,379 |
| `KA004` | Bengaluru, Karnataka | Bapuji Nagar - KSPCB | Deccan Plateau / Urban | 3,548 |
| `TG001` | Hyderabad, Telangana | Bollaram Industrial - TSPCB | Deccan Plateau / Industrial | 3,089 |
| `TN002` | Chennai, Tamil Nadu | Manali Village - TNPCB | Coromandel Coast / Petrochemical | 3,180 |
| `CH001` | Chandigarh, Union Territory | Sector-25 - CPCC | Himalayan Foothills / Urban | 3,037 |
| `AS001` | Guwahati, Assam | Railway Colony - APCB | Brahmaputra Valley | 2,456 |

### Strict Scientific Data Availability & Proxy Rules:
- **No Fabricated Telemetry:** For city centers without direct in-situ CAAQMS instrumentation (e.g., Vijayawada, Guntur), the platform strictly reports `"Historical model data unavailable"`.
- **Explicit Proxy Opt-In:** Users can opt-in to analyze regional air parcels using the closest operational CAAQMS monitor. Results are strictly labeled `"Nearby-station proxy forecast"`, reporting the actual station ID, location, and great-circle distance in kilometers.

---

## 3. Verified Experimental Benchmarks

### 3.1 Longitudinal Multi-Station Live Portfolio (236,289 Sequences Across 15 Stations)

| Model Architecture | Telemetry Requirements | $\text{PM}_{2.5}$ MAE ($\mu\text{g/m}^3$) | $\text{PM}_{2.5}$ RMSE | $\text{PM}_{2.5}$ $R^2$ | Gain vs Persist | $\text{PM}_{10}$ MAE ($\mu\text{g/m}^3$) | $\text{PM}_{10}$ RMSE | $\text{PM}_{10}$ $R^2$ | Gain vs Persist |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Hour Persistence Baseline** | Telemetry at $t$ only | 6.95 | 14.49 | 0.7844 | 0.00% | 17.16 | 38.70 | 0.8089 | 0.00% |
| **Sensor + Meteorology (Adapted LSTM)** | $\text{PM}_{2.5}, \text{PM}_{10}$ + ERA5 Weather (27 feats) | 6.80 | 13.18 | 0.8215 | +2.11% | 16.48 | 34.65 | 0.8468 | +3.96% |
| **Sensor-Only Standalone Adapted GBDT** | Particulate Autoregression Only (49 feats) | 6.67 | 13.17 | 0.8219 | +3.97% | 16.29 | 35.24 | 0.8415 | +5.06% |
| **Unified Adapted Severity-Gated Ensemble** | **Particulates + Weather + Severity Gate** | **6.62** | **13.04** | **0.8254** | **+4.72%** | **16.14** | **34.53** | **0.8479** | **+5.92%** |

### 3.2 Historical Benchmark (404,431 Held-Out Sequences Across 90 Stations)
- **2-Layer LSTM Forecaster:** $\text{PM}_{2.5}$ MAE **10.57 $\mu\text{g/m}^3$** ($R^2 = 0.9492$); $\text{PM}_{10}$ MAE **18.86 $\mu\text{g/m}^3$** ($R^2 = 0.9301$).
- **CPCB NAQI Category Accuracy:** **77.04%** exact ($\text{PM}_{2.5}$) and **80.88%** ($\text{PM}_{10}$); **$>99.3\%$ adjacent-category accuracy** ($\pm 1$ bin).

### 3.3 Empirical Negative Result on Opportunistic Imagery
- **ResNet-50 Image Baseline:** $\text{PM}_{2.5}$ MAE **66.21 $\mu\text{g/m}^3$** ($R^2 = 0.1524$) on 79 held-out images.
- **Multimodal Early Fusion on Matched Cohort ($N=60$):** Image-Only MAE 73.67 $\mu\text{g/m}^3$ vs Multimodal Fusion MAE **85.86 $\mu\text{g/m}^3$** ($R^2 = -0.5064$).  
*Finding:* Early feature fusion degraded performance by **12.19 $\mu\text{g/m}^3$** due to lack of sub-daily timestamps amidst $166.72\ \mu\text{g/m}^3$ diurnal swings, extreme burst clustering, and $+70.74\ \mu\text{g/m}^3$ systematic label bias.

---

## 4. Key Scientific Contributions

### 4.1 Standalone Adapted GBDT (LightGBM)
Features a canonical 49-variable autoregressive schema (24 lagged particulate terms, 14 rolling statistics, 4 temporal harmonics, 3 calendar variables, 4 spatial coordinates). Designed specifically for zero reliance on secondary trace gases ($\text{NO}_x, \text{CO}, \text{SO}_2, \text{O}_3$), eliminating failure modes caused by telemetry dropouts.

### 4.2 Distribution-Free Split-Conformal Uncertainty Calibration
Guarantees valid finite-sample marginal coverage across 118,145 test instances:
- $\text{PM}_{2.5}$: **92.87%** empirical coverage at 80% nominal (width: 24.12 $\mu\text{g/m}^3$), **96.76%** at 90% nominal, and **98.38%** at 95% nominal.
- $\text{PM}_{10}$: **93.66%** empirical coverage at 80% nominal (width: 60.25 $\mu\text{g/m}^3$), **97.23%** at 90% nominal, and **98.71%** at 95% nominal.

### 4.3 Rigorous Statistical Robustness
- **Stationary Block Bootstrap** ($B=5,000$, block length $L=24\text{h}$): Pooled $\text{PM}_{2.5}$ difference $-0.28\ \mu\text{g/m}^3$ (95% CI $[-0.30, -0.25]$, Wilcoxon $p = 4.98 \times 10^{-12}$); pooled $\text{PM}_{10}$ difference $-0.87\ \mu\text{g/m}^3$ ($p = 7.57 \times 10^{-24}$).
- **Multiple Testing Corrections**: Passes Bonferroni FWER ($p < 0.0028$) and Benjamini-Hochberg FDR ($q < 0.05$).
- **Hazardous Regimes**: In the *Very Poor* regime ($N=4,847$), GBDT reduces MAE by **2.48 $\mu\text{g/m}^3$** (29.14 vs 31.62 $\mu\text{g/m}^3$, $p = 3.71 \times 10^{-25}$).

### 4.4 Spatial-Temporal Cross-Station Advection
- **Mumbai Marine Corridor (17.2 km)**: Peak correlation occurs at an **advective lag of $\tau = +1\text{ hour}$** ($r = 0.5385$) driven by coastal sea breezes. Incorporating neighbor lags reduces MAE to **6.05 $\mu\text{g/m}^3$** (**$+0.86\%$ gain** over single-station GBDT).
- **Inland Corridors (Delhi & Kolkata)**: Synchronous correlation peaks at $\tau = 0$. Local emissions dominate; spatial neighbor concatenation introduces noise without physics-informed wind gating.

---

## 5. Website Architecture & Navigation

The upgraded application (`app.py`) features 8 dedicated modules:
1. **🏠 Overview (Dual-Mode Summary):** Hero banner, current weather, dynamic greeting based on Asia/Kolkata timezone, and quick navigation.
2. **🛡️ Live Air Quality:** Station-level telemetry inspection across CAAQMS stations with data freshness indicators.
3. **📈 Individual Forecasting Methods (Mode A):** Sensor-Only LSTM, Sensor + Weather, ResNet-50 Vision, and 1-hour persistence baselines.
4. **⚡ Unified Hybrid Forecast (Mode B):** Severity-gated ensemble combining deep recurrent sequences and LightGBM with missing-modality fallback.
5. **⚖️ Side-by-Side Method Comparison:** Comprehensive matrix evaluating all forecasting methods simultaneously for any selected station.
6. **🗺️ Interactive 3D Station Map:** Interactive geospatial map of 90 CPCB continuous monitoring stations with hover inspection.
7. **📊 Model Performance & Benchmark Matrices:** Classification confusion matrices, 168-hour continuous trajectories, and multi-station/seasonal benchmark charts.
8. **ℹ️ Data Sources & System Health:** Checkpoint integrity verification (`best_lstm_model.pt`, `adapted_gbdt_pm25.joblib`), live telemetry health, and audit registry.

---

## 6. Quick Start & Reproducibility Instructions

### Prerequisites
- Python 3.10 or 3.11
- Windows PowerShell or Bash
- Existing virtual environment: `.venv`

### Step 1: Launch the Interactive Dashboard
```bash
streamlit run app.py
```
The application opens in your browser at `http://localhost:8501`.

### Step 2: Run Full Automated Smoke Test (All 8 Sections)
```bash
python tests/test_eight_sections_smoke.py
```
*(Verifies data dependencies, model checkpoints, fallback mechanisms, and UI components across all 8 dashboard sections)*

### Step 3: Run Phase 13 Spatial Advection Tests
```bash
pytest tests/test_phase13_spatial_advection.py -v
```

### Step 4: Run Hourly Autonomous Live Monitoring Cycle
```bash
python -m src.services.longitudinal_collector_service
```

---

## 7. Academic Research Publication Package (`paper/`)

The academic research manuscript is ready in the `paper/` directory:
- **Master LaTeX Manuscript:** [`paper/main.tex`](paper/main.tex) (ACM / IEEE / NeurIPS standard).
- **Markdown Companion:** [`paper/manuscript.md`](paper/manuscript.md).
- **BibTeX Bibliography:** [`paper/references.bib`](paper/references.bib) (18 verified peer-reviewed citations).
- **Modular Tables:** [`paper/tables/`](paper/tables/) (7 standalone `.tex` tables).
- **High-Resolution Figures:** [`paper/figures/`](paper/figures/) (7 publication figures).
- **Pre-Submission Checklist:** [`paper/pre_submission_checklist.md`](paper/pre_submission_checklist.md).
- **Compilation:**
  ```bash
  cd paper/
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

---

## 8. Repository Layout

```text
├── app.py                                   # Operational Streamlit Web Application (8 Sections)
├── config.yaml                              # Global project hyperparameters
├── requirements.txt                         # Dependencies
├── README.md                                # Comprehensive documentation & guide
│
├── src/
│   ├── services/
│   │   ├── spatial_advection_service.py     # Cross-station Haversine, bearing & wind advection
│   │   ├── longitudinal_collector_service.py# Autonomous hourly live-telemetry collector
│   │   ├── live_air_quality_service.py      # Real-time CPCB / OpenAQ telemetry ingest
│   │   ├── sensor_forecast_service.py       # Operational LSTM & GBDT inference
│   │   ├── image_estimation_service.py      # ResNet-50 visual PM estimation
│   │   ├── hybrid_prediction_service.py     # Severity-gated hybrid ensemble
│   │   └── model_comparison_service.py      # Multi-method benchmark service
│   ├── prediction/
│   │   ├── aqi_advisor.py                   # CPCB NAQI calculation & health precautions
│   │   ├── lstm_forecaster.py               # 2-layer LSTM sequence forecaster
│   │   └── image_estimator.py               # ResNet-50 optical regression
│   └── data/
│
├── tests/
│   ├── test_phase13_spatial_advection.py    # Spatial advection test suite (6/6 passed)
│   ├── test_phase12_autonomous_monitoring.py# Multi-station & autonomous monitoring tests
│   ├── test_eight_sections_smoke.py         # End-to-end smoke test for all 8 dashboard sections
│   └── test_standalone_gbdt_inference.py    # 49-feature schema & LightGBM inference tests
│
├── models/
│   ├── adapted_gbdt/                        # Standalone Adapted GBDT checkpoints (PM2.5, PM10)
│   ├── lstm_baseline/                       # Trained 2-layer LSTM checkpoint & scalers
│   └── pm25vision_cnn/                      # Trained ResNet-50 checkpoint
│
├── paper/
│   ├── main.tex                             # Master conference manuscript source
│   ├── manuscript.md                        # Markdown paper companion
│   ├── references.bib                       # 18 verified BibTeX citations
│   ├── tables/                              # 7 modular LaTeX table files
│   ├── figures/                             # 7 high-resolution publication figures
│   └── pre_submission_checklist.md          # Pre-submission quality audit report
│
├── experiments/
│   ├── phase12/                             # Multi-station longitudinal live trial artifacts
│   ├── phase13/                             # Spatial advection analysis artifacts & reports
│   └── recent_data_audit/                   # Phase 6B–11 feasibility & audit reports
│
└── docs/
    └── DEPLOYMENT_AND_VERIFICATION_GUIDE.md # Production launch & ops instructions
```
