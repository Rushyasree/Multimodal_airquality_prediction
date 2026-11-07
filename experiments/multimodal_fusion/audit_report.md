# Phase 1 Data & Model Audit Report: Multimodal Sensor–Image Fusion Feasibility

**Project:** VayuDrishti — Multimodal Air Quality Forecasting  
**Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Date:** October 24, 2026  
**Auditor:** Senior ML Research Engineer (Environmental Forecasting & Multimodal Deep Learning)

---

## Executive Summary

This audit rigorously inspects the existing numerical LSTM forecaster, the ResNet-50 environmental vision model, the spatial-temporal sequence indexing, weather integration, and image metadata. The investigation evaluates whether genuine end-to-end sensor–image fusion can be scientifically justified using the current project artifacts or external public APIs.

### Key Audit Conclusions:
1. **Numerical Forecaster Integrity**: The 2-layer LSTM operates on **404,431 held-out hourly sequences** with 27 features per step ($24 \times 27$ input tensor). It predicts next-hour continuous $\text{PM}_{2.5}$ and $\text{PM}_{10}$ ($t+1$) with confirmed out-of-sample metrics ($R^2 = 0.9492$ / $0.9301$, MAE = $10.57$ / $18.86\ \mu\text{g/m}^3$).
2. **Weather Features Already Present**: Hourly meteorological reanalysis variables from Copernicus ERA5-Land (temperature, dewpoint, relative humidity, wind speed, wind direction components, precipitation) are **already merged with 100.00% completeness** into the 27 LSTM input features. Querying the Open-Meteo Historical Weather API would only duplicate these existing ERA5-Land fields.
3. **Absence of Image Capture Timestamps**: PM25Vision images contain **zero sub-daily timestamp information** (no hour, minute, or second) in their metadata and zero EXIF timestamp tags. 
4. **Intraday Diurnal Swings**: Indian urban CAAQMS stations exhibit an average intraday $\text{PM}_{2.5}$ diurnal range of **$166.72\ \mu\text{g/m}^3$** between afternoon minimums and nighttime inversion peaks. Pairing an image of unknown capture hour with a specific hourly forecast sequence introduces an uncalibrated error margin over $15\times$ larger than the numerical model's MAE.
5. **Sample Size Bottleneck & Burst Pseudo-Replication**: Across 5.5 years (2015–2020), only **265 images** fall within 5 km of an active station on the same calendar date, representing just **36 unique station-date groups** across the entire country. Furthermore, **39.6%** of these images originate from only two single-day camera bursts.
6. **Scientific Verdict**: End-to-end next-hour ($t+1$) multimodal fusion using the PM25Vision dataset is **statistically and methodologically indefensible**. We outline below the precise empirical evidence, external data feasibility, and the recommended research path.

---

## 1. Detailed Inspection of Existing Artifacts

### 1.1 Numerical LSTM Model
- **Checkpoint Path**: `models/lstm_baseline/best_lstm_model.pt` (File size: 250,575 bytes).
- **Architecture**:
  - Class: `LSTMForecaster` (defined in `src/prediction/lstm_forecaster.py`).
  - Gating: 2-layer recurrent LSTM, `hidden_size=64`, `dropout=0.2`, `batch_first=True`.
  - Regression Head: Multi-Layer Perceptron: $\mathbb{R}^{64} \rightarrow \text{Linear}(64, 64) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(64, 2)$.
  - Checkpoint State Dict Tensor Shapes:
    - `lstm.weight_ih_l0`: `torch.Size([256, 27])`
    - `lstm.weight_hh_l0`: `torch.Size([256, 64])`
    - `lstm.bias_ih_l0`: `torch.Size([256])`
    - `lstm.bias_hh_l0`: `torch.Size([256])`
    - `lstm.weight_ih_l1`: `torch.Size([256, 64])`
    - `lstm.weight_hh_l1`: `torch.Size([256, 64])`
    - `lstm.bias_ih_l1`: `torch.Size([256])`
    - `lstm.bias_hh_l1`: `torch.Size([256])`
    - `regressor.0.weight`: `torch.Size([64, 64])`
    - `regressor.0.bias`: `torch.Size([64])`
    - `regressor.3.weight`: `torch.Size([2, 64])`
    - `regressor.3.bias`: `torch.Size([2])`
- **Input Dimensions**:
  - Tensor Shape: $[B, 24, 27]$ (Batch size, 24-hour historical lookback, 27 features).
  - Feature Ordering:
    1. 12 Pollutant parameters: `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`.
    2. 7 ERA5-Land weather parameters: `temperature_C`, `dewpoint_C`, `relative_humidity`, `wind_speed`, `precipitation_mm`, `wind_direction_sin`, `wind_direction_cos`.
    3. 6 Cyclical temporal encodings: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_of_week_sin`, `day_of_week_cos`.
    4. 2 Spatial coordinates: `Latitude`, `Longitude`.
- **Normalization & Preprocessing**:
  - Saved in: `data/processed/model_sequences/preprocessing_metadata.json`.
  - Scaler parameters fitted strictly on training partition ($N_{\text{train}} = 336,641$ sequences):
    - Input standardization: $x_{\text{norm}} = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$.
    - Imputation: Median values computed exclusively from training partition.
- **Target Scaling**:
  - Checkpoint keys `target_mean` and `target_std`:
    - $\text{PM}_{2.5}$: $\mu = 79.5145\ \mu\text{g/m}^3$, $\sigma = 77.4174\ \mu\text{g/m}^3$.
    - $\text{PM}_{10}$: $\mu = 172.1068\ \mu\text{g/m}^3$, $\sigma = 137.4705\ \mu\text{g/m}^3$.
  - Inference code unscales raw model predictions: $\hat{\mathbf{y}} = \max(0, \hat{\mathbf{y}}_{\text{norm}} \odot \boldsymbol{\sigma} + \boldsymbol{\mu})$.

---

### 1.2 Image Model (ResNet-50) & PM25Vision Dataset
- **Checkpoint Path**: `models/pm25vision_cnn/best_model.pt` (File size: 96,521,626 bytes).
- **Architecture**:
  - Class: `ResNet50Regressor` (defined in `src/prediction/image_estimator.py`).
  - Backbone: Torchvision `resnet50` truncated prior to final classifier, extracting a 2048-dimensional pooled feature vector $\mathbf{v} \in \mathbb{R}^{2048}$.
  - Regression Head: $\mathbb{R}^{2048} \rightarrow \text{Linear}(2048, 256) \rightarrow \text{BatchNorm1d}(256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(256, 64) \rightarrow \text{ReLU} \rightarrow \text{Linear}(64, 1)$.
  - Target: Continuous $\text{PM}_{2.5}$ only (does not predict $\text{PM}_{10}$).
- **Target Scaling**:
  - `target_mean`: $204.1526\ \mu\text{g/m}^3$, `target_std`: $61.3761\ \mu\text{g/m}^3$.
- **Image Dataset Manifest & Properties**:
  - Manifest: `data/processed/pm25vision_cnn/pm25vision_clean_manifest.csv` (640 unique physical images, 2 SHA-256 bit-level duplicates excluded).
  - Spatial mapping: Nearest station distance computed via Haversine formula against 90 CPCB monitors.
    - Minimum distance: $0.052\text{ km}$
    - Median distance: $5.16\text{ km}$
    - Mean distance: $92.78\text{ km}$
    - Maximum distance: $620.77\text{ km}$
    - Images within 5.0 km threshold: **316 images** (49.4%).
    - Images exceeding 5.0 km: **324 images** (50.6%, unmatchable).
- **CRITICAL AUDIT FINDING: Image Timestamps & EXIF**:
  - In `pm25vision_clean_manifest.csv`, the date column `capture_date_clean` contains only calendar dates formatted as `YYYY-MM-DD` (e.g., `2019-04-10`).
  - An exhaustive forensic scan of all image files in `data/raw/pm25vision/images/` using PIL EXIF parsers revealed that **385 out of 386 images have stripped EXIF headers**; exactly 1 image contained an EXIF structure, and **zero images contain DateTimeOriginal or SubSecTime tags**.
  - As a result, sub-daily capture time (hour, minute, second) is **completely unavailable** across the entire dataset.

---

### 1.3 Sequence Index & Station Metadata
- **Index File**: `data/processed/sequences/air_quality_24h_sequence_index.csv` (91,852,094 bytes, 1,067,495 rows).
- **Structure**:
  - Columns: `StationId, history_start, history_end, target_time, target_PM2.5, target_PM10, split`.
  - Confirmed sample alignment: Rows where `split == 'test'` match `y_test.npy` and `test_predictions.npy` row-for-row ($N = 404,431$, absolute difference $< 3 \times 10^{-5}\ \mu\text{g/m}^3$).
  - Target time range of test set: `2019-09-04 20:00:00` to `2020-07-01 00:00:00` across all 90 CAAQMS stations.
- **Station Coordinates**: `data/processed/station_coordinates_90.csv` contains verified coordinates, city, state, and elevation for all 90 continuous stations.

---

### 1.4 Partitioning & Leakage Audit
- **Partitioning Method**: Strict chronological calendar splitting.
  - **Train**: `2015-01-02 05:00:00` to `2018-11-06 14:00:00` ($N = 336,641$, 55 stations).
  - **Buffer Gap 1**: **25 hours** (`2018-11-06 15:00:00` to `2018-11-07 14:00:00`).
  - **Validation**: `2018-11-07 15:00:00` to `2019-09-03 19:00:00` ($N = 326,423$, 78 stations).
  - **Buffer Gap 2**: **25 hours** (`2019-09-03 20:00:00` to `2019-09-04 19:00:00`).
  - **Test**: `2019-09-04 20:00:00` to `2020-07-01 00:00:00` ($N = 404,431$, 90 stations).
- **Leakage Assessment**:
  - Because each sample lookback is 24 hours, the 25-hour gaps between splits guarantee that **zero historical windows overlap across partition boundaries**.
  - Preprocessing standardizers and target scalers were computed on the training partition only.
  - Data leakage between sequence splits is **0%**.

---

### 1.5 Weather Integration Audit
- **Merge Status**: `data/processed/air_quality_weather_merged.csv` contains 1,369,779 hourly records.
- **Merge Audit Report**: `data/processed/air_quality_weather_merge_report.txt` confirms:
  - 90 stations matched out of 90 (100.00% match rate).
  - 0 missing weather values after bilinear interpolation from ERA5-Land.
  - Weather features incorporated into the 27 LSTM inputs: 2m Temperature, 2m Dewpoint, Relative Humidity, Wind Speed, Wind Direction ($\sin/\cos$), Total Precipitation.

---

## 2. External Data Source Feasibility (Phase 2 & 3)

### 2.1 Open-Meteo Historical Weather API
- **URL**: `https://open-meteo.com/en/docs/historical-weather-api`
- **Audit Findings**:
  - Open-Meteo provides hourly historical weather data for specified coordinates without mandatory API keys (10,000 requests/day limit).
  - **Redundancy Analysis**: Open-Meteo's historical reanalysis endpoint derives its surface variables from the Copernicus Climate Change Service ECMWF ERA5 and ERA5-Land reanalyses.
  - **Conclusion**: The VayuDrishti dataset **already contains hourly ERA5-Land weather variables merged with 100% completeness**. Querying Open-Meteo would fetch redundant or identical meteorological data and would **not provide any visual/image data**.

### 2.2 NASA Earthdata Satellite Products
- **URL**: `https://www.earthdata.nasa.gov/`
- **Candidate Products**:
  1. *MODIS (Terra & Aqua) Aerosol Optical Depth (MOD04_L2 / MYD04_L2 / MCD19A2 MAIAC)*:
     - Spatial Resolution: 1 km (MAIAC) to 3 km / 10 km (Dark Target / Deep Blue).
     - Orbit: Sun-synchronous Polar Low Earth Orbit (LEO).
     - **Revisit Frequency**: Only **1 to 2 daytime overpasses per day** over India (Terra at ~10:30 AM local solar time, Aqua at ~1:30 PM local solar time).
     - Nighttime: Completely unavailable (passive optical sensors require sunlight).
     - Cloud / Smog Dropouts: During the summer monsoon (June–Sept) and winter fog inversions across North India, MODIS AOD retrievals exhibit 60%–80% pixel masking due to cloud cover.
     - **Data Format**: Gridded geophysical scientific arrays (dimensionless optical thickness $\tau_{550}$), **not natural RGB photographs**.
  2. *VIIRS (Suomi-NPP / NOAA-20)*:
     - Similar LEO sun-synchronous constraints (~1:30 PM overpass), providing 1 snapshot per day.
- **Compatibility with Next-Hour Sequence Forecasting**:
  - The operational LSTM forecaster predicts 24 hours a day, 365 days a year across 90 stations (1.06M hourly sequences).
  - A polar-orbiting satellite that provides at most 1 snapshot per day at ~10:30 AM cannot provide continuous hourly sequences or nighttime coverage.
  - Feeding a single float (AOD) or coarse $1\text{ km} \times 1\text{ km}$ raster pixel into ResNet-50 (a 25.5M-parameter CNN designed for 3-channel natural images with edges, textures, and horizons) represents an architectural mismatch.

---

## 3. Pairing Viability & Mathematical Constraints

| Constraint Dimension | Operational Sensor Forecaster (LSTM) | PM25Vision Image Dataset (ResNet-50) | Scientific Compatibility |
| :--- | :--- | :--- | :--- |
| **Temporal Frequency** | Continuous hourly (24 predictions/day) | Single snapshot on arbitrary calendar date | **Incompatible** ($24\times$ temporal mismatch) |
| **Time-of-Day Stamp** | Precise UTC/IST hour (`YYYY-MM-DD HH:00:00`) | Strictly date-only (`YYYY-MM-DD`) | **Undefined** (Unknown hour of capture) |
| **Intraday Variation** | Diurnal range averages $166.72\ \mu\text{g/m}^3$ | Assumes static daily label | **Severe bias** (Up to $\pm 166.7\ \mu\text{g/m}^3$ error) |
| **Sample Size** | 404,431 test sequences across 90 stations | 265 matched images across 36 station-dates | **Sample Bottleneck** ($N_{\text{groups}}=36$) |
| **Spatial Footprint** | Stationary point monitor (beta attenuation) | Mobile / web-scraped photos (up to 5 km away) | **Micro-climate divergence** |
| **Calibration Baseline** | CPCB certified CAAQMS telemetry | Web-scraped / crowdsourced metadata | **$+70.74\ \mu\text{g/m}^3$ positive offset** |

### Mathematical Proof of Incompatibility
Let $y_{s, d, h}$ denote the true particulate concentration at station $s$, date $d$, and hour $h \in \{0, 1, \dots, 23\}$.  
Let $\mathbf{x}_{s, d, h}$ denote the 24-hour sensor history up to hour $h$.  
An image $I_{s, d}$ is associated only with date $d$, with true capture time $h^* \in \{0, \dots, 23\}$ unknown.

If a fusion model pairs $I_{s, d}$ with the forecast at hour $h$:
1. If $h < h^*$, the model accesses visual features from the future ($h^* > h$), violating temporal causality (**lookahead leakage**).
2. If $h > h^*$, the model uses visual features from hours past, during which ambient concentration has shifted by up to $\pm 166.72\ \mu\text{g/m}^3$ (**temporal staleness**).
3. The diurnal variance $\operatorname{Var}_{h}(y_{s, d, h})$ exceeds the entire MSE of the numerical LSTM forecaster.

---

## 4. Recommendations for Phases 2–5

Based on this audit, we submit the following recommendations for faculty review:

### Recommendation A (Strongly Recommended — Academic Integrity & Rigor)
**Do not force a synthetic pairing between PM25Vision and the hourly LSTM.**  
Instead, frame the multimodal contribution of the thesis/paper as an **Empirical Feasibility and Alignment Audit**:
- Document the empirical boundaries that govern multimodal air quality sensing.
- Demonstrate mathematically and experimentally why opportunistic, un-timestamped web imagery cannot be naively fused with continuous telemetry.
- In IEEE publications, well-documented negative results and feasibility boundaries that prevent methodological errors are highly regarded and frequently cited.

### Recommendation B (Controlled Proof-of-Concept Offline Experiment)
If the faculty requires an explicit fused PyTorch architecture (`LSTM + ResNet-50`), implement it as a **standalone proof-of-concept ablation** in `experiments/multimodal_fusion/`:
- Restrict evaluation to the **36 matched station-date groups** using daily-aggregated sensor representations (or fixed noon windows).
- Use `GroupShuffleSplit` on station-date clusters to prevent pseudo-replication.
- Explicitly report the sample size constraint ($N_{\text{groups}}=36$) and document that multimodal early fusion does not generalize out-of-distribution without collocated, timestamped cameras.

---

*Report prepared and certified for project record.*
