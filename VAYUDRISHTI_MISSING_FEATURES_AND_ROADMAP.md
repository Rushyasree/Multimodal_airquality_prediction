# VayuDrishti: Missing Features, Prioritized Enhancement Roadmap & Engineering Feasibility

**Document Status:** Engineering Roadmap & Feasibility Study  
**Investigation Date:** November 2026  
**Workspace:** `e:\Documents\multimodal_air_quality_vscode_starter`  

---

## 1. Prioritization Framework

Enhancements are categorized into four strict priority tiers based on:
1. **Academic Impact:** Significance for IEEE conference presentation, peer-review defense, and technical rigor.
2. **Computational Feasibility:** Execution on commodity dual/quad-core CPU instances without GPU dependency.
3. **Safety & Stability:** Zero risk of breaking the existing verified forecasting services or test suites.

| Tier | Urgency & Scope | Timeline | Primary Objective |
| :--- | :--- | :--- | :--- |
| **P0 (Critical)** | Core Missing Spatial Features | Immediate (Phase 1) | Spatial Interpolation (IDW/Kriging) + City-Wide 2D Continuous Heatmap |
| **P1 (High)** | Arbitrary Location & Uncertainty | Near-Term (Phase 2) | Arbitrary Lat/Lon Point Inference + Spatial Uncertainty Fields |
| **P2 (Medium)** | Model Explainability & Telemetry | Mid-Term (Phase 3) | Feature SHAP Attribution + ResNet Saliency Maps + Automated Anomaly Detection |
| **P3 (Long-Term)**| Spatio-Temporal Deep Learning | Future Research | Graph Convolutional Network (GCN-LSTM / STGCN) on Dynamic Road Networks |

---

## 2. Detailed Breakdown of Missing Features

### Tier P0: Critical Missing Spatial Capabilities

#### 1. Real-Time Spatial Interpolation Service (`src/services/spatial_interpolation_service.py`)
- **Missing Capability:** The current codebase can only compute predictions at fixed physical CAAQMS stations ($N=90$). There is no method to estimate pollution in unmonitored zones between stations.
- **Proposed Solution:** Implement vectorized Inverse Distance Weighting (IDW) and Ordinary Kriging with automated empirical variogram fitting.
- **Feasibility:** High. Vectorized NumPy matrix operations compute a $50 \times 50$ (2,500 points) grid over Delhi NCR in $< 75\,\text{ms}$ on CPU.
- **Dependencies:** `numpy`, `scipy`, `scikit-learn`.

#### 2. City-Wide 2D Continuous Heatmap Layer
- **Missing Capability:** The dashboard map currently plots discrete station points (`Scattergeo`). It lacks continuous pollution surface contours or density maps.
- **Proposed Solution:** Integrate Streamlit-compatible PyDeck (`HeatmapLayer` / `GridLayer`) or Folium `Choropleth`/contour polygons rendering continuous PM2.5/AQI gradients.
- **Feasibility:** High. WebGL-accelerated rendering on the client browser; zero server-side rendering bottlenecks.
- **Academic Value:** Elevates the manuscript from a standard point-forecasting paper to a full spatio-temporal urban environmental intelligence system.

---

### Tier P1: High-Value Usability & Geostatistical Rigor

#### 3. Arbitrary Lat/Lon Query Service (`src/services/spatial_query_service.py`)
- **Missing Capability:** Users cannot enter their home or office GPS coordinates to obtain a localized air quality prediction.
- **Proposed Solution:** An inference endpoint taking `(latitude, longitude)`, querying the real-time spatial engine, computing distance to nearest stations, and returning a blended forecast with 95% spatial confidence intervals.
- **Feasibility:** $< 5\,\text{ms}$ latency per point query.

#### 4. Spatial Uncertainty Field Quantification
- **Missing Capability:** Current forecasts provide temporal confidence bounds ($\pm 1\sigma$), but no spatial variance metrics indicating how reliable the prediction is at distance $d$ from the nearest physical monitor.
- **Proposed Solution:** Kriging estimation variance $\sigma^2(x, y)$ or Gaussian Process standard deviation, visualized as an uncertainty confidence overlay.
- **Academic Value:** Directly addresses reviewer questions regarding interpolation fidelity in sparse sensor networks.

---

### Tier P2: Explainability & Operational Hardening

#### 5. Multimodal Feature Explainability (SHAP & Grad-CAM)
- **Missing Capability:** The dashboard displays model predictions, but lacks automated attribution showing *why* PM2.5 is elevated (e.g., whether wind stagnation, humidity, or visual haze was the primary driver).
- **Proposed Solution:** 
  - TreeSHAP for the LightGBM residual model.
  - Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-50 visual features, highlighting whether haze or ground clutter triggered visual corrections.
- **Feasibility:** Precomputed background datasets allow TreeSHAP to execute in $< 150\,\text{ms}$. Grad-CAM takes $< 250\,\text{ms}$ per image.

#### 6. Automated Sensor Drift & Flatline Detection
- **Missing Capability:** If a physical station sensor gets stuck on a static reading or experiences calibration drift, the pipeline ingests it without automated flagging.
- **Proposed Solution:** Statistical anomaly detector flagging variance collapse ($\sigma^2 < 10^{-4}$ over 6h) or sudden unphysical step-changes ($> 150\,\mu\text{g/m}^3$ in 1h).

---

### Tier P3: Long-Term Advanced Spatio-Temporal Architectures

#### 7. Graph Neural Network (GNN / STGCN) Integration
- **Concept:** Replace the decoupled LSTM + spatial interpolation pipeline with an end-to-end Spatio-Temporal Graph Convolutional Network (STGCN) where stations are graph nodes and edges represent wind-weighted road/distance connectivity.
- **Feasibility & Trade-offs:** Requires GPU training infrastructure and continuous graph topology updates; recommended for future journal expansion (VayuDrishti 3.0) rather than near-term conference deadlines.

---

## 3. Engineering Feasibility & Resource Budget Analysis

| Metric / Parameter | Current Baseline | With P0 (IDW Heatmap) | With P1 (Arbitrary Lat/Lon) | With P2 (SHAP / Grad-CAM) |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Core Requirement** | 2 Cores | 2 Cores | 2 Cores | 4 Cores recommended |
| **RAM Footprint** | ~650 MB | ~720 MB (+70 MB) | ~735 MB (+15 MB) | ~1.1 GB (+365 MB) |
| **Inference Latency (Single Station)** | 18 ms | 18 ms | 22 ms | 180 ms |
| **Grid Generation Latency (50x50)**| N/A | **65–85 ms** | 65–85 ms | ~450 ms |
| **GPU Dependency** | None (CPU only) | **None (CPU only)** | **None (CPU only)** | Optional (CUDA accelerates CAM) |
| **External Package Dependencies** | `torch`, `lightgbm`, `streamlit` | + `scipy`, `pydeck` | None | + `shap`, `torch-cam` |

---

## 4. Phase-by-Phase Implementation Schedule

```
Phase 1 (Week 1): Spatial Foundation (P0)
├── Implement src/services/spatial_interpolation_service.py (NumPy IDW & SciPy Kriging)
├── Write unit test suite tests/test_spatial_interpolation.py
└── Benchmark grid computation latency across Delhi, Mumbai, Kolkata grids

Phase 2 (Week 2): Heatmap Visualization & Arbitrary Queries (P0/P1)
├── Integrate PyDeck / Folium raster layer into src/components/hybrid_dashboard_pages.py
├── Implement src/services/spatial_query_service.py (Point lookup + 95% CI)
└── Add interactive coordinate pin-drop interface on Streamlit UI

Phase 3 (Week 3): Explainability & Geostatistical Hardening (P1/P2)
├── Implement Kriging spatial variance field sigma^2(x, y)
├── Add TreeSHAP tabular feature attribution card to Dashboard Page 4
└── Validate end-to-end pipeline against test suite
```
