# VayuDrishti: Proposed Future Architecture & Enhancement Specification

**Document Status:** Architectural Evolution Blueprint  
**Target Release:** VayuDrishti 2.0  
**Target Capabilities:** Continuous Spatial Interpolation, City-Wide Heatmaps, Arbitrary Lat/Lon Inference  

---

## 1. Executive Summary & Vision

The current VayuDrishti implementation excels at point-based multi-station temporal forecasting and multimodal sensor-vision fusion. However, urban decision-makers and citizens require air quality predictions across continuous urban space—including unmonitored neighborhoods, transit corridors, and school zones.

The **Proposed Future Architecture** bridges this critical gap. It introduces an asynchronous, real-time **Spatial Interpolation & Surface Mapping Engine** downstream of the multi-station forecasting pathways. This preserves 100% of the verified temporal and multimodal model pipelines while enabling continuous 2D raster heatmaps, spatial uncertainty quantification, and arbitrary coordinate queries.

```
+------------------------------------------------------------------------------------------------------------------+
|                                  PROPOSED FUTURE SYSTEM ARCHITECTURE (v2.0)                                      |
+------------------------------------------------------------------------------------------------------------------+

  [ LIVE MULTI-STATION TELEMETRY ]           [ SATELLITE & CORRIDOR WIND FIELDS ]          [ USER ARBITRARY COORDINATE ]
     (90 CPCB/APPCB Stations)                        (Open-Meteo & ERA5)                        (Lat, Lon Query: e.g. Connaught Pl.)
                 |                                             |                                          |
                 v                                             v                                          |
   +---------------------------+                 +---------------------------+                            |
   | Temporal Forecasting Core |                 | Spatial Transport Core    |                            |
   | - Pathway 1 (GBDT Ensemble)|                | - Wind Transect Advection |                            |
   | - Pathway 2 (Hybrid Fusion)|                | - Boundary Layer Decay    |                            |
   +---------------------------+                 +---------------------------+                            |
                 |                                             |                                          |
                 +----------------------+----------------------+                                          |
                                        |                                                                 |
                                        v                                                                 |
                   [ Multi-Station Synchronous Forecast Matrix ]                                          |
                   Y_hat in R^{N x 2} (PM2.5, PM10 at all N stations)                                     |
                                        |                                                                 |
                                        v                                                                 |
                   +-----------------------------------------------+                                      |
                   |    REAL-TIME SPATIAL INTERPOLATION ENGINE     |                                      |
                   |    - Inverse Distance Weighting (IDW)         |                                      |
                   |    - Ordinary Kriging with Variogram Fitting  |                                      |
                   |    - Anisotropic Wind-Advection Warping       |<-------------------------------------+
                   +-----------------------------------------------+                                      |
                                        |                                                                 |
                 +----------------------+----------------------+                                          |
                 |                                             |                                          |
                 v                                             v                                          v
   [ 2D Raster Grid Generator ]                 [ Spatial Uncertainty Field ]               [ Arbitrary Lat/Lon Forecaster ]
   - 50x50 / 100x100 Grid Mesh                  - Kriging Variance Sigma^2(x, y)            - Nearest Neighbor Distance Decay
   - GeoJSON / GeoTIFF Export                   - Station Density Confidence Mask           - Fallback to Spatial Prior
   - Latency: < 85 ms on CPU                    - Unmonitored Territory Warning             - Point Prediction + 95% CI
                 |                                             |                                          |
                 +----------------------+----------------------+                                          |
                                        |                                                                 |
                                        v                                                                 v
                   +--------------------------------------------------------------------------------------+
                   |                            ENHANCED PRESENTATION LAYER                                |
                   | - Streamlit PyDeck / Folium Dynamic Continuous Heatmap Layer                          |
                   | - Interactive Coordinate Pin-Drop Querying                                            |
                   | - High-Resolution Spatial Risk Contours & Policy Advisory                             |
                   +--------------------------------------------------------------------------------------+
```

---

## 2. Mermaid Future Architecture Flowchart

```mermaid
flowchart TD
    subgraph INGEST ["1. Telemetry Ingest & Preprocessing (Preserved)"]
        AQ["OpenAQ CPCB Ingest (90 Stations)"]
        WX["Open-Meteo Wind & PBL Vectors"]
        BUF["Thread-Safe In-Memory Buffer"]
        AQ --> BUF
        WX --> BUF
    end

    subgraph FORECASTING ["2. Multi-Station Forecasting Core (Preserved)"]
        P1["Pathway 1: Sensor-Only GBDT Ensemble"]
        P2["Pathway 2: Residual Multimodal Hybrid"]
        BUF --> P1
        BUF --> P2
        P1 --> MATRIX["Synchronous Station Forecast Matrix: Y_hat in R^(N x 2)"]
        P2 --> MATRIX
    end

    subgraph SPATIAL_ENGINE ["3. Real-Time Spatial Interpolation Engine (NEW)"]
        MATRIX --> ENGINE{"Interpolation Algorithm Selector"}
        WX -->|Wind Vectors (u, v)| ANISO["Anisotropic Wind Distortion Kernel"]
        ANISO --> ENGINE
        
        ENGINE -->|Default: Fast CPU| IDW["Modified Shepards IDW (p=2.0)"]
        ENGINE -->|Rigorous Geostatistics| KRIG["Ordinary Kriging (Spherical Variogram)"]
        ENGINE -->|Non-Linear Spatial ML| GPR["Gaussian Process Regressor (RBF + WhiteKernel)"]
        
        IDW --> SYNTH["Continuous Field Synthesizer"]
        KRIG --> SYNTH
        GPR --> SYNTH
    end

    subgraph OUTPUTS ["4. Continuous Spatial Outputs & Services (NEW)"]
        SYNTH --> RASTER["50x50 Urban Meshgrid (GeoTIFF / GeoJSON)"]
        SYNTH --> UNCERTAINTY["Spatial Variance Field: sigma^2(x, y)"]
        
        QUERY["Arbitrary Lat/Lon Query Service (src/services/spatial_query_service.py)"]
        SYNTH --> QUERY
        USER_PIN["User Coordinate Pin-Drop (Lat, Lon)"] --> QUERY
        
        QUERY --> POINT_PRED["Point AQI Forecast + 95% Spatial Confidence Bounds"]
    end

    subgraph VISUALIZATION ["5. Interactive Heatmap Dashboard (NEW)"]
        RASTER --> HEATMAP["PyDeck / Folium Continuous 2D Heatmap"]
        UNCERTAINTY --> SENSORS["Optimal Sensor Placement Planner"]
        HEATMAP --> DASH["Streamlit 2.0 Spatial Intelligence Dashboard"]
        POINT_PRED --> DASH
    end
```

---

## 3. Mathematical Specifications of the Spatial Interpolation Engine

### 3.1 Inverse Distance Weighting (IDW) with Anisotropic Wind Warping
For fast, CPU-friendly real-time interpolation across an urban bounding box:
$$\hat{Z}(x_0) = \frac{\sum_{i=1}^N w_i(x_0) \, Z(x_i)}{\sum_{i=1}^N w_i(x_0)}$$

#### Weight Formulation with Wind Anisotropy:
Standard Euclidean distance is distorted along the ambient wind direction vector $\vec{v}_{\text{wind}}$ to account for downwind advective transport:
$$d_i^*(x_0) = \sqrt{\left(\Delta x_i - \beta v_x\right)^2 + \left(\Delta y_i - \beta v_y\right)^2}$$
$$w_i(x_0) = \frac{1}{\left(d_i^*(x_0) + \epsilon\right)^p}$$
*where $p \in [1.5, 2.5]$ is the distance decay power (typically $p=2.0$), $\epsilon = 0.05\,\text{km}$ prevents division by zero at station coordinates, and $\beta$ is the advective scaling parameter.*

### 3.2 Ordinary Kriging with Spatial Uncertainty Quantification
For geostatistically optimal, unbiased linear estimation with minimum variance:
$$\hat{Z}(x_0) = \sum_{i=1}^N \lambda_i \, Z(x_i) \quad \text{s.t.} \quad \sum_{i=1}^N \lambda_i = 1$$
The Kriging weights $\boldsymbol{\lambda}$ are solved via the linear system:
$$\begin{bmatrix}
\gamma(x_1, x_1) & \cdots & \gamma(x_1, x_N) & 1 \\
\vdots & \ddots & \vdots & \vdots \\
\gamma(x_N, x_1) & \cdots & \gamma(x_N, x_N) & 1 \\
1 & \cdots & 1 & 0
\end{bmatrix}
\begin{bmatrix}
\lambda_1 \\ \vdots \\ \lambda_N \\ \mu
\end{bmatrix}
=
\begin{bmatrix}
\gamma(x_1, x_0) \\ \vdots \\ \gamma(x_N, x_0) \\ 1
\end{bmatrix}$$
*where $\gamma(h)$ is the fitted empirical variogram (e.g., Spherical or Exponential), and $\mu$ is the Lagrange multiplier.*

#### Spatial Variance (Uncertainty Field):
$$\sigma^2(x_0) = \sum_{i=1}^N \lambda_i \, \gamma(x_i, x_0) + \mu$$
This produces a **continuous uncertainty map** indicating where forecasts are highly reliable (near clusters of physical stations) and where uncertainty is high (in unmonitored periphery zones).

---

## 4. Implementation Blueprint & File Roadmap

To implement this architecture safely without breaking existing verified services, the following three modular additions are planned:

| Component File Path | Module Responsibility | Algorithmic Design | Compute Profile |
| :--- | :--- | :--- | :--- |
| `src/services/spatial_interpolation_service.py` | Core spatial calculation engine | Vectorized NumPy IDW and SciPy/PyKrige Ordinary Kriging | CPU: $< 85\,\text{ms}$ for $50 \times 50$ grid |
| `src/services/spatial_query_service.py` | Arbitrary Lat/Lon inference API | Accepts `(lat, lon)`, queries interpolation engine, computes spatial confidence interval | CPU: $< 5\,\text{ms}$ per query |
| `src/components/spatial_heatmap_view.py` | UI visualization component | PyDeck `HexagonLayer` / Folium `Choropleth` or contour overlay on OpenStreetMap | Client-side WebGL rendering |

---

## 5. Architectural Benefits for IEEE Conference Publication

1. **Closes the Research Gap:** Directly fulfills the "Spatio-Temporal" scope claimed in the paper's title without resorting to synthetic or unverified claims.
2. **Defensible Scientific Grounding:** Kriging variance provides formal error bounds for urban air quality management, transforming the system into a decision-support tool for municipal governments.
3. **No GPU Requirement:** The spatial interpolation layer executes entirely on CPU using vectorized linear algebra, preserving VayuDrishti's low-resource deployment capability.
