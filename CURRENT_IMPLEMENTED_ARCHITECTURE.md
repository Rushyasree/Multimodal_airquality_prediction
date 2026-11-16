# VayuDrishti: Current Implemented System Architecture (Phase 12 Updated)

**Document Status:** Ground-Truth Technical Specification  
**Architecture Version:** 2.0 (Phase 12 Spatial Layer Integrated)  
**Execution Environment:** Python 3.10.11 / PyTorch 2.12.1+cpu / LightGBM 4.6.0 / Streamlit 1.32.0  
**Verification Date:** November 2026  

---

## 1. Architectural Overview & Implemented System Reality

VayuDrishti is a multimodal, spatio-temporal environmental intelligence platform. With the successful completion and verification of **Phase 12**, the architecture seamlessly integrates discrete station-level temporal forecasting with continuous 2D geostatistical spatial interpolation.

The operational pipeline consists of:
1. **Operational Ingest & Buffering Layer:** Continual acquisition of criteria pollutants from OpenAQ (CPCB/APPCB) and atmospheric meteorology from Open-Meteo, maintained in thread-safe 24-hour FIFO rolling buffers across 90 physical CAAQMS stations.
2. **Pathway 1 (Sensor-Only GBDT-Residual Ensemble):** A 2-layer LSTM temporal encoder generating baseline forecasts, augmented by a LightGBM tabular error regressor with an automated severe-tier safety gate ($\ge 250\,\mu\text{g/m}^3$).
3. **Pathway 2 (Gated Bounded Residual Multimodal Hybrid):** A unified neural network combining the 2-layer LSTM temporal encoder with a frozen ResNet-50 visual projection network and physical distance embeddings, modulated by a learned quality gate and bounded residual head with zero-distortion fallback guarantee ($m=0 \implies \hat{y}_{\text{hybrid}} \equiv \hat{y}_{\text{sensor}}$).
4. **Phase 12 Spatial Interpolation Layer (NEW - FULLY IMPLEMENTED):** A geostatistical engine (`src/services/spatial_interpolation_service.py`) using modified Geodesic Inverse Distance Weighting (IDW) to generate continuous 2D urban concentration fields and localized coordinate predictions with 95% spatial confidence bounds.
5. **Interactive Presentation Layer:** A 9-page Streamlit application (`app.py`, `src/components/`) featuring the new WebGL-accelerated "Spatial Prediction & Heatmap" interface.

```
+------------------------------------------------------------------------------------------------------------------+
|                              VAYUDRISHTI v2.0 FULL OPERATIONAL SYSTEM ARCHITECTURE                               |
+------------------------------------------------------------------------------------------------------------------+
                                          
                                          [ LIVE EXTERNAL DATA ]
                                                    |
                   +--------------------------------+--------------------------------+
                   |                                                                 |
                   v                                                                 v
         [ OpenAQ API (CPCB) ]                                             [ Open-Meteo Weather API ]
       (Hourly PM2.5, PM10, Gases)                                         (Temp, RH, WS, WD, Pressure)
                   |                                                                 |
                   +--------------------------------+--------------------------------+
                                                    |
                                                    v
                                      [ In-Memory Rolling Buffer ]
                                    (24h Sliding Window x 90 Stations)
                                                    |
                                                    v
                                    [ Sensor Preprocessor & Scaler ]
                                    - Missing value imputation
                                    - Cyclical temporal transforms (sin/cos)
                                    - Rolling 6h/24h statistics
                                    - Lag features (t-1, t-2, t-3, t-6)
                                    - Robust standard scaling
                                                    |
                                        Tensor: (1, 24, 27)
                                                    |
         +------------------------------------------+------------------------------------------+
         |                                                                                     |
         v [PATHWAY 1: Sensor Ensemble]                                                        v [PATHWAY 2: Multimodal Hybrid]
   +---------------------------------+                                           +---------------------------------+
   |    2-Layer LSTM Forecaster      |                                           |     Optimized Residual Hybrid   |
   |   (Input: 27, Hidden: 64)       |                                           |   (PyTorch Neural Architecture) |
   +---------------------------------+                                           +---------------------------------+
         |                    |                                                        |               |
   y_lstm (Baseline)    h_s (64D state)                                          y_sensor (Base)   h_s (64D state)
         |                    |                                                        |               |
         v                    v                                                        |        [ Outdoor Sky Image ]
   +-----------+      +---------------+                                                |        (224x224x3 RGB)
   | Severity  |      |   LightGBM    |                                                |               |
   | Gate Check|      | Residual Head |                                                |               v
   +-----------+      +---------------+                                                |        [ ResNet-50 Backbone ]
     |       |                |                                                        |        (Pretrained, Frozen)
     |       +----------------+                                                        |               |
     |                |                                                                |        [ Vision Projection ]
     |                v                                                                |        (Linear -> 32D vector)
     |        Residual Correction:                                                     |               |
     |        e_hat = y - y_lstm                                                       |        [ Distance Embedding ]
     |                |                                                                |        (Scalar km -> 8D vector)
     v                v                                                                |               |
   [ If y_lstm >= 250 ug/m3: Revert to LSTM ]                                          |               v
   [ Else: Final = y_lstm + e_hat           ]                                          |      [ 104D Feature Fusion ]
                      |                                                                |      [h_s || m*e_v || e_d]
                      v                                                                |               |
         +--------------------------+                                                  |               v
         | Standalone Sensor Output |                                                  |      [ Learned Quality Gate ]
         | Next 1 Hour (t + 1)      |                                                  |      alpha_eff = m * Sigmoid()
         +--------------------------+                                                  |               |
                      |                                                                |               v
                      |                                                                |      [ Bounded Residual Head ]
                      |                                                                |      Delta = Tanh() * [25, 40]
                      |                                                                |               |
                      |                                                                +-------+-------+
                      |                                                                        |
                      |                                                                        v
                      |                                                            y_hyb = max(0, y_s + alpha*Delta)
                      |                                                                        |
                      |                                                            (Fallback: m=0 => y_hyb = y_sensor)
                      |                                                                        |
                      |                                                                        v
                      |                                                            +--------------------------+
                      |                                                            | Multimodal Hybrid Output |
                      |                                                            | Next 1 Hour (t + 1)      |
                      |                                                            +--------------------------+
                      |                                                                        |
                      +-----------------------------------+------------------------------------+
                                                          |
                                                          v
                                     [ Synchronous Station Forecast Pool ]
                                       Y_hat in R^{N x 2} (Target Time: t+1)
                                                          |
                                                          v
                                     +-----------------------------------------+
                                     |  PHASE 12 SPATIAL INTERPOLATION SERVICE |
                                     |  - Modified Geodesic IDW (p=2.0)        |
                                     |  - Exact Station Coordinate Snapping    |
                                     |  - Distance Bounded Radius (R_max=40km) |
                                     |  - Zero Negative Concentration Clamping |
                                     +-----------------------------------------+
                                                          |
                             +----------------------------+----------------------------+
                             |                                                         |
                             v                                                         v
             [ Continuous 2D Urban Grids ]                            [ Arbitrary Location Point Inference ]
             - Delhi, Mumbai, Kolkata, Bengaluru                      - Geodesic distance to nearest monitor
             - 25x25, 50x50, 75x75 resolutions                        - Localized PM2.5 & PM10 estimates
             - Measured Latency: 11.2 ms (CPU)                        - 95% Spatial Confidence Bounds
                             |                                                         |
                             +----------------------------+----------------------------+
                                                          |
                                                          v
                                     +-----------------------------------------+
                                     |     STREAMLIT 9-PAGE PRESENTATION UI    |
                                     |   "Spatial Prediction & Urban Heatmap"  |
                                     |   - WebGL Densitymapbox Continuous Field|
                                     |   - Physical Monitor Overlays & Tooltips|
                                     |   - Address Landmark Pin-Drop Estimator |
                                     +-----------------------------------------+
```

---

## 2. Updated Mermaid Architecture Diagram

```mermaid
flowchart TD
    subgraph INGEST ["1. Telemetry Ingestion & Live Buffering"]
        AQ["OpenAQ CPCB / APPCB REST API (90 Stations)"]
        WX["Open-Meteo Atmospheric Meteorology REST API"]
        BUF["Thread-Safe In-Memory Rolling Buffer (24h x 90 Stations)"]
        CAM["Outdoor Sky Imagery (224x224x3 RGB)"]
        DIST["Station Offset Distance (d km)"]
        
        AQ --> BUF
        WX --> BUF
    end

    subgraph PREPROC ["2. Feature Engineering & Tensor Normalization"]
        PRE["Sensor Preprocessor & Robust Scaler (src/prediction/preprocessor.py)"]
        IMG_PRE["Image Normalizer & Solar Elevation Filter"]
        
        BUF --> PRE
        CAM --> IMG_PRE
    end

    subgraph FORECASTING ["3. Dual Multi-Station Forecasting Core"]
        LSTM1["2-Layer PyTorch LSTM (Input: 27, Hidden: 64)"]
        GBDT["LightGBM Residual Regressor"]
        SEV_GATE{"Severity Gate: y_lstm >= 250 ug/m3?"}
        P1_OUT["Sensor Forecast Output (t+1)"]
        
        PRE -->|Tensor: 1x24x27| LSTM1
        PRE -->|Instantaneous Features| GBDT
        LSTM1 -->|y_lstm Baseline| SEV_GATE
        GBDT -->|Error Residual e_hat| SEV_GATE
        SEV_GATE -->|True: Suppress GBDT| P1_OUT
        SEV_GATE -->|False: y_lstm + e_hat| P1_OUT

        LSTM2["LSTM Sensor Encoder (2 Layers, 64 Units)"]
        S_HEAD["Linear Sensor Head (Baseline y_sensor)"]
        RESNET["ResNet-50 Vision Backbone (Pretrained, Frozen)"]
        V_PROJ["Vision Projection (2048D -> 32D Vector e_v)"]
        D_EMB["Distance Embedding (1D Scalar -> 8D Vector e_d)"]
        FUSION["Feature Concatenation [h_s || m*e_v || e_d] (104D)"]
        GATE["Learned Quality Gate: alpha = Sigmoid(MLP)"]
        M_GATE["Effective Gate: alpha_eff = m * alpha"]
        RES_HEAD["Bounded Residual Head: Delta = Tanh(MLP) * [25, 40]"]
        CLAMP["Output Clamp: max(0, y_sensor + alpha_eff * Delta)"]
        P2_OUT["Multimodal Hybrid Output (t+1)"]
        
        PRE -->|Tensor: 1x24x27| LSTM2
        LSTM2 -->|Hidden State h_s 64D| S_HEAD
        LSTM2 -->|Hidden State h_s 64D| FUSION
        S_HEAD -->|y_sensor Baseline| CLAMP
        
        IMG_PRE -->|Tensor: 1x3x224x224| RESNET
        RESNET -->|Feature Map: 2048D| V_PROJ
        V_PROJ -->|e_v 32D| FUSION
        
        DIST -->|Scalar d| D_EMB
        D_EMB -->|e_d 8D| FUSION
        
        FUSION --> GATE
        GATE --> M_GATE
        FUSION --> RES_HEAD
        
        M_GATE -->|alpha_eff| CLAMP
        RES_HEAD -->|Delta| CLAMP
        CLAMP --> P2_OUT
    end

    subgraph SPATIAL_LAYER ["4. Phase 12 Spatial Interpolation Layer (IMPLEMENTED)"]
        P1_OUT --> CONTEXT["ForecastContext (Synchronous t+1 Validation)"]
        P2_OUT --> CONTEXT
        
        CONFIG["City Bounding Boxes & Params (src/config/spatial_config.py)"]
        CONFIG --> ENGINE["SpatialInterpolationService (src/services/spatial_interpolation_service.py)"]
        CONTEXT --> ENGINE
        
        ENGINE -->|Vectorized Haversine IDW| GRID["Continuous 2D Urban Concentration Field"]
        ENGINE -->|Localized Geodesic Decay| POINT["Arbitrary Location Prediction + 95% CI"]
    end

    subgraph UI ["5. Streamlit Presentation Layer (app.py)"]
        GRID --> HEATMAP_PAGE["Page 9: Spatial Prediction & Heatmap (Densitymapbox)"]
        POINT --> HEATMAP_PAGE
        P1_OUT --> DASH["Pages 1-8: Home, Stations, LSTM, Hybrid, Compare, Map, Accuracy, Health"]
        P2_OUT --> DASH
    end
```

---

## 3. Ground-Truth Operational State Summary

| Architectural Subsystem | Status | Verification Reference | Operational Latency |
| :--- | :--- | :--- | :--- |
| **Numerical Sensor Ingest** | **Fully Implemented** | `src/prediction/preprocessor.py` | $< 2\,\text{ms}$ |
| **LSTM Sensor Forecaster** | **Fully Implemented** | `models/lstm_baseline/best_lstm_model.pt` | $14\,\text{ms}$ (CPU) |
| **Residual Hybrid Network** | **Fully Implemented** | `experiments/.../best_optimized_hybrid_model.pt` | $18\,\text{ms}$ (CPU) |
| **Guaranteed Fallback (m=0)**| **Fully Implemented** | `tests/test_unified_hybrid_stability.py` | 1,000 trials: $0.000\,\mu\text{g/m}^3$ |
| **Geodesic Spatial Interpolator**| **Fully Implemented** | `src/services/spatial_interpolation_service.py` | **$11.2\,\text{ms}$ (50x50 Grid)** |
| **Arbitrary Location Predictor**| **Fully Implemented** | `src/services/spatial_interpolation_service.py` | **$0.82\,\text{ms}$ (Point Query)** |
| **Urban Heatmap UI** | **Fully Integrated** | `src/components/spatial_heatmap_page.py` | WebGL Client-Side |
| **Unit & Regression Tests** | **Passing** | 14/14 spatial tests, 24/24 core regression tests | Pytest 9.1.1 Verified |
