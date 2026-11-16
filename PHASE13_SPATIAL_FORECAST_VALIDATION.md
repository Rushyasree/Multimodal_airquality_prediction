# VayuDrishti Phase 13: Spatial Forecast Validation & Comparative Evaluation

**Document Status:** Empirical Spatial Forecasting Experiment  
**Experiment Script:** `experiments/spatial_prediction/validate_spatial_forecasting.py`  
**Dataset Artifacts:**  
- Granular Results: `experiments/spatial_prediction/spatial_forecast_validation_results.csv` ($N = 1,564$ holdout instances)  
- Summary Metrics: `experiments/spatial_prediction/spatial_forecast_validation_metrics.json`  
**Date:** November 2026  
**Execution Environment:** Python 3.10.11 / PyTorch 2.12.1+cpu / NumPy 2.2.6  

---

## 1. Motivation: Decoupling Observation Mapping from Forward Forecasting

In Phase 12, the initial Leave-One-Station-Out (LOSO) experiment evaluated **Spatial Observation Interpolation**—interpolating contemporaneous sensor observations $y(s_i, t)$ to recover an omitted sensor's measurement $y(s, t)$.

While valuable for sensor failure recovery, urban decision-makers and citizens asking *"What will the air quality be at my house next hour?"* require **Spatial Forecast Interpolation**—taking the *forward-looking forecasts* generated at physical stations ($\hat{y}(s_i, t+1)$) and interpolating them across continuous urban space to predict unmonitored locations at $t+1$.

Phase 13 conducts this rigorous experiment on the locked test set.

---

## 2. Experimental Design & Methodology

```
+---------------------------------------------------------------------------------------------------------+
|                                  SPATIAL FORECAST EVALUATION PROTOCOL                                   |
+---------------------------------------------------------------------------------------------------------+

  [ Common Forecast Issue Time (t) ]
                 |
                 v
  [ Ingest Causal 24h Sequences: X(s_i, t-23...t) for all stations ]
                 |
                 v
  [ Compute Next-Hour Forward Predictions: y_hat(s_i, t+1) via Pretrained LSTM ]
                 |
  +--------------+----------------------------------------------------------+
  |                                                                         |
  v (At Target Station s)                                                   v (At Neighboring Stations s_i != s)
[ Direct Temporal Forecast ]                                     [ Pool of Forward Forecasts ]
  y_hat_temporal(s, t+1)                                           { y_hat(s_i, t+1) | s_i != s }
  (Direct Station Sensor)                                                   |
                                                                            +--------------------+
                                                                            |                    |
                                                                            v                    v
                                                            [ Spatial IDW Interpolation ] [ Nearest Neighbor ]
                                                              y_hat_IDW(s, t+1)             y_hat_NN(s, t+1)
                                                                            |                    |
  +-------------------------------------------------------------------------+--------------------+
  |
  v
[ Compare All Three Methods against Ground-Truth Observed Concentration y(s, t+1) ]
```

### Strict Temporal and Spatial Safeguards:
1. **Identical Target Timestamp:** All models predict the exact identical target hour $t+1$.
2. **Zero Target Leakage:** The target station's sequence and direct forecast $\hat{y}_{\text{temporal}}(s, t+1)$ are strictly withheld from spatial interpolation inputs.
3. **Strictly Causal History:** Sequence lookback is strictly $t-23 \to t$; no future observations are accessible.

---

## 3. Empirical Results: Head-to-Head Comparison

Evaluated across **1,564 spatial forecast holdout instances** across 60 synchronous multi-station test slices in Delhi NCR.

```
+---------------------------------------------------------------------------------------------------------+
|                                    SPATIAL FORECAST EVALUATION BENCHMARK                                |
+-------------------+--------------------+------------------------+-------------------+-------------------+
| Criteria Pollutant| Forecasting Method | Formulation            | MAE (µg/m³)       | RMSE (µg/m³)      |
+-------------------+--------------------+------------------------+-------------------+-------------------+
| PM2.5             | Direct Station LSTM| Dedicated local sensor | 13.61 µg/m³       | 26.54 µg/m³       |
|                   | IDW Spatial Forecast| Neighbor forecasts IDW | 24.30 µg/m³       | 42.92 µg/m³       |
|                   | Nearest Neighbor FC| Single closest station | 28.83 µg/m³       | 49.62 µg/m³       |
+-------------------+--------------------+------------------------+-------------------+-------------------+
| PM10              | Direct Station LSTM| Dedicated local sensor | 22.19 µg/m³       | 38.65 µg/m³       |
|                   | IDW Spatial Forecast| Neighbor forecasts IDW | 44.41 µg/m³       | 68.73 µg/m³       |
|                   | Nearest Neighbor FC| Single closest station | 55.31 µg/m³       | 84.18 µg/m³       |
+-------------------+--------------------+------------------------+-------------------+-------------------+
```

---

## 4. Key Scientific Insights & Paper Recommendations

1. **The Role of Dedicated Physical Stations:**
   - When a location has a physical monitoring station, the **Direct Temporal LSTM Forecast** provides the highest accuracy ($\text{MAE} = 13.61\,\mu\text{g/m}^3$ for PM2.5). Local historical autoregression captures specific micro-scale emissions that cannot be inferred from a distance.
2. **The Power of Spatial Interpolation for Unmonitored Territory:**
   - In unmonitored zones where local sensors do not exist, **IDW Spatial Forecast Interpolation** reduces error from $28.83\,\mu\text{g/m}^3$ (nearest-neighbor fallback) to **$24.30\,\mu\text{g/m}^3$** (**$15.71\%$ error reduction** for PM2.5, **$19.71\%$ error reduction** for PM10).
3. **Decoupled Terminology for Academic Paper:**
   - The manuscript should explicitly distinguish between **Monitored Station Forecasting** ($\text{MAE} \approx 10\text{--}13\,\mu\text{g/m}^3$) and **Unmonitored Spatial Estimation** ($\text{MAE} \approx 24\,\mu\text{g/m}^3$). Conflating these two distinct operational scenarios would mislead reviewers.
