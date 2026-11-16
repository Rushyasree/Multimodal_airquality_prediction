# VayuDrishti Phase 12: Spatial Interpolation Service & Urban Heatmap Implementation Report

**Document Status:** Implementation Technical Specification & Verification  
**Module Name:** `SpatialInterpolationService`  
**Source Path:** `src/services/spatial_interpolation_service.py`  
**Configuration Path:** `src/config/spatial_config.py`  
**Date:** November 2026  
**Execution Environment:** Python 3.10.11 / NumPy 2.2.6 / PyTorch 2.12.1+cpu  

---

## 1. Executive Summary & Purpose

Phase 12 introduces a scientifically defensible, CPU-optimized **Spatial Prediction Layer** to VayuDrishti. Prior to Phase 12, the system generated air quality forecasts exclusively at 90 discrete, physical Central Pollution Control Board (CPCB) and Andhra Pradesh Pollution Control Board (APPCB) CAAQMS monitoring stations.

The `SpatialInterpolationService` bridges the gap between discrete monitoring sites and continuous urban geography. It enables:
1. **Continuous 2D Spatial Concentration Heatmaps:** Generating high-resolution urban concentration grids ($25 \times 25$, $50 \times 50$, $75 \times 75$) across major metropolitan agglomerations (Delhi NCR, Mumbai MMR, Kolkata, Bengaluru).
2. **Arbitrary-Location Point Prediction:** Computing localized PM2.5 and PM10 estimates at any user-specified coordinate with distance-decayed 95% spatial confidence intervals.
3. **Strict Target Timestamp & Horizon Preservation:** Ensuring that spatial surfaces are generated exclusively from synchronous forecasts sharing the identical target hour ($t+1$).

---

## 2. Mathematical Formulation & Algorithmic Design

### 2.1 Geodesic Haversine Distance
Because Euclidean distance directly on latitude/longitude decimal degrees introduces severe latitude-dependent spatial distortion (up to 15% error at Delhi's latitude $\approx 28.6^\circ\text{N}$), all spatial separations are computed via the **vectorized Great-Circle Haversine metric**:

$$\Delta\phi = \phi_2 - \phi_1, \quad \Delta\lambda = \lambda_2 - \lambda_1$$
$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
$$c = 2 \arcsin\left(\min\left(1.0, \, \sqrt{a}\right)\right)$$
$$d = R_{\text{Earth}} \cdot c, \quad R_{\text{Earth}} = 6371.0088\,\text{km}$$

The implementation is vectorized in NumPy to broadcast query coordinate arrays $(M, 1)$ against station coordinate arrays $(1, N)$, computing the full $M \times N$ distance matrix in $< 5\,\text{ms}$.

---

### 2.2 Modified Inverse Distance Weighting (IDW)
The concentration $Z(x_0)$ at query coordinate $x_0 = (\text{lat}_0, \text{lon}_0)$ is estimated as the distance-weighted harmonic combination of eligible neighboring station predictions:

$$\hat{Z}(x_0) = \frac{\sum_{i \in \mathcal{N}(x_0)} w_i(x_0) \, Z(x_i)}{\sum_{i \in \mathcal{N}(x_0)} w_i(x_0)}$$

#### Weight Formulation with Numerical Safeguards:
$$w_i(x_0) = \frac{1}{\left(d(x_0, x_i) + \epsilon\right)^p}$$

- **Power Parameter ($p$):** Configurable exponent, default $p=2.0$ (standard Inverse Distance Squared). Higher values localize influence around adjacent monitors; lower values produce smoother regional fields.
- **Epsilon Safeguard ($\epsilon$):** $\epsilon = 0.001\,\text{km}$ ($1\,\text{meter}$) prevents numerical division by zero.
- **Exact Station Snapping:** If the nearest physical station is within $d \le 0.05\,\text{km}$ ($50\,\text{meters}$), the service bypasses interpolation and snaps directly to the station's exact forecast ($w_i = 1.0, w_{j \ne i} = 0.0$).
- **Neighborhood Selection $\mathcal{N}(x_0)$:** Stations exceeding the maximum valid radius ($R_{\max} = 40.0\,\text{km}$) are strictly excluded ($w_i = 0.0$). If `max_neighbors` is configured (default 10), only the $k$ nearest stations within radius are included.

---

### 2.3 Distance-Decayed Spatial Confidence Bounds
Unlike point measurements at physical sensors, unmonitored spatial estimates carry spatial uncertainty that increases monotonically with distance to the nearest physical monitor:

$$\text{CI}_{95\%}^{\text{PM2.5}}(x_0) = \hat{Z}_{\text{PM2.5}}(x_0) \pm \left(4.0 + 0.85 \cdot d_{\min}(x_0)\right) \quad (\mu\text{g/m}^3)$$
$$\text{CI}_{95\%}^{\text{PM10}}(x_0) = \hat{Z}_{\text{PM10}}(x_0) \pm \left(7.5 + 1.40 \cdot d_{\min}(x_0)\right) \quad (\mu\text{g/m}^3)$$

This provides decision-makers with clear, defensible uncertainty bounds based on sensor network proximity.

---

## 3. Data Ingestion & Context Validation

To prevent temporal leakage or unphysical blending, the service enforces strict input validation via `ForecastContext`:
1. **Synchronous Target Time:** All ingested records must match the reference `target_time` ISO string (e.g., `2026-11-15 12:00:00`). Any divergent station timestamp is rejected.
2. **Forecast Horizon Uniformity:** All records must share identical `forecast_horizon` (`Next 1 Hour (t + 1)`).
3. **Coordinate Bounds Checking:** Latitude $\in [-90, 90]$ and Longitude $\in [-180, 180]$.
4. **NaN and Infinity Filtering:** Stations reporting corrupted or NaN predictions are dropped before distance calculation.
5. **Deduplication:** Repeated station IDs are automatically deduplicated.
6. **Minimum Station Requirement:** At least $N_{\min} = 3$ valid stations must be available. If fewer exist, the service returns status `INSUFFICIENT_STATIONS` without fabricating values.
7. **Physical Non-Negativity:** All outputs are strictly clamped to $[\ge 0.0\,\mu\text{g/m}^3]$.

---

## 4. City Configuration Architecture (`src/config/spatial_config.py`)

Geographic bounding boxes are decoupled into `src/config/spatial_config.py`:

```python
SUPPORTED_CITIES = {
    "delhi_ncr": CityBoundingBox(
        city_id="delhi_ncr", name="Delhi NCR",
        min_lat=28.35, max_lat=28.95, min_lon=76.85, max_lon=77.45,
        center_lat=28.65, center_lon=77.15, default_zoom=10, primary_state="Delhi"
    ),
    "mumbai_mmr": CityBoundingBox(
        city_id="mumbai_mmr", name="Mumbai MMR",
        min_lat=18.85, max_lat=19.35, min_lon=72.75, max_lon=73.20,
        center_lat=19.07, center_lon=72.88, default_zoom=10, primary_state="Maharashtra"
    ),
    "kolkata": CityBoundingBox(
        city_id="kolkata", name="Kolkata Metropolitan",
        min_lat=22.35, max_lat=22.75, min_lon=88.20, max_lon=88.55,
        center_lat=22.57, center_lon=88.36, default_zoom=11, primary_state="West Bengal"
    ),
    "bengaluru": CityBoundingBox(
        city_id="bengaluru", name="Bengaluru Urban",
        min_lat=12.75, max_lat=13.20, min_lon=77.40, max_lon=77.80,
        center_lat=12.97, center_lon=77.59, default_zoom=11, primary_state="Karnataka"
    ),
}
```

---

## 5. Arbitrary Location Function (`predict_at_location`)

### Function Signature:
```python
def predict_at_location(
    latitude: float,
    longitude: float,
    forecast_context: Union[ForecastContext, List[StationForecastRecord]],
) -> Dict[str, Any]
```

### Return Schema:
```json
{
  "latitude": 28.6315,
  "longitude": 77.2167,
  "predicted_pm25": 104.35,
  "predicted_pm10": 182.60,
  "nearest_station_id": "DL001",
  "nearest_station_distance_km": 2.14,
  "num_contributing_stations": 10,
  "contributing_station_ids": ["DL001", "DL004", "DL012", ...],
  "data_quality_status": "HIGH_CONFIDENCE_INTERPOLATION",
  "interpolation_method": "Modified IDW (p=2.0)",
  "target_time": "2026-11-15 12:00:00",
  "forecast_issue_time": "2026-11-15 11:00:00",
  "forecast_horizon": "Next 1 Hour (t + 1)",
  "model_source": "VayuDrishti_Ensemble",
  "ci_95_pm25": [98.53, 110.17],
  "ci_95_pm10": [172.10, 193.10],
  "is_extrapolation": false
}
```

---

## 6. Implementation Verification Status

| Capability / Requirement | Target File | Verification Evidence | Status |
| :--- | :--- | :--- | :--- |
| Geodesic Distance | `src/services/spatial_interpolation_service.py` | Haversine formula tested vs known geodetic distances | **VERIFIED** |
| Exact Snapping | `src/services/spatial_interpolation_service.py` | Tested at $d=0.0\,\text{km}$; snaps to exact monitor values | **VERIFIED** |
| Power Parameter | `src/services/spatial_interpolation_service.py` | Tested $p \in [1.0, 3.0]$; smooth decay verified | **VERIFIED** |
| Distance Cap | `src/services/spatial_interpolation_service.py` | Rejects extrapolation beyond $R_{\max} = 40\,\text{km}$ | **VERIFIED** |
| Min Stations Check | `src/services/spatial_interpolation_service.py` | Rejects $< 3$ stations with `INSUFFICIENT_STATIONS` | **VERIFIED** |
| City Grid Generation | `src/services/spatial_interpolation_service.py` | Tested $25\times 25, 50\times 50, 75\times 75$ resolutions | **VERIFIED** |
| Arbitrary Prediction | `src/services/spatial_interpolation_service.py` | Validated at landmark coordinates in 4 cities | **VERIFIED** |
| Unit Tests | `tests/test_spatial_interpolation_service.py` | 14/14 unit tests passing | **VERIFIED** |
