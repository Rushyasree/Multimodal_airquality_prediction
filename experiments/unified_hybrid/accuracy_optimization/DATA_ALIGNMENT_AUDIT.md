# VayuDrishti: Multimodal Data Alignment and Integrity Audit

**Audit Date**: October 2026  
**Artifact Location**: `experiments/unified_hybrid/accuracy_optimization/DATA_ALIGNMENT_AUDIT.md`  
**Dataset Analyzed**: `experiments/unified_hybrid/paired_dataset_hybrid.csv` (199 multimodal pairs)  
**Associated Cache Artifacts**:
- `experiments/multimodal_fusion/cached_sequences.npz` (33 station-date sequence groups, 24h x 27 features)
- `experiments/multimodal_fusion/cached_image_features.npz` (199 ResNet-50 feature vectors, 2048-dim)

---

## 1. Executive Summary

This audit establishes the rigorous empirical and scientific foundation for multimodal accuracy optimization in the VayuDrishti project. Prior fusion experiments (`experiments/multimodal_fusion/` and `experiments/unified_hybrid/`) suffered from severe negative transfer:
- Sensor-Only LSTM baseline (pretrained on 1.37M rows) achieves **PM2.5 MAE = 9.04 µg/m³** ($R^2 = 0.9625$) on the untouched 97-sample test set.
- Prior fusion models collapsed to **PM2.5 MAE = 40.94 – 58.94 µg/m³** ($R^2 < 0.35$).

This audit reveals the precise mathematical and operational reasons for this discrepancy, analyzes the multimodal alignment constraints across space and time, and establishes the strict design criteria required to achieve state-of-the-art hybrid predictive performance without negative transfer.

---

## 2. Dataset Split and Group Disjointness

To prevent spatial-temporal leakage, data partitioning is performed strictly at the **`station_date_group`** level. All images and sequences captured at a particular station on a specific calendar date reside in exactly one partition.

| Split | Sample Count ($N$) | Unique Station-Date Groups | Unique Stations | Geodesic Distance (Mean ± Std) | Geodesic Distance (Min – Max) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 81 | 18 | 12 | 0.73 ± 0.50 km | 0.05 – 2.56 km |
| **VAL** | 21 | 6 | 6 | 1.07 ± 0.63 km | 0.25 – 2.16 km |
| **TEST** | 97 | 9 | 8 | 0.71 ± 0.44 km | 0.10 – 1.81 km |
| **TOTAL** | **199** | **33** | **18** | **0.76 ± 0.50 km** | **0.05 – 2.56 km** |

### Group Disjointness Verification
- `Train ∩ Val Groups`: **0**
- `Train ∩ Test Groups`: **0**
- `Val ∩ Test Groups`: **0**

Target distributions across splits exhibit significant natural variance, reflecting real-world seasonal and meteorological variations:
- **Train Targets**: PM2.5 = 175.1 ± 195.1 µg/m³ (Range: 30.0 – 473.0), PM10 = 363.2 ± 373.4 µg/m³
- **Val Targets**: PM2.5 = 78.1 ± 38.0 µg/m³ (Range: 45.0 – 148.1), PM10 = 177.6 ± 69.2 µg/m³
- **Test Targets**: PM2.5 = 110.8 ± 80.9 µg/m³ (Range: 22.6 – 500.2), PM10 = 245.8 ± 163.3 µg/m³

---

## 3. Spatial and Temporal Alignment Analysis

### 3.1 Spatial Alignment
- The maximum distance between any camera capture location and the corresponding CPCB continuous monitoring station is **2.56 km** (median: 0.60 km, 75th percentile: 1.03 km).
- In urban atmospheres, atmospheric aerosol concentrations maintain high spatial autocorrelation over distances $< 3$ km, confirming that spatial proximity is sufficient for paired multimodal modeling.

### 3.2 Temporal Alignment and Sub-Daily Resolution Limitation
- **Sensor Sequences**: Continuous, hourly resolved time series capturing 24 consecutive hours of telemetry (27 features: PM2.5, PM10, gases, meteorological variables, temporal cyclical encodings, and coordinates).
- **Camera Imagery**: Public camera images sourced from the PM25Vision dataset contain valid date metadata (`capture_date`), but lack sub-daily EXIF timestamps.
- **Consequence**: Pairing a static daytime camera capture with a dynamic hourly telemetry window creates a phase discrepancy. If an image captured at 11:00 AM is paired with a target at 16:00 PM, diurnally shifting boundary layers and localized transient plumes mean the visual haze signal represents an integrated daytime optical depth rather than instantaneous next-hour concentration.

---

## 4. Root Cause of Negative Transfer in Prior Fusion Work

Why did earlier fusion models achieve MAE > 40 µg/m³ when the pretrained LSTM achieved 9.04 µg/m³?

1. **Destructive Regressor Head Replacement**:
   - The pretrained sensor baseline (`models/lstm_baseline/best_lstm_model.pt`) learned its representations across **1,371,118 rows** and 90 continuous stations.
   - Previous fusion training pipelines (`train_v2.py`, `train_hybrid.py`) extracted the LSTM backbone but completely discarded the pretrained regression head (`regressor.0`, `regressor.3`).
   - A randomly initialized joint MLP head (`Linear(129, 64) -> Linear(64, 32) -> Linear(32, 2)`) was trained from scratch on only **81 training samples**.
   - Training a deep MLP regression head from scratch on 81 samples with target values from 30 to 473 µg/m³ caused catastrophic overfitting and failure to generalize.

2. **Forced Concatenation Without Residual Bounds**:
   - In previous models, predictions were computed as $\hat{y} = \text{MLP}([e_{\text{sensor}}, e_{\text{image}}])$.
   - Because all information flowed through the newly initialized MLP, even when the visual gate closed ($\alpha \to 0$), the sensor representation was still processed through an uncalibrated, overfitted head, destroying sensor accuracy.

3. **Multi-Task Scale Imbalance**:
   - PM10 concentrations (mean ~363 µg/m³ in train) are ~2.5x larger in magnitude than PM2.5 (mean ~175 µg/m³).
   - Computing unnormalized SmoothL1 loss caused gradients to be dominated by PM10 errors, degrading PM2.5 convergence.

---

## 5. Architectural Requirements for Accuracy Optimization

To resolve these failure modes, the optimized hybrid architecture must satisfy:

1. **Intact Pretrained Sensor Regressor**:
   - Preserve the full pretrained `LSTMForecaster` (backbone and regressor head) as the primary base predictor $\hat{y}_{\text{sensor}}$.
   - Guarantees baseline PM2.5 MAE of $\le 9.04$ µg/m³.

2. **Residual Formulation with Zero-Centered Initialization**:
   - Visual and fusion components must predict a bounded **residual correction**:
     $$\hat{y}_{\text{hybrid}} = \hat{y}_{\text{sensor}} + \alpha \cdot \Delta_{\text{correction}}$$
   - Initialize the residual weights and biases near zero so that at epoch 0, $\hat{y}_{\text{hybrid}} \equiv \hat{y}_{\text{sensor}}$.

3. **Quality-Aware Gating with Explicit Modality Fallback**:
   - Gating factor $\alpha \in [0, 1]$ modulates the magnitude of the visual correction based on sensor features, image representation, and spatial distance $d$.
   - When visual data is absent or degraded ($m_{\text{visual}} = 0$), $\alpha$ is strictly 0.0, guaranteeing deterministic fallback to pure telemetry with zero performance penalty.

4. **Normalized Multi-Target Loss**:
   - Balance PM2.5 and PM10 gradients using target standard deviations ($\sigma_{2.5} \approx 77.42$, $\sigma_{10} \approx 137.47$).

5. **Validation-Only Early Stopping**:
   - Model selection strictly monitors validation loss on the 21 validation samples (6 disjoint groups); zero test loss monitoring.
