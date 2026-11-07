# VayuDrishti: Accuracy Optimization v2 — Data and Baseline Audit

**Artifact Location**: `experiments/accuracy_optimization_v2/DATA_AND_BASELINE_AUDIT.md`  
**Date**: October 2026  
**Dataset Analyzed**: CPCB Multi-City Hourly Air Quality & ERA5-Land Weather (1,369,779 cleaned records, 90 monitoring stations across India)  
**Historical Sequence Partitions**:
- Training Set: 336,641 sequences ($[336641, 24, 27]$, 70% chronological split, cutoff: 2018-11-06 14:00:00)
- Validation Set: 326,423 sequences ($[326423, 24, 27]$, 15% chronological split)
- Held-Out Test Benchmark: 404,431 sequences ($[404431, 24, 27]$, 15% chronological split, strictly untouched)
- Matched Image Cohort: 199 paired samples (97 held-out test instances across 9 station-date groups)

---

## 1. Audit Checklist & Verification Findings (Phase 1)

| Audit Check | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :--- |
| **1. Feature Availability at $t$** | Every input feature must be available at or before forecast issue time $t$. | All 27 channels in the 24h history $[t-23, t]$ represent sensor, meteorological, cyclical time, and station coordinate variables measured at or before $t$. | **PASS** |
| **2. Rolling Feature Past-Only** | Rolling features must compute strictly from historical steps ($k \le 23$). | Rolling statistics (3h, 6h, 12h, 24h) and lag values ($t-1$ to $t-23$) are extracted exclusively from the lookback window. The target timestep $t+1$ is never accessed. | **PASS** |
| **3. Training-Only Scaling & Imputation** | Scalers and medians must be fitted on training split only. | `preprocessing_metadata.json` confirms `training_preprocessing_cutoff` = `2018-11-06 14:00:00`. `scaler_means`, `scaler_stds`, and medians were fitted strictly prior to this cutoff. | **PASS** |
| **4. Station Boundary Integrity** | Sequences and lags must not cross unrelated station histories. | `data/create_sequences.py` groups strictly by `StationId` and verifies that all consecutive steps are contiguous 1-hour deltas (`diff == pd.Timedelta(hours=1)`). | **PASS** |
| **5. Forecast Horizon & Target Alignment** | Target must be the actual observation one hour after final input ($t+1$). | Verified by assertion `target_time - last_history_time == pd.Timedelta(hours=1)`. The forecast horizon is strictly $h = 1$ hour. | **PASS** |
| **6. Chronological Test Isolation** | Test period must remain untouched with zero boundary overlap. | Sequences must satisfy `sequence_start >= split_start and sequence_end < split_end`. Zero sequences cross split boundaries. Test set is strictly locked. | **PASS** |
| **7. Identical Target Rows** | Persistence, LSTM, and candidate models must be evaluated on identical rows. | All comparisons on the large test benchmark use the exact same 404,431 rows in `y_test.npy`. Image-matched comparisons use the exact same 97 test rows. | **PASS** |

---

## 2. Full Historical Benchmark Baseline Record ($N = 404,431$)

Before any algorithmic modification or retraining, baseline metrics are recorded across the entire historical held-out test benchmark:

| Model / Baseline | Target Horizon | Evaluated Test Instances ($N$) | PM2.5 MAE (µg/m³) | PM2.5 RMSE (µg/m³) | PM2.5 $R^2$ | PM10 MAE (µg/m³) | PM10 RMSE (µg/m³) | PM10 $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Hour Persistence** ($y_{t+1} = y_t$) | $t + 1$ | 404,431 | 10.6783 | 19.8072 | 0.9423 | 19.7384 | 36.2374 | 0.9115 |
| **24-Hour Seasonal Persistence** ($y_{t+1} = y_{t-23}$) | $t + 1$ | 404,431 | 28.9914 | 53.4336 | 0.5797 | 50.9172 | 84.3962 | 0.5200 |
| **168-Hour Weekly Persistence** ($y_{t+1} = y_{t-167}$)* | $t + 1$ | 377,481 (Eligible subset) | 45.1216 | 81.7984 | 0.0403 | 75.0273 | 121.4629 | 0.0222 |
| **Pretrained 2-Layer LSTM Baseline** | $t + 1$ | 404,431 | 10.5719 | 18.5789 | 0.9492 | 18.8643 | 32.2027 | 0.9301 |

*\*Note: 168-hour persistence requires an observation 7 days prior at the identical station. 377,481 of the 404,431 test sequences (93.3%) possess valid preceding weekly timestamps.*

---

## 3. Strict Separation of Benchmarks

To eliminate scientific ambiguity, two evaluation benchmarks are maintained independently throughout Optimization v2:

1. **Large Historical Benchmark ($N = 404,431$)**:
   - Spans 90 CPCB continuous monitoring stations across all Indian meteorological zones from late 2019 through mid-2020.
   - Evaluates pure numerical sensor forecasting, weather integration, and gradient-boosted temporal learning.
2. **Matched-Image Multimodal Benchmark ($N = 97$)**:
   - Spans 9 station-date groups with co-located outdoor photographs ($\le 2.56$ km).
   - Evaluates visual residual fusion and modality fallback behavior.
   - Any metric computed on the 97-sample cohort must never be conflated with the 404k historical benchmark.
