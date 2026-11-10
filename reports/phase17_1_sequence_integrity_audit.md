# Phase 17.1 — Forecast Time Alignment & Provenance Validation Audit Report

> **Audit Date**: November 10, 2026  
> **Baseline Commit**: `46f24ecc83c97a797018d0f2cd769fa71908471e`  
> **Target Dataset**: `data/processed/dashboard_demo_sequences.csv`  
> **Preprocessing Metadata**: `data/processed/model_sequences/preprocessing_metadata.json`  

---

## 1. Executive Summary & Purpose

Phase 17.1 addresses data provenance, temporal alignment, and multi-model input integrity across the VayuDrishti dashboard. The primary objectives of this phase are:

1. **Sequence Integrity Audit**: Audit all sequences in `data/processed/dashboard_demo_sequences.csv` for row counts, frequency, gaps, duplicate timestamps, and coverage.
2. **Strict Provenance Categorization**: Clearly separate **live physical telemetry**, **regional model proxies**, and **historical benchmark demo sequences**. Historical benchmark sequences are **never** labeled as live telemetry.
3. **Temporal Horizon Alignment**: Ensure that forecast issue time is determined by actual sequence end time ($t$), with predicted target time set strictly to $t + 1\text{ hr}$. Historical sequences are labeled as `HISTORICAL DEMO FORECAST`.
4. **Comparison Tensor Parity**: Enforce that both the Sensor-Only LSTM and Unified Hybrid Model ingest the identical antecedent $(1, 24, 27)$ normalized tensor, verified via SHA-256 tensor digest.
5. **Photographic Inference Disclosure**: Explicitly disclose calendar-date precision limitations of optical images and delineate image-based haze estimation from causal sequential forecasting.
6. **Preservation of Model Weights and Baselines**: All trained model checkpoints, Phase 8/9 metrics, and benchmark evaluation tables remain completely unmodified.

---

## 2. Dashboard Demo Sequences Audit

- **Total Rows**: 576 rows
- **Total Monitored Stations**: 12 stations
- **Total Scenarios**: 24 distinct scenarios (2 scenarios per station, each consisting of exactly 24 contiguous hourly observations)
- **Input Tensor Dimensions**: $(1, 24, 27)$
- **Dataset Classification**: `Curated Benchmark Evaluation Sequences (Static Historical Test Holdouts)`

### 2.1 Station-Level Inventory

| Station ID | Row Count | Scenarios | Unique Timestamps | Duplicates | Earliest Timestamp | Latest Timestamp | Coverage Period & Nature |
|:---|:---:|:---:|:---:|:---:|:---|:---|:---|
| **AP001** | 48 | 2 | 48 | 0 | 2026-04-27 15:00:00 | 2026-12-04 13:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **AP005** | 48 | 2 | 48 | 0 | 2026-04-09 21:00:00 | 2026-11-29 06:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **DL001** | 48 | 2 | 48 | 0 | 2026-04-20 13:00:00 | 2026-11-15 23:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **DL008** | 48 | 2 | 48 | 0 | 2026-04-21 02:00:00 | 2026-11-20 01:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **DL024** | 48 | 2 | 48 | 0 | 2026-04-22 17:00:00 | 2026-11-20 08:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **KA006** | 48 | 2 | 48 | 0 | 2026-04-25 10:00:00 | 2026-12-04 16:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **KA011** | 48 | 2 | 48 | 0 | 2026-04-25 15:00:00 | 2026-11-20 17:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **MH008** | 48 | 2 | 48 | 0 | 2026-04-26 12:00:00 | 2026-11-21 08:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **BR005** | 48 | 2 | 48 | 0 | 2026-04-24 16:00:00 | 2026-11-22 03:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **CH001** | 48 | 2 | 48 | 0 | 2026-04-23 20:00:00 | 2026-11-29 00:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **TG006** | 48 | 2 | 48 | 0 | 2026-04-28 07:00:00 | 2026-11-27 15:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |
| **WB008** | 48 | 2 | 48 | 0 | 2026-04-29 11:00:00 | 2026-11-28 07:00:00 | Two distinct 24-hour discontinuous seasonal episodes (Holdout Test Evaluation Windows) |

### 2.2 Scenario-Level Audit

| Scenario ID | Station | Rows | Hourly? | Missing Hours | Dups | Window Start | Window End (Issue Time) | Target Time ($t+1$) | Aligned? |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---:|
| `SCENARIO_01_AP001_20261204_14` | **AP001** | 24 | True | 0 | 0 | 2026-12-03 14:00:00 | 2026-12-04 13:00:00 | 2026-12-04 14:00:00 | YES |
| `SCENARIO_02_AP001_20260428_15` | **AP001** | 24 | True | 0 | 0 | 2026-04-27 15:00:00 | 2026-04-28 14:00:00 | 2026-04-28 15:00:00 | YES |
| `SCENARIO_03_AP005_20261129_07` | **AP005** | 24 | True | 0 | 0 | 2026-11-28 07:00:00 | 2026-11-29 06:00:00 | 2026-11-29 07:00:00 | YES |
| `SCENARIO_04_AP005_20260410_21` | **AP005** | 24 | True | 0 | 0 | 2026-04-09 21:00:00 | 2026-04-10 20:00:00 | 2026-04-10 21:00:00 | YES |
| `SCENARIO_05_DL001_20261116_00` | **DL001** | 24 | True | 0 | 0 | 2026-11-15 00:00:00 | 2026-11-15 23:00:00 | 2026-11-16 00:00:00 | YES |
| `SCENARIO_06_DL001_20260421_13` | **DL001** | 24 | True | 0 | 0 | 2026-04-20 13:00:00 | 2026-04-21 12:00:00 | 2026-04-21 13:00:00 | YES |
| `SCENARIO_07_DL008_20261120_02` | **DL008** | 24 | True | 0 | 0 | 2026-11-19 02:00:00 | 2026-11-20 01:00:00 | 2026-11-20 02:00:00 | YES |
| `SCENARIO_08_DL008_20260422_02` | **DL008** | 24 | True | 0 | 0 | 2026-04-21 02:00:00 | 2026-04-22 01:00:00 | 2026-04-22 02:00:00 | YES |
| `SCENARIO_09_DL024_20261120_09` | **DL024** | 24 | True | 0 | 0 | 2026-11-19 09:00:00 | 2026-11-20 08:00:00 | 2026-11-20 09:00:00 | YES |
| `SCENARIO_10_DL024_20260423_17` | **DL024** | 24 | True | 0 | 0 | 2026-04-22 17:00:00 | 2026-04-23 16:00:00 | 2026-04-23 17:00:00 | YES |
| `SCENARIO_11_KA006_20261204_17` | **KA006** | 24 | True | 0 | 0 | 2026-12-03 17:00:00 | 2026-12-04 16:00:00 | 2026-12-04 17:00:00 | YES |
| `SCENARIO_12_KA006_20260426_10` | **KA006** | 24 | True | 0 | 0 | 2026-04-25 10:00:00 | 2026-04-26 09:00:00 | 2026-04-26 10:00:00 | YES |
| `SCENARIO_13_KA011_20261120_18` | **KA011** | 24 | True | 0 | 0 | 2026-11-19 18:00:00 | 2026-11-20 17:00:00 | 2026-11-20 18:00:00 | YES |
| `SCENARIO_14_KA011_20260426_15` | **KA011** | 24 | True | 0 | 0 | 2026-04-25 15:00:00 | 2026-04-26 14:00:00 | 2026-04-26 15:00:00 | YES |
| `SCENARIO_15_MH008_20261121_09` | **MH008** | 24 | True | 0 | 0 | 2026-11-20 09:00:00 | 2026-11-21 08:00:00 | 2026-11-21 09:00:00 | YES |
| `SCENARIO_16_MH008_20260427_12` | **MH008** | 24 | True | 0 | 0 | 2026-04-26 12:00:00 | 2026-04-27 11:00:00 | 2026-04-27 12:00:00 | YES |
| `SCENARIO_17_BR005_20261122_04` | **BR005** | 24 | True | 0 | 0 | 2026-11-21 04:00:00 | 2026-11-22 03:00:00 | 2026-11-22 04:00:00 | YES |
| `SCENARIO_18_BR005_20260425_16` | **BR005** | 24 | True | 0 | 0 | 2026-04-24 16:00:00 | 2026-04-25 15:00:00 | 2026-04-25 16:00:00 | YES |
| `SCENARIO_19_CH001_20261129_01` | **CH001** | 24 | True | 0 | 0 | 2026-11-28 01:00:00 | 2026-11-29 00:00:00 | 2026-11-29 01:00:00 | YES |
| `SCENARIO_20_CH001_20260424_20` | **CH001** | 24 | True | 0 | 0 | 2026-04-23 20:00:00 | 2026-04-24 19:00:00 | 2026-04-24 20:00:00 | YES |
| `SCENARIO_21_TG006_20261127_16` | **TG006** | 24 | True | 0 | 0 | 2026-11-26 16:00:00 | 2026-11-27 15:00:00 | 2026-11-27 16:00:00 | YES |
| `SCENARIO_22_TG006_20260429_07` | **TG006** | 24 | True | 0 | 0 | 2026-04-28 07:00:00 | 2026-04-29 06:00:00 | 2026-04-29 07:00:00 | YES |
| `SCENARIO_23_WB008_20261128_08` | **WB008** | 24 | True | 0 | 0 | 2026-11-27 08:00:00 | 2026-11-28 07:00:00 | 2026-11-28 08:00:00 | YES |
| `SCENARIO_24_WB008_20260430_11` | **WB008** | 24 | True | 0 | 0 | 2026-04-29 11:00:00 | 2026-04-30 10:00:00 | 2026-04-30 11:00:00 | YES |

---

## 3. Sequence Validation Specification (`src/prediction/sequence_validator.py`)

The `validate_forecast_sequence` function enforces rigorous pre-inference validation before passing data to any neural network or GBDT branch:

```python
def validate_forecast_sequence(
    df: pd.DataFrame,
    station_id: str,
    max_age_hours: Optional[float] = None,
    require_exact_station: bool = True,
    forecast_issue_time: Optional[datetime] = None,
) -> SequenceValidationResult:
```

### Validation Checks Enforced

1. **Dimensional Window**: Strictly requires 24 contiguous hourly rows ($24 \times 1\text{ hr}$). Sequences with $< 24$ or $> 24$ rows are rejected.
2. **Temporal Order**: Timestamps must be strictly monotonically increasing ($t_i > t_{i-1}$).
3. **Duplicate Timestamps**: Zero duplicate timestamps allowed.
4. **Regular Hourly Cadence**: Delta between consecutive timestamps must equal exactly $1\text{ hour}$ (3600 seconds). Any missing hour or timestamp jump flags the sequence as invalid.
5. **Station Identity Integrity**: Verifies that sequence records originate from the target station. If `require_exact_station=True` and a regional proxy (e.g., AP001 proxy for Nellore) is passed, the sequence is rejected for station-specific live inference and classified as `REGIONAL_PROXY`.
6. **Criteria Pollutant Completeness**: Non-null criteria pollutants (PM2.5, PM10) must be present with at most 6 missing values (permitted for standard imputation).
7. **Feature Schema & Scaler Compatibility**: Sequence must map into the 27 canonical feature channels with matching z-score scaler means and standard deviations from `preprocessing_metadata.json`.
8. **Freshness & Age Threshold**: When `max_age_hours` is supplied (default 3.0 hours for live operations), sequences older than `max_age_hours` or sequences with future timestamps are flagged. If fresh, they receive `LIVE FORECAST (VALIDATED INPUT)`; otherwise, they are designated `HISTORICAL DEMO FORECAST`.

---

## 4. Multi-Model Input Array Integrity & Tensor Parity

To prevent input discrepancy between comparison models in Section 4 (*Model Comparison*), the following array integrity mechanism is implemented:

1. **Single Resolution**: `ModelComparisonService.compare_all` resolves `df_to_use` once at the beginning of the comparison pipeline.
2. **Unified Preprocessing**: Both `SensorForecastService.get_forecast()` and `HybridPredictionService.predict_hybrid()` utilize `LSTMForecasterService.preprocess_sequence(df_to_use, station_id)`.
3. **Cryptographic Tensor Digest**: Both services compute `hashlib.sha256(seq_arr.tobytes()).hexdigest()` on the standardized $(1, 24, 27)$ tensor.
4. **Parity Assertion**: The comparison service asserts that `sensor_digest_lstm == sensor_digest_hybrid`. A synchronization status badge (`Strictly Synchronized (Identical Tensor Digest)`) is exposed only when both digests match.

---

## 5. Explicit Dashboard Provenance States

The application strictly enforces six mutually exclusive data and prediction states across all UI components:

| State Identifier | Category | Definition & Verification Criteria |
|:---|:---|:---|
| `LIVE MEASURED OBSERVATION` | Ground Truth Observation | Physical instrument measurement received from CPCB/APPCB monitoring station within last 3 hours. |
| `LIVE FORECAST (VALIDATED INPUT)` | Model Forecast | Continuous 24h antecedent sequence ending $\le 3$ hours ago, strictly passing all validation checks. |
| `HISTORICAL DEMO FORECAST` | Offline Demonstration | Forecast computed using curated historical benchmark sequence ($t > 3\text{ hrs}$ in the past). Never called live. |
| `REGIONAL MODEL ESTIMATE` | Atmospheric Model | Open-Meteo European Centre / GFS numerical reanalysis estimate when ground station telemetry is offline. |
| `MODELED BASELINE` | Calibrated Calculation | Offline deterministic mathematical reference based on diurnal cycles. |
| `IMAGE-BASED EXPERIMENTAL ESTIMATE` | Computer Vision | Single-frame daytime outdoor photograph analyzed via ResNet-50. Calendar-date precision only (asynchronous prior). |

---

## 6. Photographic Data Limitations & Provenance Disclosure

- **Camera Dataset Provenance**: Trained on public PM25Vision / Shanghai outdoor camera datasets.
- **Temporal Granularity**: Optical photographs possess **calendar-date precision only**. Sub-daily EXIF hours are absent in public benchmarks.
- **Functional Delineation**: Image inference operates as an **asynchronous optical haze prior**, completely distinct from causal next-hour sequential forecasting.
- **Fallback Mechanism**: When optical images are unavailable, the Unified Hybrid Model suppresses the visual channel through learned gating (mask = 0.0), maintaining pure sensor accuracy ($R^2 = 0.949$).

---

## 7. Audit Conclusion & Safeguards

1. **Weights Untouched**: All neural network checkpoints (`best_lstm_model.pt`, `best_model.pt`, `best_optimized_hybrid_model.pt`, `best_gbdt_pm25.joblib`) remain 100% untouched.
2. **Metrics Preserved**: Historical evaluation metrics from Phase 8 ($R^2 = 0.949$ on 404,431 sequences) and Phase 9 ($R^2 = 0.963$ on 97 matched samples) are strictly maintained.
3. **No Artificial Inflation**: No claims of improved accuracy are made from provenance validation alone.
4. **Full Transparency**: Historical sequences are explicitly acknowledged as curated benchmark test episodes.
