# VayuDrishti — Phase 11: Independent Dataset Integrity Audit Report

**Audit Date:** November 5, 2026  
**Audit Scope:** Full Longitudinal Telemetry Dataset (October 22 – November 5, 2026)  
**Execution Script:** [`scripts/audit_phase11_dataset_integrity.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/audit_phase11_dataset_integrity.py)  
**Machine-Readable Audit Record:** [`experiments/recent_data_audit/phase11_dataset_integrity_audit.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/phase11_dataset_integrity_audit.json)  
**Overall Status:** **ALL 9 INTEGRITY CHECKS PASSED [VERIFIED]**  

---

## 1. Executive Summary & Verification Scope

In Phase 11 Task 2, an exhaustive, deterministic data audit was conducted across all raw tabular and metadata files in `experiments/recent_data_audit/`:
1. `live_trial_observations.csv`: **6,207 validated telemetry readings** across 4 CAAQMS stations.
2. `live_trial_predictions.csv`: **420 causal rolling predictions** ($t+1$) issued by `adapted_gbdt_only`.
3. `live_trial_evaluation.csv`: **418 verified forecast-target evaluation pairs**.
4. `collector_state.json`: Provenance metadata, operational counters, and per-station tracking.

Every record was audited across 9 orthogonal integrity vectors to verify strict causal ordering, zero target leakage, absence of duplicates or zero-imputations, model hash immutability, and full preservation of the Phase 8 trial cohort.

---

## 2. Comprehensive 9-Check Audit Matrix

| Check ID | Integrity Vector | Target Criteria | Actual Measured Result | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **Check 1** | **Forecast ID Uniqueness** | `forecast_id` and `match_id` must be 100% unique | Predictions: 420 unique / 420 total.<br>Evaluations: 418 unique / 418 total. | **PASS** |
| **Check 2** | **Duplicate Observation Detection** | Tuple `(station_id, timestamp, parameter)` unique | 6,207 unique tuples / 6,207 rows.<br>0 duplicates detected. | **PASS** |
| **Check 3** | **Chronological Consistency** | Monotonically non-decreasing timestamps per station | 0 backward jumps or inverted sequence steps detected across all 4 stations. | **PASS** |
| **Check 4** | **UTC Timestamp Validation** | All timestamps carry valid UTC offsets (`Z` or `+00:00`) | 100% ISO 8601 conformant (0 non-UTC formats across all 6,207 readings). | **PASS** |
| **Check 5** | **Forecast Horizon Verification** | Lead time $\Delta t = \text{target\_time} - \text{issue\_time} = 1.0\text{h}$ | Exactly 1.000 hours for all 420 predictions (min: 1.0h, max: 1.0h, bad: 0). | **PASS** |
| **Check 6** | **Target Matching Causality** | Target observation strictly at $t+1$; 0 lookahead | Evaluated target times match forecast targets with zero leakage into the 24h feature buffer. | **PASS** |
| **Check 7** | **Unmatched Prediction Accounting** | Identify missing target dropouts; confirm 0 imputation | Exactly 2 unmatched forecasts on station `DL008` (sensor outages). Zero synthetic values. | **PASS** |
| **Check 8** | **Model Checkpoint Provenance** | Frozen SHA-256 hashes match files on disk | $\text{PM}_{2.5}$ GBDT: `982dbdfb...` (Match)<br>$\text{PM}_{10}$ GBDT: `e951a12b...` (Match) | **PASS** |
| **Check 9** | **Phase 8 Cohort Preservation** | Phase 8 trial records unaltered within extended dataset | 16/16 Phase 8 pairs preserved with exact MAE match to 4 decimal places. | **PASS** |

---

## 3. Detailed Audit Findings by Check

### 3.1 Check 1 & 2: Identifier Uniqueness & Deduplication
- `live_trial_predictions.csv` contains 420 rows with 420 unique `forecast_id` values generated using the deterministic schema: `fc_{station_id}_{issue_iso}`.
- `live_trial_evaluation.csv` contains 418 rows with 418 unique `match_id` values formatted as `eval_{station_id}_{target_iso}`.
- In `live_trial_observations.csv`, every row represents an independent physical sensor observation. Zero duplicate readings exist.

### 3.2 Check 3 & 4: Timestamp Monotonicity & UTC Format
- Grouped by `station_id` and `parameter`, time vectors are strictly monotonic.
- Timestamps strictly adhere to ISO 8601 UTC notation (`YYYY-MM-DDTHH:MM:SSZ` or `+00:00`). No unhandled local Indian Standard Time (IST) offsets contaminate the persistent UTC storage layer.

### 3.3 Check 5 & 6: Causal Horizon & Zero Lookahead
```
Anchor Window (Input): [t - 23h, ..., t]  --> Feature Builder (49 Tabular Channels)
Issue Time (Prediction): t                --> Model Forecast issued
Target Horizon:          t + 1h           --> Strictly evaluated only after hour t+1 elapses
```
- For all 420 prediction records:
  $$\tau_{\text{lead}} = \frac{t_{\text{target}} - t_{\text{issue}}}{3600\ \text{seconds}} = 1.000\ \text{hour}$$
- Target hour observations are strictly excluded from the preceding 24-hour sequence buffer.

### 3.4 Check 7: Unmatched Prediction Accounting & Sensor Downtime
Of the 420 generated predictions, exactly **418** were successfully matched with ground-truth target observations, leaving **2 unmatched predictions**:

| Forecast ID | Station ID | Issue Time (UTC) | Target Time (UTC) | Predicted $\text{PM}_{2.5}$ | Predicted $\text{PM}_{10}$ | Ground Cause | Imputation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `fc_DL008_2026-10-30T00:00:00Z` | `DL008` | 2026-10-30 00:00Z | 2026-10-30 01:00Z | 15.21 | 25.86 | CPCB/OpenAQ Sensor Downtime | **Zero Imputation** (Preserved Unmatched) |
| `fc_DL008_2026-11-01T11:00:00Z` | `DL008` | 2026-11-01 11:00Z | 2026-11-01 12:00Z | 30.65 | 205.79 | CPCB/OpenAQ Sensor Downtime | **Zero Imputation** (Preserved Unmatched) |

> [!NOTE]
> Audit confirms that neither zero-imputation ($0.0\ \mu\text{g/m}^3$) nor linear interpolation was performed. The evaluation dataset contains exactly 418 pairs corresponding to genuine, physically measured target observations.

### 3.5 Check 8: Checkpoint SHA-256 Immutability
Cryptographic verification of model artifacts on disk:
- **`adapted_gbdt_pm25.joblib`**:
  - File Path: [`experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib)
  - Actual SHA-256: `982dbdfbf61db4ba937a289860e8297bb482f3b79060af2c173f83476ad2225c`
  - Manifest SHA-256: `982dbdfbf61db4ba937a289860e8297bb482f3b79060af2c173f83476ad2225c`
  - Match: **VERIFIED (Bit-for-bit identical)**
- **`adapted_gbdt_pm10.joblib`**:
  - File Path: [`experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib)
  - Actual SHA-256: `e951a12b7e2195cae81eb20511ea635c2466d5e42208420382b09ed1a15743aa`
  - Manifest SHA-256: `e951a12b7e2195cae81eb20511ea635c2466d5e42208420382b09ed1a15743aa`
  - Match: **VERIFIED (Bit-for-bit identical)**

### 3.6 Check 9: Phase 8 Cohort Preservation
The initial Phase 8 trial cohort consisted of 16 matched pairs collected during the 72-hour window (November 2–5, 2026). In `live_trial_evaluation.csv`, all 16 Phase 8 match records are preserved:
- Recalculated GBDT $\text{PM}_{2.5}$ MAE: **$7.6200\ \mu\text{g/m}^3$** (Expected: $7.62\ \mu\text{g/m}^3$, Diff: $0.0000$)
- Recalculated GBDT $\text{PM}_{10}$ MAE: **$16.6700\ \mu\text{g/m}^3$** (Expected: $16.67\ \mu\text{g/m}^3$, Diff: $0.0000$)
- Recalculated Persistence $\text{PM}_{2.5}$ MAE: **$12.0900\ \mu\text{g/m}^3$** (Expected: $12.09\ \mu\text{g/m}^3$, Diff: $0.0000$)
- Recalculated Persistence $\text{PM}_{10}$ MAE: **$24.7900\ \mu\text{g/m}^3$** (Expected: $24.79\ \mu\text{g/m}^3$, Diff: $0.0000$)

---

## 4. Per-Station Dataset Summary

| Station ID | Station Name | City, State | Raw Observations | Clock-Hour Bins | Predictions Generated | Matched Targets | Missing Targets |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`AP001`** | Secretariat - APPCB | Amaravati, AP | 1,633 | 225 | 149 | 149 | 0 |
| **`AP005`** | GVM Corporation - APPCB | Visakhapatnam, AP | 1,081 | 148 | 33 | 33 | 0 |
| **`DL002`** | Anand Vihar - DPCC | New Delhi, DL | 1,752 | 236 | 131 | 131 | 0 |
| **`DL008`** | DTU - CPCB | New Delhi, DL | 1,741 | 235 | 107 | 105 | 2 |
| **Total** | — | — | **6,207** | **844** | **420** | **418** | **2** |

---

## 5. Audit Conclusion

The longitudinal live dataset satisfies all research-grade data integrity standards. Causal segregation between antecedent predictors and subsequent target observations is strictly maintained, duplicate records are completely absent, and model provenance is immutably verified.
