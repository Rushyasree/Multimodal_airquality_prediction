# VayuDrishti Phase 6 — Shadow Observation Log Schema & Governance

**Document Version:** 1.0  
**Date:** October 2026  
**Format:** JSON Lines (`.jsonl`), UTF-8 encoded  
**Location:** `experiments/recent_data_audit/logs/`

---

## 1. Overview and Architecture

The observational trial uses two append-only structured JSONL log stores:
1. `shadow_forecasts.jsonl`: Records dual forecast predictions (Production vs Shadow) issued at time $T$ for target hour $T+1$.
2. `matched_evaluations.jsonl`: Records the verified paired evaluation when ground truth telemetry arrives at $T+1$ and passes validation.

All records are immutable. Each entry is written on a single line, terminated by `\n`. In-memory deduplication indices prevent duplicate entries from being written.

---

## 2. Issued Forecast Schema: `shadow_forecasts.jsonl`

### 2.1 Complete JSON Example
```json
{
  "record_id": "fc-00b87c7e-8c33-4ea2-9749-d758c0e29471",
  "dedup_key": "c4ca4238a0b923820dcc509a6f75849b",
  "issue_time_utc": "2026-10-27T06:00:00Z",
  "target_time_utc": "2026-10-27T07:00:00Z",
  "station_id": "AP001",
  "data_source": "OpenAQ_v3_CAAQMS",
  "historical_forecast": {
    "pm25": 48.72,
    "pm10": 98.45,
    "unit": "ug/m3",
    "model_version": "LSTM-2Layer Baseline v1.2",
    "model_mode": "severity_gated",
    "gate_triggered": false
  },
  "shadow_forecast": {
    "pm25": 46.10,
    "pm10": 93.80,
    "unit": "ug/m3",
    "model_version": "VayuDrishti Adapted Severity-Gated Ensemble v2.0",
    "model_mode": "adapted_severity_gated",
    "gate_triggered": false,
    "experimental": true
  },
  "execution": {
    "status": "success",
    "latency_ms": 138.4,
    "error": null
  },
  "telemetry_freshness": {
    "last_observed_timestamp_utc": "2026-10-27T05:00:00Z",
    "staleness_minutes": 60,
    "missing_features_count": 0
  }
}
```

### 2.2 Field Definitions & Constraints

| Field Path | Type | Nullable | Description & Constraints |
| :--- | :--- | :---: | :--- |
| `record_id` | string | No | Unique UUID identifying the forecast issuance (`fc-<uuid4>`). |
| `dedup_key` | string | No | MD5 hash of `station_id:issue_time_utc:t+1`. Enforces uniqueness. |
| `issue_time_utc` | string | No | ISO 8601 UTC timestamp of forecast generation. |
| `target_time_utc` | string | No | ISO 8601 UTC timestamp of target hour ($t+1$). |
| `station_id` | string | No | Official CAAQMS station identifier (e.g. `AP001`, `DL001`). |
| `data_source` | string | No | Ingestion source identifier (e.g. `OpenAQ_v3_CAAQMS`). |
| `historical_forecast.pm25` | float | No | Production next-hour PM2.5 forecast ($0.0 \le \text{val} \le 1000.0 \ \mu\text{g/m}^3$). |
| `historical_forecast.pm10` | float | No | Production next-hour PM10 forecast ($0.0 \le \text{val} \le 1500.0 \ \mu\text{g/m}^3$). |
| `historical_forecast.model_version`| string | No | Semantic version of production checkpoint. |
| `historical_forecast.model_mode` | string | No | Production mode (`severity_gated`). |
| `historical_forecast.gate_triggered`| bool | No | Whether severe threshold ($> 250 \ \mu\text{g/m}^3$) engaged. |
| `shadow_forecast.pm25` | float | No | Experimental adapted PM2.5 forecast. |
| `shadow_forecast.pm10` | float | No | Experimental adapted PM10 forecast. |
| `shadow_forecast.model_version` | string | No | Semantic version of adapted model. |
| `shadow_forecast.model_mode` | string | No | Experimental mode (`adapted_severity_gated`). |
| `shadow_forecast.experimental` | bool | No | Strictly `true` to guarantee experimental segregation. |
| `execution.latency_ms` | float | No | Total end-to-end execution latency in milliseconds. |
| `execution.status` | string | No | `"success"` or `"error"`. |
| `telemetry_freshness.staleness_minutes` | int | No | Elapsed minutes between newest sensor input and issue time. |

---

## 3. Matched Ground-Truth Schema: `matched_evaluations.jsonl`

### 3.1 Complete JSON Example
```json
{
  "match_id": "eval-39df16a2-e6e7-4940-a1cf-8302f8149e83",
  "forecast_record_id": "fc-00b87c7e-8c33-4ea2-9749-d758c0e29471",
  "dedup_key": "eval:c4ca4238a0b923820dcc509a6f75849b",
  "matched_at_utc": "2026-10-27T07:08:12Z",
  "station_id": "AP001",
  "target_time_utc": "2026-10-27T07:00:00Z",
  "observation": {
    "pm25": 45.20,
    "pm10": 91.50,
    "unit": "ug/m3",
    "observation_time_utc": "2026-10-27T07:00:00Z",
    "source": "OpenAQ_v3_Live"
  },
  "errors": {
    "historical": {
      "ae_pm25": 3.52,
      "ae_pm10": 6.95,
      "se_pm25": 12.3904,
      "se_pm10": 48.3025,
      "bias_pm25": 3.52,
      "bias_pm10": 6.95
    },
    "shadow": {
      "ae_pm25": 0.90,
      "ae_pm10": 2.30,
      "se_pm25": 0.8100,
      "se_pm10": 5.2900,
      "bias_pm25": 0.90,
      "bias_pm10": 2.30
    },
    "paired_difference": {
      "delta_ae_pm25": -2.62,
      "delta_ae_pm10": -4.65,
      "delta_se_pm25": -11.5804,
      "delta_se_pm10": -43.0125,
      "shadow_won_pm25": true,
      "shadow_won_pm10": true
    }
  },
  "severity_tier": "moderate"
}
```

### 3.2 Key Definitions
- `delta_ae_pm25`: $\text{AE}_{\text{shadow}} - \text{AE}_{\text{historical}}$. Negative values indicate the adapted shadow model had lower error than production.
- `shadow_won_pm25`: Boolean indicator ($\text{AE}_{\text{shadow}} < \text{AE}_{\text{historical}}$).
- `severity_tier`: Categorized by actual PM2.5:
  - `"good"`: $0 \le \text{PM}_{2.5} \le 30$
  - `"satisfactory"`: $31 \le \text{PM}_{2.5} \le 60$
  - `"moderate"`: $61 \le \text{PM}_{2.5} \le 90$
  - `"poor"`: $91 \le \text{PM}_{2.5} \le 120$
  - `"very_poor"`: $121 \le \text{PM}_{2.5} \le 250$
  - `"severe"`: $\text{PM}_{2.5} > 250$

---

## 4. Secret Sanitization & Privacy Policy

The logging engine enforces automatic data scrubbing before any byte touches disk:
1. **API Keys & Headers:** All fields named `api_key`, `token`, `secret`, `authorization`, `password`, or matching regular expressions `(Bearer\s+[A-Za-z0-9_\-\.]+)|([A-Fa-f0-9]{32,64})` are automatically replaced with `[REDACTED]`.
2. **IP Addresses & Client Headers:** Client user-agent strings, internal IP addresses, and session IDs are omitted entirely.
3. **Station Coordinates:** Only official CPCB / CAAQMS station IDs are logged; internal network addresses of local loggers are discarded.

---

## 5. Log Maintenance & Operations

### 5.1 Size Budgeting & Bounding
- Standard daily log volume: 85 stations $\times 24$ hours $\approx 2,040$ records/day.
- Average record size: $\approx 650$ bytes.
- Daily uncompressed disk footprint: $\approx 1.3 \text{ MB/day}$.
- Annual footprint: $< 500 \text{ MB}$.
- **Storage Cap:** The runner monitors log directory size. If total size exceeds 5 GB, logs are compressed into monthly `.jsonl.gz` archives.

### 5.2 Verification Commands
To inspect and validate log integrity from the shell:

```powershell
# 1. Count logged forecasts
Get-Content experiments/recent_data_audit/logs/shadow_forecasts.jsonl | Measure-Object -Line

# 2. Count matched evaluations
Get-Content experiments/recent_data_audit/logs/matched_evaluations.jsonl | Measure-Object -Line

# 3. Verify zero unhandled errors in shadow execution
Get-Content experiments/recent_data_audit/logs/shadow_forecasts.jsonl | Select-String '"status": "error"'

# 4. Check real-time trial statistics via Python
python -c "from src.services.shadow_observation_runner import ShadowObservationRunner; print(ShadowObservationRunner().get_trial_statistics())"
```
