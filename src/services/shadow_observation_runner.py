"""
src/services/shadow_observation_runner.py

Production-grade, opt-in observational trial runner for VayuDrishti Phase 6.
Provides:
1. Controlled opt-in execution of adapted shadow model alongside historical production model.
2. Structured, append-only JSONL logging with deduplication.
3. Strict time-aligned matching of 1-hour-ahead forecasts to ground-truth observations.
4. Paired evaluation computation on verified matched pairs.
5. Strict credential redaction and operational kill-switch safeguards.
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger("VayuDrishti.ShadowRunner")

ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "experiments/recent_data_audit/logs"
CONFIG_FILE = ROOT_DIR / "experiments/recent_data_audit/shadow_trial_config.json"

DEFAULT_FORECASTS_LOG = LOGS_DIR / "shadow_forecasts.jsonl"
DEFAULT_MATCHED_LOG = LOGS_DIR / "matched_evaluations.jsonl"

from src.services.sensor_forecast_service import SensorForecastService


class ShadowObservationRunner:
    """Safe, opt-in observational runner for live shadow forecasting trial."""

    def __init__(
        self,
        forecasts_log_path: Optional[Path] = None,
        matched_log_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        service: Optional[SensorForecastService] = None,
    ):
        self.forecasts_log_path = Path(forecasts_log_path or DEFAULT_FORECASTS_LOG)
        self.matched_log_path = Path(matched_log_path or DEFAULT_MATCHED_LOG)
        self.config_path = Path(config_path or CONFIG_FILE)

        self.forecasts_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.matched_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.service = service if service is not None else SensorForecastService()
        self._known_forecast_keys: set = set()
        self._known_matched_keys: set = set()

        self._load_dedup_index()

    def check_live_readiness(
        self,
        station_id: str,
        buffer: Optional[Any] = None,
        registry: Optional[Any] = None,
        anchor_time_utc: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ) -> Dict[str, Any]:
        """Validates all multi-stage readiness prerequisites before enabling or invoking shadow scoring.
        
        Requires:
        1. Valid station mapping in official catalog.
        2. Complete, model-compatible 24h contiguous input window in rolling buffer.
        3. Sufficient data freshness (within staleness threshold).
        4. Successful dry-run inference on both production and shadow models.
        """
        from src.data.live_rolling_buffer import LiveRollingBufferRegistry

        reg = registry or LiveRollingBufferRegistry()
        if not reg.is_station_recognized(station_id):
            return {
                "is_ready": False,
                "status": "not_ready",
                "reason": "station_mapping_failed",
                "station_id": station_id,
                "detail": f"Station '{station_id}' does not map to a recognized CAAQMS monitoring station in catalog.",
                "trial_enabled": self.is_enabled(),
            }

        if buffer is None:
            buffer = reg.get_or_create_buffer(station_id)

        buf_readiness = buffer.check_readiness(
            anchor_time_utc=anchor_time_utc,
            current_system_time=anchor_time_utc if anchor_time_utc is not None else None,
        )
        if not buf_readiness.get("is_ready", False):
            return {
                "is_ready": False,
                "status": "not_ready",
                "reason": buf_readiness.get("reason", "insufficient_history"),
                "station_id": station_id,
                "detail": buf_readiness.get("detail", "Buffer verification failed"),
                "buffer_status": buf_readiness,
                "trial_enabled": self.is_enabled(),
            }

        df_24h = buf_readiness.get("dataframe_24h")
        if df_24h is None or len(df_24h) != 24:
            return {
                "is_ready": False,
                "status": "not_ready",
                "reason": "insufficient_history",
                "station_id": station_id,
                "detail": "Verified 24-hour sequence dataframe could not be constructed.",
                "trial_enabled": self.is_enabled(),
            }

        # Dry-run inference verification (catches shape/scaling issues safely)
        try:
            dry_run = self.service.get_forecast(
                station_id=station_id,
                df_24h=df_24h,
                model_mode="severity_gated",
                enable_shadow=True,
            )
            inference_ok = (dry_run.get("pm25") is not None and dry_run.get("pm10") is not None)
            shadow_run = dry_run.get("shadow_forecast") or {}
            shadow_ok = (shadow_run.get("pm25") is not None and shadow_run.get("pm10") is not None)
        except Exception as e:
            return {
                "is_ready": False,
                "status": "not_ready",
                "reason": "inference_execution_failed",
                "station_id": station_id,
                "detail": f"Dry-run inference raised an exception: {e}",
                "trial_enabled": self.is_enabled(),
            }

        if not (inference_ok and shadow_ok):
            return {
                "is_ready": False,
                "status": "not_ready",
                "reason": "inference_output_invalid",
                "station_id": station_id,
                "detail": "Model dry-run failed to return valid PM2.5/PM10 forecast values.",
                "trial_enabled": self.is_enabled(),
            }

        return {
            "is_ready": True,
            "status": "ready",
            "reason": None,
            "station_id": station_id,
            "anchor_hour_utc": buf_readiness.get("anchor_hour_utc"),
            "target_hour_utc": buf_readiness.get("target_hour_utc"),
            "staleness_hours": buf_readiness.get("staleness_hours"),
            "trial_enabled": self.is_enabled(),
            "historical_ready": buf_readiness.get("historical_ready", True),
            "adapted_ready": buf_readiness.get("adapted_ready", True),
            "observed_channels": buf_readiness.get("observed_channels", []),
            "unobserved_channels_imputed": buf_readiness.get("unobserved_channels_imputed", []),
            "dataframe_24h": df_24h,
        }

    def record_forecast_from_buffer(
        self,
        station_id: str,
        buffer: Optional[Any] = None,
        registry: Optional[Any] = None,
        anchor_time_utc: Optional[Union[str, datetime, pd.Timestamp]] = None,
        data_source: str = "Live CAAQMS Buffer",
    ) -> Dict[str, Any]:
        """Safely checks buffer readiness before executing or logging a forecast.
        
        If readiness gates fail, returns structured reason without logging or calling forecasting models.
        """
        if not self.is_enabled():
            return {
                "status": "trial_disabled",
                "production_forecast": None,
                "shadow_recorded": False,
            }

        readiness = self.check_live_readiness(
            station_id=station_id,
            buffer=buffer,
            registry=registry,
            anchor_time_utc=anchor_time_utc,
        )

        if not readiness.get("is_ready", False):
            return {
                "status": "not_ready",
                "reason": readiness.get("reason"),
                "detail": readiness.get("detail"),
                "readiness": readiness,
                "shadow_recorded": False,
            }

        df_24h = readiness["dataframe_24h"]
        anchor_hour = readiness.get("anchor_hour_utc")

        return self.record_forecast(
            station_id=station_id,
            df_24h=df_24h,
            issue_time_utc=anchor_hour,
            data_source=data_source,
        )

    def _load_dedup_index(self) -> None:
        """Loads existing deduplication keys from disk to prevent duplicate entries."""
        if self.forecasts_log_path.exists():
            try:
                with open(self.forecasts_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            k = rec.get("dedup_key")
                            if k:
                                self._known_forecast_keys.add(k)
            except Exception as e:
                logger.warning(f"Error indexing forecasts log: {e}")

        if self.matched_log_path.exists():
            try:
                with open(self.matched_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            k = rec.get("dedup_key")
                            if k:
                                self._known_matched_keys.add(k)
            except Exception as e:
                logger.warning(f"Error indexing matched log: {e}")

    def is_enabled(self) -> bool:
        """Returns True only if explicitly enabled via environment variable or local config."""
        env_val = os.environ.get("VAYUDRISHTI_SHADOW_TRIAL_ENABLED", "").lower().strip()
        if env_val in ("true", "1", "yes"):
            return True

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return bool(cfg.get("shadow_trial_enabled", False))
            except Exception:
                pass
        return False

    def enable_trial(self, max_records: int = 10000) -> None:
        """Explicitly enables the shadow trial via local configuration file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "shadow_trial_enabled": True,
            "max_records": max_records,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_version": "Phase6-v1.0",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def disable_trial(self) -> None:
        """Operational kill-switch: immediately disables the shadow observation trial."""
        cfg = {
            "shadow_trial_enabled": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_version": "Phase6-v1.0",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def _sanitize_record(self, d: Any) -> Any:
        """Recursively redacts API keys, tokens, and credentials from log dictionaries."""
        if isinstance(d, dict):
            sanitized = {}
            for k, v in d.items():
                k_lower = str(k).lower()
                if any(sec in k_lower for sec in ["api_key", "token", "secret", "password", "authorization"]):
                    sanitized[k] = "[REDACTED]"
                else:
                    sanitized[k] = self._sanitize_record(v)
            return sanitized
        elif isinstance(d, list):
            return [self._sanitize_record(item) for item in d]
        return d

    def record_forecast(
        self,
        station_id: str,
        df_24h: pd.DataFrame,
        issue_time_utc: Optional[str] = None,
        data_source: str = "CAAQMS Telemetry",
    ) -> Dict[str, Any]:
        """Executes a dual forecast and appends to the shadow forecasts log if trial is enabled.

        Returns:
            Dictionary containing official historical forecast and execution status.
        """
        # 1. Determine issue and target timestamps
        now_dt = datetime.now(timezone.utc)
        if issue_time_utc is None:
            if "Datetime" in df_24h.columns and len(df_24h) > 0:
                last_dt = pd.to_datetime(df_24h["Datetime"].iloc[-1])
                if last_dt.tzinfo is None:
                    last_dt = last_dt.tz_localize("UTC")
                issue_dt = last_dt
            else:
                issue_dt = now_dt
        else:
            issue_dt = pd.to_datetime(issue_time_utc)
            if issue_dt.tzinfo is None:
                issue_dt = issue_dt.tz_localize("UTC")

        target_dt = issue_dt + timedelta(hours=1)
        issue_iso = issue_dt.strftime("%Y-%m-%dT%H:00:00Z")
        target_iso = target_dt.strftime("%Y-%m-%dT%H:00:00Z")

        # Deduplication key
        dedup_key = f"{station_id}_{issue_iso}_t+1"

        # 2. Check if trial is enabled
        trial_active = self.is_enabled()

        # 3. Execute forecast (with shadow enabled only if trial is active)
        t0 = time.perf_counter()
        fc_res = self.service.get_forecast(
            station_id=station_id,
            df_24h=df_24h,
            model_mode="severity_gated",
            enable_shadow=trial_active,
        )
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)

        # 4. If trial is disabled, return production forecast cleanly without logging
        if not trial_active:
            return {
                "status": "trial_disabled",
                "production_forecast": fc_res,
                "shadow_recorded": False,
            }

        # 5. Check deduplication
        if dedup_key in self._known_forecast_keys:
            return {
                "status": "duplicate_skipped",
                "production_forecast": fc_res,
                "shadow_recorded": False,
                "dedup_key": dedup_key,
            }

        # 6. Build structured append-only record
        record_id = str(uuid.uuid4())
        shadow_payload = fc_res.get("shadow_forecast") or {}

        record = {
            "record_id": record_id,
            "dedup_key": dedup_key,
            "station_id": station_id,
            "issue_time_utc": issue_iso,
            "target_time_utc": target_iso,
            "forecast_horizon": "t+1 (next 1 hour)",
            "historical_forecast": {
                "pm25": fc_res.get("pm25"),
                "pm10": fc_res.get("pm10"),
                "model_version": fc_res.get("model_version"),
                "metrics": fc_res.get("metrics"),
                "unit": fc_res.get("unit", "ug/m3"),
            },
            "shadow_forecast": {
                "pm25": shadow_payload.get("pm25"),
                "pm10": shadow_payload.get("pm10"),
                "model_version": shadow_payload.get("model_version"),
                "components": shadow_payload.get("components"),
                "experimental": shadow_payload.get("experimental", True),
                "unit": shadow_payload.get("unit", "ug/m3"),
            },
            "telemetry_metadata": {
                "data_source": data_source,
                "input_freshness": fc_res.get("input_freshness"),
                "feature_schema_version": "v2-49feat",
                "trial_version": "Phase6-v1.0",
            },
            "execution": {
                "status": shadow_payload.get("status", "error" if "error" in shadow_payload else "success"),
                "latency_ms": latency_ms,
                "fallback_used": bool(fc_res.get("fallback_note")),
                "error": shadow_payload.get("error"),
            },
            "status": "pending_observation",
            "logged_at_utc": now_dt.isoformat(),
        }

        # 7. Write append-only record with secret sanitization
        sanitized = self._sanitize_record(record)
        with open(self.forecasts_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized) + "\n")

        self._known_forecast_keys.add(dedup_key)

        return {
            "status": "recorded",
            "record_id": record_id,
            "dedup_key": dedup_key,
            "production_forecast": fc_res,
            "shadow_recorded": True,
        }

    def match_forecast_to_observation(
        self,
        record_id: str,
        actual_pm25: float,
        actual_pm10: float,
        observation_time_utc: str,
        station_id: str,
        observation_source: str = "CPCB Ground Telemetry",
    ) -> Dict[str, Any]:
        """Matches a target hour observation to a recorded forecast and computes paired errors.

        Strict validation rules:
        - Observation hour must match forecast target_time_utc.
        - Station IDs must match.
        - Physical bounds check: 0 <= PM2.5 <= 1000, 0 <= PM10 <= 1500.
        - Duplicate matching rejected.
        """
        # Find the pending forecast record
        forecast_rec = None
        if self.forecasts_log_path.exists():
            with open(self.forecasts_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec.get("record_id") == record_id:
                            forecast_rec = rec
                            break

        if not forecast_rec:
            return {"status": "error", "reason": f"Forecast record {record_id} not found."}

        # 1. Station ID validation
        if forecast_rec["station_id"] != station_id:
            return {
                "status": "rejected",
                "reason": f"Station ID mismatch: forecast station {forecast_rec['station_id']} != observation {station_id}.",
            }

        # 2. Timestamp alignment check
        target_iso = forecast_rec["target_time_utc"]
        obs_dt = pd.to_datetime(observation_time_utc)
        if obs_dt.tzinfo is None:
            obs_dt = obs_dt.tz_localize("UTC")
        obs_iso = obs_dt.strftime("%Y-%m-%dT%H:00:00Z")

        if target_iso != obs_iso:
            return {
                "status": "rejected",
                "reason": f"Temporal misalignment: forecast target {target_iso} != observation {obs_iso}.",
            }

        # 3. Deduplication check
        match_dedup_key = f"match_{record_id}_{obs_iso}"
        if match_dedup_key in self._known_matched_keys:
            return {"status": "duplicate_skipped", "reason": "Observation already matched to this forecast."}

        # 4. Physical plausibility check
        try:
            f_pm25 = float(actual_pm25)
            f_pm10 = float(actual_pm10)
        except (ValueError, TypeError):
            return {"status": "rejected", "reason": "Actual observations contain non-numeric values."}

        if not (0.0 <= f_pm25 <= 1000.0 and 0.0 <= f_pm10 <= 1500.0):
            return {
                "status": "rejected",
                "reason": f"Actual observations ({f_pm25}, {f_pm10}) violate physical plausibility bounds.",
            }

        # 5. Extract predictions
        hist_p25 = float(forecast_rec["historical_forecast"]["pm25"])
        hist_p10 = float(forecast_rec["historical_forecast"]["pm10"])
        shadow_p25 = float(forecast_rec["shadow_forecast"]["pm25"])
        shadow_p10 = float(forecast_rec["shadow_forecast"]["pm10"])

        # 6. Compute paired error metrics
        ae_hist_25 = abs(hist_p25 - f_pm25)
        ae_shadow_25 = abs(shadow_p25 - f_pm25)
        delta_ae_25 = ae_shadow_25 - ae_hist_25  # Negative indicates shadow is better

        ae_hist_10 = abs(hist_p10 - f_pm10)
        ae_shadow_10 = abs(shadow_p10 - f_pm10)
        delta_ae_10 = ae_shadow_10 - ae_hist_10

        se_hist_25 = (hist_p25 - f_pm25) ** 2
        se_shadow_25 = (shadow_p25 - f_pm25) ** 2
        delta_se_25 = se_shadow_25 - se_hist_25

        se_hist_10 = (hist_p10 - f_pm10) ** 2
        se_shadow_10 = (shadow_p10 - f_pm10) ** 2
        delta_se_10 = se_shadow_10 - se_hist_10

        match_id = str(uuid.uuid4())
        matched_rec = {
            "match_id": match_id,
            "dedup_key": match_dedup_key,
            "forecast_record_id": record_id,
            "station_id": station_id,
            "target_time_utc": target_iso,
            "observation_time_utc": obs_iso,
            "actual_pm25": f_pm25,
            "actual_pm10": f_pm10,
            "observation_source": observation_source,
            "historical_pred": {"pm25": hist_p25, "pm10": hist_p10},
            "shadow_pred": {"pm25": shadow_p25, "pm10": shadow_p10},
            "paired_errors": {
                "ae_historical_pm25": round(ae_hist_25, 4),
                "ae_shadow_pm25": round(ae_shadow_25, 4),
                "delta_ae_pm25": round(delta_ae_25, 4),
                "ae_historical_pm10": round(ae_hist_10, 4),
                "ae_shadow_pm10": round(ae_shadow_10, 4),
                "delta_ae_pm10": round(delta_ae_10, 4),
                "se_historical_pm25": round(se_hist_25, 4),
                "se_shadow_pm25": round(se_shadow_25, 4),
                "delta_se_pm25": round(delta_se_25, 4),
                "se_historical_pm10": round(se_hist_10, 4),
                "se_shadow_pm10": round(se_shadow_10, 4),
                "delta_se_pm10": round(delta_se_10, 4),
            },
            "matched_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # 7. Write to matched evaluations log
        with open(self.matched_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._sanitize_record(matched_rec)) + "\n")

        self._known_matched_keys.add(match_dedup_key)

        return {
            "status": "matched",
            "match_id": match_id,
            "delta_ae_pm25": round(delta_ae_25, 4),
            "delta_ae_pm10": round(delta_ae_10, 4),
            "shadow_improved_pm25": delta_ae_25 < 0,
            "shadow_improved_pm10": delta_ae_10 < 0,
        }

    def get_trial_statistics(self) -> Dict[str, Any]:
        """Calculates operational and predictive metrics across matched cases."""
        n_issued = len(self._known_forecast_keys)
        n_matched = len(self._known_matched_keys)
        missing_target_rate = round(1.0 - (n_matched / n_issued), 4) if n_issued > 0 else 0.0

        if not self.matched_log_path.exists() or n_matched == 0:
            return {
                "trial_status": "IMPLEMENTED BUT NOT YET OBSERVED",
                "is_enabled": self.is_enabled(),
                "forecasts_issued": n_issued,
                "forecasts_matched": 0,
                "coverage_rate": 0.0,
                "missing_target_rate": missing_target_rate,
                "metrics": None,
            }

        # Load matched records
        records = []
        with open(self.matched_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return {
                "trial_status": "IMPLEMENTED BUT NOT YET OBSERVED",
                "is_enabled": self.is_enabled(),
                "forecasts_issued": n_issued,
                "forecasts_matched": 0,
                "coverage_rate": 0.0,
                "missing_target_rate": missing_target_rate,
                "metrics": None,
            }

        ae_hist_25 = [r["paired_errors"]["ae_historical_pm25"] for r in records]
        ae_shadow_25 = [r["paired_errors"]["ae_shadow_pm25"] for r in records]
        delta_ae_25 = [r["paired_errors"]["delta_ae_pm25"] for r in records]

        ae_hist_10 = [r["paired_errors"]["ae_historical_pm10"] for r in records]
        ae_shadow_10 = [r["paired_errors"]["ae_shadow_pm10"] for r in records]
        delta_ae_10 = [r["paired_errors"]["delta_ae_pm10"] for r in records]

        se_hist_25 = [r["paired_errors"]["se_historical_pm25"] for r in records]
        se_shadow_25 = [r["paired_errors"]["se_shadow_pm25"] for r in records]

        se_hist_10 = [r["paired_errors"]["se_historical_pm10"] for r in records]
        se_shadow_10 = [r["paired_errors"]["se_shadow_pm10"] for r in records]

        bias_hist_25 = [r["historical_pred"]["pm25"] - r["actual_pm25"] for r in records]
        bias_shadow_25 = [r["shadow_pred"]["pm25"] - r["actual_pm25"] for r in records]

        bias_hist_10 = [r["historical_pred"]["pm10"] - r["actual_pm10"] for r in records]
        bias_shadow_10 = [r["shadow_pred"]["pm10"] - r["actual_pm10"] for r in records]

        metrics = {
            "sample_count": len(records),
            "pm25": {
                "historical_mae": round(float(np.mean(ae_hist_25)), 4),
                "shadow_mae": round(float(np.mean(ae_shadow_25)), 4),
                "delta_mae": round(float(np.mean(delta_ae_25)), 4),
                "historical_rmse": round(float(np.sqrt(np.mean(se_hist_25))), 4),
                "shadow_rmse": round(float(np.sqrt(np.mean(se_shadow_25))), 4),
                "historical_mean_bias": round(float(np.mean(bias_hist_25)), 4),
                "shadow_mean_bias": round(float(np.mean(bias_shadow_25)), 4),
                "win_rate_pct": round(float(np.mean([d < 0 for d in delta_ae_25])) * 100.0, 2),
            },
            "pm10": {
                "historical_mae": round(float(np.mean(ae_hist_10)), 4),
                "shadow_mae": round(float(np.mean(ae_shadow_10)), 4),
                "delta_mae": round(float(np.mean(delta_ae_10)), 4),
                "historical_rmse": round(float(np.sqrt(np.mean(se_hist_10))), 4),
                "shadow_rmse": round(float(np.sqrt(np.mean(se_shadow_10))), 4),
                "historical_mean_bias": round(float(np.mean(bias_hist_10)), 4),
                "shadow_mean_bias": round(float(np.mean(bias_shadow_10)), 4),
                "win_rate_pct": round(float(np.mean([d < 0 for d in delta_ae_10])) * 100.0, 2),
            },
        }

        return {
            "trial_status": "OBSERVATION IN PROGRESS" if len(records) < 500 else "OBSERVATION COHORT READY",
            "is_enabled": self.is_enabled(),
            "forecasts_issued": n_issued,
            "forecasts_matched": len(records),
            "coverage_rate": round(len(records) / n_issued, 4) if n_issued > 0 else 0.0,
            "missing_target_rate": missing_target_rate,
            "metrics": metrics,
        }
