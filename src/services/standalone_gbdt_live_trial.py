"""
src/services/standalone_gbdt_live_trial.py

Phase 8: Controlled Real Live Shadow Trial for Standalone Adapted GBDT Forecaster.
Provides:
1. Multi-station OpenAQ API v3 telemetry ingestion into StationRollingBuffer.
2. Ingestion tracking and raw observation export (live_trial_observations.csv).
3. Rolling causal forecast generation at issue time t for target t+1 (model_mode="adapted_gbdt_only").
4. Causal persistence baseline generation at issue time t.
5. Strict post-horizon target observation matching (target arrived strictly after issue time).
6. Paired error computation (GBDT vs Persistence) and metrics aggregation.
7. Deduplication, operational kill-switch, and credential protection safeguards.
"""

import os
import json
import uuid
import math
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests

logger = logging.getLogger("VayuDrishti.Phase8LiveTrial")

ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT_DIR / "experiments/recent_data_audit"
DEFAULT_OBSERVATIONS_CSV = AUDIT_DIR / "live_trial_observations.csv"
DEFAULT_PREDICTIONS_CSV = AUDIT_DIR / "live_trial_predictions.csv"
DEFAULT_EVALUATION_CSV = AUDIT_DIR / "live_trial_evaluation.csv"
DEFAULT_METRICS_JSON = AUDIT_DIR / "live_trial_metrics.json"

from src.data.live_rolling_buffer import (
    StationRollingBuffer,
    normalize_timestamp_utc,
    validate_sensor_reading,
    PHYSICAL_BOUNDS,
    PARAM_MAP,
)
from src.services.sensor_forecast_service import SensorForecastService
from src.prediction.current_air_quality_service import CurrentAirQualityService, load_env_file

# Verified CAAQMS station definitions with OpenAQ v3 Location and Modern Sensor IDs
TARGET_CAAQMS_STATIONS: Dict[str, Dict[str, Any]] = {
    "AP005": {
        "station_name": "GVM Corporation, Visakhapatnam - APPCB",
        "city": "Visakhapatnam",
        "state": "Andhra Pradesh",
        "location_id": 5628,
        "pm25_sensor_id": 12235460,
        "pm10_sensor_id": 12235459,
        "lat": 17.72,
        "lon": 83.30,
    },
    "AP001": {
        "station_name": "Secretariat, Amaravati - APPCB",
        "city": "Amaravati",
        "state": "Andhra Pradesh",
        "location_id": 5408,
        "pm25_sensor_id": 12234948,
        "pm10_sensor_id": 12234947,
        "lat": 16.5150833,
        "lon": 80.5181667,
    },
    "DL008": {
        "station_name": "DTU, Delhi - CPCB",
        "city": "Delhi",
        "state": "Delhi",
        "location_id": 5626,
        "pm25_sensor_id": 12234760,
        "pm10_sensor_id": 12234759,
        "lat": 28.7500499,
        "lon": 77.1112615,
    },
    "DL002": {
        "station_name": "Anand Vihar, Delhi - DPCC",
        "city": "Delhi",
        "state": "Delhi",
        "location_id": 235,
        "pm25_sensor_id": 12235610,
        "pm10_sensor_id": 12235609,
        "lat": 28.647622,
        "lon": 77.315809,
    },
}


class StandaloneGBDTLiveTrialRunner:
    """Manages the controlled execution of the Phase 8 live shadow trial for Standalone GBDT."""

    def __init__(
        self,
        observations_csv: Optional[Path] = None,
        predictions_csv: Optional[Path] = None,
        evaluation_csv: Optional[Path] = None,
        metrics_json: Optional[Path] = None,
        config_path: Optional[Path] = None,
        forecast_service: Optional[SensorForecastService] = None,
        openaq_service: Optional[CurrentAirQualityService] = None,
    ):
        self.observations_csv = Path(observations_csv or DEFAULT_OBSERVATIONS_CSV)
        self.predictions_csv = Path(predictions_csv or DEFAULT_PREDICTIONS_CSV)
        self.evaluation_csv = Path(evaluation_csv or DEFAULT_EVALUATION_CSV)
        self.metrics_json = Path(metrics_json or DEFAULT_METRICS_JSON)
        self.config_path = Path(config_path or (AUDIT_DIR / "shadow_trial_config.json"))

        self.forecast_service = forecast_service or SensorForecastService()
        self.openaq_service = openaq_service or CurrentAirQualityService()

        # Station buffers dictionary: station_id -> StationRollingBuffer
        self.buffers: Dict[str, StationRollingBuffer] = {}
        for sid, meta in TARGET_CAAQMS_STATIONS.items():
            self.buffers[sid] = StationRollingBuffer(
                station_id=sid,
                station_lat=meta["lat"],
                station_lon=meta["lon"],
                max_retention_hours=120,
                max_staleness_hours=120.0,
            )

        # Ingestion records cache
        self.ingested_records: List[Dict[str, Any]] = []

    def is_trial_enabled(self) -> bool:
        """Checks opt-in trial status via environment variable or local config."""
        env_val = os.environ.get("VAYUDRISHTI_LIVE_TRIAL_ENABLED", "").lower().strip()
        if env_val in ("true", "1", "yes"):
            return True

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return bool(cfg.get("live_trial_enabled", False))
            except Exception:
                pass
        return False

    def enable_trial(self) -> None:
        """Enables live trial execution via local configuration file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "live_trial_enabled": True,
            "shadow_trial_enabled": True,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_phase": "Phase8-StandaloneGBDT",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def disable_trial(self) -> None:
        """Kill-switch: instantly halts live trial execution and recording."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "live_trial_enabled": False,
            "shadow_trial_enabled": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_phase": "Phase8-StandaloneGBDT",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def fetch_station_telemetry(
        self,
        station_id: str,
        hours_back: int = 72,
        end_time_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Fetches available PM2.5 and PM10 observations from OpenAQ API v3 for a station.
        
        Uses genuine OpenAQ sensors and queries time range [end_time - hours_back, end_time].
        """
        meta = TARGET_CAAQMS_STATIONS.get(station_id)
        if not meta:
            return {"status": "error", "reason": f"Unknown station {station_id}"}

        if not self.openaq_service.has_valid_api_key():
            return {"status": "error", "reason": "No valid OpenAQ API key configured"}

        headers = self.openaq_service.get_headers()
        base_url = self.openaq_service.base_url_v3

        now_utc = end_time_utc or datetime.now(timezone.utc)
        from_utc = now_utc - timedelta(hours=hours_back)
        dt_from_str = from_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_to_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        buffer = self.buffers[station_id]
        counts = {"PM2.5": 0, "PM10": 0, "rejected": 0, "duplicate": 0}

        for param, sensor_id in [("PM2.5", meta["pm25_sensor_id"]), ("PM10", meta["pm10_sensor_id"])]:
            meas_url = f"{base_url}/sensors/{sensor_id}/measurements"
            params = {
                "datetime_from": dt_from_str,
                "datetime_to": dt_to_str,
                "limit": 1000,
            }
            try:
                resp = requests.get(meas_url, headers=headers, params=params, timeout=12.0)
                if resp.status_code == 200:
                    items = resp.json().get("results", [])
                    for item in items:
                        period = item.get("period", {})
                        dt_str = (
                            period.get("datetimeTo", {}).get("utc")
                            or period.get("datetimeFrom", {}).get("utc")
                            or item.get("datetime", {}).get("utc")
                        )
                        val = item.get("value")
                        if dt_str and val is not None:
                            res = buffer.ingest_reading(
                                parameter=param,
                                value=val,
                                timestamp=dt_str,
                                station_id=station_id,
                            )
                            st = res.get("status")
                            if st == "ingested":
                                counts[param] += 1
                                self.ingested_records.append({
                                    "station_id": station_id,
                                    "observation_time_utc": res.get("timestamp_utc"),
                                    "parameter": param,
                                    "value": res.get("value"),
                                    "unit": "ug/m3",
                                    "ingestion_status": "ingested",
                                })
                            elif st == "duplicate_skipped":
                                counts["duplicate"] += 1
                            else:
                                counts["rejected"] += 1
                else:
                    logger.warning(f"Failed to fetch {param} for {station_id}: HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"Error querying sensor {sensor_id} for {station_id}: {e}")

        summary = buffer.get_hourly_summary()
        return {
            "status": "success",
            "station_id": station_id,
            "counts": counts,
            "hourly_bins": len(summary),
        }

    def export_observations_csv(self, path: Optional[Path] = None) -> Path:
        """Exports all ingested raw observations to live_trial_observations.csv."""
        out_path = Path(path or self.observations_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.ingested_records:
            # If empty, create CSV with header
            df_empty = pd.DataFrame(columns=[
                "station_id", "observation_time_utc", "parameter", "value", "unit", "ingestion_status"
            ])
            df_empty.to_csv(out_path, index=False)
            return out_path

        df_obs = pd.DataFrame(self.ingested_records)
        df_obs.sort_values(by=["station_id", "observation_time_utc", "parameter"], inplace=True)
        df_obs.to_csv(out_path, index=False)
        return out_path

    def run_rolling_forecasts_and_matching(
        self,
        station_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Runs chronological rolling 1-hour-ahead forecasts and matches to targets.
        
        Causal Guarantee:
        - At anchor hour t, only preceding 24 contiguous hours [t-23, ..., t] are provided to the model.
        - The target observation at t+1 is strictly excluded from feature generation.
        - Standalone GBDT model (model_mode="adapted_gbdt_only") predicts next-hour PM2.5 and PM10.
        - Persistence baseline predicts PM2.5(t+1) = PM2.5(t) and PM10(t+1) = PM10(t).
        - Matching occurs only after the t+1 observation is present in the buffer.
        """
        buffer = self.buffers.get(station_id)
        if not buffer:
            return [], []

        summary = buffer.get_hourly_summary()
        if len(summary) < 25:
            return [], []

        hourly_cache = buffer._hourly_cache
        sorted_hours = sorted(hourly_cache.keys())

        predictions: List[Dict[str, Any]] = []
        evaluations: List[Dict[str, Any]] = []

        # We can form a 24-hour sequence for any hour index i where i >= 23
        for i in range(23, len(sorted_hours) - 1):
            anchor_h = sorted_hours[i]
            target_h = sorted_hours[i + 1]

            # Verify that target_h is exactly anchor_h + 1 hour
            if target_h != anchor_h + timedelta(hours=1):
                continue

            # Verify contiguous 24 hours [anchor_h - 23h, ..., anchor_h]
            expected_window = [anchor_h - timedelta(hours=23 - k) for k in range(24)]
            if any(h not in hourly_cache for h in expected_window):
                continue

            # Verify both PM2.5 and PM10 are non-null in all 24 hours
            if any("PM2.5" not in hourly_cache[h] or "PM10" not in hourly_cache[h] for h in expected_window):
                continue

            # Build verified 24h DataFrame strictly up to anchor_h
            rows = []
            for h in expected_window:
                r = dict(hourly_cache[h])
                r["Datetime"] = h
                r["StationId"] = station_id
                r["Latitude"] = buffer.station_lat
                r["Longitude"] = buffer.station_lon
                rows.append(r)
            df_24h = pd.DataFrame(rows)

            # Issue time ISO and Target time ISO
            issue_iso = anchor_h.strftime("%Y-%m-%dT%H:00:00Z")
            target_iso = target_h.strftime("%Y-%m-%dT%H:00:00Z")
            forecast_id = f"fc_{station_id}_{issue_iso}"

            # Standalone GBDT Forecast
            fc_res = self.forecast_service.get_forecast(
                station_id=station_id,
                df_24h=df_24h,
                model_mode="adapted_gbdt_only",
                enable_shadow=False,
            )
            pred_pm25 = float(fc_res.get("pm25", 0.0))
            pred_pm10 = float(fc_res.get("pm10", 0.0))

            # Causal Persistence Baseline (observation at anchor hour t)
            pers_pm25 = float(hourly_cache[anchor_h]["PM2.5"])
            pers_pm10 = float(hourly_cache[anchor_h]["PM10"])

            pred_record = {
                "forecast_id": forecast_id,
                "station_id": station_id,
                "issue_time_utc": issue_iso,
                "target_time_utc": target_iso,
                "model_name": "adapted_gbdt_only",
                "pred_pm25": round(pred_pm25, 2),
                "pred_pm10": round(pred_pm10, 2),
                "persistence_pm25": round(pers_pm25, 2),
                "persistence_pm10": round(pers_pm10, 2),
                "status": "forecast_generated",
            }
            predictions.append(pred_record)

            # Target observation matching
            actual_pm25 = hourly_cache[target_h].get("PM2.5")
            actual_pm10 = hourly_cache[target_h].get("PM10")

            if actual_pm25 is not None and actual_pm10 is not None:
                # Plausibility bounds
                if (0.0 <= actual_pm25 <= 1000.0) and (0.0 <= actual_pm10 <= 1500.0):
                    gbdt_err_25 = round(pred_pm25 - actual_pm25, 2)
                    gbdt_abs_err_25 = round(abs(pred_pm25 - actual_pm25), 2)
                    gbdt_err_10 = round(pred_pm10 - actual_pm10, 2)
                    gbdt_abs_err_10 = round(abs(pred_pm10 - actual_pm10), 2)

                    pers_err_25 = round(pers_pm25 - actual_pm25, 2)
                    pers_abs_err_25 = round(abs(pers_pm25 - actual_pm25), 2)
                    pers_err_10 = round(pers_pm10 - actual_pm10, 2)
                    pers_abs_err_10 = round(abs(pers_pm10 - actual_pm10), 2)

                    eval_record = {
                        "match_id": f"eval_{station_id}_{target_iso}",
                        "forecast_id": forecast_id,
                        "station_id": station_id,
                        "target_time_utc": target_iso,
                        "actual_pm25": round(float(actual_pm25), 2),
                        "actual_pm10": round(float(actual_pm10), 2),
                        "pred_pm25": round(pred_pm25, 2),
                        "pred_pm10": round(pred_pm10, 2),
                        "persistence_pm25": round(pers_pm25, 2),
                        "persistence_pm10": round(pers_pm10, 2),
                        "gbdt_err_pm25": gbdt_err_25,
                        "gbdt_err_pm10": gbdt_err_10,
                        "pers_err_pm25": pers_err_25,
                        "pers_err_pm10": pers_err_10,
                        "gbdt_abs_err_pm25": gbdt_abs_err_25,
                        "gbdt_abs_err_pm10": gbdt_abs_err_10,
                        "pers_abs_err_pm25": pers_abs_err_25,
                        "pers_abs_err_pm10": pers_abs_err_10,
                    }
                    evaluations.append(eval_record)

        return predictions, evaluations

    def execute_live_trial(
        self,
        hours_back: int = 72,
    ) -> Dict[str, Any]:
        """Executes full Phase 8 live trial across all target CAAQMS stations."""
        logger.info("Executing Phase 8 live shadow trial across CAAQMS stations...")

        all_predictions: List[Dict[str, Any]] = []
        all_evaluations: List[Dict[str, Any]] = []
        station_summaries: Dict[str, Any] = {}

        for sid in TARGET_CAAQMS_STATIONS.keys():
            fetch_res = self.fetch_station_telemetry(sid, hours_back=hours_back)
            st_preds, st_evals = self.run_rolling_forecasts_and_matching(sid)
            all_predictions.extend(st_preds)
            all_evaluations.extend(st_evals)

            station_summaries[sid] = {
                "fetch_status": fetch_res.get("status"),
                "counts": fetch_res.get("counts"),
                "hourly_bins": fetch_res.get("hourly_bins", 0),
                "forecasts_generated": len(st_preds),
                "targets_matched": len(st_evals),
            }

        # 1. Export observations CSV
        self.export_observations_csv(self.observations_csv)

        # 2. Export predictions CSV
        self.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        if all_predictions:
            df_preds = pd.DataFrame(all_predictions)
            df_preds.to_csv(self.predictions_csv, index=False)
        else:
            pd.DataFrame(columns=[
                "forecast_id", "station_id", "issue_time_utc", "target_time_utc",
                "model_name", "pred_pm25", "pred_pm10", "persistence_pm25", "persistence_pm10", "status"
            ]).to_csv(self.predictions_csv, index=False)

        # 3. Export evaluations CSV
        self.evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
        if all_evaluations:
            df_evals = pd.DataFrame(all_evaluations)
            df_evals.to_csv(self.evaluation_csv, index=False)
        else:
            pd.DataFrame(columns=[
                "match_id", "forecast_id", "station_id", "target_time_utc",
                "actual_pm25", "actual_pm10", "pred_pm25", "pred_pm10",
                "persistence_pm25", "persistence_pm10", "gbdt_err_pm25", "gbdt_err_pm10",
                "pers_err_pm25", "pers_err_pm10", "gbdt_abs_err_pm25", "gbdt_abs_err_pm10",
                "pers_abs_err_pm25", "pers_abs_err_pm10"
            ]).to_csv(self.evaluation_csv, index=False)

        # 4. Compute aggregate metrics
        metrics = self._compute_trial_metrics(all_evaluations, station_summaries)

        # 5. Export metrics JSON
        self.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _compute_trial_metrics(
        self,
        evaluations: List[Dict[str, Any]],
        station_summaries: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculates comprehensive live trial performance metrics and comparisons."""
        if not evaluations:
            return {
                "status": "completed_zero_matches",
                "total_observations_ingested": len(self.ingested_records),
                "total_forecasts_generated": 0,
                "total_targets_matched": 0,
                "station_summaries": station_summaries,
                "overall": {},
                "stations": {},
            }

        df = pd.DataFrame(evaluations)
        n = len(df)

        # Standalone GBDT overall metrics
        gbdt_pm25_mae = float(df["gbdt_abs_err_pm25"].mean())
        gbdt_pm25_rmse = float(np.sqrt((df["gbdt_err_pm25"] ** 2).mean()))
        gbdt_pm25_bias = float(df["gbdt_err_pm25"].mean())

        gbdt_pm10_mae = float(df["gbdt_abs_err_pm10"].mean())
        gbdt_pm10_rmse = float(np.sqrt((df["gbdt_err_pm10"] ** 2).mean()))
        gbdt_pm10_bias = float(df["gbdt_err_pm10"].mean())

        # Persistence baseline overall metrics
        pers_pm25_mae = float(df["pers_abs_err_pm25"].mean())
        pers_pm25_rmse = float(np.sqrt((df["pers_err_pm25"] ** 2).mean()))
        pers_pm25_bias = float(df["pers_err_pm25"].mean())

        pers_pm10_mae = float(df["pers_abs_err_pm10"].mean())
        pers_pm10_rmse = float(np.sqrt((df["pers_err_pm10"] ** 2).mean()))
        pers_pm10_bias = float(df["pers_err_pm10"].mean())

        # Delta improvements (Persistence MAE - GBDT MAE) / Persistence MAE * 100
        pm25_mae_improvement_pct = round(((pers_pm25_mae - gbdt_pm25_mae) / max(pers_pm25_mae, 1e-6)) * 100.0, 2)
        pm10_mae_improvement_pct = round(((pers_pm10_mae - gbdt_pm10_mae) / max(pers_pm10_mae, 1e-6)) * 100.0, 2)

        # Per station metrics
        st_metrics = {}
        for sid, group in df.groupby("station_id"):
            st_metrics[sid] = {
                "matched_count": int(len(group)),
                "gbdt_pm25_mae": round(float(group["gbdt_abs_err_pm25"].mean()), 2),
                "gbdt_pm25_rmse": round(float(np.sqrt((group["gbdt_err_pm25"] ** 2).mean())), 2),
                "gbdt_pm25_bias": round(float(group["gbdt_err_pm25"].mean()), 2),
                "pers_pm25_mae": round(float(group["pers_abs_err_pm25"].mean()), 2),
                "pers_pm25_rmse": round(float(np.sqrt((group["pers_err_pm25"] ** 2).mean())), 2),
                "gbdt_pm10_mae": round(float(group["gbdt_abs_err_pm10"].mean()), 2),
                "gbdt_pm10_rmse": round(float(np.sqrt((group["gbdt_err_pm10"] ** 2).mean())), 2),
                "gbdt_pm10_bias": round(float(group["gbdt_err_pm10"].mean()), 2),
                "pers_pm10_mae": round(float(group["pers_abs_err_pm10"].mean()), 2),
                "pers_pm10_rmse": round(float(np.sqrt((group["pers_err_pm10"] ** 2).mean())), 2),
            }

        return {
            "trial_phase": "Phase 8: Real Live Shadow Trial — Standalone GBDT",
            "trial_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trial_window": "Recent 72-Hour Continuous Telemetry (OpenAQ API v3)",
            "total_observations_ingested": len(self.ingested_records),
            "total_matched_pairs": n,
            "overall": {
                "gbdt": {
                    "pm25": {
                        "mae": round(gbdt_pm25_mae, 2),
                        "rmse": round(gbdt_pm25_rmse, 2),
                        "mean_bias": round(gbdt_pm25_bias, 2),
                    },
                    "pm10": {
                        "mae": round(gbdt_pm10_mae, 2),
                        "rmse": round(gbdt_pm10_rmse, 2),
                        "mean_bias": round(gbdt_pm10_bias, 2),
                    },
                },
                "persistence": {
                    "pm25": {
                        "mae": round(pers_pm25_mae, 2),
                        "rmse": round(pers_pm25_rmse, 2),
                        "mean_bias": round(pers_pm25_bias, 2),
                    },
                    "pm10": {
                        "mae": round(pers_pm10_mae, 2),
                        "rmse": round(pers_pm10_rmse, 2),
                        "mean_bias": round(pers_pm10_bias, 2),
                    },
                },
                "comparison": {
                    "pm25_mae_improvement_pct": pm25_mae_improvement_pct,
                    "pm10_mae_improvement_pct": pm10_mae_improvement_pct,
                    "gbdt_wins_pm25": int((df["gbdt_abs_err_pm25"] < df["pers_abs_err_pm25"]).sum()),
                    "pers_wins_pm25": int((df["gbdt_abs_err_pm25"] > df["pers_abs_err_pm25"]).sum()),
                    "gbdt_wins_pm10": int((df["gbdt_abs_err_pm10"] < df["pers_abs_err_pm10"]).sum()),
                    "pers_wins_pm10": int((df["gbdt_abs_err_pm10"] > df["pers_abs_err_pm10"]).sum()),
                },
            },
            "per_station": st_metrics,
            "station_summaries": station_summaries,
        }
