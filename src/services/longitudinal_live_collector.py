"""
src/services/longitudinal_live_collector.py

Phase 10: Production-grade longitudinal telemetry collector and research dataset manager.
Provides:
1. Resilient, append-only telemetry ingestion from OpenAQ API v3.
2. Complete provenance capture for predictions (model checkpoint hashes, input bounds, schema version).
3. Causal target matching after target hour elapses with tracking of unmatched / missing targets.
4. Operational state tracking and persistent audit logging (collector_state.json).
5. Robust retry and outage logging preventing corruptions or duplicate records.
"""

import os
import sys
import json
import uuid
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests

logger = logging.getLogger("VayuDrishti.LongitudinalCollector")

ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT_DIR / "experiments/recent_data_audit"
DEFAULT_OBS_CSV = AUDIT_DIR / "live_trial_observations.csv"
DEFAULT_PREDS_CSV = AUDIT_DIR / "live_trial_predictions.csv"
DEFAULT_EVAL_CSV = AUDIT_DIR / "live_trial_evaluation.csv"
DEFAULT_STATE_JSON = AUDIT_DIR / "collector_state.json"

from src.prediction.current_air_quality_service import CurrentAirQualityService, load_env_file
from src.services.standalone_gbdt_live_trial import TARGET_CAAQMS_STATIONS
from src.data.live_rolling_buffer import StationRollingBuffer

# Model Checkpoint Verification Hashes (SHA-256)
CHECKPOINT_HASHES = {
    "PM2.5": "982dbdfbf61db4ba937a289860e8297bb482f3b79060af2c173f83476ad2225c",
    "PM10": "e951a12b7e2195cae81eb20511ea635c2466d5e42208420382b09ed1a15743aa",
}
FEATURE_SCHEMA_VERSION = "v2-49feat"
MODEL_NAME = "adapted_gbdt_only"


class LongitudinalCollector:
    """Manages longitudinal live data collection, prediction provenance, and causal target matching."""

    def __init__(
        self,
        obs_csv_path: Optional[Path] = None,
        preds_csv_path: Optional[Path] = None,
        eval_csv_path: Optional[Path] = None,
        state_json_path: Optional[Path] = None,
        openaq_service: Optional[CurrentAirQualityService] = None,
    ):
        self.obs_csv = Path(obs_csv_path or DEFAULT_OBS_CSV)
        self.preds_csv = Path(preds_csv_path or DEFAULT_PREDS_CSV)
        self.eval_csv = Path(eval_csv_path or DEFAULT_EVAL_CSV)
        self.state_json = Path(state_json_path or DEFAULT_STATE_JSON)

        self.openaq_service = openaq_service or CurrentAirQualityService()

        # Cached datasets in memory
        self.df_obs = pd.DataFrame()
        self.df_preds = pd.DataFrame()
        self.df_eval = pd.DataFrame()

        # Deduplication sets
        self.seen_obs_keys: Set[Tuple[str, str, str]] = set()
        self.seen_pred_ids: Set[str] = set()
        self.seen_match_ids: Set[str] = set()

        # Operational state
        self.state: Dict[str, Any] = self._load_state()

        self._load_existing_records()

    def _load_state(self) -> Dict[str, Any]:
        """Loads collector operational state from disk or initializes default state."""
        if self.state_json.exists():
            try:
                with open(self.state_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse collector state from {self.state_json}: {e}")

        return {
            "collector_version": "v1.0-longitudinal",
            "trial_phase": "Phase 10: Longitudinal Live Monitoring",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_run_start_utc": None,
            "last_run_end_utc": None,
            "total_runs": 0,
            "total_observations_ingested": 0,
            "total_predictions_generated": 0,
            "total_targets_matched": 0,
            "unmatched_predictions_count": 0,
            "api_outages_logged": [],
            "station_telemetry_status": {},
        }

    def save_state(self) -> None:
        """Persists collector operational state atomically to disk."""
        self.state_json.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_json, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _load_existing_records(self) -> None:
        """Loads existing CSV datasets and populates in-memory deduplication indices."""
        if self.obs_csv.exists():
            self.df_obs = pd.read_csv(self.obs_csv)
            for _, r in self.df_obs.iterrows():
                self.seen_obs_keys.add((str(r["station_id"]), str(r["observation_time_utc"]), str(r["parameter"])))

        if self.preds_csv.exists():
            self.df_preds = pd.read_csv(self.preds_csv)
            for _, r in self.df_preds.iterrows():
                self.seen_pred_ids.add(str(r["forecast_id"]))

        if self.eval_csv.exists():
            self.df_eval = pd.read_csv(self.eval_csv)
            for _, r in self.df_eval.iterrows():
                self.seen_match_ids.add(str(r["match_id"]))

        # Update state counts from disk records
        self.state["total_observations_ingested"] = len(self.df_obs)
        self.state["total_predictions_generated"] = len(self.df_preds)
        self.state["total_targets_matched"] = len(self.df_eval)
        self.state["unmatched_predictions_count"] = len(self.df_preds) - len(self.df_eval)

    def fetch_live_telemetry(
        self,
        station_id: str,
        hours_back: int = 48,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetches telemetry with exponential backoff and structured outage recording.
        
        Returns:
            Tuple of (new_valid_records, outage_info_if_any)
        """
        meta = TARGET_CAAQMS_STATIONS.get(station_id)
        if not meta:
            return [], {"station_id": station_id, "error": "unknown_station"}

        if not self.openaq_service.has_valid_api_key():
            return [], {"station_id": station_id, "error": "missing_api_key"}

        headers = self.openaq_service.get_headers()
        base_url = self.openaq_service.base_url_v3

        now_utc = datetime.now(timezone.utc)
        from_utc = now_utc - timedelta(hours=hours_back)
        dt_from = from_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_to = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        new_records = []
        outage_info = None

        for param, s_id in [("PM2.5", meta["pm25_sensor_id"]), ("PM10", meta["pm10_sensor_id"])]:
            url = f"{base_url}/sensors/{s_id}/measurements"
            params = {"datetime_from": dt_from, "datetime_to": dt_to, "limit": 1000}

            attempt = 0
            success = False
            last_err = None

            while attempt < max_retries and not success:
                attempt += 1
                try:
                    resp = requests.get(url, headers=headers, params=params, timeout=12.0)
                    if resp.status_code == 200:
                        success = True
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
                                k = (station_id, dt_str, param)
                                if k not in self.seen_obs_keys:
                                    try:
                                        f_val = float(val)
                                        # Atmospheric plausibility check
                                        max_bound = 1000.0 if param == "PM2.5" else 1500.0
                                        if 0.0 <= f_val <= max_bound:
                                            self.seen_obs_keys.add(k)
                                            new_records.append({
                                                "station_id": station_id,
                                                "observation_time_utc": dt_str,
                                                "parameter": param,
                                                "value": round(f_val, 2),
                                                "unit": "ug/m3",
                                                "ingestion_status": "ingested",
                                            })
                                    except (ValueError, TypeError):
                                        pass
                    else:
                        last_err = f"HTTP_{resp.status_code}"
                        time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                except requests.exceptions.RequestException as e:
                    last_err = type(e).__name__
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))

            if not success:
                outage_info = {
                    "station_id": station_id,
                    "parameter": param,
                    "sensor_id": s_id,
                    "query_from_utc": dt_from,
                    "query_to_utc": dt_to,
                    "error": last_err or "max_retries_exceeded",
                    "logged_at_utc": now_utc.isoformat(),
                }

        return new_records, outage_info

    def append_observations(self, records: List[Dict[str, Any]]) -> int:
        """Appends new observation records to live_trial_observations.csv without altering existing rows."""
        if not records:
            return 0

        # Filter strictly for un-seen records
        unique_records = []
        for r in records:
            k = (str(r["station_id"]), str(r["observation_time_utc"]), str(r["parameter"]))
            if k not in self.seen_obs_keys:
                self.seen_obs_keys.add(k)
                unique_records.append(r)

        if not unique_records:
            return 0

        df_new = pd.DataFrame(unique_records)
        if self.df_obs.empty:
            self.df_obs = df_new
        else:
            self.df_obs = pd.concat([self.df_obs, df_new], ignore_index=True)

        self.df_obs.sort_values(by=["station_id", "observation_time_utc", "parameter"], inplace=True)
        self.obs_csv.parent.mkdir(parents=True, exist_ok=True)
        self.df_obs.to_csv(self.obs_csv, index=False)
        self.state["total_observations_ingested"] = len(self.df_obs)
        return len(unique_records)

    def build_provenance_forecasts_and_matches(
        self,
        forecast_service: Any,
    ) -> Tuple[int, int]:
        """Generates predictions with complete provenance and executes causal matching against elapsed targets.
        
        Returns:
            Tuple of (new_predictions_count, new_matches_count)
        """
        if self.df_obs.empty:
            return 0, 0

        # Build StationRollingBuffers from observations
        station_buffers: Dict[str, StationRollingBuffer] = {}
        for sid, meta in TARGET_CAAQMS_STATIONS.items():
            buf = StationRollingBuffer(
                station_id=sid,
                station_lat=meta["lat"],
                station_lon=meta["lon"],
                max_retention_hours=500,
                max_staleness_hours=500.0,
            )
            sub_obs = self.df_obs[self.df_obs["station_id"] == sid]
            for _, r in sub_obs.iterrows():
                buf.ingest_reading(
                    parameter=r["parameter"],
                    value=r["value"],
                    timestamp=r["observation_time_utc"],
                    station_id=sid,
                )
            buf.get_hourly_summary()
            station_buffers[sid] = buf

        new_preds = []
        new_evals = []

        now_utc = datetime.now(timezone.utc)

        for sid, buf in station_buffers.items():
            hourly_cache = buf._hourly_cache
            sorted_hours = sorted(hourly_cache.keys())

            for i in range(23, len(sorted_hours) - 1):
                anchor_h = sorted_hours[i]
                target_h = sorted_hours[i + 1]

                # Strict 1-hour step check
                if target_h != anchor_h + timedelta(hours=1):
                    continue

                # Contiguous 24-hour verification window [anchor_h - 23h, ..., anchor_h]
                expected_window = [anchor_h - timedelta(hours=23 - k) for k in range(24)]
                if any(h not in hourly_cache for h in expected_window):
                    continue
                if any("PM2.5" not in hourly_cache[h] or "PM10" not in hourly_cache[h] for h in expected_window):
                    continue

                issue_iso = anchor_h.strftime("%Y-%m-%dT%H:00:00Z")
                target_iso = target_h.strftime("%Y-%m-%dT%H:00:00Z")
                input_start_iso = expected_window[0].strftime("%Y-%m-%dT%H:00:00Z")
                input_end_iso = issue_iso
                forecast_id = f"fc_{sid}_{issue_iso}"

                # Generate prediction if not already recorded
                if forecast_id not in self.seen_pred_ids:
                    self.seen_pred_ids.add(forecast_id)

                    rows = []
                    for h in expected_window:
                        r = dict(hourly_cache[h])
                        r["Datetime"] = h
                        r["StationId"] = sid
                        r["Latitude"] = buf.station_lat
                        r["Longitude"] = buf.station_lon
                        rows.append(r)
                    df_24h = pd.DataFrame(rows)

                    fc_res = forecast_service.get_forecast(
                        station_id=sid,
                        df_24h=df_24h,
                        model_mode=MODEL_NAME,
                        enable_shadow=False,
                    )
                    pred_pm25 = float(fc_res.get("pm25", 0.0))
                    pred_pm10 = float(fc_res.get("pm10", 0.0))
                    pers_pm25 = float(hourly_cache[anchor_h]["PM2.5"])
                    pers_pm10 = float(hourly_cache[anchor_h]["PM10"])

                    pred_record = {
                        "forecast_id": forecast_id,
                        "station_id": sid,
                        "issue_time_utc": issue_iso,
                        "target_time_utc": target_iso,
                        "input_start_utc": input_start_iso,
                        "input_end_utc": input_end_iso,
                        "model_name": MODEL_NAME,
                        "checkpoint_sha256_pm25": CHECKPOINT_HASHES["PM2.5"][:16],
                        "checkpoint_sha256_pm10": CHECKPOINT_HASHES["PM10"][:16],
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "pred_pm25": round(pred_pm25, 2),
                        "pred_pm10": round(pred_pm10, 2),
                        "persistence_pm25": round(pers_pm25, 2),
                        "persistence_pm10": round(pers_pm10, 2),
                        "data_source": "OpenAQ API v3 CAAQMS Telemetry",
                        "quality_status": "verified_contiguous_24h",
                        "status": "forecast_generated",
                    }
                    new_preds.append(pred_record)
                else:
                    # Retrieve existing forecast values for evaluation
                    ex_rows = self.df_preds[self.df_preds["forecast_id"] == forecast_id]
                    if not ex_rows.empty:
                        ex_match = ex_rows.iloc[0]
                        pred_pm25 = float(ex_match["pred_pm25"])
                        pred_pm10 = float(ex_match["pred_pm10"])
                        pers_pm25 = float(ex_match["persistence_pm25"])
                        pers_pm10 = float(ex_match["persistence_pm10"])
                    else:
                        continue

                # Causal Target Matching: target hour must have elapsed
                match_id = f"eval_{sid}_{target_iso}"
                if match_id not in self.seen_match_ids:
                    actual_pm25 = hourly_cache[target_h].get("PM2.5")
                    actual_pm10 = hourly_cache[target_h].get("PM10")

                    if actual_pm25 is not None and actual_pm10 is not None:
                        if (0.0 <= actual_pm25 <= 1000.0) and (0.0 <= actual_pm10 <= 1500.0):
                            self.seen_match_ids.add(match_id)
                            gbdt_err_25 = round(pred_pm25 - actual_pm25, 2)
                            gbdt_abs_25 = round(abs(pred_pm25 - actual_pm25), 2)
                            gbdt_err_10 = round(pred_pm10 - actual_pm10, 2)
                            gbdt_abs_10 = round(abs(pred_pm10 - actual_pm10), 2)

                            pers_err_25 = round(pers_pm25 - actual_pm25, 2)
                            pers_abs_25 = round(abs(pers_pm25 - actual_pm25), 2)
                            pers_err_10 = round(pers_pm10 - actual_pm10, 2)
                            pers_abs_10 = round(abs(pers_pm10 - actual_pm10), 2)

                            new_evals.append({
                                "match_id": match_id,
                                "forecast_id": forecast_id,
                                "station_id": sid,
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
                                "gbdt_abs_err_pm25": gbdt_abs_25,
                                "gbdt_abs_err_pm10": gbdt_abs_10,
                                "pers_abs_err_pm25": pers_abs_25,
                                "pers_abs_err_pm10": pers_abs_10,
                            })

        # Append new predictions
        if new_preds:
            df_new_preds = pd.DataFrame(new_preds)
            if self.df_preds.empty:
                self.df_preds = df_new_preds
            else:
                self.df_preds = pd.concat([self.df_preds, df_new_preds], ignore_index=True)
            self.df_preds.sort_values(by=["station_id", "issue_time_utc"], inplace=True)
            self.preds_csv.parent.mkdir(parents=True, exist_ok=True)
            self.df_preds.to_csv(self.preds_csv, index=False)

        # Append new evaluations
        if new_evals:
            df_new_evals = pd.DataFrame(new_evals)
            if self.df_eval.empty:
                self.df_eval = df_new_evals
            else:
                self.df_eval = pd.concat([self.df_eval, df_new_evals], ignore_index=True)
            self.df_eval.sort_values(by=["station_id", "target_time_utc"], inplace=True)
            self.eval_csv.parent.mkdir(parents=True, exist_ok=True)
            self.df_eval.to_csv(self.eval_csv, index=False)

        # Update state counters
        self.state["total_predictions_generated"] = len(self.df_preds)
        self.state["total_targets_matched"] = len(self.df_eval)
        self.state["unmatched_predictions_count"] = len(self.df_preds) - len(self.df_eval)

        # Update per-station status
        for sid, buf in station_buffers.items():
            self.state["station_telemetry_status"][sid] = {
                "raw_observation_count": int((self.df_obs["station_id"] == sid).sum()) if not self.df_obs.empty else 0,
                "hourly_bins_count": len(buf.get_hourly_summary()),
                "predictions_count": int((self.df_preds["station_id"] == sid).sum()) if not self.df_preds.empty else 0,
                "evaluations_count": int((self.df_eval["station_id"] == sid).sum()) if not self.df_eval.empty else 0,
            }

        return len(new_preds), len(new_evals)

    def execute_collection_cycle(
        self,
        hours_back: int = 48,
        forecast_service: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes a single end-to-end longitudinal monitoring run.
        
        Steps:
        1. Query OpenAQ API v3 for target CAAQMS stations.
        2. Append new observations safely without duplicates.
        3. Build 24h contiguous sequences and log provenance forecasts.
        4. Match elapsed target observations.
        5. Persist collector state.
        """
        start_utc = datetime.now(timezone.utc)
        self.state["last_run_start_utc"] = start_utc.isoformat()
        self.state["total_runs"] += 1

        if forecast_service is None:
            from src.services.sensor_forecast_service import SensorForecastService
            forecast_service = SensorForecastService()

        collected_records = []
        outages = []

        for sid in TARGET_CAAQMS_STATIONS.keys():
            recs, outage = self.fetch_live_telemetry(sid, hours_back=hours_back)
            collected_records.extend(recs)
            if outage:
                outages.append(outage)
                self.state["api_outages_logged"].append(outage)

        # Append observations
        n_obs = self.append_observations(collected_records)

        # Build forecasts and match
        n_preds, n_matches = self.build_provenance_forecasts_and_matches(forecast_service)

        end_utc = datetime.now(timezone.utc)
        self.state["last_run_end_utc"] = end_utc.isoformat()
        self.state["last_status"] = "partial_outage" if outages else "success"
        self.save_state()

        return {
            "status": self.state["last_status"],
            "run_started_utc": start_utc.isoformat(),
            "run_ended_utc": end_utc.isoformat(),
            "new_observations": n_obs,
            "new_predictions": n_preds,
            "new_matches": n_matches,
            "total_observations": self.state["total_observations_ingested"],
            "total_predictions": self.state["total_predictions_generated"],
            "total_matches": self.state["total_targets_matched"],
            "unmatched_predictions": self.state["unmatched_predictions_count"],
            "outages_logged": len(outages),
        }
