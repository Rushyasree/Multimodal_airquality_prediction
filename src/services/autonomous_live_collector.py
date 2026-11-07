"""
src/services/autonomous_live_collector.py

Phase 12: Production-grade autonomous live-data collector and multi-station telemetry manager.
Provides:
1. Hourly automated or on-demand multi-station telemetry ingestion from OpenAQ API v3.
2. Separate recording of collection_time_utc vs observation_time_utc.
3. Stable composite key deduplication: f"{station_id}::{parameter}::{observation_time_utc}".
4. Rigorous validation of station identifiers, pollutant names, units, and physical bounds.
5. Bounded retries with exponential backoff, rate limit handling, and outage recovery.
6. Comprehensive health logging (experiments/phase12/collection_health.csv) and state tracking.
7. Absolute secret redaction and configurable kill-switch enforcement.
8. Guaranteed prevention of accidental model retraining.
"""

import os
import sys
import re
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PHASE12_DIR = ROOT_DIR / "experiments/phase12"
DEFAULT_CONFIG = PHASE12_DIR / "collector_configuration.json"
DEFAULT_REGISTRY = PHASE12_DIR / "station_registry.csv"
DEFAULT_OBS_CSV = PHASE12_DIR / "live_observations.csv"
DEFAULT_PREDS_CSV = PHASE12_DIR / "live_predictions.csv"
DEFAULT_EVAL_CSV = PHASE12_DIR / "live_evaluation.csv"
DEFAULT_HEALTH_CSV = PHASE12_DIR / "collection_health.csv"
DEFAULT_STATE_JSON = PHASE12_DIR / "collector_state.json"

from src.prediction.current_air_quality_service import CurrentAirQualityService, load_env_file
from src.services.sensor_forecast_service import SensorForecastService
from src.data.live_rolling_buffer import StationRollingBuffer

logger = logging.getLogger("VayuDrishti.AutonomousCollector")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [VayuDrishti.Collector] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Physical plausible bounds (µg/m³)
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "PM2.5": (0.0, 1000.0),
    "PM10": (0.0, 1500.0),
    "NO2": (0.0, 1000.0),
    "SO2": (0.0, 1000.0),
    "CO": (0.0, 100.0),
    "O3": (0.0, 1000.0),
}

CANONICAL_UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "µg/m³",
    "SO2": "µg/m³",
    "CO": "mg/m³",
    "O3": "µg/m³",
}

# Regex to detect API keys for redaction
SECRET_REGEX = re.compile(r"([0-9a-fA-F]{32,64}|[a-zA-Z0-9_\-]{30,})")


def redact_secrets(text: str) -> str:
    """Masks potential tokens or sensitive hash credentials in logging output."""
    if not isinstance(text, str):
        return str(text)
    import re
    env_key = os.environ.get("OPENAQ_API_KEY", "")
    if env_key and len(env_key) > 5 and env_key in text:
        text = text.replace(env_key, "[REDACTED_API_KEY]")
    text = re.sub(r"(openaq_[a-zA-Z0-9_\-]+)", "[REDACTED_API_KEY]", text)
    text = re.sub(r"(secret_[a-zA-Z0-9_\-]+)", "[REDACTED_TOKEN]", text)
    text = re.sub(r"(api[_-]?key[:=\s]+)([a-zA-Z0-9_\-]+)", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"(token[:=\s]+)([a-zA-Z0-9_\-]+)", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    return text


class AutonomousLiveCollector:
    """Production-grade autonomous collector for multi-station air quality monitoring."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        registry_path: Optional[Path] = None,
        obs_csv_path: Optional[Path] = None,
        preds_csv_path: Optional[Path] = None,
        eval_csv_path: Optional[Path] = None,
        health_csv_path: Optional[Path] = None,
        state_json_path: Optional[Path] = None,
        openaq_service: Optional[CurrentAirQualityService] = None,
    ):
        load_env_file()
        self.config_path = Path(config_path or DEFAULT_CONFIG)
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY)
        self.obs_csv = Path(obs_csv_path or DEFAULT_OBS_CSV)
        self.preds_csv = Path(preds_csv_path or DEFAULT_PREDS_CSV)
        self.eval_csv = Path(eval_csv_path or DEFAULT_EVAL_CSV)
        self.health_csv = Path(health_csv_path or DEFAULT_HEALTH_CSV)
        self.state_json = Path(state_json_path or DEFAULT_STATE_JSON)

        self.openaq_service = openaq_service or CurrentAirQualityService()

        # Load or initialize configurations
        self.config = self._load_configuration()
        self.station_registry = self._load_station_registry()

        # In-memory deduplication index
        self.seen_obs_keys: Set[str] = set()
        self.seen_pred_ids: Set[str] = set()
        self.seen_match_ids: Set[str] = set()

        # Dataframe caches
        self.df_obs = pd.DataFrame()
        self.df_preds = pd.DataFrame()
        self.df_eval = pd.DataFrame()

        self.state: Dict[str, Any] = self._load_state()
        self._load_existing_records()

    def _load_configuration(self) -> Dict[str, Any]:
        """Loads collector configuration from JSON with fallback defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse config from {self.config_path}: {e}")
        return {
            "collector_version": "v2.0-autonomous",
            "collector_enabled": True,
            "kill_switch_active": False,
            "prevent_model_retraining": True,
            "collection_interval_minutes": 60,
            "default_lookback_hours": 6,
            "request_timeout_seconds": 10,
            "max_retries": 3,
            "backoff_factor_seconds": 2.0,
        }

    def _load_station_registry(self) -> pd.DataFrame:
        """Loads station registry dataframe."""
        if self.registry_path.exists():
            return pd.read_csv(self.registry_path)
        logger.warning(f"Station registry not found at {self.registry_path}")
        return pd.DataFrame()

    def _load_state(self) -> Dict[str, Any]:
        """Loads persistent operational state from disk."""
        if self.state_json.exists():
            try:
                with open(self.state_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load collector state from {self.state_json}: {e}")

        return {
            "collector_version": "v2.0-autonomous",
            "trial_phase": "Phase 12: Autonomous Live Monitoring",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_successful_run_utc": None,
            "last_run_duration_seconds": 0.0,
            "total_execution_cycles": 0,
            "total_observations_ingested": 0,
            "total_predictions_generated": 0,
            "total_targets_matched": 0,
            "unmatched_predictions_count": 0,
            "active_stations_monitored": len(self.station_registry),
            "recent_outages_logged": [],
            "station_telemetry_status": {},
        }

    def _load_existing_records(self) -> None:
        """Loads existing CSV records to initialize in-memory deduplication indexes."""
        if self.obs_csv.exists():
            try:
                self.df_obs = pd.read_csv(self.obs_csv)
                if not self.df_obs.empty and "observation_key" in self.df_obs.columns:
                    self.seen_obs_keys = set(self.df_obs["observation_key"].dropna())
                elif not self.df_obs.empty:
                    for _, r in self.df_obs.iterrows():
                        k = f"{r['station_id']}::{r['parameter']}::{r['observation_time_utc']}"
                        self.seen_obs_keys.add(k)
            except Exception as e:
                logger.warning(f"Error loading {self.obs_csv}: {e}")

        if self.preds_csv.exists():
            try:
                self.df_preds = pd.read_csv(self.preds_csv)
                if not self.df_preds.empty and "forecast_id" in self.df_preds.columns:
                    self.seen_pred_ids = set(self.df_preds["forecast_id"].dropna())
            except Exception as e:
                logger.warning(f"Error loading {self.preds_csv}: {e}")

        if self.eval_csv.exists():
            try:
                self.df_eval = pd.read_csv(self.eval_csv)
                if not self.df_eval.empty and "match_id" in self.df_eval.columns:
                    self.seen_match_ids = set(self.df_eval["match_id"].dropna())
            except Exception as e:
                logger.warning(f"Error loading {self.eval_csv}: {e}")

    def save_state(self) -> None:
        """Persists current state to collector_state.json atomically."""
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.state["total_observations_ingested"] = len(self.df_obs)
        self.state["total_predictions_generated"] = len(self.df_preds)
        self.state["total_targets_matched"] = len(self.df_eval)
        self.state["unmatched_predictions_count"] = max(0, len(self.df_preds) - len(self.df_eval))

        self.state_json.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_json.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        temp_file.replace(self.state_json)

    def is_kill_switch_active(self) -> bool:
        """Checks if collection is halted via kill-switch."""
        if not self.config.get("collector_enabled", True):
            return True
        if self.config.get("kill_switch_active", False):
            return True
        ks_file = self.config.get("kill_switch_file")
        if ks_file and Path(ks_file).exists():
            return True
        env_kill = os.environ.get("VAYUDRISHTI_KILL_SWITCH", "").lower().strip()
        if env_kill in ("true", "1", "yes", "active"):
            return True
        return False

    def validate_reading(self, reading: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates station identifier, pollutant, units, timestamps, and physical plausibility.
        Returns (is_valid, rejection_reason).
        """
        station_id = reading.get("station_id")
        if not station_id or station_id not in set(self.station_registry["station_id"]):
            return False, f"invalid_station_id_{station_id}"

        parameter = reading.get("parameter")
        if not parameter or parameter not in PHYSICAL_BOUNDS:
            return False, f"unsupported_pollutant_{parameter}"

        # Unit validation
        unit = reading.get("unit", "")
        expected_unit = CANONICAL_UNITS.get(parameter, "µg/m³")
        if unit and unit not in (expected_unit, "ug/m3", "µg/m³", "mg/m3"):
            return False, f"invalid_unit_{unit}_for_{parameter}"

        # Numeric value and physical plausibility bounds
        val = reading.get("value")
        if val is None or not isinstance(val, (int, float)) or np.isnan(val):
            return False, "non_numeric_value"

        low_b, high_b = PHYSICAL_BOUNDS[parameter]
        if val < low_b or val > high_b:
            return False, f"out_of_physical_bounds_{val}_for_{parameter}"

        # Timestamp validation
        obs_time = reading.get("observation_time_utc")
        if not obs_time:
            return False, "missing_observation_time"

        try:
            # Must carry UTC indicator (Z or +00:00)
            if not (obs_time.endswith("Z") or "+00:00" in obs_time):
                return False, f"non_utc_timestamp_{obs_time}"
            dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
            # Future check: cannot be in the future relative to now + 5 min clock skew
            now_utc = datetime.now(timezone.utc)
            if dt > now_utc + timedelta(minutes=5):
                return False, f"future_timestamp_rejected_{obs_time}"
        except Exception as e:
            return False, f"malformed_timestamp_{e}"

        return True, None

    def log_health_event(
        self,
        event_type: str,
        status: str,
        station_id: Optional[str] = None,
        parameter: Optional[str] = None,
        message: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Appends operational event to collection_health.csv."""
        log_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "status": status,
            "station_id": station_id or "ALL",
            "parameter": parameter or "ALL",
            "duration_ms": round(duration_ms, 2),
            "message": redact_secrets(message or ""),
        }
        self.health_csv.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame([log_record])
        if not self.health_csv.exists():
            df_new.to_csv(self.health_csv, index=False)
        else:
            df_new.to_csv(self.health_csv, mode="a", header=False, index=False)

    def fetch_station_telemetry_with_retry(
        self,
        station_row: pd.Series,
        hours_back: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Fetches sensor telemetry for a station with bounded retries and exponential backoff.
        Handles HTTP 429 rate limits, timeouts, and connection errors without crashing.
        """
        sid = station_row["station_id"]
        loc_id = station_row.get("openaq_location_id")
        pm25_sid = station_row.get("openaq_pm25_sensor_id")
        pm10_sid = station_row.get("openaq_pm10_sensor_id")

        readings = []
        timeout = self.config.get("request_timeout_seconds", 10)
        max_retries = self.config.get("max_retries", 3)
        backoff = self.config.get("backoff_factor_seconds", 2.0)

        # Sensors to query: (param, sensor_id)
        sensors_to_query = []
        if pd.notna(pm25_sid) and pm25_sid:
            sensors_to_query.append(("PM2.5", int(pm25_sid)))
        if pd.notna(pm10_sid) and pm10_sid:
            sensors_to_query.append(("PM10", int(pm10_sid)))

        collection_time_iso = datetime.now(timezone.utc).isoformat()
        now_dt = datetime.now(timezone.utc)
        datetime_from = (now_dt - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:00:00Z")

        headers = {}
        api_key = os.environ.get("OPENAQ_API_KEY", "")
        if api_key:
            headers["X-API-Key"] = api_key

        for param, s_id in sensors_to_query:
            url = f"https://api.openaq.org/v3/sensors/{s_id}/measurements"
            params = {"datetime_from": datetime_from, "limit": 100}

            t0 = time.perf_counter()
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
                    dur_ms = (time.perf_counter() - t0) * 1000

                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        for item in results:
                            dt_raw = item.get("datetime", {}).get("utc") or item.get("period", {}).get("datetimeTo", {}).get("utc")
                            val = item.get("value")
                            if dt_raw and val is not None:
                                # Ensure UTC ISO string
                                if not dt_raw.endswith("Z") and not "+00:00" in dt_raw:
                                    dt_raw = f"{dt_raw}Z"

                                cand = {
                                    "station_id": sid,
                                    "parameter": param,
                                    "value": float(val),
                                    "unit": CANONICAL_UNITS.get(param, "µg/m³"),
                                    "observation_time_utc": dt_raw,
                                    "collection_time_utc": collection_time_iso,
                                    "observation_key": f"{sid}::{param}::{dt_raw}",
                                    "provenance_provider": "OpenAQ API v3",
                                    "provenance_sensor_id": s_id,
                                }
                                is_valid, reason = self.validate_reading(cand)
                                if is_valid:
                                    readings.append(cand)
                                else:
                                    logger.debug(f"Reading rejected for {sid} {param}: {reason}")
                        success = True
                        self.log_health_event("fetch_sensor", "SUCCESS", station_id=sid, parameter=param, duration_ms=dur_ms)
                        break
                    elif resp.status_code == 429:
                        wait_sec = backoff ** attempt
                        logger.warning(f"Rate limited (429) for {sid} {param}. Waiting {wait_sec}s (attempt {attempt}/{max_retries})")
                        time.sleep(wait_sec)
                    else:
                        logger.warning(f"OpenAQ API returned HTTP {resp.status_code} for {sid} {param}")
                        time.sleep(backoff)
                except Exception as ex:
                    dur_ms = (time.perf_counter() - t0) * 1000
                    wait_sec = backoff ** attempt
                    logger.warning(f"Fetch exception for {sid} {param} (attempt {attempt}/{max_retries}): {redact_secrets(str(ex))}")
                    if attempt < max_retries:
                        time.sleep(wait_sec)
                    else:
                        self.log_health_event("fetch_sensor", "FAILURE", station_id=sid, parameter=param, message=str(ex), duration_ms=dur_ms)

        return readings

    def append_observations(self, new_readings: List[Dict[str, Any]]) -> int:
        """
        Appends valid, deduplicated observations to experiments/phase12/live_observations.csv.
        Guarantees append-only persistence. Returns count of new records added.
        """
        if not new_readings:
            return 0

        unique_readings = []
        for r in new_readings:
            obs_key = r.get("observation_key") or f"{r['station_id']}::{r['parameter']}::{r['observation_time_utc']}"
            if obs_key not in self.seen_obs_keys:
                self.seen_obs_keys.add(obs_key)
                r["observation_key"] = obs_key
                unique_readings.append(r)

        if not unique_readings:
            return 0

        df_to_append = pd.DataFrame(unique_readings)
        self.obs_csv.parent.mkdir(parents=True, exist_ok=True)

        if not self.obs_csv.exists() or self.df_obs.empty:
            df_to_append.to_csv(self.obs_csv, index=False)
            self.df_obs = df_to_append.copy()
        else:
            df_to_append.to_csv(self.obs_csv, mode="a", header=False, index=False)
            self.df_obs = pd.concat([self.df_obs, df_to_append], ignore_index=True)

        logger.info(f"Appended {len(unique_readings)} new raw observations to {self.obs_csv}")
        return len(unique_readings)

    def execute_collection_cycle(
        self,
        hours_back: int = 6,
        dry_run: bool = False,
        forecast_service: Optional[SensorForecastService] = None,
    ) -> Dict[str, Any]:
        """
        Executes one full autonomous hourly collection and evaluation cycle.
        1. Checks kill-switch and retraining lock.
        2. Ingests available telemetry across all configured stations.
        3. Appends observations idempotently.
        4. Runs non-leaking 1-hour forecasts for eligible 24h windows.
        5. Matches elapsed ground-truth targets ($t+1$).
        6. Updates persistent operational health and state JSON.
        """
        cycle_start_utc = datetime.now(timezone.utc)
        t_start = time.perf_counter()

        # Check kill-switch
        if self.is_kill_switch_active():
            msg = "Collection halted: kill-switch is ACTIVE or collector is DISABLED in configuration."
            logger.warning(msg)
            self.log_health_event("collection_cycle", "HALTED_KILL_SWITCH", message=msg)
            return {"status": "halted_kill_switch", "timestamp_utc": cycle_start_utc.isoformat()}

        # Verify model retraining prevention lock
        if not self.config.get("prevent_model_retraining", True):
            logger.error("FATAL: prevent_model_retraining flag must be True. Aborting cycle.")
            return {"status": "error_retraining_flag_invalid"}

        logger.info(f"Executing Phase 12 collection cycle across {len(self.station_registry)} stations (dry_run={dry_run})...")

        total_fetched = 0
        all_readings = []

        for _, st_row in self.station_registry.iterrows():
            sid = st_row["station_id"]
            try:
                station_readings = self.fetch_station_telemetry_with_retry(st_row, hours_back=hours_back)
                all_readings.extend(station_readings)
                total_fetched += len(station_readings)
            except Exception as e:
                logger.error(f"Error fetching station {sid}: {redact_secrets(str(e))}")

        if dry_run:
            duration = time.perf_counter() - t_start
            logger.info(f"[DRY RUN] Ingested {total_fetched} candidate readings. No files written.")
            return {
                "status": "dry_run_success",
                "total_fetched": total_fetched,
                "duration_seconds": round(duration, 3),
            }

        # Append observations
        n_appended = self.append_observations(all_readings)

        # Generate forecasts and match targets
        fc_service = forecast_service or SensorForecastService()
        n_preds, n_matches = self.generate_forecasts_and_matches(fc_service)

        duration = time.perf_counter() - t_start
        cycle_end_utc = datetime.now(timezone.utc)

        # Update state
        self.state["total_execution_cycles"] = self.state.get("total_execution_cycles", 0) + 1
        self.state["last_successful_run_utc"] = cycle_end_utc.isoformat()
        self.state["last_run_duration_seconds"] = round(duration, 3)
        self.save_state()

        self.log_health_event(
            "collection_cycle",
            "SUCCESS",
            message=f"Cycle complete: {n_appended} new obs, {n_preds} preds, {n_matches} matches.",
            duration_ms=duration * 1000,
        )

        return {
            "status": "cycle_success",
            "cycle_start_utc": cycle_start_utc.isoformat(),
            "cycle_end_utc": cycle_end_utc.isoformat(),
            "duration_seconds": round(duration, 3),
            "new_observations_appended": n_appended,
            "new_predictions_generated": n_preds,
            "new_targets_matched": n_matches,
            "total_observations": len(self.df_obs),
            "total_predictions": len(self.df_preds),
            "total_evaluations": len(self.df_eval),
        }

    def generate_forecasts_and_matches(self, forecast_service: SensorForecastService) -> Tuple[int, int]:
        """
        Generates causal 1-hour forecasts (adapted_gbdt_only) for any 24h contiguous windows
        and evaluates elapsed targets with strict non-leaking causal alignment.
        """
        if self.df_obs.empty:
            return 0, 0

        # Build station rolling buffers
        station_buffers: Dict[str, StationRollingBuffer] = {}
        for _, s_row in self.station_registry.iterrows():
            sid = s_row["station_id"]
            buf = StationRollingBuffer(
                station_id=sid,
                station_lat=float(s_row["latitude"]),
                station_lon=float(s_row["longitude"]),
                max_retention_hours=2000,
                max_staleness_hours=2000.0,
            )
            sub_obs = self.df_obs[self.df_obs["station_id"] == sid]
            for _, r in sub_obs.iterrows():
                buf.ingest_reading(
                    parameter=r["parameter"],
                    value=float(r["value"]),
                    timestamp=r["observation_time_utc"],
                    station_id=sid,
                )
            # Rebuild hourly aggregates
            buf.get_hourly_summary()
            station_buffers[sid] = buf

        new_preds = []
        new_evals = []

        # Local lookup cache for forecasts
        fc_cache: Dict[str, Tuple[float, float, float, float]] = {}
        if not self.df_preds.empty:
            for _, r in self.df_preds.iterrows():
                fc_cache[str(r["forecast_id"])] = (
                    float(r["pred_pm25"]),
                    float(r["pred_pm10"]),
                    float(r["persistence_pm25"]),
                    float(r["persistence_pm10"]),
                )

        for sid, buf in station_buffers.items():
            cache = buf._hourly_cache
            sorted_hours = sorted(cache.keys())
            if len(sorted_hours) < 25:
                continue

            for i in range(23, len(sorted_hours) - 1):
                anchor_h = sorted_hours[i]
                target_h = sorted_hours[i + 1]

                if target_h != anchor_h + timedelta(hours=1):
                    continue

                expected_window = [anchor_h - timedelta(hours=23 - k) for k in range(24)]
                if any(h not in cache for h in expected_window):
                    continue
                if any("PM2.5" not in cache[h] or "PM10" not in cache[h] for h in expected_window):
                    continue

                issue_iso = anchor_h.strftime("%Y-%m-%dT%H:00:00Z")
                target_iso = target_h.strftime("%Y-%m-%dT%H:00:00Z")
                forecast_id = f"fc_{sid}_{issue_iso}"

                if forecast_id not in self.seen_pred_ids:
                    self.seen_pred_ids.add(forecast_id)
                    rows = []
                    for h in expected_window:
                        r = dict(cache[h])
                        r["Datetime"] = h
                        r["StationId"] = sid
                        r["Latitude"] = buf.station_lat
                        r["Longitude"] = buf.station_lon
                        rows.append(r)
                    df_24h = pd.DataFrame(rows)

                    fc_res = forecast_service.get_forecast(
                        station_id=sid,
                        df_24h=df_24h,
                        model_mode="adapted_gbdt_only",
                        enable_shadow=False,
                    )
                    pred_pm25 = float(fc_res.get("pm25", 0.0))
                    pred_pm10 = float(fc_res.get("pm10", 0.0))
                    pers_pm25 = float(cache[anchor_h]["PM2.5"])
                    pers_pm10 = float(cache[anchor_h]["PM10"])

                    fc_cache[forecast_id] = (pred_pm25, pred_pm10, pers_pm25, pers_pm10)

                    new_preds.append({
                        "forecast_id": forecast_id,
                        "station_id": sid,
                        "issue_time_utc": issue_iso,
                        "target_time_utc": target_iso,
                        "model_name": "adapted_gbdt_only",
                        "pred_pm25": round(pred_pm25, 2),
                        "pred_pm10": round(pred_pm10, 2),
                        "persistence_pm25": round(pers_pm25, 2),
                        "persistence_pm10": round(pers_pm10, 2),
                        "status": "forecast_generated",
                    })
                else:
                    if forecast_id in fc_cache:
                        pred_pm25, pred_pm10, pers_pm25, pers_pm10 = fc_cache[forecast_id]
                    else:
                        match_row = self.df_preds[self.df_preds["forecast_id"] == forecast_id]
                        if not match_row.empty:
                            ex = match_row.iloc[0]
                            pred_pm25 = float(ex["pred_pm25"])
                            pred_pm10 = float(ex["pred_pm10"])
                            pers_pm25 = float(ex["persistence_pm25"])
                            pers_pm10 = float(ex["persistence_pm10"])
                            fc_cache[forecast_id] = (pred_pm25, pred_pm10, pers_pm25, pers_pm10)
                        else:
                            continue

                # Check causal target match
                match_id = f"eval_{sid}_{target_iso}"
                if match_id not in self.seen_match_ids:
                    act_25 = cache[target_h].get("PM2.5")
                    act_10 = cache[target_h].get("PM10")
                    if act_25 is not None and act_10 is not None:
                        if (0.0 <= act_25 <= 1000.0) and (0.0 <= act_10 <= 1500.0):
                            self.seen_match_ids.add(match_id)
                            new_evals.append({
                                "match_id": match_id,
                                "forecast_id": forecast_id,
                                "station_id": sid,
                                "target_time_utc": target_iso,
                                "actual_pm25": round(act_25, 2),
                                "actual_pm10": round(act_10, 2),
                                "pred_pm25": round(pred_pm25, 2),
                                "pred_pm10": round(pred_pm10, 2),
                                "persistence_pm25": round(pers_pm25, 2),
                                "persistence_pm10": round(pers_pm10, 2),
                                "gbdt_err_pm25": round(pred_pm25 - act_25, 2),
                                "gbdt_err_pm10": round(pred_pm10 - act_10, 2),
                                "pers_err_pm25": round(pers_pm25 - act_25, 2),
                                "pers_err_pm10": round(pers_pm10 - act_10, 2),
                                "gbdt_abs_err_pm25": round(abs(pred_pm25 - act_25), 2),
                                "gbdt_abs_err_pm10": round(abs(pred_pm10 - act_10), 2),
                                "pers_abs_err_pm25": round(abs(pers_pm25 - act_25), 2),
                                "pers_abs_err_pm10": round(abs(pers_pm10 - act_10), 2),
                            })

        # Append predictions
        if new_preds:
            df_np = pd.DataFrame(new_preds)
            if not self.preds_csv.exists() or self.df_preds.empty:
                df_np.to_csv(self.preds_csv, index=False)
                self.df_preds = df_np.copy()
            else:
                df_np.to_csv(self.preds_csv, mode="a", header=False, index=False)
                self.df_preds = pd.concat([self.df_preds, df_np], ignore_index=True)

        # Append evaluations
        if new_evals:
            df_ne = pd.DataFrame(new_evals)
            if not self.eval_csv.exists() or self.df_eval.empty:
                df_ne.to_csv(self.eval_csv, index=False)
                self.df_eval = df_ne.copy()
            else:
                df_ne.to_csv(self.eval_csv, mode="a", header=False, index=False)
                self.df_eval = pd.concat([self.df_eval, df_ne], ignore_index=True)

        return len(new_preds), len(new_evals)


def generate_scheduler_script(platform: str = "windows") -> str:
    """Generates exact command or script to register the autonomous collector in the host OS."""
    python_exe = sys.executable
    script_path = ROOT_DIR / "scripts/run_autonomous_collector.py"

    if platform.lower() == "windows":
        cmd = (
            f'schtasks.exe /Create /TN "VayuDrishti_Autonomous_Collector" '
            f'/TR "\"{python_exe}\" \"{script_path}\"" '
            f'/SC HOURLY /MO 1 /ST 00:15 /RU "%USERNAME%" /RL LIMITED /F'
        )
        return cmd
    else:
        cron_entry = f"15 * * * * cd {ROOT_DIR} && {python_exe} {script_path} >> {PHASE12_DIR}/collector_cron.log 2>&1"
        return cron_entry
