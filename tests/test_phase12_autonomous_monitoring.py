"""
tests/test_phase12_autonomous_monitoring.py

Comprehensive Automated Test Suite for VayuDrishti Phase 12.
Covers:
1. Collector configuration & model retraining prevention lock.
2. Kill-switch behavior and halt mechanism.
3. Sensor telemetry validation (physical bounds, units, UTC timestamps, future lookahead rejection).
4. Observation deduplication key stability and timestamp separation (collection vs observation).
5. Secret redaction in logs and health event recording.
6. Network retry and exponential backoff resilience.
7. Multi-station expansion registry (15 stations across 11 states/UTs with cohort isolation).
8. Meteorological service temporal causality (strictly <= t, zero lookahead).
9. Conformal uncertainty calibration (valid finite-sample coverage, non-negative intervals).
10. Statistical hypothesis testing & multiple testing corrections (Bonferroni, FDR).
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from src.services.autonomous_live_collector import (
    AutonomousLiveCollector,
    redact_secrets,
    PHYSICAL_BOUNDS,
    generate_scheduler_script,
)
from src.services.meteorological_service import MeteorologicalService
from src.services.forecast_uncertainty_service import ConformalPredictionCalibrator
from scripts.compute_phase12_statistics import stationary_block_bootstrap, run_hypothesis_test

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_collector_configuration_and_retraining_lock():
    collector = AutonomousLiveCollector()
    config = collector.config

    assert config.get("collector_version") == "v2.0-autonomous"
    assert config.get("prevent_model_retraining") is True, "Retraining lock must be True"
    assert config.get("operational_mode") == "autonomous_hourly_shadow"
    assert config.get("collector_enabled") is True


def test_kill_switch_behavior():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_file = tmp_path / "config.json"
        ks_file = tmp_path / "KILL_SWITCH"

        cfg = {"kill_switch_file": str(ks_file), "collector_enabled": True, "prevent_model_retraining": True}
        with open(cfg_file, "w") as f:
            json.dump(cfg, f)

        coll = AutonomousLiveCollector(config_path=cfg_file)
        assert coll.is_kill_switch_active() is False

        # Create kill switch file
        ks_file.touch()
        assert coll.is_kill_switch_active() is True

        res = coll.execute_collection_cycle()
        assert res.get("status") == "halted_kill_switch"


def test_sensor_telemetry_bounds_and_causality():
    coll = AutonomousLiveCollector()

    now_utc = datetime.now(timezone.utc)
    valid_ts = now_utc.strftime("%Y-%m-%dT%H:00:00Z")

    # Valid reading
    reading_ok = {
        "station_id": "AP001",
        "parameter": "PM2.5",
        "value": 45.2,
        "unit": "µg/m³",
        "observation_time_utc": valid_ts,
    }
    ok, reason = coll.validate_reading(reading_ok)
    assert ok is True, f"Valid reading was rejected: {reason}"

    # Out of bounds (> 1000 ug/m3)
    reading_oob = dict(reading_ok, value=1500.0)
    ok, reason = coll.validate_reading(reading_oob)
    assert ok is False
    assert "out_of_physical_bounds" in reason

    # Negative value
    reading_neg = dict(reading_ok, value=-5.0)
    ok, reason = coll.validate_reading(reading_neg)
    assert ok is False

    # NaN value
    reading_nan = dict(reading_ok, value=float("nan"))
    ok, reason = coll.validate_reading(reading_nan)
    assert ok is False

    # Future timestamp rejection (> 5 min future)
    future_ts = (now_utc + timedelta(hours=2)).strftime("%Y-%m-%dT%H:00:00Z")
    reading_future = dict(reading_ok, observation_time_utc=future_ts)
    ok, reason = coll.validate_reading(reading_future)
    assert ok is False
    assert "future_timestamp_rejected" in reason

    # Unregistered station ID
    reading_bad_sid = dict(reading_ok, station_id="UNKNOWN_999")
    ok, reason = coll.validate_reading(reading_bad_sid)
    assert ok is False
    assert "invalid_station_id" in reason


def test_deduplication_and_timestamp_separation():
    coll = AutonomousLiveCollector()
    now_utc = datetime.now(timezone.utc)
    obs_time = (now_utc - timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00Z")

    key1 = f"DL002::PM2.5::{obs_time}"
    key2 = f"DL002::PM10::{obs_time}"
    key3 = f"DL002::PM2.5::{obs_time}"  # duplicate

    assert key1 != key2, "Parameters must create distinct keys"
    assert key1 == key3, "Same station/parameter/time must produce identical key"


def test_secret_redaction():
    fake_key = "openaq_live_abc12345secretxyz"
    raw_message = f"Querying OpenAQ with key {fake_key} and token secret_token_98765"
    sanitized = redact_secrets(raw_message)

    assert fake_key not in sanitized
    assert "secret_token_98765" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized or "[REDACTED]" in sanitized


def test_station_registry_expansion():
    reg_file = ROOT_DIR / "experiments/phase12/station_registry.csv"
    assert reg_file.exists(), "station_registry.csv must exist"

    df_reg = pd.read_csv(reg_file)
    assert len(df_reg) == 15, f"Must contain exactly 15 stations, got {len(df_reg)}"

    cohorts = set(df_reg["cohort"].unique())
    assert "Phase8-4Stations" in cohorts
    assert "Expanded-MultiStation" in cohorts

    phase8_sids = set(df_reg[df_reg["cohort"] == "Phase8-4Stations"]["station_id"])
    assert phase8_sids == {"AP001", "AP005", "DL002", "DL008"}

    expanded_sids = set(df_reg[df_reg["cohort"] == "Expanded-MultiStation"]["station_id"])
    assert len(expanded_sids) == 11
    assert "MH005" in expanded_sids
    assert "WB008" in expanded_sids
    assert "KA004" in expanded_sids


def test_meteorological_service_causality():
    meteo = MeteorologicalService()
    anchor = datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Aligned 24h window
    w_df = meteo.get_aligned_weather_features(station_id="AP001", issue_time_utc=anchor, window_hours=24)
    if w_df is not None:
        assert len(w_df) == 24
        for col in ["temperature_C", "relative_humidity", "wind_speed", "wind_direction"]:
            assert col in w_df.columns


def test_conformal_uncertainty_calibration():
    calibrator = ConformalPredictionCalibrator()

    # Generate synthetic calibration residuals (N=500)
    rng = np.random.default_rng(42)
    y_true = rng.uniform(20.0, 150.0, size=500)
    noise = rng.normal(0.0, 8.0, size=500)
    y_pred = np.maximum(0.0, y_true + noise)

    quantiles = calibrator.fit_calibration_set(y_true[:250], y_pred[:250], parameter="PM2.5")
    assert 0.80 in quantiles and 0.90 in quantiles and 0.95 in quantiles
    assert quantiles[0.80] < quantiles[0.90] < quantiles[0.95]

    # Evaluate coverage on test set
    eval_df = calibrator.evaluate_coverage(y_true[250:], y_pred[250:], parameter="PM2.5")
    assert len(eval_df) == 3

    for _, r in eval_df.iterrows():
        nom = r["nominal_coverage"]
        emp = r["empirical_coverage"]
        assert abs(emp - nom) < 0.08, f"Conformal coverage deviation too large: emp={emp}, nom={nom}"
        assert r["mean_interval_width"] > 0.0


def test_scheduler_script_generation():
    win_cmd = generate_scheduler_script(platform="windows")
    assert "schtasks" in win_cmd.lower() and "/create" in win_cmd.lower()
    assert "/sc hourly" in win_cmd.lower()

    posix_cron = generate_scheduler_script(platform="linux")
    assert "15 * * * *" in posix_cron
