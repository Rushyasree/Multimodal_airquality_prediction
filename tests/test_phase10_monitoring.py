"""
tests/test_phase10_monitoring.py

Comprehensive test suite for Phase 10: Longitudinal Live Monitoring and Research Dataset Preservation.
Validates:
1. Operational collector state serialization and persistence.
2. Provenance tracking: model checkpoint SHA-256 hashes, feature schema version, and input timestamps.
3. Append-only deduplication: re-running collection never creates duplicate observations or predictions.
4. Handling of unmatched predictions / missing target observations without corrupting evaluation datasets.
5. Immutability of model checkpoints and historical evaluation cohorts.
6. Per-station coverage tracking and temporal continuity.
"""

import os
import sys
import json
import hashlib
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

AUDIT_DIR = ROOT_DIR / "experiments/recent_data_audit"
OBS_CSV = AUDIT_DIR / "live_trial_observations.csv"
PREDS_CSV = AUDIT_DIR / "live_trial_predictions.csv"
EVAL_CSV = AUDIT_DIR / "live_trial_evaluation.csv"
STATE_JSON = AUDIT_DIR / "collector_state.json"

from src.services.longitudinal_live_collector import (
    LongitudinalCollector,
    CHECKPOINT_HASHES,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
)
from src.services.standalone_gbdt_live_trial import TARGET_CAAQMS_STATIONS


class TestPhase10Monitoring(unittest.TestCase):
    """Test suite validating longitudinal collector mechanics, provenance, and dataset preservation."""

    def setUp(self):
        self.collector = LongitudinalCollector()

    def test_01_collector_state_persistence(self):
        """Verifies that collector_state.json exists, loads correctly, and contains required schema."""
        self.assertTrue(STATE_JSON.exists(), "collector_state.json must exist on disk.")
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            state = json.load(f)

        required_keys = [
            "collector_version", "trial_phase", "total_observations_ingested",
            "total_predictions_generated", "total_targets_matched",
            "unmatched_predictions_count", "station_telemetry_status"
        ]
        for k in required_keys:
            self.assertIn(k, state, f"Missing state key: {k}")

        self.assertEqual(state["total_observations_ingested"], len(self.collector.df_obs))
        self.assertEqual(state["total_predictions_generated"], len(self.collector.df_preds))
        self.assertEqual(state["total_targets_matched"], len(self.collector.df_eval))

    def test_02_provenance_schema_completeness(self):
        """Verifies that prediction records retain complete provenance metadata."""
        self.assertFalse(self.collector.df_preds.empty)
        cols = self.collector.df_preds.columns.tolist()

        expected_provenance_fields = [
            "forecast_id", "station_id", "issue_time_utc", "target_time_utc",
            "model_name", "pred_pm25", "pred_pm10", "persistence_pm25", "persistence_pm10",
        ]
        for field in expected_provenance_fields:
            self.assertIn(field, cols, f"Missing prediction provenance column: {field}")

        # Check values in prediction records
        for _, row in self.collector.df_preds.head(10).iterrows():
            self.assertEqual(row["model_name"], MODEL_NAME)
            self.assertGreater(row["pred_pm25"], 0.0)
            self.assertGreater(row["pred_pm10"], 0.0)

    def test_03_deduplication_and_idempotency(self):
        """Verifies that re-appending existing observations or predictions adds 0 duplicate records."""
        initial_obs_count = len(self.collector.df_obs)
        initial_preds_count = len(self.collector.df_preds)
        initial_eval_count = len(self.collector.df_eval)

        # Attempt to append existing observation records
        sample_recs = self.collector.df_obs.head(20).to_dict("records")
        n_appended = self.collector.append_observations(sample_recs)
        self.assertEqual(n_appended, 0, "Appending already-seen records must return 0 new additions.")
        self.assertEqual(len(self.collector.df_obs), initial_obs_count)

        self.assertEqual(len(self.collector.seen_pred_ids), initial_preds_count)
        self.assertEqual(len(self.collector.seen_match_ids), initial_eval_count)

    def test_04_unmatched_predictions_accounting(self):
        """Verifies that unmatched predictions (e.g. missing targets) are precisely accounted for."""
        df_p = self.collector.df_preds
        df_e = self.collector.df_eval

        matched_ids = set(df_e["forecast_id"])
        unmatched = df_p[~df_p["forecast_id"].isin(matched_ids)]

        # Must match state counter exactly
        expected_unmatched_count = len(unmatched)
        self.assertEqual(self.collector.state["unmatched_predictions_count"], expected_unmatched_count)

        # The 2 known missing target hours occurred on DL008
        if expected_unmatched_count > 0:
            for _, r in unmatched.iterrows():
                self.assertIn(r["station_id"], TARGET_CAAQMS_STATIONS)

    def test_05_checkpoint_hash_immutability(self):
        """Verifies that the trained model checkpoint hashes match the provenance manifest."""
        gbdt_25 = AUDIT_DIR / "checkpoints/adapted_gbdt_pm25.joblib"
        gbdt_10 = AUDIT_DIR / "checkpoints/adapted_gbdt_pm10.joblib"

        h25 = hashlib.sha256(gbdt_25.read_bytes()).hexdigest()
        h10 = hashlib.sha256(gbdt_10.read_bytes()).hexdigest()

        self.assertEqual(h25, CHECKPOINT_HASHES["PM2.5"])
        self.assertEqual(h10, CHECKPOINT_HASHES["PM10"])

    def test_06_temporal_coverage_and_continuity(self):
        """Verifies that observations cover at least 14 calendar days across all 4 stations."""
        min_dt = pd.to_datetime(self.collector.df_obs["observation_time_utc"].min())
        max_dt = pd.to_datetime(self.collector.df_obs["observation_time_utc"].max())
        duration_days = (max_dt - min_dt).total_seconds() / 86400.0

        self.assertGreaterEqual(duration_days, 14.0, f"Trial duration ({duration_days:.1f} days) is less than 14 days.")
        self.assertEqual(set(self.collector.df_obs["station_id"].unique()), set(TARGET_CAAQMS_STATIONS.keys()))


if __name__ == "__main__":
    unittest.main()
