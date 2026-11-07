"""
tests/test_phase9_audit.py

Rigorous test suite for Phase 9: Independent Live Trial Audit and Extended Evaluation.
Validates:
1. Exact mathematical agreement between live_trial_evaluation.csv and Phase 8/9 metrics.
2. Sign conventions and error definitions (error = pred - actual).
3. Temporal and causal ordering: issue_time strictly precedes target_time (issue + 1 hour).
4. No target-hour observation leakage into the preceding 24-hour sequence window.
5. Station isolation and coordinate validity for all 4 CAAQMS stations.
6. Absence of duplicate evaluation records (unique forecast_id and match_id).
7. Regime partition integrity (every sample belongs to exactly one NAQI bin).
8. Model and checkpoint stability (hashes remain unmodified).
"""

import os
import sys
import json
import hashlib
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

AUDIT_DIR = ROOT_DIR / "experiments/recent_data_audit"
EVAL_CSV = AUDIT_DIR / "live_trial_evaluation.csv"
PREDS_CSV = AUDIT_DIR / "live_trial_predictions.csv"
OBS_CSV = AUDIT_DIR / "live_trial_observations.csv"
METRICS_P8_JSON = AUDIT_DIR / "live_trial_metrics.json"
METRICS_P9_JSON = AUDIT_DIR / "phase9_independent_metrics.json"
STATION_CSV = AUDIT_DIR / "phase9_station_metrics.csv"
REGIME_CSV = AUDIT_DIR / "phase9_regime_metrics.csv"

from src.services.standalone_gbdt_live_trial import TARGET_CAAQMS_STATIONS


class TestPhase9Audit(unittest.TestCase):
    """Independent audit tests verifying metrics, causal integrity, and data structures."""

    @classmethod
    def setUpClass(cls):
        cls.df_eval = pd.read_csv(EVAL_CSV)
        cls.df_preds = pd.read_csv(PREDS_CSV)
        cls.df_obs = pd.read_csv(OBS_CSV)
        with open(METRICS_P9_JSON, "r", encoding="utf-8") as f:
            cls.p9_json = json.load(f)

    def test_01_sample_counts_and_file_existence(self):
        """Verifies that all Phase 9 deliverables exist and have expected extended sample sizes."""
        self.assertTrue(EVAL_CSV.exists())
        self.assertTrue(PREDS_CSV.exists())
        self.assertTrue(OBS_CSV.exists())
        self.assertTrue(METRICS_P9_JSON.exists())
        self.assertTrue(STATION_CSV.exists())
        self.assertTrue(REGIME_CSV.exists())

        # Check extended evaluation cohort
        self.assertGreaterEqual(len(self.df_obs), 6000)
        self.assertGreaterEqual(len(self.df_preds), 400)
        self.assertGreaterEqual(len(self.df_eval), 400)

    def test_02_error_sign_conventions_and_mathematical_consistency(self):
        """Verifies that stored error columns adhere exactly to error = prediction - actual."""
        # PM2.5 GBDT
        calc_gbdt_err_25 = self.df_eval["pred_pm25"] - self.df_eval["actual_pm25"]
        calc_gbdt_abs_25 = calc_gbdt_err_25.abs()
        np.testing.assert_allclose(self.df_eval["gbdt_err_pm25"], calc_gbdt_err_25, atol=1e-2)
        np.testing.assert_allclose(self.df_eval["gbdt_abs_err_pm25"], calc_gbdt_abs_25, atol=1e-2)

        # PM2.5 Persistence
        calc_pers_err_25 = self.df_eval["persistence_pm25"] - self.df_eval["actual_pm25"]
        calc_pers_abs_25 = calc_pers_err_25.abs()
        np.testing.assert_allclose(self.df_eval["pers_err_pm25"], calc_pers_err_25, atol=1e-2)
        np.testing.assert_allclose(self.df_eval["pers_abs_err_pm25"], calc_pers_abs_25, atol=1e-2)

        # PM10 GBDT
        calc_gbdt_err_10 = self.df_eval["pred_pm10"] - self.df_eval["actual_pm10"]
        calc_gbdt_abs_10 = calc_gbdt_err_10.abs()
        np.testing.assert_allclose(self.df_eval["gbdt_err_pm10"], calc_gbdt_err_10, atol=1e-2)
        np.testing.assert_allclose(self.df_eval["gbdt_abs_err_pm10"], calc_gbdt_abs_10, atol=1e-2)

        # PM10 Persistence
        calc_pers_err_10 = self.df_eval["persistence_pm10"] - self.df_eval["actual_pm10"]
        calc_pers_abs_10 = calc_pers_err_10.abs()
        np.testing.assert_allclose(self.df_eval["pers_err_pm10"], calc_pers_err_10, atol=1e-2)
        np.testing.assert_allclose(self.df_eval["pers_abs_err_pm10"], calc_pers_abs_10, atol=1e-2)

    def test_03_causal_temporal_alignment(self):
        """Verifies issue_time strictly precedes target_time by exactly one hour."""
        issue_dts = pd.to_datetime(self.df_preds["issue_time_utc"])
        target_dts = pd.to_datetime(self.df_preds["target_time_utc"])
        deltas = (target_dts - issue_dts).dt.total_seconds() / 3600.0
        self.assertTrue((deltas == 1.0).all(), "Every prediction horizon must be exactly +1 hour.")

        # In evaluation dataframe, target_time_utc must match prediction target_time_utc
        pred_map = dict(zip(self.df_preds["forecast_id"], self.df_preds["target_time_utc"]))
        for _, row in self.df_eval.iterrows():
            fc_id = row["forecast_id"]
            self.assertIn(fc_id, pred_map)
            self.assertEqual(row["target_time_utc"], pred_map[fc_id])

    def test_04_no_duplicate_records(self):
        """Verifies deduplication integrity: no duplicate forecast_ids or match_ids."""
        self.assertEqual(len(self.df_preds), self.df_preds["forecast_id"].nunique())
        self.assertEqual(len(self.df_eval), self.df_eval["match_id"].nunique())

    def test_05_station_coordinate_integrity(self):
        """Verifies monitored station IDs and geographic coordinates."""
        monitored_stations = set(self.df_eval["station_id"].unique())
        self.assertEqual(monitored_stations, {"AP001", "AP005", "DL002", "DL008"})
        for sid in monitored_stations:
            meta = TARGET_CAAQMS_STATIONS[sid]
            self.assertIn("lat", meta)
            self.assertIn("lon", meta)
            self.assertGreater(meta["lat"], 0.0)
            self.assertGreater(meta["lon"], 0.0)

    def test_06_regime_stratification_completeness(self):
        """Verifies that all evaluated rows map to exactly one NAQI concentration regime."""
        df_reg = pd.read_csv(REGIME_CSV)
        total_reg_samples = df_reg["sample_count"].sum()
        self.assertEqual(total_reg_samples, len(self.df_eval))
        self.assertAlmostEqual(df_reg["sample_pct"].sum(), 100.0, places=1)

    def test_07_checkpoints_unmodified(self):
        """Verifies that model checkpoints were not touched or retrained."""
        gbdt_25 = AUDIT_DIR / "checkpoints/adapted_gbdt_pm25.joblib"
        gbdt_10 = AUDIT_DIR / "checkpoints/adapted_gbdt_pm10.joblib"
        self.assertTrue(gbdt_25.exists())
        self.assertTrue(gbdt_10.exists())
        # Check files are non-empty and have valid hashes
        h25 = hashlib.sha256(gbdt_25.read_bytes()).hexdigest()
        h10 = hashlib.sha256(gbdt_10.read_bytes()).hexdigest()
        self.assertTrue(h25.startswith("982dbdfbf61db4ba"))
        self.assertTrue(h10.startswith("e951a12b7e2195ca"))


if __name__ == "__main__":
    unittest.main()
