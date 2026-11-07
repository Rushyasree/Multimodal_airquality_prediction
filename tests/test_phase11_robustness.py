"""
tests/test_phase11_robustness.py

Unit and statistical validation tests for Phase 11:
Longitudinal Monitoring Validation & Statistical Robustness.
Validates:
1. Dataset integrity audit passes all 9 checks.
2. Station-stratified block bootstrap sensitivity metrics (24h, 48h, 72h).
3. Correct bracketing of zero in pooled PM2.5 confidence intervals.
4. Multiple-comparison adjustments (Bonferroni & Benjamini-Hochberg FDR) across 16 tests.
5. Strict distinction between Confirmatory and Exploratory hypotheses.
6. Immutability of model checkpoint SHA-256 hashes.
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
INTEGRITY_JSON = AUDIT_DIR / "phase11_dataset_integrity_audit.json"
ROBUSTNESS_JSON = AUDIT_DIR / "phase11_robustness_metrics.json"
EVAL_CSV = AUDIT_DIR / "live_trial_evaluation.csv"
STATE_JSON = AUDIT_DIR / "collector_state.json"


class TestPhase11Robustness(unittest.TestCase):
    """Test suite validating Phase 11 statistical robustness, sensitivity, and integrity audits."""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(INTEGRITY_JSON.exists(), f"Missing {INTEGRITY_JSON}")
        cls.assertTrue(ROBUSTNESS_JSON.exists(), f"Missing {ROBUSTNESS_JSON}")
        with open(INTEGRITY_JSON, "r", encoding="utf-8") as f:
            cls.integrity = json.load(f)
        with open(ROBUSTNESS_JSON, "r", encoding="utf-8") as f:
            cls.robustness = json.load(f)

    def test_01_dataset_integrity_all_checks_passed(self):
        """Verifies that all 9 integrity checks passed without defects."""
        self.assertTrue(self.integrity.get("all_checks_passed"), "Not all integrity checks passed.")
        checks = self.integrity.get("checks", {})
        self.assertEqual(len(checks), 9)
        for check_name, check_data in checks.items():
            self.assertEqual(check_data.get("status"), "PASS", f"Check {check_name} did not PASS.")

    def test_02_block_bootstrap_sensitivity_block_lengths(self):
        """Verifies that block bootstrap was executed for 24h, 48h, and 72h block lengths."""
        sens = self.robustness.get("block_bootstrap_sensitivity", {})
        for blk in ["24h", "48h", "72h"]:
            self.assertIn(blk, sens)
            blk_data = sens[blk]
            self.assertIn("pooled", blk_data)
            self.assertIn("per_station", blk_data)

            # Check pooled delta metrics
            pooled = blk_data["pooled"]
            for m in ["delta_mae_pm25", "delta_mae_pm10", "delta_rmse_pm25", "delta_rmse_pm10"]:
                self.assertIn(m, pooled)
                ci = pooled[m]["ci_95"]
                self.assertEqual(len(ci), 2)
                self.assertLess(ci[0], ci[1])

    def test_03_primary_confirmatory_hypothesis_insignificance(self):
        """
        Verifies that the primary confirmatory hypothesis (pooled PM2.5 delta_MAE < 0)
        honestly reports non-significance (confidence intervals bracket zero).
        """
        conf = self.robustness.get("primary_confirmatory_hypothesis", {})
        self.assertFalse(conf.get("achieves_statistical_significance"))

        # Verify that all 3 block bootstrap CIs bracket zero
        for blk_key in ["bootstrap_24h_ci", "bootstrap_48h_ci", "bootstrap_72h_ci"]:
            ci = conf[blk_key]
            self.assertLess(ci[0], 0.0, f"Lower bound should be negative: {ci}")
            self.assertGreater(ci[1], 0.0, f"Upper bound should be positive (bracketing zero): {ci}")

    def test_04_multiple_comparison_corrections_structure(self):
        """Verifies the 16 subgroup multiple-comparison hypothesis tests."""
        m_comp = self.robustness.get("multiple_comparison_corrections", {})
        self.assertEqual(m_comp.get("total_hypotheses"), 16)
        self.assertAlmostEqual(m_comp.get("bonferroni_threshold"), 0.05 / 16, places=5)

        hyps = m_comp.get("hypotheses_evaluated", [])
        self.assertEqual(len(hyps), 16)

        # Check ranking and monotonicity of BH critical values
        for i, h in enumerate(hyps):
            self.assertIn("bonferroni_significant", h)
            self.assertIn("bh_fdr_significant", h)
            self.assertEqual(h["rank"], i + 1)
            self.assertAlmostEqual(h["bh_critical_value"], ((i + 1) / 16) * 0.05, places=4)

    def test_05_confirmatory_vs_exploratory_demarcation(self):
        """Verifies clear separation of confirmatory vs exploratory findings."""
        demarc = self.robustness.get("confirmatory_vs_exploratory_demarcation", {})
        self.assertIn("confirmatory_findings", demarc)
        self.assertIn("exploratory_findings", demarc)
        self.assertGreater(len(demarc["confirmatory_findings"]), 0)
        self.assertGreater(len(demarc["exploratory_findings"]), 0)

    def test_06_checkpoint_hashes_unmodified(self):
        """Verifies SHA-256 hashes of adapted GBDT models on disk match frozen hashes."""
        checkpoints = {
            "PM2.5": ("checkpoints/adapted_gbdt_pm25.joblib", "982dbdfbf61db4ba937a289860e8297bb482f3b79060af2c173f83476ad2225c"),
            "PM10": ("checkpoints/adapted_gbdt_pm10.joblib", "e951a12b7e2195cae81eb20511ea635c2466d5e42208420382b09ed1a15743aa"),
        }
        for name, (rel_path, expected_hash) in checkpoints.items():
            full_path = AUDIT_DIR / rel_path
            self.assertTrue(full_path.exists())
            h = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(h, expected_hash, f"Hash mismatch for {name}")


if __name__ == "__main__":
    unittest.main()
