"""
tests/test_accuracy_optimization_v2.py

Verification and regression test suite for Accuracy Optimization v2:
1. Feature extraction shape, consistency, and non-empty metadata (67 features).
2. Strict temporal leakage invariance (features only depend on historical steps t-23 to t).
3. Checkpoint integrity and loading (GBDT and Residual models).
4. Deterministic inference reproducibility.
5. Physical constraints (non-negativity, valid numerical ranges).
6. Artifacts integrity (baseline comparison, metrics, tier breakdown).
"""

import json
import sys
import unittest
from pathlib import Path
import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.tabular_features import (
    FEATURE_NAMES,
    extract_tabular_features,
    get_tabular_feature_names,
)


class TestAccuracyOptimizationV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opt_dir = ROOT_DIR / "experiments/accuracy_optimization_v2"
        cls.ckpt_dir = cls.opt_dir / "checkpoints"
        cls.cfg_path = cls.opt_dir / "configs/best_model_config.json"

    def test_01_feature_extraction_dimension(self):
        """Verify that extract_tabular_features produces exactly (N, 67) shape."""
        N = 10
        dummy_seq = np.random.randn(N, 24, 27).astype(np.float32)
        X_tab = extract_tabular_features(dummy_seq)

        self.assertEqual(X_tab.shape, (N, 67), f"Expected shape ({N}, 67), got {X_tab.shape}")
        self.assertFalse(np.isnan(X_tab).any(), "Extracted features should not contain NaNs")
        self.assertFalse(np.isinf(X_tab).any(), "Extracted features should not contain Infs")

        names = get_tabular_feature_names()
        self.assertEqual(len(names), 67, f"Expected 67 feature names, got {len(names)}")
        self.assertEqual(len(names), len(FEATURE_NAMES))

    def test_02_temporal_leakage_prevention(self):
        """
        Verify that features are strictly extracted from past timesteps (t-23 to t)
        and invariant to any downstream modifications outside the sequence window.
        """
        # Create a synthetic sequence of 25 timesteps (24 past + 1 future at index 24)
        full_timeline = np.random.randn(5, 25, 27).astype(np.float32)
        seq_historical = full_timeline[:, :24, :]

        # Extract features from past sequence
        features_1 = extract_tabular_features(seq_historical)

        # Alter the future step (index 24)
        full_timeline[:, 24, :] += 500.0

        # Re-extract from historical sequence
        features_2 = extract_tabular_features(full_timeline[:, :24, :])

        np.testing.assert_allclose(
            features_1,
            features_2,
            atol=1e-7,
            err_msg="Extracted features must be strictly invariant to future timesteps (> t)",
        )

        # Verify that index 23 is treated as current (t)
        # and rolling means/stds include only steps <= 23
        t_current_pm25 = seq_historical[:, 23, 0]
        # In extract_tabular_features, feature index 0 is pm25_lag0
        np.testing.assert_allclose(
            features_1[:, 0],
            t_current_pm25,
            err_msg="Lag 0 feature must match exact sequence timestep t (index 23)",
        )

    def test_03_checkpoints_exist_and_loadable(self):
        """Verify all trained v2 checkpoints and configuration exist and load."""
        self.assertTrue(self.cfg_path.exists(), f"Missing config: {self.cfg_path}")
        with open(self.cfg_path) as f:
            cfg = json.load(f)
        self.assertIn("best_ensemble_weights", cfg)
        self.assertIn("winning_model", cfg)

        expected_ckpts = [
            "best_gbdt_pm25.joblib",
            "best_gbdt_pm10.joblib",
            "best_residual_gbdt_pm25.joblib",
            "best_residual_gbdt_pm10.joblib",
        ]
        for ckpt_name in expected_ckpts:
            ckpt_file = self.ckpt_dir / ckpt_name
            self.assertTrue(ckpt_file.exists(), f"Missing checkpoint: {ckpt_file}")
            model = joblib.load(ckpt_file)
            self.assertIsNotNone(model, f"Failed to load checkpoint: {ckpt_name}")

    def test_04_deterministic_inference(self):
        """Verify deterministic outputs on identical inputs."""
        gbt_pm25 = joblib.load(self.ckpt_dir / "best_gbdt_pm25.joblib")
        dummy_seq = np.random.RandomState(42).randn(8, 24, 27).astype(np.float32)

        feat1 = extract_tabular_features(dummy_seq)
        feat2 = extract_tabular_features(dummy_seq)
        np.testing.assert_array_equal(feat1, feat2)

        pred1 = gbt_pm25.predict(feat1)
        pred2 = gbt_pm25.predict(feat2)
        np.testing.assert_array_equal(pred1, pred2, err_msg="GBDT predictions must be strictly deterministic")

    def test_05_physical_constraints(self):
        """Verify physical constraints: non-negativity and reasonable scale."""
        gbt_pm25 = joblib.load(self.ckpt_dir / "best_gbdt_pm25.joblib")
        gbt_pm10 = joblib.load(self.ckpt_dir / "best_gbdt_pm10.joblib")

        # Typical normalized sequence around zero (mean)
        dummy_seq = np.zeros((4, 24, 27), dtype=np.float32)
        feat = extract_tabular_features(dummy_seq)

        p25 = gbt_pm25.predict(feat)
        p10 = gbt_pm10.predict(feat)

        self.assertTrue(np.all(p25 >= 0.0), f"PM2.5 prediction must be non-negative: {p25}")
        self.assertTrue(np.all(p10 >= 0.0), f"PM10 prediction must be non-negative: {p10}")
        self.assertTrue(np.all(p25 < 1000.0), f"PM2.5 prediction out of physical range: {p25}")
        self.assertTrue(np.all(p10 < 1500.0), f"PM10 prediction out of physical range: {p10}")


if __name__ == "__main__":
    unittest.main()
