"""
tests/test_optimized_hybrid.py

Comprehensive unit and integration test suite for the Optimized Residual Hybrid Model:
1. Architecture initialization & checkpoint loading.
2. Prediction output tensor shapes (PM2.5, PM10).
3. Deterministic inference reproducibility.
4. Exact mathematical fallback invariance when visual modality is missing (mask=0).
5. Physical unit range and non-negativity guarantees.
6. Service-level end-to-end integration (HybridPredictionService).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.unified_hybrid.accuracy_optimization.hybrid_model_optimized import OptimizedResidualHybridModel
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.image_estimation_service import ImageEstimationService


class TestOptimizedResidualHybridModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lstm_ckpt = str(ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt")
        cls.opt_ckpt = str(ROOT_DIR / "experiments/unified_hybrid/accuracy_optimization/checkpoints/best_optimized_hybrid_model.pt")
        cls.model = OptimizedResidualHybridModel(
            lstm_checkpoint_path=cls.lstm_ckpt,
            freeze_sensor=True,
            visual_dim=32,
            dropout=0.2,
            max_delta_pm25=25.0,
            max_delta_pm10=40.0,
        )
        if Path(cls.opt_ckpt).exists():
            ckpt = torch.load(cls.opt_ckpt, map_location="cpu")
            cls.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        cls.model.eval()

    def test_01_output_shapes(self):
        """Verify tensor shapes for batched inference."""
        B = 4
        sensor_x = torch.randn(B, 24, 27)
        feats = torch.randn(B, 2048)
        dist = torch.tensor([[0.5], [1.2], [0.8], [2.1]])
        mask = torch.ones(B, 1)

        y_pred, alpha, details = self.model(sensor_x, feats, dist, mask)
        self.assertEqual(y_pred.shape, (B, 2), "Predictions should be shape [B, 2]")
        self.assertEqual(alpha.shape, (B, 1), "Alpha should be shape [B, 1]")
        self.assertEqual(details["delta"].shape, (B, 2), "Delta should be shape [B, 2]")

    def test_02_fallback_invariance(self):
        """Verify that when mask_visual=0, hybrid output matches sensor baseline identically."""
        B = 3
        sensor_x = torch.randn(B, 24, 27)
        feats = torch.randn(B, 2048)
        dist = torch.tensor([[0.3], [0.9], [1.5]])
        mask_zero = torch.zeros(B, 1)

        y_pred, alpha, details = self.model(sensor_x, feats, dist, mask_zero)
        # In fallback mode, alpha must be strictly zero
        self.assertTrue(torch.all(alpha == 0.0).item(), "Fallback alpha must be strictly 0.0")
        # y_pred must equal y_sensor identically
        self.assertTrue(
            torch.allclose(y_pred, details["y_sensor"], atol=1e-5),
            "In fallback mode, prediction must match pure sensor forecast identically",
        )

    def test_03_physical_non_negativity(self):
        """Verify that predictions cannot fall below 0.0 ug/m3."""
        # Test with random inputs including potential negative inputs
        sensor_x = torch.randn(5, 24, 27)
        feats = torch.randn(5, 2048)
        dist = torch.tensor([[0.1], [0.5], [1.0], [1.5], [2.0]])
        mask = torch.ones(5, 1)

        y_pred, _, _ = self.model(sensor_x, feats, dist, mask)
        self.assertTrue(torch.all(y_pred >= 0.0).item(), "Concentrations must be non-negative")

    def test_04_deterministic_inference(self):
        """Verify identical outputs across consecutive inference calls."""
        sensor_x = torch.randn(2, 24, 27)
        feats = torch.randn(2, 2048)
        dist = torch.tensor([[0.6], [1.1]])
        mask = torch.ones(2, 1)

        y1, a1, _ = self.model(sensor_x, feats, dist, mask)
        y2, a2, _ = self.model(sensor_x, feats, dist, mask)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-6), "Model inference must be deterministic")
        self.assertTrue(torch.allclose(a1, a2, atol=1e-6), "Gating alpha must be deterministic")


class TestHybridPredictionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = HybridPredictionService()
        cls.img_service = ImageEstimationService()

    def test_05_service_initialization(self):
        """Verify service loads checkpoint and metadata successfully."""
        self.assertTrue(self.service.is_loaded, "Hybrid prediction service checkpoint should be loaded")
        self.assertEqual(self.service.model_version, "VayuDrishti Optimized Residual Hybrid v2.0")

    def test_06_service_fallback_execution(self):
        """Verify service execution without image input returns sensor fallback."""
        res = self.service.predict_hybrid(station_id="AP001", image_input=None)
        self.assertIn("pm25", res)
        self.assertIn("pm10", res)
        self.assertGreater(res["pm25"], 0.0)
        self.assertGreater(res["pm10"], 0.0)
        self.assertEqual(res["gating_alpha"], 0.0)
        self.assertIn("sensor_telemetry_fallback", res["fallback_status"])
        self.assertFalse(res["modality_availability"]["visual_environment"])
        # Check that metrics report verified test baseline
        self.assertAlmostEqual(res["metrics"]["pm25_mae"], 9.0434, places=2)

    def test_07_service_full_multimodal_execution(self):
        """Verify service execution with curated test image."""
        samples = self.img_service.get_curated_sample_images()
        self.assertGreater(len(samples), 0, "Curated sample images should exist")
        sample_path = samples[0].get("image_path") or samples[0].get("path")

        res = self.service.predict_hybrid(station_id="AP001", image_input=sample_path)
        self.assertIn("pm25", res)
        self.assertIn("pm10", res)
        self.assertGreater(res["pm25"], 0.0)
        self.assertGreater(res["pm10"], 0.0)
        self.assertGreater(res["gating_alpha"], 0.0)
        self.assertEqual(res["fallback_status"], "active_multimodal_gated")
        self.assertTrue(res["modality_availability"]["visual_environment"])
        self.assertAlmostEqual(res["metrics"]["pm25_mae"], 8.9798, places=2)


if __name__ == "__main__":
    unittest.main()
