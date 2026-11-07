"""
tests/test_environmental_vision.py

Comprehensive test suite for the VayuDrishti AI Environmental Vision system:
- Image format validation (JPG, JPEG, PNG, WebP)
- Rejection of invalid file types and extensions
- Corrupted file and broken header detection
- Maximum file size enforcement (15 MB limit)
- Minimum resolution boundary validation (16x16 px)
- Deep ResNet-50 regression inference, scaling, and unit verification
- Authentic computed optical feature analytics (brightness, contrast, sharpness, saturation)
- English and Telugu bilingual interpretations and precautionary guidance
- Scientific disclaimers and rejection of fabricated confidence percentages
- Clean exception handling when model checkpoints are missing
- Comparative analysis between optical estimates and CPCB ground station observations
- Curated benchmark test sample retrieval
"""

import sys
from pathlib import Path
import io
import unittest
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.image_estimator import ImageEstimatorService, ResNet50Regressor


class TestEnvironmentalVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ImageEstimatorService()

    # 1. Valid Image Formats (JPG, PNG, WebP)
    def test_01_valid_image_formats(self):
        for fmt in ["JPEG", "PNG", "WEBP"]:
            img_byte_arr = io.BytesIO()
            test_img = Image.new("RGB", (128, 128), color=(90, 140, 210))
            test_img.save(img_byte_arr, format=fmt)
            img_byte_arr.seek(0)

            opened = self.service.validate_and_open_image(img_byte_arr)
            self.assertIsInstance(opened, Image.Image)
            self.assertEqual(opened.mode, "RGB")
            self.assertEqual(opened.size, (128, 128))

    # 2. Rejection of Invalid File Types
    def test_02_invalid_file_types(self):
        # Text bytes instead of image
        fake_text = b"This is a text file purporting to be an image."
        with self.assertRaises(ValueError):
            self.service.validate_and_open_image(fake_text)

        # BMP format (not in supported set JPEG, PNG, WEBP)
        bmp_io = io.BytesIO()
        test_img = Image.new("RGB", (64, 64), color=(50, 100, 150))
        test_img.save(bmp_io, format="BMP")
        bmp_io.seek(0)
        with self.assertRaises(ValueError) as ctx:
            self.service.validate_and_open_image(bmp_io)
        self.assertIn("Unsupported image format", str(ctx.exception))

    # 3. Corrupt File Handling
    def test_03_corrupt_images(self):
        # Truncated JPEG stream
        corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01" + b"\x00" * 20
        with self.assertRaises(ValueError) as ctx:
            self.service.validate_and_open_image(corrupt_bytes)
        self.assertIn("Invalid or corrupted image file", str(ctx.exception))

    # 4. Oversized Image Enforcement (15 MB Limit)
    def test_04_oversized_images(self):
        # Simulate byte buffer exceeding 15 MB
        oversized_bytes = b"0" * (15 * 1024 * 1024 + 1024)
        with self.assertRaises(ValueError) as ctx:
            self.service.validate_and_open_image(oversized_bytes)
        self.assertIn("exceeds maximum limit of 15 MB", str(ctx.exception))

    # 5. Minimum Resolution Boundary Validation (< 16x16)
    def test_05_minimum_resolution(self):
        tiny_img = Image.new("RGB", (12, 12), color=(200, 100, 50))
        with self.assertRaises(ValueError) as ctx:
            self.service.validate_and_open_image(tiny_img)
        self.assertIn("Image resolution too small", str(ctx.exception))

    # 6. Deep ResNet-50 Inference, Scaling, and Units
    def test_06_successful_inference_and_units(self):
        sample_img = Image.new("RGB", (224, 224), color=(140, 180, 220))
        res = self.service.estimate_pm25(sample_img)

        self.assertIn("estimated_PM25", res)
        self.assertTrue(np.isfinite(res["estimated_PM25"]))
        self.assertGreaterEqual(res["estimated_PM25"], 0.0)
        self.assertEqual(res["unit"], "ug/m3")
        self.assertIn("visual_category", res)
        self.assertIn("severity_level", res)
        self.assertIn("category_color", res)
        self.assertNotIn("official_aqi", res)  # Must never report arbitrary AQI from an image!

    # 7. Authentic Computed Optical Features (Actually Calculated)
    def test_07_computed_optical_features(self):
        # Create an image with known contrast and edge gradients
        gradient_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        gradient_arr[:, :50] = [30, 60, 90]
        gradient_arr[:, 50:] = [220, 230, 240]
        grad_img = Image.fromarray(gradient_arr, mode="RGB")

        features = self.service.compute_optical_features(grad_img)
        self.assertIn("brightness", features)
        self.assertIn("contrast", features)
        self.assertIn("sharpness_laplacian", features)
        self.assertIn("saturation_pct", features)
        self.assertIn("haze_assessment", features)

        # Contrast and sharpness should be positive non-zero for gradient image
        self.assertGreater(features["contrast"], 0.0)
        self.assertGreater(features["sharpness_laplacian"], 0.0)
        self.assertTrue(0.0 <= features["brightness"] <= 255.0)
        self.assertTrue(0.0 <= features["saturation_pct"] <= 100.0)

    # 8. English & Telugu Bilingual Interpretations
    def test_08_bilingual_interpretations(self):
        test_img = Image.new("RGB", (224, 224), color=(100, 120, 160))
        res = self.service.estimate_pm25(test_img)

        self.assertIn("plain_english_interpretation", res)
        self.assertIn("plain_telugu_interpretation", res)
        self.assertIn("precautionary_guidance_en", res)
        self.assertIn("precautionary_guidance_te", res)

        # Check non-empty strings and Telugu characters present in Telugu string
        self.assertTrue(len(res["plain_english_interpretation"]) > 10)
        self.assertTrue(any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in res["plain_telugu_interpretation"]))
        self.assertTrue(any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in res["precautionary_guidance_te"]))

    # 9. Scientific Disclaimer & No Fabricated Confidence Percentages
    def test_09_scientific_disclaimer_and_no_fake_confidence(self):
        test_img = Image.new("RGB", (224, 224), color=(80, 110, 150))
        res = self.service.estimate_pm25(test_img)

        # Disclaimer must be present and explicit
        self.assertIn("disclaimer", res)
        self.assertIn("optical estimate", res["disclaimer"].lower())
        self.assertIn("caaqms", res["disclaimer"].lower())

        # Confidence statement must acknowledge continuous point regression limitations
        self.assertIn("confidence_statement", res)
        self.assertIn("point regression", res["confidence_statement"].lower())
        self.assertNotIn("confidence_percent", res)  # No fake confidence percentages!

    # 10. Missing Checkpoint Error Handling
    def test_10_missing_checkpoint_handling(self):
        with self.assertRaises(FileNotFoundError):
            ImageEstimatorService(model_path="non_existent_resnet_model.pt")

    # 11. Comparison with Ground Station Readings
    def test_11_compare_with_station_readings(self):
        estimated_pm25 = 75.4
        mock_station_obs = {
            "station_name": "GVM Corporation, Visakhapatnam - APPCB",
            "city": "Visakhapatnam",
            "distance_km": 4.2,
            "pollutants": {"PM2.5": 48.0, "PM10": 92.0},
            "aqi": 85,
            "aqi_tier": "Satisfactory",
            "observation_time_ist": "2026-10-15 10:00 AM IST",
            "provider": "OpenAQ API v3",
        }

        comparison = self.service.compare_with_station(estimated_pm25, mock_station_obs)
        self.assertEqual(comparison["model_estimate_pm25"], 75.4)
        self.assertEqual(comparison["station_pm25"], 48.0)
        self.assertEqual(comparison["difference_ugm3"], 27.4)
        self.assertEqual(comparison["percentage_difference"], 57.1)
        self.assertEqual(comparison["station_name"], "GVM Corporation, Visakhapatnam - APPCB")
        self.assertIn("caution_note", comparison)

    # 12. Curated Test Benchmark Samples Loading
    def test_12_curated_samples_loading(self):
        samples = self.service.get_curated_sample_images(max_samples=5)
        self.assertIsInstance(samples, list)
        self.assertTrue(len(samples) > 0)
        for s in samples:
            self.assertIn("image_id", s)
            self.assertIn("actual_label", s)
            self.assertIn("image_path", s)
            self.assertTrue(Path(s["image_path"]).exists())

    # 13. Truthful Handling When Station Measurement is Missing (No Invention of Values)
    def test_13_truthful_missing_station_observation(self):
        estimated_pm25 = 62.3
        # Station where PM2.5 sensor is offline or not reported
        mock_station_missing_pm25 = {
            "station_name": "Tirupati Monitoring Point - APPCB",
            "city": "Tirupati",
            "distance_km": 12.8,
            "pollutants": {"NO2": 24.0, "SO2": 8.0},  # No PM2.5!
            "observation_time_ist": "2026-10-16 08:30 PM IST",
            "provider": "CPCB / APPCB",
        }

        comparison = self.service.compare_with_station(estimated_pm25, mock_station_missing_pm25)
        self.assertEqual(comparison["model_estimate_pm25"], 62.3)
        self.assertIsNone(comparison["station_pm25"])
        self.assertIsNone(comparison["difference_ugm3"])  # Must NEVER invent a fake difference!
        self.assertIsNone(comparison["percentage_difference"])
        self.assertEqual(comparison["station_name"], "Tirupati Monitoring Point - APPCB")
        self.assertEqual(comparison["distance_km"], 12.8)


if __name__ == "__main__":
    unittest.main()
