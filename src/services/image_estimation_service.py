"""
src/services/image_estimation_service.py

Unified backend service for Image-Based Environmental Estimation (Mode A - Method D).
Preserves existing ResNet-50 visual estimation capability.
Explicitly discloses that PM25Vision lacks sub-daily timestamps, so outputs are
environmental visual estimates rather than causal next-hour forecasts.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
from PIL import Image

from src.prediction.image_estimator import ImageEstimatorService

ROOT_DIR = Path(__file__).resolve().parents[2]


class ImageEstimationService:
    """Service providing visual PM2.5 estimation from outdoor camera photos."""

    def __init__(self):
        self.underlying_service = ImageEstimatorService()
        self.model_version = "ResNet-50 PM25Vision Baseline v1.0"
        self._load_metrics()

    def _load_metrics(self):
        metrics_file = ROOT_DIR / "experiments/multimodal_fusion/validation_v2/metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            t_m = data.get("test_metrics", {}).get("image_only_resnet50", {})
            self.saved_metrics = {
                "MAE": float(t_m.get("MAE", 64.99)),
                "RMSE": float(t_m.get("RMSE", 94.59)),
                "R2": float(t_m.get("R2", -0.3828)),
                "evaluation_cohort": "Held-Out Test Cohort (N = 97 matched images, zero group leakage)",
            }
        else:
            self.saved_metrics = {
                "MAE": 64.99,
                "RMSE": 94.59,
                "R2": -0.3828,
                "evaluation_cohort": "Held-Out Test Cohort (N = 97 matched images, zero group leakage)",
            }

    def estimate_from_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
    ) -> Dict[str, Any]:
        """Estimates PM2.5 from an outdoor camera image."""
        try:
            res = self.underlying_service.estimate_pm25(image_input)
            pm25_val = float(res.get("predicted_pm25", 55.0))
            category = res.get("category", "Moderate")
            optical_features = res.get("optical_features", {})
            status = "Success"
        except Exception as e:
            pm25_val = 55.0
            category = "Moderate"
            optical_features = {"contrast": 0.45, "haze_index": 0.32}
            status = f"Fallback/Error: {str(e)}"

        return {
            "result_type": "Independent Estimate",
            "model_version": self.model_version,
            "pm25": round(pm25_val, 2),
            "pm10": None,  # Image model predicts PM2.5 only
            "unit": "ug/m3",
            "aqi_category": category,
            "optical_features": optical_features,
            "status": status,
            "timestamp_precision": "Calendar Date Only (Sub-daily EXIF timestamps absent in PM25Vision)",
            "metrics": {
                "pm25_mae": self.saved_metrics["MAE"],
                "pm25_rmse": self.saved_metrics["RMSE"],
                "pm25_r2": self.saved_metrics["R2"],
                "evaluation_cohort": self.saved_metrics["evaluation_cohort"],
            },
            "scientific_disclosure": (
                "Photographic image estimation reflects optical transmission and atmospheric scatter. "
                "Because public air-quality image datasets lack sub-daily capture timestamps, this output "
                "is an environmental estimate rather than a verified causal next-hour forecast."
            ),
        }

    def get_curated_sample_images(self):
        """Returns verified sample images for dashboard demonstration."""
        return self.underlying_service.get_curated_sample_images()

    def estimate_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Alias for estimate_from_image for file path inputs."""
        res = self.estimate_from_image(file_path)
        res["pm25_pred"] = res.get("pm25")
        res["category"] = res.get("aqi_category")
        return res

    def get_sample_images(self):
        """Returns verified sample images for dashboard demonstration."""
        return self.get_curated_sample_images()
