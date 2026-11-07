"""
src/prediction/image_estimator.py

Production-ready inference service for image-based PM2.5 concentration estimation
using the trained ResNet-50 transfer learning checkpoint.
"""

from pathlib import Path
import io
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from scipy import ndimage


class ResNet50Regressor(nn.Module):
    """ResNet-50 backbone with MLP regression head matching the trained checkpoint."""

    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        base_model = models.resnet50(weights=None)
        modules = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.regressor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return torch.flatten(feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.regressor(feats)


class ImageEstimatorService:
    """Encapsulates image loading, validation, preprocessing, ResNet-50 inference,

    and PM2.5 estimation with scientific disclaimer formatting.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        test_csv_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path or root / "models/pm25vision_cnn/best_model.pt")
        self.test_csv_path = Path(test_csv_path or root / "data/processed/pm25vision_cnn/test.csv")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"ResNet-50 checkpoint not found: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location=self.device)
        self.model = ResNet50Regressor(pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.target_mean = float(ckpt.get("target_mean", 204.1526))
        self.target_std = float(ckpt.get("target_std", 61.3761))
        self.best_epoch = int(ckpt.get("epoch", 8))
        self.test_metrics = ckpt.get("test_metrics", {"MAE": 66.21, "RMSE": 80.68, "R2": 0.152})

    MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 Megabytes
    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

    def validate_and_open_image(
        self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]
    ) -> Image.Image:
        """Validates input type, file format, size, and dimensions, returning a verified RGB PIL Image."""
        try:
            # 1. File size checks for raw bytes and streams
            if isinstance(image_input, bytes):
                if len(image_input) > self.MAX_IMAGE_BYTES:
                    raise ValueError(f"Image size ({len(image_input)/(1024*1024):.1f} MB) exceeds maximum limit of 15 MB.")
                stream = io.BytesIO(image_input)
                img = Image.open(stream)
            elif isinstance(image_input, io.BytesIO):
                image_input.seek(0, io.SEEK_END)
                size_b = image_input.tell()
                image_input.seek(0)
                if size_b > self.MAX_IMAGE_BYTES:
                    raise ValueError(f"Image size ({size_b/(1024*1024):.1f} MB) exceeds maximum limit of 15 MB.")
                img = Image.open(image_input)
            elif isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    raise FileNotFoundError(f"Image file does not exist: {path}")
                if path.stat().st_size > self.MAX_IMAGE_BYTES:
                    raise ValueError(f"Image size ({path.stat().st_size/(1024*1024):.1f} MB) exceeds maximum limit of 15 MB.")
                img = Image.open(path)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")

            # 2. Format validation
            fmt = getattr(img, "format", None)
            if fmt and str(fmt).upper() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {fmt}. Only JPG, JPEG, PNG, and WebP images are supported.")

            # 3. Header verification
            img.verify()

            # Reopen after verify() since verify() closes image pointer
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input)
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, io.BytesIO):
                image_input.seek(0)
                img = Image.open(image_input)

            img = img.convert("RGB")

            # 4. Resolution validation
            if img.width < 16 or img.height < 16:
                raise ValueError(
                    f"Image resolution too small ({img.width}x{img.height}). Minimum is 16x16."
                )

            return img

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError, TypeError)):
                raise
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")

    @staticmethod
    def compute_optical_features(img: Image.Image) -> Dict[str, Any]:
        """
        Calculates authentic physical optical metrics from the image pixels:
        - Mean Brightness: average luma intensity (0–255)
        - Contrast: standard deviation of luminance (0–128)
        - Sharpness / Haze Metric: 2D Laplacian variance on grayscale
        - Saturation: mean color saturation in HSV space (0–100%)
        """
        rgb = np.array(img, dtype=np.float32)
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # 2D Laplacian variance as optical sharpness / edge definition metric
        laplacian = ndimage.laplace(gray)
        sharpness = float(np.var(laplacian))

        # Color saturation in HSV space: (max - min) / max
        max_c = np.max(rgb, axis=2)
        min_c = np.min(rgb, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)
        saturation_pct = float(np.mean(sat) * 100.0)

        if sharpness < 150.0:
            haze_assessment = "High Optical Haze / Low Edge Definition"
            haze_desc = "Diffuse optical scattering, heavy visual softening, low structural gradient definition."
        elif sharpness < 600.0:
            haze_assessment = "Moderate Atmospheric Haze"
            haze_desc = "Partial visual extinction with discernible horizon and skyline features."
        else:
            haze_assessment = "Clear Visibility / Sharp Contours"
            haze_desc = "High contrast edge transitions, crisp horizon definition, minimal atmospheric scattering."

        return {
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness_laplacian": round(sharpness, 1),
            "saturation_pct": round(saturation_pct, 1),
            "haze_assessment": haze_assessment,
            "haze_description": haze_desc,
        }

    def estimate_pm25(
        self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]
    ) -> Dict[str, Any]:
        """Preprocesses the image and runs ResNet-50 inference to predict image-associated PM2.5."""
        img = self.validate_and_open_image(image_input)

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            norm_pred = self.model(tensor).item()

        # Inverse transform to continuous PM2.5 concentration in ug/m3
        raw_pm25 = norm_pred * self.target_std + self.target_mean
        pm25_val = float(np.clip(raw_pm25, 0.0, None))

        # Determine visual AQI severity category (CPCB 24h standard breakpoints)
        if pm25_val <= 30.0:
            category = "Good (0-30 ug/m3)"
            category_telugu = "మంచిది (0–30 µg/m³)"
            severity = "Low"
            cat_color = "#10b981"
            precaution_en = "Air quality poses minimal risk. Enjoy standard outdoor activities."
            precaution_te = "గాలి నాణ్యత చాలా బాగుంది. సాధారణ బహిరంగ కార్యకలాపాలను కొనసాగించవచ్చు."
        elif pm25_val <= 60.0:
            category = "Satisfactory (31-60 ug/m3)"
            category_telugu = "సంతృప్తికరం (31–60 µg/m³)"
            severity = "Mild"
            cat_color = "#65a30d"
            precaution_en = "Acceptable air quality. Unusually sensitive individuals should observe for mild discomfort."
            precaution_te = "గాలి నాణ్యత ఆమోదయోగ్యం. అసాధారణ శ్వాసకోశ సమస్యలు ఉన్నవారు జాగ్రత్తగా ఉండాలి."
        elif pm25_val <= 90.0:
            category = "Moderate (61-90 ug/m3)"
            category_telugu = "మితమైనది (61–90 µg/m³)"
            severity = "Moderate"
            cat_color = "#facc15"
            precaution_en = "Mild breathing irritation possible for individuals with asthma or lung conditions."
            precaution_te = "ఉబ్బసం లేదా ఊపిరితిత్తుల సమస్యలు ఉన్న వ్యక్తులు స్వల్ప అసౌకర్యానికి గురికావచ్చు."
        elif pm25_val <= 120.0:
            category = "Poor (91-120 ug/m3)"
            category_telugu = "పేలవమైనది (91–120 µg/m³)"
            severity = "Unhealthy"
            cat_color = "#f97316"
            precaution_en = "Breathing discomfort to most people on prolonged exposure. Limit intense outdoor exercise."
            precaution_te = "ఎక్కువసేపు బహిరంగ ప్రదేశాల్లో ఉండటం వల్ల శ్వాస తీసుకోవడంలో అసౌకర్యం కలగవచ్చు."
        elif pm25_val <= 250.0:
            category = "Very Poor (121-250 ug/m3)"
            category_telugu = "చాలా పేలవమైనది (121–250 µg/m³)"
            severity = "Severe"
            cat_color = "#ef4444"
            precaution_en = "Respiratory illness on prolonged exposure. Consider wearing an N95 mask outdoors."
            precaution_te = "తీవ్రమైన శ్వాసకోశ సమస్యలు రావచ్చు. N95 మాస్క్‌లను ధరించండి మరియు ఇంట్లోనే ఉండండి."
        else:
            category = "Severe (>250 ug/m3)"
            category_telugu = "తీవ్రమైనది (>250 µg/m³)"
            severity = "Hazardous"
            cat_color = "#991b1b"
            precaution_en = "Hazardous outdoor air pollution. Healthy people may experience symptoms. Avoid strenuous outdoor exposure."
            precaution_te = "ప్రమాదకరమైన వాయు కాలుష్యం. బహిరంగ కార్యకలాపాలను పూర్తిగా నివారించండి."

        # Compute authentic optical features from image
        optical = self.compute_optical_features(img)

        return {
            "estimated_PM25": round(pm25_val, 2),
            "unit": "ug/m3",
            "visual_category": category,
            "visual_category_telugu": category_telugu,
            "category_color": cat_color,
            "severity_level": severity,
            "image_dimensions": f"{img.width}x{img.height} px",
            "model": "ResNet-50 Transfer Learning (Image-Only Baseline)",
            "model_epoch": self.best_epoch,
            "optical_features": optical,
            "plain_english_interpretation": (
                f"The fine-tuned ResNet-50 model analyzes optical extinction and horizon contrast to estimate "
                f"an ambient PM2.5 level of ~{pm25_val:.1f} µg/m³ ({category})."
            ),
            "plain_telugu_interpretation": (
                f"దృశ్యమాన కాంట్రాస్ట్ మరియు వాతావరణ పొగమంచు ఆధారంగా ResNet-50 మోడల్ "
                f"PM2.5 గాఢతను సుమారు {pm25_val:.1f} µg/m³ ({category_telugu}) గా అంచనా వేసింది."
            ),
            "precautionary_guidance_en": precaution_en,
            "precautionary_guidance_te": precaution_te,
            "confidence_statement": (
                "Continuous point regression model. It does not output Bayesian probability intervals. "
                "Empirical test benchmark on held-out test data: MAE 66.21 µg/m³, RMSE 80.68 µg/m³, R² 0.152."
            ),
            "test_benchmark": {
                "test_MAE": 66.21,
                "test_RMSE": 80.68,
                "test_R2": 0.152,
                "test_sample_count": 79,
            },
            "disclaimer": (
                "IMPORTANT SCIENTIFIC NOTICE: This value is an optical estimate derived from "
                "outdoor visual contrast and atmospheric haze features. It is sensitive to lighting, "
                "camera optics, weather, and fog, and is NOT a guaranteed substitute for a "
                "calibrated regulatory CAAQMS ground station."
            ),
        }

    def compare_with_station(
        self,
        estimated_pm25: float,
        station_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Scientifically compares an image-derived optical PM2.5 estimate with a ground station observation.
        Highlights discrepancies, units, physical measurement methods, and temporal/spatial cautions.
        """
        station_name = station_obs.get("station_name", "Unknown Station")
        city_name = station_obs.get("city", "Unknown City")
        distance_km = float(station_obs.get("distance_km", 0.0))
        pollutants = station_obs.get("pollutants", {})
        station_pm25 = pollutants.get("PM2.5")
        station_aqi = station_obs.get("aqi")
        station_aqi_tier = station_obs.get("aqi_tier", "N/A")
        station_time = station_obs.get("observation_time_ist", "N/A")
        provider = station_obs.get("provider", "Regulatory CAAQMS")

        delta = None
        pct_diff = None
        if station_pm25 is not None:
            delta = round(float(estimated_pm25) - float(station_pm25), 2)
            if float(station_pm25) > 0:
                pct_diff = round((delta / float(station_pm25)) * 100.0, 1)

        caution_note = (
            "Scientific Integrity Caution: A photograph captures line-of-sight optical extinction "
            "over a broad visual cone, whereas a CAAQMS ground station measures localized mass concentration "
            "via beta-attenuation or gravimetric absorption. Timing offsets, humidity, and camera exposure "
            "will inherently create discrepancies between optical estimates and ground sensors."
        )

        return {
            "model_estimate_pm25": estimated_pm25,
            "station_pm25": station_pm25,
            "difference_ugm3": delta,
            "percentage_difference": pct_diff,
            "station_name": station_name,
            "city": city_name,
            "distance_km": distance_km,
            "station_aqi": station_aqi,
            "station_aqi_tier": station_aqi_tier,
            "station_time": station_time,
            "provider": provider,
            "caution_note": caution_note,
        }

    def get_curated_sample_images(self, max_samples: int = 10) -> List[Dict[str, Union[str, float]]]:
        """Returns sample images from the held-out test split for interactive dashboard evaluation."""
        if not self.test_csv_path.exists():
            return []

        try:
            df = pd.read_csv(self.test_csv_path)
            samples = []
            for _, row in df.head(max_samples).iterrows():
                p = Path(row["image_path"])
                if p.exists():
                    samples.append({
                        "image_id": str(row["image_id"]),
                        "image_path": str(p),
                        "actual_label": float(row["pm25_label"]),
                        "capture_date": str(row.get("capture_date_clean", "Unknown")),
                        "city": str(row.get("nearest_station_city", "Unknown")),
                        "station_name": str(row.get("nearest_station_name", "Unknown")),
                    })
            return samples
        except Exception:
            return []
