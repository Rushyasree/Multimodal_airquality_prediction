"""
src/services/model_comparison_service.py

Unified backend service for Side-by-Side Model Comparison (Section 4).
Compares:
1. Live Air Quality Data (Observation)
2. Sensor-Only LSTM (Independent Forecast)
3. Image-Only Model (Independent Estimate)
4. Weather Context (Meteorological Context)
5. Unified Hybrid Model (Integrated Multimodal Prediction)
6. Persistence Baseline (Reference Comparison)
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from PIL import Image

from src.services.live_air_quality_service import LiveAirQualityService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.image_estimation_service import ImageEstimationService
from src.services.hybrid_prediction_service import HybridPredictionService

ROOT_DIR = Path(__file__).resolve().parents[2]


class ModelComparisonService:
    """Service generating side-by-side comparative views across all individual and hybrid methods."""

    def __init__(self):
        self.live_service = LiveAirQualityService()
        self.sensor_service = SensorForecastService()
        self.image_service = ImageEstimationService()
        self.hybrid_service = HybridPredictionService()

    def compare_all(
        self,
        station_id: str = "AP001",
        city: str = "Amaravati",
        df_24h: Optional[pd.DataFrame] = None,
        image_input: Optional[Union[str, Path, bytes, Image.Image]] = None,
    ) -> Dict[str, Any]:
        """Runs and aggregates all prediction methods and live observations side-by-side."""

        # 1. Live observation
        live_res = self.live_service.get_live_observation(station_id=station_id, city=city)

        # 2. Sensor-only LSTM
        sensor_res = self.sensor_service.get_forecast(station_id=station_id, df_24h=df_24h)

        # 3. Image-only ResNet-50
        if image_input is not None:
            image_res = self.image_service.estimate_from_image(image_input)
        else:
            # Use sample curated image if none provided
            samples = self.image_service.get_curated_sample_images()
            if samples:
                img_p = samples[0].get("image_path") or samples[0].get("path")
                image_res = self.image_service.estimate_from_image(img_p)
            else:
                image_res = {
                    "result_type": "Independent Estimate",
                    "model_version": "ResNet-50 PM25Vision Baseline v1.0",
                    "pm25": 55.0,
                    "pm10": None,
                    "unit": "ug/m3",
                    "timestamp_precision": "Calendar Date Only (Sub-daily EXIF hours absent in PM25Vision)",
                    "status": "Simulated Sample",
                    "metrics": {"pm25_mae": 60.99, "pm25_rmse": 88.20, "pm25_r2": -0.2023},
                }

        # 4. Weather Context
        weather_dict = live_res.get("weather", {})
        temp = weather_dict.get("temperature", 28.5)
        hum = weather_dict.get("humidity", 65.0)
        wind = weather_dict.get("wind_speed", 2.8)

        # 5. Unified Hybrid Model
        hybrid_res = self.hybrid_service.predict_hybrid(
            station_id=station_id,
            df_24h=df_24h,
            image_input=image_input,
        )

        # Build comparison summary list
        rows = [
            {
                "Method": "Live Ground Station",
                "Result Type": "Actual Observation",
                "PM2.5 (ug/m3)": live_res.get("pm25"),
                "PM10 (ug/m3)": live_res.get("pm10"),
                "Timestamp / Horizon": f"Observed: {live_res.get('observation_time', 'N/A')}",
                "Data Source": live_res.get("data_source", "CPCB / OpenAQ"),
                "Model Version": "N/A (Physical Instrument)",
                "Input Freshness": live_res.get("freshness_status", "Fresh"),
                "Modality Availability": "Ground Station Telemetry",
                "Reported MAE (ug/m3)": "N/A (Ground Truth)",
                "Reported R2": "N/A",
            },
            {
                "Method": "Sensor-Only LSTM",
                "Result Type": "Independent Forecast",
                "PM2.5 (ug/m3)": sensor_res.get("pm25"),
                "PM10 (ug/m3)": sensor_res.get("pm10"),
                "Timestamp / Horizon": f"Target: {sensor_res.get('target_time')} (t+1)",
                "Data Source": "24h Continuous Telemetry",
                "Model Version": sensor_res.get("model_version"),
                "Input Freshness": sensor_res.get("input_freshness"),
                "Modality Availability": "Sensor Sequence (27 features)",
                "Reported MAE (ug/m3)": f"PM2.5: {sensor_res['metrics']['pm25_mae']:.1f} | PM10: {sensor_res['metrics']['pm10_mae']:.1f}",
                "Reported R2": f"PM2.5: {sensor_res['metrics']['pm25_r2']:.4f} | PM10: {sensor_res['metrics']['pm10_r2']:.4f}",
            },
            {
                "Method": "Image-Only ResNet-50",
                "Result Type": "Independent Estimate",
                "PM2.5 (ug/m3)": image_res.get("pm25"),
                "PM10 (ug/m3)": "Not Supported",
                "Timestamp / Horizon": "Optical Snapshot (Calendar Date)",
                "Data Source": "Outdoor Camera Photo",
                "Model Version": image_res.get("model_version"),
                "Input Freshness": image_res.get("timestamp_precision"),
                "Modality Availability": "Single RGB Image",
                "Reported MAE (ug/m3)": f"PM2.5: {image_res['metrics']['pm25_mae']:.1f}",
                "Reported R2": f"PM2.5: {image_res['metrics']['pm25_r2']:.4f}",
            },
            {
                "Method": "Weather / Context",
                "Result Type": "Meteorological Context",
                "PM2.5 (ug/m3)": "N/A (Weather Only)",
                "PM10 (ug/m3)": "N/A (Weather Only)",
                "Timestamp / Horizon": "Current Ambient Weather",
                "Data Source": "Open-Meteo / Ground Weather Sensor",
                "Model Version": "Atmospheric Observation",
                "Input Freshness": "Current Weather Telemetry",
                "Modality Availability": f"T: {temp}°C | RH: {hum}% | Wind: {wind} m/s",
                "Reported MAE (ug/m3)": "N/A (Contextual Variable)",
                "Reported R2": "N/A",
            },
            {
                "Method": "Unified Hybrid Model",
                "Result Type": "Integrated Multimodal Prediction",
                "PM2.5 (ug/m3)": hybrid_res.get("pm25"),
                "PM10 (ug/m3)": hybrid_res.get("pm10"),
                "Timestamp / Horizon": f"Target: {hybrid_res.get('target_time')} (t+1)",
                "Data Source": "Fused Telemetry + Vision",
                "Model Version": hybrid_res.get("model_version"),
                "Input Freshness": hybrid_res.get("input_freshness"),
                "Modality Availability": (
                    f"Sensor: Yes | Weather: Yes | Image: {'Yes' if hybrid_res['modality_availability']['visual_environment'] else 'Fallback'}"
                ),
                "Reported MAE (ug/m3)": f"PM2.5: {hybrid_res['metrics']['pm25_mae']:.1f} | PM10: {hybrid_res['metrics']['pm10_mae']:.1f}",
                "Reported R2": f"PM2.5: {hybrid_res['metrics']['pm25_r2']:.4f} | PM10: {hybrid_res['metrics']['pm10_r2']:.4f}",
            },
        ]

        df_comparison = pd.DataFrame(rows)

        return {
            "comparison_table": df_comparison,
            "live_data": live_res,
            "sensor_forecast": sensor_res,
            "image_estimate": image_res,
            "hybrid_prediction": hybrid_res,
            "weather_context": {
                "temperature": temp,
                "humidity": hum,
                "wind_speed": wind,
            },
            "scientific_takeaway": (
                "Live air-quality measurements represent ground-truth physical observations. "
                "Sensor-only LSTM provides high-accuracy next-hour continuous forecasting (PM2.5 R2 = 0.949 on 404k sequences). "
                "The Unified Hybrid Model integrates sensor, weather, and visual features with learned gating and graceful fallback. "
                "Because public camera datasets lack sub-daily timestamps, sensor-only telemetry remains superior to image-fused outputs."
            ),
        }
