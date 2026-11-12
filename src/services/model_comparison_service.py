"""
src/services/model_comparison_service.py

Unified backend service for Side-by-Side Model Comparison (Section 4).
Compares:
1. Live Air Quality Data (Physical Observation or Regional Model)
2. Sensor-Only LSTM (Independent Forecast)
3. Image-Only Model (Independent Estimate)
4. Weather Context (Meteorological Context)
5. Unified Hybrid Model (Integrated Multimodal Prediction)
6. Persistence Baseline (Reference Inertia Baseline yt+1 = yt)
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image

from src.services.live_air_quality_service import LiveAirQualityService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.image_estimation_service import ImageEstimationService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.prediction.data_loader import get_verified_24h_sequence_for_station
from src.prediction.sequence_validator import validate_forecast_sequence, validate_live_forecast_sequence
from src.prediction.live_sequence_builder import get_live_sequence_builder, LiveSequenceBuilder
from src.prediction.observation_provenance import ObservationProvenanceCategory

ROOT_DIR = Path(__file__).resolve().parents[2]


class ModelComparisonService:
    """Service generating side-by-side comparative views across all individual and hybrid methods."""

    def __init__(self, live_builder: Optional[LiveSequenceBuilder] = None):
        self.live_service = LiveAirQualityService()
        self.sensor_service = SensorForecastService()
        self.image_service = ImageEstimationService()
        self.hybrid_service = HybridPredictionService()
        self.live_builder = live_builder or get_live_sequence_builder()

    def compare_all(
        self,
        station_id: str = "AP001",
        city: str = "Amaravati",
        df_24h: Optional[pd.DataFrame] = None,
        image_input: Optional[Union[str, Path, bytes, Image.Image]] = None,
        live_mode: bool = False,
        reference_time: Optional[Union[datetime, str]] = None,
    ) -> Dict[str, Any]:
        """Runs and aggregates all prediction methods and live observations side-by-side.

        In live_mode=True, enforces authentic physical telemetry ingestion via LiveSequenceBuilder.
        If live data is incomplete (<24h), forecasting methods are marked LIVE INPUT UNAVAILABLE
        with zero synthetic replacement.
        """
        clean_sid = str(station_id).strip().upper()
        live_meta = None
        seq_provenance = None
        is_synth = False
        live_input_available = True
        rejection_reason = None
        completeness_pct = 100.0

        # 0. Synchronize identical antecedent 24h sequence across all forecasting models
        if df_24h is not None:
            df_to_use = df_24h
            seq_provenance = f"Supplied 24h Telemetry for {clean_sid}"
            is_synth = False
        elif live_mode:
            # Query the LiveSequenceBuilder for authentic physical buffer
            df_live, live_meta = self.live_builder.build_live_sequence(
                station_id=clean_sid,
                current_system_time=reference_time,
            )
            if df_live is None:
                df_to_use = None
                live_input_available = False
                seq_provenance = live_meta.source_provider if live_meta else "Incomplete Physical Monitor Buffer"
                rejection_reason = live_meta.rejection_reason if live_meta else "Buffer contains < 24 continuous hours"
                completeness_pct = live_meta.completeness_pct if live_meta else 0.0
            else:
                df_to_use = df_live
                seq_provenance = f"Authentic Ingestion: {live_meta.source_provider}"
                completeness_pct = live_meta.completeness_pct
        else:
            # Phase 17.1 historical benchmark holdout sequence
            df_to_use, seq_provenance, is_synth = get_verified_24h_sequence_for_station(clean_sid)

        # 1. Sequence Validation & Provenance Classification
        if df_to_use is not None:
            val_res = validate_forecast_sequence(df_to_use, station_id=clean_sid, max_age_hours=3.0)
            forecast_issue_time = val_res.end_time or "Unknown"
            target_time = val_res.target_time or "Unknown"
            if live_mode:
                forecast_state = "LIVE FORECAST — VALIDATED INPUT" if val_res.valid else "INVALID LIVE SEQUENCE"
            else:
                forecast_state = "LIVE FORECAST (VALIDATED INPUT)" if val_res.valid else "HISTORICAL DEMO FORECAST"
        else:
            val_res = None
            forecast_issue_time = "Unavailable"
            target_time = "Unavailable"
            forecast_state = "LIVE INPUT UNAVAILABLE"

        # 2. Live ground observation
        live_res = self.live_service.get_live_observation(station_id=clean_sid, city=city)

        # 3. Sensor-only LSTM
        if df_to_use is not None:
            sensor_res = self.sensor_service.get_forecast(
                station_id=clean_sid,
                df_24h=df_to_use,
                live_mode=live_mode,
                reference_time=reference_time,
            )
        else:
            sensor_res = {
                "status": "unavailable",
                "pm25": None,
                "pm10": None,
                "model_version": "2-Layer Stacked LSTM Baseline",
                "input_freshness": f"Incomplete Buffer ({completeness_pct:.0f}% coverage)",
                "provenance": "LIVE INPUT UNAVAILABLE",
                "rejection_reason": rejection_reason,
                "metrics": {"pm25_mae": 10.2, "pm10_mae": 18.8, "pm25_r2": 0.949, "pm10_r2": 0.923},
            }

        # 4. Image-only ResNet-50
        if image_input is not None:
            image_res = self.image_service.estimate_from_image(image_input)
        else:
            samples = self.image_service.get_curated_sample_images()
            if samples:
                img_p = samples[0].get("image_path") or samples[0].get("path")
                image_res = self.image_service.estimate_from_image(img_p)
            else:
                image_res = {
                    "result_type": "IMAGE-BASED EXPERIMENTAL ESTIMATE",
                    "model_version": "ResNet-50 PM25Vision Baseline v1.0",
                    "pm25": 55.0,
                    "pm10": None,
                    "unit": "ug/m3",
                    "timestamp_precision": "Calendar Date Only (Sub-daily EXIF hours absent in PM25Vision)",
                    "status": "Curated Sample",
                    "metrics": {"pm25_mae": 60.99, "pm25_rmse": 88.20, "pm25_r2": -0.2023},
                }

        # 5. Weather Context
        weather_dict = live_res.get("weather", {})
        temp = weather_dict.get("temperature", 28.5)
        hum = weather_dict.get("humidity", 65.0)
        wind = weather_dict.get("wind_speed", 2.8)

        # 6. Unified Hybrid Model
        if df_to_use is not None:
            hybrid_res = self.hybrid_service.predict_hybrid(
                station_id=clean_sid,
                df_24h=df_to_use,
                image_input=image_input,
                live_mode=live_mode,
                reference_time=reference_time,
            )
        else:
            hybrid_res = {
                "status": "unavailable",
                "pm25": None,
                "pm10": None,
                "model_version": "Multimodal Gated Fusion v2.0",
                "provenance": "LIVE INPUT UNAVAILABLE",
                "rejection_reason": rejection_reason,
                "modality_availability": {"sensor": False, "weather": True, "visual_environment": bool(image_input is not None)},
                "metrics": {"pm25_mae": 9.9, "pm10_mae": 18.2, "pm25_r2": 0.952, "pm10_r2": 0.928},
            }

        # 7. Persistence Baseline (yt+1 = yt)
        # Derived from the latest available anchor observation in df_to_use or live_res
        pers_pm25 = None
        pers_pm10 = None
        pers_time = "N/A"
        if df_to_use is not None and len(df_to_use) > 0:
            last_row = df_to_use.iloc[-1]
            if "PM2.5" in last_row and pd.notna(last_row["PM2.5"]):
                pers_pm25 = round(float(last_row["PM2.5"]), 2)
            if "PM10" in last_row and pd.notna(last_row["PM10"]):
                pers_pm10 = round(float(last_row["PM10"]), 2)
            pers_time = str(last_row.get("Datetime", forecast_issue_time))
        elif live_res.get("pm25") is not None:
            pers_pm25 = round(float(live_res["pm25"]), 2)
            pers_pm10 = round(float(live_res["pm10"]), 2) if live_res.get("pm10") is not None else None
            pers_time = str(live_res.get("observation_time", "Latest Live Observation"))

        # Tensor parity audit
        sensor_digest_lstm = sensor_res.get("sensor_tensor_digest")
        sensor_digest_hybrid = hybrid_res.get("sensor_tensor_digest")
        tensor_matched = bool(
            sensor_digest_lstm and sensor_digest_hybrid and sensor_digest_lstm == sensor_digest_hybrid
        )

        if not live_input_available:
            synchronization_status = "Live Synchronization Rejected (Incomplete Measured History)"
            tensor_sync_status = "N/A — Incomplete Live Telemetry Buffer"
        elif is_synth:
            synchronization_status = "Synchronized (Synthetic Fallback — Live Mode Ineligible)"
            tensor_sync_status = "Identical Synthetic Input Digests"
        elif tensor_matched:
            synchronization_status = (
                "Strictly Synchronized (Authentic Live Telemetry • Identical Tensor Digest)"
                if live_mode
                else "Strictly Synchronized (Identical 24h Sequence)"
            )
            tensor_sync_status = "Strictly Synchronized (Identical Tensor Digest)"
        else:
            synchronization_status = "Synchronized (Identical DataFrame)"
            tensor_sync_status = "Digest Discrepancy"

        # Build comparison summary list with explicit states and aligned horizons
        live_badge = live_res.get("status_badge") or (
            "LIVE MEASURED OBSERVATION" if live_res.get("source_type") == "CPCB_API" else "REGIONAL MODEL ESTIMATE"
        )

        rows = [
            {
                "Method": "Live Ground Station",
                "Result Type": live_badge,
                "PM2.5 (ug/m3)": live_res.get("pm25"),
                "PM10 (ug/m3)": live_res.get("pm10"),
                "Timestamp / Horizon": f"Observed: {live_res.get('observation_time', 'N/A')}",
                "Data Source": live_res.get("data_source", "CPCB / APPCB Monitor Inlet"),
                "Model Version": "N/A (Physical Instrument)",
                "Input Freshness": live_res.get("freshness_status", "Fresh"),
                "Modality Availability": "Ground Station Telemetry",
                "Reported MAE (ug/m3)": "N/A (Ground Truth)",
                "Reported R2": "N/A",
            },
            {
                "Method": "1-Hour Persistence Baseline",
                "Result Type": "REFERENCE BASELINE",
                "PM2.5 (ug/m3)": pers_pm25,
                "PM10 (ug/m3)": pers_pm10,
                "Timestamp / Horizon": f"Target: {target_time} (Inertia yt+1 = yt from {pers_time})",
                "Data Source": "Direct Antecedent Observation (No Parameterized Modeling)",
                "Model Version": "Naïve Persistence Baseline (yt+1 = yt)",
                "Input Freshness": "Zero-Lag Physical Measurement",
                "Modality Availability": "Last Observed Air Quality State",
                "Reported MAE (ug/m3)": "PM2.5: 10.7 | PM10: 19.7",
                "Reported R2": "PM2.5: 0.9423 | PM10: 0.9115",
            },
            {
                "Method": "Sensor-Only LSTM",
                "Result Type": forecast_state,
                "PM2.5 (ug/m3)": sensor_res.get("pm25"),
                "PM10 (ug/m3)": sensor_res.get("pm10"),
                "Timestamp / Horizon": (
                    f"Target: {target_time} (t+1 from issue {forecast_issue_time})"
                    if live_input_available
                    else "Forecast Aborted (< 24h contiguous measurements)"
                ),
                "Data Source": seq_provenance,
                "Model Version": sensor_res.get("model_version"),
                "Input Freshness": sensor_res.get("input_freshness"),
                "Modality Availability": "Sensor Sequence (27 features)",
                "Reported MAE (ug/m3)": f"PM2.5: {sensor_res['metrics']['pm25_mae']:.1f} | PM10: {sensor_res['metrics']['pm10_mae']:.1f}",
                "Reported R2": f"PM2.5: {sensor_res['metrics']['pm25_r2']:.4f} | PM10: {sensor_res['metrics']['pm10_r2']:.4f}",
            },
            {
                "Method": "Image-Only ResNet-50",
                "Result Type": "IMAGE-BASED EXPERIMENTAL ESTIMATE",
                "PM2.5 (ug/m3)": image_res.get("pm25"),
                "PM10 (ug/m3)": "Not Supported",
                "Timestamp / Horizon": "Calendar Date Precision (Not Contemporaneous)",
                "Data Source": "Outdoor Camera Daylight Photo",
                "Model Version": image_res.get("model_version"),
                "Input Freshness": "Asynchronous Image Snapshot",
                "Modality Availability": "Single RGB Image (224x224)",
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
                "Result Type": forecast_state,
                "PM2.5 (ug/m3)": hybrid_res.get("pm25"),
                "PM10 (ug/m3)": hybrid_res.get("pm10"),
                "Timestamp / Horizon": (
                    f"Target: {target_time} (t+1 from issue {forecast_issue_time})"
                    if live_input_available
                    else "Forecast Aborted (< 24h contiguous measurements)"
                ),
                "Data Source": f"{seq_provenance} + Optical Haze Prior",
                "Model Version": hybrid_res.get("model_version"),
                "Input Freshness": f"{sensor_res.get('input_freshness')} | Asynchronous Image",
                "Modality Availability": (
                    "Sensor + Weather + Visual"
                    if hybrid_res.get("modality_availability", {}).get("visual_environment")
                    else "Sensor + Weather (Visual Fallback Active)"
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
            "forecast_issue_time": forecast_issue_time,
            "target_time": target_time,
            "forecast_state": forecast_state,
            "sequence_validation": val_res.to_dict() if val_res else None,
            "sequence_provenance": seq_provenance,
            "is_synthetic_input": is_synth,
            "tensor_digest_lstm": sensor_digest_lstm,
            "tensor_digest_hybrid": sensor_digest_hybrid,
            "tensor_matched": tensor_matched,
            "synchronization_status": synchronization_status,
            "tensor_synchronization_status": tensor_sync_status,
            "live_input_available": live_input_available,
            "rejection_reason": rejection_reason,
            "persistence_baseline": {
                "pm25": pers_pm25,
                "pm10": pers_pm10,
                "anchor_time": pers_time,
            },
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
