"""
src/services package

Unified service layer for VayuDrishti:
- LiveAirQualityService: actual ground observations
- SensorForecastService: independent LSTM forecasting
- ImageEstimationService: independent visual PM2.5 estimation
- HybridPredictionService: unified multimodal prediction with fallback
- ModelComparisonService: comparative evaluation and side-by-side view
- AutonomousLiveCollector: autonomous hourly live-data collection
- MeteorologicalService: genuine aligned meteorological telemetry
- ConformalPredictionCalibrator: calibrated forecast uncertainty
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "LiveAirQualityService":
        from src.services.live_air_quality_service import LiveAirQualityService
        return LiveAirQualityService
    elif name == "SensorForecastService":
        from src.services.sensor_forecast_service import SensorForecastService
        return SensorForecastService
    elif name == "ImageEstimationService":
        from src.services.image_estimation_service import ImageEstimationService
        return ImageEstimationService
    elif name == "HybridPredictionService":
        from src.services.hybrid_prediction_service import HybridPredictionService
        return HybridPredictionService
    elif name == "ModelComparisonService":
        from src.services.model_comparison_service import ModelComparisonService
        return ModelComparisonService
    elif name == "AutonomousLiveCollector":
        from src.services.autonomous_live_collector import AutonomousLiveCollector
        return AutonomousLiveCollector
    elif name == "MeteorologicalService":
        from src.services.meteorological_service import MeteorologicalService
        return MeteorologicalService
    elif name == "ConformalPredictionCalibrator":
        from src.services.forecast_uncertainty_service import ConformalPredictionCalibrator
        return ConformalPredictionCalibrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LiveAirQualityService",
    "SensorForecastService",
    "ImageEstimationService",
    "HybridPredictionService",
    "ModelComparisonService",
    "AutonomousLiveCollector",
    "MeteorologicalService",
    "ConformalPredictionCalibrator",
]
