"""Prediction and inference modules for numerical forecasting, image-based estimation,
CPCB AQI calculations, and live service safety.
"""

from .aqi_advisor import (
    calculate_sub_index,
    calculate_overall_cpcb_aqi,
    get_aqi_category,
    BREAKPOINTS,
    AQI_CATEGORIES,
)
from .live_service import LiveAirQualityService
from .current_air_quality_service import CurrentAirQualityService, validate_coordinates

__all__ = [
    "LSTMForecasterService",
    "LSTMForecaster",
    "prepare_forecast_display_table",
    "ImageEstimatorService",
    "ResNet50Regressor",
    "calculate_sub_index",
    "calculate_overall_cpcb_aqi",
    "get_aqi_category",
    "BREAKPOINTS",
    "AQI_CATEGORIES",
    "LiveAirQualityService",
    "CurrentAirQualityService",
    "validate_coordinates",
]

_LAZY_MODULES = {
    "LSTMForecasterService": ".lstm_forecaster",
    "LSTMForecaster": ".lstm_forecaster",
    "prepare_forecast_display_table": ".lstm_forecaster",
    "ImageEstimatorService": ".image_estimator",
    "ResNet50Regressor": ".image_estimator",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
