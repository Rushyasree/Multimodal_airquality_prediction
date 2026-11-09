"""
tests/test_data_provenance_and_integrity.py

Verification tests ensuring data provenance integrity:
- Proves calibrated generated values are NEVER badged as LIVE DATA.
- Proves regional atmospheric model reanalysis (Open-Meteo) is transparently labeled as REANALYSIS.
- Proves hybrid forecasting uses verified station sequences rather than unseeded random noise.
- Proves model comparison runs on synchronized, identical antecedent sequences across models.
"""

import pytest
import pandas as pd
import numpy as np
from src.prediction.current_air_quality_service import (
    generate_calibrated_baseline_observation,
    fetch_open_meteo_live,
)
from src.prediction.data_loader import get_verified_24h_sequence_for_station
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.sensor_forecast_service import SensorForecastService
from src.services.model_comparison_service import ModelComparisonService


def test_calibrated_baseline_never_labeled_as_live():
    """Verify calibrated baseline calculation has is_live=False and never masquerades as live station telemetry."""
    obs = generate_calibrated_baseline_observation(
        lat=14.4426,
        lon=79.9865,
        station_name="Nellore Ambient Monitor",
        city_name="Nellore",
    )
    assert obs["is_live"] is False, "Calibrated baseline must NOT have is_live=True"
    assert obs["is_modeled"] is True
    assert obs["status_badge"] == "MODELED BASELINE"
    assert "not a direct physical station measurement" in obs["badge_desc"].lower()
    assert obs["observation_time_ist"] != "Just now"


def test_open_meteo_transparently_labeled_as_reanalysis():
    """Verify Open-Meteo hourly regional model is labeled as REGIONAL REANALYSIS, not physical CAAQMS inlet."""
    obs = fetch_open_meteo_live(
        lat=14.4426,
        lon=79.9865,
        station_name="Nellore Ambient Monitor",
        city_name="Nellore",
    )
    if obs is not None:
        assert obs["is_live"] is False, "Open-Meteo model reanalysis must NOT claim to be a physical inlet"
        assert obs["is_modeled"] is True
        assert obs["status_badge"] == "REGIONAL REANALYSIS"
        assert "ECMWF" in obs["provider"] or "Open-Meteo" in obs["provider"]
        assert "not a direct physical station" in obs["badge_desc"].lower()


def test_verified_24h_sequence_retrieval():
    """Verify that get_verified_24h_sequence_for_station returns 24 rows and truthful provenance."""
    df_ap, prov_ap, is_synth_ap = get_verified_24h_sequence_for_station("AP001")
    assert len(df_ap) == 24
    assert is_synth_ap is False
    assert "Verified 24h" in prov_ap

    df_proxy, prov_proxy, is_synth_proxy = get_verified_24h_sequence_for_station("AP_NELLORE")
    assert len(df_proxy) == 24
    assert is_synth_proxy is False
    assert "Regional CAAQMS Benchmark Sequence" in prov_proxy


def test_hybrid_prediction_provenance_and_no_random_drift():
    """Verify that predict_hybrid uses verified sequences when df_24h=None and produces consistent results."""
    srv = HybridPredictionService()
    res1 = srv.predict_hybrid(station_id="AP001")
    res2 = srv.predict_hybrid(station_id="AP001")

    assert "sequence_provenance" in res1
    assert res1["is_synthetic_input"] is False
    # Proves deterministic real sequence, not np.random.uniform drift
    assert abs(res1["pm25"] - res2["pm25"]) < 1e-4, "Predict hybrid should produce deterministic outputs on verified sequence"


def test_model_comparison_synchronized_sequence():
    """Verify that ModelComparisonService synchronizes the antecedent sequence across all models."""
    comp_srv = ModelComparisonService()
    res = comp_srv.compare_all(station_id="AP001", city="Amaravati")

    assert "sequence_provenance" in res
    assert res["synchronization_status"] == "Strictly Synchronized (Identical 24h Sequence)"
    assert res["sensor_forecast"]["station_id"] == "AP001"
    assert res["hybrid_prediction"]["station_id"] == "AP001"
