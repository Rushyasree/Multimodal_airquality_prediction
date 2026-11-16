"""
tests/test_phase13_spatial_validation.py

Unit and integration test suite for Phase 13:
- Spatial Leave-One-Out validation integrity (zero target leakage)
- Same-timestamp constraint verification
- Conformal prediction intervals and coverage scaling
- Out-of-bounds and extrapolation boundary enforcement
- Spatial forecast vs observation interpolation separation
"""

import json
import os
import pytest
import numpy as np
import pandas as pd

from src.services.spatial_interpolation_service import (
    SpatialInterpolationService,
    StationForecastRecord,
    ForecastContext,
)
from src.config.spatial_config import SpatialInterpolationConfig, SUPPORTED_CITIES


@pytest.fixture
def mock_station_records():
    """Generates 5 synthetic station records at a shared target timestamp."""
    stations = [
        ("SITE_01", 28.6200, 77.2100, 120.0, 210.0),
        ("SITE_02", 28.6500, 77.2300, 140.0, 240.0),
        ("SITE_03", 28.5800, 77.1900, 110.0, 190.0),
        ("SITE_04", 28.6100, 77.2800, 130.0, 220.0),
        ("SITE_05", 28.7000, 77.1500, 160.0, 280.0),
    ]
    return [
        StationForecastRecord(
            station_id=sid,
            latitude=lat,
            longitude=lon,
            predicted_pm25=p25,
            predicted_pm10=p10,
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="VayuDrishti_Ensemble",
        )
        for sid, lat, lon, p25, p10 in stations
    ]


@pytest.fixture
def spatial_service():
    config = SpatialInterpolationConfig(
        power=2.0,
        min_stations=2,
        max_distance_km=45.0,
        max_neighbors=10,
        exact_match_threshold_km=0.05,
    )
    return SpatialInterpolationService(config=config)


# ---------------------------------------------------------------------------
# TEST 1: Strict Held-Out Station Exclusion (Zero Target Leakage)
# ---------------------------------------------------------------------------
def test_held_out_station_exclusion_zero_leakage(spatial_service, mock_station_records):
    """
    Verify that when evaluating an arbitrary target location (or station),
    the held-out station's data is strictly excluded from interpolation inputs.
    """
    target_sid = "SITE_01"
    target_record = [r for r in mock_station_records if r.station_id == target_sid][0]

    # Input pool strictly excludes target_record
    input_records = [r for r in mock_station_records if r.station_id != target_sid]
    assert len(input_records) == len(mock_station_records) - 1
    assert all(r.station_id != target_sid for r in input_records)

    ctx = ForecastContext.from_records(input_records)
    res = spatial_service.predict_at_location(target_record.latitude, target_record.longitude, ctx)

    # Check that SITE_01 was NOT used
    assert target_sid not in res["contributing_station_ids"]
    assert res["nearest_station_id"] != target_sid
    # Check that prediction is non-null and valid
    assert res["predicted_pm25"] is not None
    assert res["predicted_pm10"] is not None


# ---------------------------------------------------------------------------
# TEST 2: Same-Timestamp Temporal Integrity Enforcement
# ---------------------------------------------------------------------------
def test_same_timestamp_temporal_integrity(spatial_service, mock_station_records):
    """
    ForecastContext must filter out and isolate records with mismatched target timestamps
    to prevent temporal cross-contamination.
    """
    mismatched_records = list(mock_station_records)
    # Tamper with one station's timestamp
    mismatched_records.append(
        StationForecastRecord(
            station_id="SITE_LEAK",
            latitude=28.6300,
            longitude=28.2200,
            predicted_pm25=999.0,
            predicted_pm10=999.0,
            target_time="2026-11-15 18:00:00",  # 6 hours later!
            issue_time="2026-11-15 17:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="VayuDrishti_Ensemble",
        )
    )

    ctx = ForecastContext.from_records(mismatched_records)
    sids_in_ctx = [r.station_id for r in ctx.records]
    assert "SITE_LEAK" not in sids_in_ctx
    assert len(ctx.records) == len(mock_station_records)


# ---------------------------------------------------------------------------
# TEST 3: Conformal Prediction Interval Generation
# ---------------------------------------------------------------------------
def test_conformal_calibrated_intervals_present(spatial_service, mock_station_records):
    """
    predict_at_location must return calibrated conformal prediction intervals
    which expand monotonically with distance to nearest monitor.
    """
    ctx = ForecastContext.from_records(mock_station_records)

    # Point close to SITE_01 (~0.5 km)
    res_close = spatial_service.predict_at_location(28.6240, 77.2120, ctx)
    # Point farther away (~8.0 km)
    res_far = spatial_service.predict_at_location(28.5000, 77.2120, ctx)

    assert "ci_conformal_95_pm25" in res_close
    assert "ci_conformal_95_pm10" in res_close

    close_ci25 = res_close["ci_conformal_95_pm25"]
    far_ci25 = res_far["ci_conformal_95_pm25"]

    width_close = close_ci25[1] - close_ci25[0]
    width_far = far_ci25[1] - far_ci25[0]

    # Interval width must widen as distance increases
    assert width_far > width_close
    # Physical lower bound must be >= 0
    assert close_ci25[0] >= 0.0
    assert far_ci25[0] >= 0.0


# ---------------------------------------------------------------------------
# TEST 4: Out-of-Bounds and Extrapolation Safety
# ---------------------------------------------------------------------------
def test_out_of_bounds_extrapolation_safety(spatial_service, mock_station_records):
    """
    Points beyond max_distance_km must return UNSUPPORTED_EXTRAPOLATION
    without raising uncaught runtime exceptions.
    """
    ctx = ForecastContext.from_records(mock_station_records)

    # Point in Rajasthan (> 200 km from Delhi)
    res_remote = spatial_service.predict_at_location(26.9124, 75.7873, ctx)

    assert res_remote["data_quality_status"] == "UNSUPPORTED_EXTRAPOLATION"
    assert res_remote["predicted_pm25"] is None
    assert res_remote["predicted_pm10"] is None
    assert res_remote["is_extrapolation"] is True


# ---------------------------------------------------------------------------
# TEST 5: Verification of Saved Phase 13 Artifacts
# ---------------------------------------------------------------------------
def test_phase13_saved_metrics_artifacts_exist():
    """Verify that all Phase 13 required metrics files exist and are valid JSON."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spatial_dir = os.path.join(base_dir, "experiments", "spatial_prediction")

    files_to_check = [
        "spatial_block_bootstrap_results.json",
        "spatial_forecast_validation_metrics.json",
        "spatial_uncertainty_calibration_metrics.json",
    ]

    for fname in files_to_check:
        fpath = os.path.join(spatial_dir, fname)
        assert os.path.exists(fpath), f"Missing artifact: {fname}"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Artifact {fname} must contain a valid JSON dict"


# ---------------------------------------------------------------------------
# TEST 6: Conformal Coverage Target Compliance
# ---------------------------------------------------------------------------
def test_conformal_calibration_metrics_integrity():
    """Verify that empirical coverage metrics in JSON reflect Phase 13 requirements."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(
        base_dir, "experiments", "spatial_prediction", "spatial_uncertainty_calibration_metrics.json"
    )
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Heuristic must fail audit
    assert metrics["audit_heuristic"]["pm25_empirical_coverage_pct"] < 50.0
    # Conformal must achieve >= 95.0% target on evaluation set
    assert metrics["conformal_calibration"]["evaluation_coverage_pm25_pct"] >= 95.0
    assert metrics["conformal_calibration"]["evaluation_coverage_pm10_pct"] >= 95.0


# ---------------------------------------------------------------------------
# TEST 7: Nearest-Station Comparison Benchmark Availability
# ---------------------------------------------------------------------------
def test_nearest_station_comparison_metrics(spatial_service, mock_station_records):
    """
    predict_at_location must provide explicit nearest-station PM2.5 and PM10 values
    for direct side-by-side comparison with the IDW spatial estimate.
    """
    ctx = ForecastContext.from_records(mock_station_records)
    res = spatial_service.predict_at_location(28.6250, 77.2150, ctx)

    assert "nearest_station_pm25" in res
    assert "nearest_station_pm10" in res
    assert isinstance(res["nearest_station_pm25"], float)
    assert isinstance(res["nearest_station_pm10"], float)
    assert res["nearest_station_id"] is not None
    assert res["nearest_station_distance_km"] >= 0.0


# ---------------------------------------------------------------------------
# TEST 8: Dashboard Sidebar Navigation Route & Alias Connectivity
# ---------------------------------------------------------------------------
def test_dashboard_sidebar_route_connectivity():
    """Verify that app.py connects Spatial Prediction to both sidebar and top navbar routing."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app.py")

    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()

    # Route check
    assert '"Spatial Prediction & Heatmap"' in app_code
    assert 'render_spatial_heatmap_page' in app_code
    assert 'initial_sidebar_state="expanded"' in app_code
    assert 'NAV_ALIASES' in app_code
    assert '"Spatial Heatmap"' in app_code
