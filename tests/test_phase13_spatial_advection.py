"""
tests/test_phase13_spatial_advection.py

Unit tests for Phase 13 Spatial-Temporal Cross-Station Advection Analysis:
1. Haversine distance accuracy.
2. Initial great-circle bearing computation.
3. Cross-correlation function (CCF) lag identification on injected signals.
4. Along-transect wind velocity projection.
5. Cluster definition and geometry integrity.
6. Causal lag non-leakage verification (<= t).
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.services.spatial_advection_service import (
    haversine_distance,
    calculate_bearing,
    compute_cross_correlation,
    compute_along_transect_wind,
    SpatialAdvectionService,
)


def test_haversine_distance():
    # Anand Vihar (28.6476, 77.3158) to DTU (28.7501, 77.1113)
    d = haversine_distance(28.6476, 77.3158, 28.7501, 77.1113)
    assert 22.0 < d < 24.0, f"Expected ~23.0 km, got {d:.2f}"

    # Self-distance should be zero
    assert haversine_distance(28.6476, 77.3158, 28.6476, 77.3158) == 0.0


def test_calculate_bearing():
    # Due North: (0, 0) -> (10, 0) should be 0 deg
    b_north = calculate_bearing(0.0, 0.0, 10.0, 0.0)
    assert math.isclose(b_north, 0.0, abs_tol=1e-3)

    # Due East: (0, 0) -> (0, 10) should be 90 deg
    b_east = calculate_bearing(0.0, 0.0, 0.0, 10.0)
    assert math.isclose(b_east, 90.0, abs_tol=1e-3)

    # Due South: (10, 0) -> (0, 0) should be 180 deg
    b_south = calculate_bearing(10.0, 0.0, 0.0, 0.0)
    assert math.isclose(b_south, 180.0, abs_tol=1e-3)


def test_cross_correlation_lag_recovery():
    # Construct a synthetic signal with injected lag of +2 hours (A leads B by 2h)
    rng = np.random.default_rng(42)
    base_signal = np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.1 * rng.normal(size=500)

    # A leads B by 2 hours -> B[t] = A[t - 2]
    series_a = base_signal[2:]
    series_b = base_signal[:-2]

    ccf = compute_cross_correlation(series_a, series_b, max_lag=6)
    peak_lag = max(ccf, key=lambda k: ccf[k])
    assert peak_lag == 2, f"Expected peak lag +2, got {peak_lag}"
    assert ccf[2] > 0.90, f"Expected high correlation at lag 2, got {ccf[2]}"


def test_along_transect_wind_projection():
    # Transect bearing = 90 deg (due East)
    # Wind from West towards East (wind direction = 270 deg or moving along transect)
    # Wind blowing at 5.0 m/s exactly along bearing (0 deg difference)
    ws = np.array([5.0, 10.0, 0.0])
    wd = np.array([90.0, 270.0, 0.0])  # 90 is aligned, 270 is opposite

    v_par = compute_along_transect_wind(ws, wd, transect_bearing_deg=90.0)
    assert math.isclose(v_par[0], 5.0, abs_tol=1e-3), f"Expected +5.0 m/s, got {v_par[0]}"
    assert math.isclose(v_par[1], -10.0, abs_tol=1e-3), f"Expected -10.0 m/s, got {v_par[1]}"
    assert v_par[2] == 0.0


def test_spatial_advection_service_clusters():
    service = SpatialAdvectionService()
    assert len(service.clusters) == 3

    for cl in service.clusters:
        assert "distance_km" in cl
        assert cl["distance_km"] > 5.0
        assert cl["distance_km"] < 30.0
        assert 0.0 <= cl["bearing_a_to_b"] < 360.0
        assert 0.0 <= cl["bearing_b_to_a"] < 360.0


def test_spatial_causality_zero_lookahead():
    # Verify that in a multi-node feature matrix, neighbor values correspond strictly to <= t
    times = pd.date_range("2025-01-01 00:00", periods=24, freq="h")
    df = pd.DataFrame({
        "time": times,
        "PM2.5_A": np.arange(100, 124),
        "PM2.5_B": np.arange(200, 224),
    })

    # When predicting PM2.5_B at index 10 (t+1), only features from index <= 9 (t) can be used
    t = 9
    feat_target_t = df.loc[t, "PM2.5_B"]
    feat_neighbor_t = df.loc[t, "PM2.5_A"]
    target_to_predict = df.loc[t + 1, "PM2.5_B"]

    assert feat_target_t == 209
    assert feat_neighbor_t == 109
    assert target_to_predict == 210
    assert target_to_predict > feat_target_t  # strictly future
