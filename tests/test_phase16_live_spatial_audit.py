"""
tests/test_phase16_live_spatial_audit.py

Comprehensive Test Suite for Phase 16:
"Live Station Coverage & Spatial Reliability Audit"

Verifies:
1. Freshness classification engine (Fresh, Delayed, Stale, Offline, Unknown, Historical isolation).
2. Spatial geometry & Clark-Evans clustering index (R calculation and geometry typing).
3. 31-City live station coverage audit and reliability tier classification.
4. Output classification separation (Observation vs Station Forecast vs IDW Estimate vs Unavailable).
5. Safeguard preservation (Zero silent substitution and suppression for < 3 monitors).
"""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from src.services.city_registry_service import get_city_registry_service
from src.services.spatial_reliability_service import (
    SpatialReliabilityService,
    get_spatial_reliability_service,
    FreshnessThresholds,
)


@pytest.fixture(scope="module")
def reliability_service():
    return get_spatial_reliability_service()


def test_freshness_classification_engine(reliability_service):
    """Verifies observation freshness categorization and ensures historical data is not labeled live."""
    now_utc = datetime.now(timezone.utc)

    # 1. Fresh: <= 2.0 hours
    fresh_time = (now_utc - timedelta(minutes=45)).isoformat()
    cat, age = reliability_service.classify_freshness(fresh_time, reference_time=now_utc, is_historical=False)
    assert cat == "FRESH"
    assert 0.0 <= age <= 2.0

    # 2. Delayed: 2.0 to 6.0 hours
    delayed_time = (now_utc - timedelta(hours=3.5)).isoformat()
    cat, age = reliability_service.classify_freshness(delayed_time, reference_time=now_utc, is_historical=False)
    assert cat == "DELAYED"
    assert 2.0 < age <= 6.0

    # 3. Stale: 6.0 to 24.0 hours
    stale_time = (now_utc - timedelta(hours=14.0)).isoformat()
    cat, age = reliability_service.classify_freshness(stale_time, reference_time=now_utc, is_historical=False)
    assert cat == "STALE"
    assert 6.0 < age <= 24.0

    # 4. Offline: > 24.0 hours
    offline_time = (now_utc - timedelta(days=3)).isoformat()
    cat, age = reliability_service.classify_freshness(offline_time, reference_time=now_utc, is_historical=False)
    assert cat == "OFFLINE"
    assert age > 24.0

    # 5. Unknown: None or corrupted
    cat, age = reliability_service.classify_freshness(None)
    assert cat == "UNKNOWN"

    # 6. Historical isolation: must never be labeled live
    cat, age = reliability_service.classify_freshness(fresh_time, is_historical=True)
    assert cat == "HISTORICAL_BENCHMARK"


def test_clark_evans_clustering_index_computation(reliability_service):
    """Tests that Clark-Evans R accurately identifies clustered vs dispersed distributions."""
    area_km2 = 1000.0

    # 1. Clustered: 6 stations tightly clumped within a 1 km box
    clustered_coords = np.array([
        [28.610, 77.210],
        [28.612, 77.211],
        [28.611, 77.213],
        [28.609, 77.212],
        [28.613, 77.214],
        [28.610, 77.215],
    ])
    r_val, desc, mean_nn = reliability_service.calculate_clark_evans_index(clustered_coords, area_km2)
    assert r_val < 0.85
    assert "Clustered" in desc
    assert mean_nn < 1.0

    # 2. Dispersed: 4 stations widely separated across corners
    dispersed_coords = np.array([
        [28.40, 77.00],
        [28.40, 77.40],
        [28.80, 77.00],
        [28.80, 77.40],
    ])
    r_val, desc, mean_nn = reliability_service.calculate_clark_evans_index(dispersed_coords, area_km2)
    assert r_val > 1.15
    assert "Dispersed" in desc

    # 3. Degenerate case: < 2 stations
    single_coord = np.array([[28.50, 77.10]])
    r_val, desc, mean_nn = reliability_service.calculate_clark_evans_index(single_coord, area_km2)
    assert "Insufficient Monitors" in desc


def test_city_reliability_audit_all_31_cities(reliability_service):
    """Audits all 31 cities and verifies tiered reliability classifications and complete telemetry fields."""
    summary_df = reliability_service.audit_all_cities_reliability(live_mode=False)
    assert len(summary_df) == 31

    # Verify Delhi NCR is Tier 1
    delhi_row = summary_df[summary_df["city_id"] == "delhi_ncr"].iloc[0]
    assert delhi_row["tier_code"] == "TIER_1_CALIBRATED"
    assert delhi_row["registered_stations"] >= 90
    assert bool(delhi_row["interpolation_supported"]) is True

    # Verify Tier 2 cities (>= 10 stations)
    tier_2_cities = ["mumbai_mmr", "kolkata", "bengaluru", "hyderabad", "pune", "patna", "ahmedabad"]
    for cid in tier_2_cities:
        row = summary_df[summary_df["city_id"] == cid].iloc[0]
        assert row["tier_code"] == "TIER_2_HIGH"
        assert row["registered_stations"] >= 10
        assert bool(row["interpolation_supported"]) is True

    # Verify Tier 4 cities (3 stations)
    tier_4_cities = ["kochi", "coimbatore", "kanpur"]
    for cid in tier_4_cities:
        row = summary_df[summary_df["city_id"] == cid].iloc[0]
        assert row["tier_code"] == "TIER_4_MARGINAL"
        assert row["registered_stations"] == 3
        assert bool(row["interpolation_supported"]) is True

    # Verify Tier 5 cities (< 3 stations: safeguards active)
    tier_5_cities = ["visakhapatnam", "tirupati", "bhubaneswar", "surat", "guntur", "ranchi"]
    for cid in tier_5_cities:
        row = summary_df[summary_df["city_id"] == cid].iloc[0]
        assert row["tier_code"] == "TIER_5_INSUFFICIENT"
        assert row["registered_stations"] < 3
        assert bool(row["interpolation_supported"]) is False


def test_location_spatial_diagnostics_classification_separation(reliability_service):
    """
    Verifies clean separation of spatial outputs:
    1. Physical Observation (co-located)
    2. Interpolated Spatial Estimate (multi-station IDW)
    3. Station-Level Forecast / Fallback (sparse)
    4. Unavailable Estimate
    """
    # 1. Exact Co-location in Delhi
    res_exact = reliability_service.get_location_spatial_diagnostics(
        lat=28.6315,
        lon=77.2167,
        city_id="delhi_ncr",
    )
    assert res_exact["output_classification"] in ("Physical Observation", "Interpolated Spatial Estimate")
    assert res_exact["contributing_stations_count"] >= 3

    # 2. Sparse City (Surat, 1 station): should be Station-Level Forecast, NOT Interpolated Spatial Estimate
    res_surat = reliability_service.get_location_spatial_diagnostics(
        lat=21.1702,
        lon=72.8311,
        city_id="surat",
    )
    assert res_surat["output_classification"] == "Station-Level Forecast (Nearest Neighbor Fallback)"
    assert res_surat["contributing_stations_count"] == 1

    # 3. Zero-station city (Guntur): should be Unavailable Estimate
    res_guntur = reliability_service.get_location_spatial_diagnostics(
        lat=16.3067,
        lon=80.4365,
        city_id="guntur",
    )
    assert res_guntur["output_classification"] == "Unavailable Estimate"
    assert res_guntur["contributing_stations_count"] == 0


def test_spatial_safeguards_preservation(reliability_service):
    """Ensures that Phase 15 safeguards are strictly preserved in Phase 16."""
    registry = get_city_registry_service()
    for cid in ["visakhapatnam", "tirupati", "surat", "guntur", "ranchi"]:
        assert registry.is_interpolation_supported(cid) is False
        city_audit = reliability_service.audit_city_reliability(cid)
        assert city_audit["interpolation_supported"] is False
        assert city_audit["registered_station_count"] < 3
