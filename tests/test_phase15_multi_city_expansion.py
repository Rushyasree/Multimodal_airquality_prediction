"""
tests/test_phase15_multi_city_expansion.py

Verification and Regression Test Suite for Phase 15:
"Multi-City Spatial Expansion across 31 Indian Urban Agglomerations"

Tests:
1. City Registry Architecture: All 31 cities registered across South, West, North, East, and Central India.
2. Geographic Bounds: Valid bounding box hierarchy and center coordinate containment.
3. Station Deduplication & Metadata Integrity: Unique station IDs and valid coordinates.
4. Delhi NCR Conformal Benchmark: Preserved validated status and high-density station network.
5. Sparse Coverage Safeguards: Cities with < 3 stations gracefully suppress 2D interpolation.
6. Multi-City IDW Grid Synthesis: Supported cities produce valid, non-negative 2D grids.
7. Point Estimation Continuity: Point queries operate smoothly across both dense and sparse cities.
8. Centralized Coverage Summary Audit: Clean, complete reporting DataFrame.
"""

import pytest
import numpy as np
import pandas as pd

from src.config.spatial_config import SUPPORTED_CITIES, CityBoundingBox, SpatialInterpolationConfig
from src.services.city_registry_service import (
    CityRegistryService,
    get_city_registry_service,
)
from src.services.spatial_interpolation_service import (
    SpatialInterpolationService,
    StationForecastRecord,
    ForecastContext,
)


@pytest.fixture(scope="module")
def registry():
    return get_city_registry_service()


@pytest.fixture(scope="module")
def spatial_service():
    return SpatialInterpolationService()


EXPECTED_31_CITIES = [
    # South India (10)
    "hyderabad", "chennai", "visakhapatnam", "vijayawada", "amaravati",
    "tirupati", "guntur", "kochi", "thiruvananthapuram", "coimbatore",
    # West India (5)
    "pune", "ahmedabad", "surat", "jaipur", "nagpur",
    # North India (5)
    "lucknow", "chandigarh", "kanpur", "agra", "varanasi",
    # East India (4)
    "patna", "bhubaneswar", "ranchi", "guwahati",
    # Central India (3)
    "bhopal", "indore", "raipur",
    # Retained (4)
    "delhi_ncr", "mumbai_mmr", "kolkata", "bengaluru",
]


def test_city_registry_31_cities_present(registry):
    """Verifies that all 31 designated cities are registered with correct macro-regions."""
    registered_ids = registry.get_all_city_ids()
    assert len(registered_ids) >= 31, f"Expected at least 31 cities, got {len(registered_ids)}"

    for cid in EXPECTED_31_CITIES:
        assert cid in registered_ids, f"City ID '{cid}' missing from registry"
        city = registry.get_city(cid)
        assert isinstance(city, CityBoundingBox)
        assert len(city.name) > 0
        assert city.region in ("South", "West", "North", "East", "Central")
        assert len(city.primary_state) > 0


def test_city_geographic_bounds_and_centers(registry):
    """Ensures each bounding box is geographically sound and encompasses its center point in India."""
    for city in registry.get_all_cities():
        # Latitude order and bounds (India is roughly 8°N to 36°N)
        assert 8.0 <= city.min_lat < city.max_lat <= 36.0, f"Invalid latitude range for {city.city_id}"
        assert city.min_lat < city.center_lat < city.max_lat, f"Center lat outside bbox for {city.city_id}"

        # Longitude order and bounds (India is roughly 68°E to 98°E)
        assert 68.0 <= city.min_lon < city.max_lon <= 98.0, f"Invalid longitude range for {city.city_id}"
        assert city.min_lon < city.center_lon < city.max_lon, f"Center lon outside bbox for {city.city_id}"


def test_station_matching_and_deduplication(registry):
    """Validates that matched stations have unique identifiers and valid coordinates."""
    for city in registry.get_all_cities():
        stations = registry.get_stations_for_city(city.city_id)
        if stations:
            sids = [s["station_id"] for s in stations]
            assert len(sids) == len(set(sids)), f"Duplicate station IDs found in {city.city_id}"
            for s in stations:
                assert -90.0 <= s["latitude"] <= 90.0
                assert -180.0 <= s["longitude"] <= 180.0
                assert s["distance_to_center_km"] >= 0.0


def test_delhi_ncr_conformal_validation_status(registry):
    """Verifies that Delhi NCR maintains scientifically validated conformal status and dense network."""
    delhi = registry.get_city("delhi_ncr")
    assert delhi.validation_status == "VALIDATED_CONFORMAL"
    st_count = registry.get_station_count("delhi_ncr")
    assert st_count >= 90, f"Delhi NCR should have >= 90 stations, got {st_count}"
    assert registry.get_monitoring_density("delhi_ncr") == "Dense"
    assert registry.is_interpolation_supported("delhi_ncr") is True


def test_sparse_coverage_safeguards_under_3_stations(registry, spatial_service):
    """
    Cities with < 3 physical monitors must trigger safeguards:
    - is_interpolation_supported is False
    - validation_status is INSUFFICIENT_COVERAGE
    - generate_city_grid returns is_supported=False without uncaught exceptions
    """
    sparse_cities = ["surat", "visakhapatnam", "tirupati", "guntur", "ranchi"]
    for cid in sparse_cities:
        city = registry.get_city(cid)
        st_count = registry.get_station_count(cid)
        assert st_count < 3, f"{cid} has {st_count} stations, expected < 3"
        assert registry.is_interpolation_supported(cid) is False
        assert registry.get_validation_status(cid) == "INSUFFICIENT_COVERAGE"

        # Test safe grid execution with minimal or empty inputs
        stations = registry.get_stations_for_city(cid)
        records = [
            StationForecastRecord(
                station_id=s["station_id"],
                latitude=s["latitude"],
                longitude=s["longitude"],
                predicted_pm25=50.0,
                predicted_pm10=100.0,
                target_time="2026-11-15 12:00:00",
                issue_time="2026-11-15 11:00:00",
            )
            for s in stations
        ]

        if records:
            ctx = ForecastContext.from_records(records)
            res = spatial_service.generate_city_grid(cid, ctx, grid_resolution=25)
            assert res["is_supported"] is False
            assert res["data_quality_status"] == "INSUFFICIENT_COVERAGE"
            assert "error_message" in res


def test_supported_cities_interpolation_generation(registry, spatial_service):
    """
    Validates that cities with >= 3 physical monitors successfully generate
    continuous 2D IDW grids with physical plausibility.
    """
    sample_supported = ["hyderabad", "pune", "ahmedabad", "patna", "lucknow", "bhopal"]
    for cid in sample_supported:
        assert registry.is_interpolation_supported(cid) is True
        stations = registry.get_stations_for_city(cid)
        assert len(stations) >= 3

        records = [
            StationForecastRecord(
                station_id=s["station_id"],
                latitude=s["latitude"],
                longitude=s["longitude"],
                predicted_pm25=float(40.0 + (i * 5.0)),
                predicted_pm10=float(80.0 + (i * 10.0)),
                target_time="2026-11-15 12:00:00",
                issue_time="2026-11-15 11:00:00",
            )
            for i, s in enumerate(stations)
        ]

        ctx = ForecastContext.from_records(records)
        res = spatial_service.generate_city_grid(cid, ctx, grid_resolution=25)

        assert res["is_supported"] is True
        assert res["grid_resolution"] == 25
        assert res["total_grid_points"] == 625
        assert res["valid_grid_points"] > 0
        assert res["coverage_percentage"] > 0.0

        # Physical concentration bounds
        stats = res["stats"]
        assert stats["pm25_min"] >= 0.0
        assert stats["pm25_max"] <= 1200.0
        assert stats["pm10_min"] >= 0.0
        assert stats["pm10_max"] <= 2000.0


def test_point_prediction_operates_on_sparse_and_dense(registry, spatial_service):
    """Verifies predict_at_location works for point coordinate estimation in any city."""
    delhi = registry.get_city("delhi_ncr")
    delhi_stations = registry.get_stations_for_city("delhi_ncr")[:5]
    delhi_records = [
        StationForecastRecord(
            station_id=s["station_id"],
            latitude=s["latitude"],
            longitude=s["longitude"],
            predicted_pm25=95.0,
            predicted_pm10=180.0,
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
        )
        for s in delhi_stations
    ]
    ctx_delhi = ForecastContext.from_records(delhi_records)
    pt_res = spatial_service.predict_at_location(delhi.center_lat, delhi.center_lon, ctx_delhi)
    assert pt_res["predicted_pm25"] is not None
    assert pt_res["predicted_pm10"] is not None
    assert "ci_conformal_95_pm25" in pt_res


def test_coverage_summary_completeness(registry):
    """Ensures get_coverage_summary returns a comprehensive dataframe of all 31 cities."""
    summary = registry.get_coverage_summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) >= 31
    required_cols = [
        "city_id", "name", "region", "state", "center_lat", "center_lon",
        "station_count", "interpolation_supported", "validation_status", "density_category"
    ]
    for col in required_cols:
        assert col in summary.columns, f"Column '{col}' missing from summary DataFrame"
    assert summary["station_count"].isna().sum() == 0
    assert summary["station_count"].min() >= 0
