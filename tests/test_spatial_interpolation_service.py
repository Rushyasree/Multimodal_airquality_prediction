"""
tests/test_spatial_interpolation_service.py

Comprehensive Unit and Integration Tests for VayuDrishti Phase 12 Spatial Interpolation Layer.
Verifies mathematical correctness, geodesic distance accuracy, boundary conditions,
missing-value handling, exact-station snapping, and deterministic grid synthesis.
"""

import numpy as np
import pytest

from src.config.spatial_config import (
    SpatialInterpolationConfig,
    SUPPORTED_CITIES,
)
from src.services.spatial_interpolation_service import (
    SpatialInterpolationService,
    StationForecastRecord,
    ForecastContext,
    haversine_distance_vectorized,
)


@pytest.fixture
def base_config():
    return SpatialInterpolationConfig(
        power=2.0,
        min_stations=3,
        max_distance_km=40.0,
        max_neighbors=10,
        exact_match_threshold_km=0.05,
        epsilon_km=0.001,
        min_concentration=0.0,
        max_pm25_concentration=1000.0,
        max_pm10_concentration=1500.0,
    )


@pytest.fixture
def spatial_service(base_config):
    return SpatialInterpolationService(config=base_config)


@pytest.fixture
def synthetic_delhi_records():
    """Returns 4 well-separated synthetic test monitors within Delhi NCR."""
    return [
        StationForecastRecord(
            station_id="ST_CENTRAL",
            latitude=28.6300,
            longitude=77.2200,
            predicted_pm25=100.0,
            predicted_pm10=180.0,
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="Test_Ensemble",
        ),
        StationForecastRecord(
            station_id="ST_NORTH",
            latitude=28.7000,
            longitude=77.2200,
            predicted_pm25=120.0,
            predicted_pm10=210.0,
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="Test_Ensemble",
        ),
        StationForecastRecord(
            station_id="ST_SOUTH",
            latitude=28.5600,
            longitude=77.2200,
            predicted_pm25=80.0,
            predicted_pm10=150.0,
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="Test_Ensemble",
        ),
        StationForecastRecord(
            station_id="ST_EAST",
            latitude=28.6300,
            longitude=77.3000,
            predicted_pm25=110.0,
            predicted_pm10=195.0,
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="Test_Ensemble",
        ),
    ]


class TestSpatialInterpolationService:
    """Validation test suite for SpatialInterpolationService."""

    def test_haversine_distance_vectorized_accuracy(self):
        """Verifies geodesic Haversine distance matches known ground distances."""
        # Delhi (28.6139, 77.2090) to Mumbai (19.0760, 72.8777) is ~1148 km
        delhi_lat, delhi_lon = np.array([28.6139]), np.array([77.2090])
        mumbai_lat, mumbai_lon = np.array([19.0760]), np.array([72.8777])

        dist_km = haversine_distance_vectorized(delhi_lat, delhi_lon, mumbai_lat, mumbai_lon)[0]
        assert 1140.0 < dist_km < 1160.0

        # Coincident coordinates must yield 0.0 km
        dist_zero = haversine_distance_vectorized(delhi_lat, delhi_lon, delhi_lat, delhi_lon)[0]
        assert pytest.approx(dist_zero, abs=1e-5) == 0.0

    def test_exact_station_coordinate_snapping(self, spatial_service, synthetic_delhi_records):
        """Querying at exact station coordinates must return exact station forecast with zero distance."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)
        target = synthetic_delhi_records[0]  # ST_CENTRAL: lat=28.63, lon=77.22, pm25=100.0, pm10=180.0

        res = spatial_service.predict_at_location(target.latitude, target.longitude, ctx)

        assert res["data_quality_status"] == "EXACT_STATION_HIT"
        assert res["nearest_station_id"] == "ST_CENTRAL"
        assert res["nearest_station_distance_km"] == 0.0
        assert res["predicted_pm25"] == 100.0
        assert res["predicted_pm10"] == 180.0
        assert res["num_contributing_stations"] == 1
        assert res["is_extrapolation"] is False

    def test_two_station_midpoint_interpolation(self, spatial_service):
        """Midpoint between two symmetric stations must yield the exact arithmetic mean under IDW."""
        records = [
            StationForecastRecord(
                station_id="ST_WEST",
                latitude=28.6000,
                longitude=77.0000,
                predicted_pm25=40.0,
                predicted_pm10=80.0,
                target_time="2026-11-15 12:00:00",
            ),
            StationForecastRecord(
                station_id="ST_EAST",
                latitude=28.6000,
                longitude=77.2000,
                predicted_pm25=80.0,
                predicted_pm10=160.0,
                target_time="2026-11-15 12:00:00",
            ),
            StationForecastRecord(
                station_id="ST_FAR_DUMMY",
                latitude=28.8500,
                longitude=77.1000,
                predicted_pm25=60.0,
                predicted_pm10=120.0,
                target_time="2026-11-15 12:00:00",
            ),
        ]
        ctx = ForecastContext.from_records(records)
        # Midpoint on the longitude axis: lon = 77.1000
        res = spatial_service.predict_at_location(28.6000, 77.1000, ctx)

        assert res["data_quality_status"] in ("HIGH_CONFIDENCE_INTERPOLATION", "SPARSE_INTERPOLATION")
        # Since ST_WEST and ST_EAST are equidistant from midpoint, they dominate equally
        assert 55.0 <= res["predicted_pm25"] <= 65.0
        assert 110.0 <= res["predicted_pm10"] <= 130.0

    def test_multi_station_interpolation_bounds(self, spatial_service, synthetic_delhi_records):
        """Interpolated value must strictly stay within the convex hull of station values."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)
        # Query inside the central cluster
        res = spatial_service.predict_at_location(28.6500, 77.2300, ctx)

        min_p25 = min(r.predicted_pm25 for r in synthetic_delhi_records)
        max_p25 = max(r.predicted_pm25 for r in synthetic_delhi_records)

        assert min_p25 <= res["predicted_pm25"] <= max_p25
        assert res["data_quality_status"] == "HIGH_CONFIDENCE_INTERPOLATION"
        assert res["num_contributing_stations"] == len(synthetic_delhi_records)

    def test_simultaneous_pm25_and_pm10_prediction(self, spatial_service, synthetic_delhi_records):
        """Both PM2.5 and PM10 must be estimated simultaneously with valid numerical values."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)
        res = spatial_service.predict_at_location(28.6400, 77.2100, ctx)

        assert isinstance(res["predicted_pm25"], float)
        assert isinstance(res["predicted_pm10"], float)
        assert res["predicted_pm10"] > res["predicted_pm25"]
        assert len(res["ci_95_pm25"]) == 2
        assert len(res["ci_95_pm10"]) == 2

    def test_invalid_coordinates_handling(self, spatial_service, synthetic_delhi_records):
        """Out-of-bound latitude and longitude must return error status without crashing."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)

        res_bad_lat = spatial_service.predict_at_location(95.0, 77.0, ctx)
        assert res_bad_lat["data_quality_status"] == "INVALID_COORDINATES"
        assert res_bad_lat["predicted_pm25"] is None

        res_bad_lon = spatial_service.predict_at_location(28.0, 205.0, ctx)
        assert res_bad_lon["data_quality_status"] == "INVALID_COORDINATES"
        assert res_bad_lon["predicted_pm25"] is None

    def test_missing_and_corrupt_prediction_filtering(self, spatial_service, synthetic_delhi_records):
        """Stations with NaN or Infinite predictions must be safely filtered out."""
        corrupted = list(synthetic_delhi_records)
        corrupted.append(
            StationForecastRecord(
                station_id="ST_NAN",
                latitude=28.6200,
                longitude=77.2100,
                predicted_pm25=float("nan"),
                predicted_pm10=float("nan"),
                target_time="2026-11-15 12:00:00",
            )
        )
        corrupted.append(
            StationForecastRecord(
                station_id="ST_INF",
                latitude=28.6100,
                longitude=77.2300,
                predicted_pm25=float("inf"),
                predicted_pm10=100.0,
                target_time="2026-11-15 12:00:00",
            )
        )
        ctx = ForecastContext.from_records(corrupted)
        res = spatial_service.predict_at_location(28.6350, 77.2250, ctx)

        assert res["predicted_pm25"] is not None
        assert "ST_NAN" not in res["contributing_station_ids"]
        assert "ST_INF" not in res["contributing_station_ids"]

    def test_duplicate_station_handling(self, spatial_service, synthetic_delhi_records):
        """Duplicate station IDs must be deduplicated without error or double-counting."""
        dupes = list(synthetic_delhi_records)
        # Append exact duplicate of ST_CENTRAL
        dupes.append(synthetic_delhi_records[0])
        ctx = ForecastContext.from_records(dupes)

        res = spatial_service.predict_at_location(28.6500, 77.2300, ctx)
        assert res["num_contributing_stations"] == len(synthetic_delhi_records)

    def test_minimum_station_requirement(self, spatial_service):
        """Fewer than min_stations (e.g. 2 stations) must reject spatial surface computation."""
        records = [
            StationForecastRecord(
                station_id="ST_1",
                latitude=28.6000,
                longitude=77.0000,
                predicted_pm25=50.0,
                predicted_pm10=100.0,
                target_time="2026-11-15 12:00:00",
            ),
            StationForecastRecord(
                station_id="ST_2",
                latitude=28.6500,
                longitude=77.0500,
                predicted_pm25=60.0,
                predicted_pm10=120.0,
                target_time="2026-11-15 12:00:00",
            ),
        ]
        ctx = ForecastContext.from_records(records)
        res = spatial_service.predict_at_location(28.6200, 77.0200, ctx)

        assert res["data_quality_status"] == "INSUFFICIENT_STATIONS"
        assert res["predicted_pm25"] is None

    def test_maximum_distance_extrapolation_rejection(self, spatial_service, synthetic_delhi_records):
        """Query points exceeding max_distance_km (e.g. 50 km away) must be flagged as extrapolation."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)
        # Jaipur coordinate ~230 km away
        jaipur_lat, jaipur_lon = 26.9124, 75.7873

        res = spatial_service.predict_at_location(jaipur_lat, jaipur_lon, ctx)
        assert res["data_quality_status"] == "UNSUPPORTED_EXTRAPOLATION"
        assert res["is_extrapolation"] is True
        assert res["predicted_pm25"] is None
        assert res["nearest_station_distance_km"] > 40.0

    def test_timestamp_and_horizon_compatibility(self, synthetic_delhi_records):
        """ForecastContext must enforce identical target_time and forecast_horizon across stations."""
        inconsistent = list(synthetic_delhi_records)
        # Alter target time for one station
        inconsistent[1] = StationForecastRecord(
            station_id="ST_DIVERGENT",
            latitude=28.7000,
            longitude=77.2200,
            predicted_pm25=120.0,
            predicted_pm10=210.0,
            target_time="2026-11-15 13:00:00",  # Divergent target time!
            forecast_horizon="Next 1 Hour (t + 1)",
        )

        ctx = ForecastContext.from_records(inconsistent, expected_target_time="2026-11-15 12:00:00")
        assert len(ctx.records) == 3
        assert "ST_DIVERGENT" not in [r.station_id for r in ctx.records]

    def test_non_negative_concentration_clamping(self, spatial_service):
        """Physical concentrations must never be negative even if unphysical inputs are provided."""
        records = [
            StationForecastRecord(
                station_id="ST_NEG",
                latitude=28.6000,
                longitude=77.0000,
                predicted_pm25=-25.0,  # Negative artifact
                predicted_pm10=-50.0,
                target_time="2026-11-15 12:00:00",
            ),
            StationForecastRecord(
                station_id="ST_ZERO",
                latitude=28.6500,
                longitude=77.0500,
                predicted_pm25=0.0,
                predicted_pm10=0.0,
                target_time="2026-11-15 12:00:00",
            ),
            StationForecastRecord(
                station_id="ST_LOW",
                latitude=28.7000,
                longitude=77.1000,
                predicted_pm25=5.0,
                predicted_pm10=10.0,
                target_time="2026-11-15 12:00:00",
            ),
        ]
        ctx = ForecastContext.from_records(records)
        res = spatial_service.predict_at_location(28.6200, 77.0200, ctx)

        assert res["predicted_pm25"] >= 0.0
        assert res["predicted_pm10"] >= 0.0

    def test_city_grid_generation_structure_and_stats(self, spatial_service, synthetic_delhi_records):
        """generate_city_grid must produce complete 2D arrays with valid dimensions and metadata."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)
        grid_res = 25

        grid_out = spatial_service.generate_city_grid(
            city_id="delhi_ncr",
            forecast_context=ctx,
            grid_resolution=grid_res,
        )

        assert grid_out["city_id"] == "delhi_ncr"
        assert grid_out["grid_resolution"] == grid_res
        assert grid_out["total_grid_points"] == grid_res * grid_res

        pm25_grid = np.array(grid_out["pm25_grid"])
        pm10_grid = np.array(grid_out["pm10_grid"])
        valid_mask = np.array(grid_out["valid_mask"])

        assert pm25_grid.shape == (grid_res, grid_res)
        assert pm10_grid.shape == (grid_res, grid_res)
        assert valid_mask.shape == (grid_res, grid_res)

        assert grid_out["coverage_percentage"] > 0.0
        assert grid_out["stats"]["pm25_mean"] > 0.0
        assert grid_out["stats"]["pm10_mean"] > grid_out["stats"]["pm25_mean"]

    def test_deterministic_output_reproducibility(self, spatial_service, synthetic_delhi_records):
        """Repeated identical calls must yield mathematically identical results."""
        ctx = ForecastContext.from_records(synthetic_delhi_records)

        res1 = spatial_service.predict_at_location(28.6500, 77.2300, ctx)
        res2 = spatial_service.predict_at_location(28.6500, 77.2300, ctx)

        assert res1["predicted_pm25"] == res2["predicted_pm25"]
        assert res1["predicted_pm10"] == res2["predicted_pm10"]
        assert res1["nearest_station_distance_km"] == res2["nearest_station_distance_km"]
