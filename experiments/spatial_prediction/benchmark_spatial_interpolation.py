"""
experiments/spatial_prediction/benchmark_spatial_interpolation.py

Rigorous Empirical Latency & Memory Benchmark for VayuDrishti Phase 12 Spatial Interpolation Layer.
Measures execution times across grid resolutions, arbitrary point queries, and station counts.
"""

import sys
import os
import time
import platform
import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.services.spatial_interpolation_service import (
    SpatialInterpolationService,
    StationForecastRecord,
    ForecastContext,
)
from src.config.spatial_config import SpatialInterpolationConfig

BENCHMARK_JSON = Path(__file__).resolve().parent / "benchmark_results.json"


def run_benchmarks():
    print("=" * 76)
    print("VAYUDRISHTI PHASE 12: EMPIRICAL SPATIAL INTERPOLATION BENCHMARKS")
    print("=" * 76)

    # 1. System Environment
    cpu_info = platform.processor() or "x86_64 Compatible"
    py_ver = sys.version.split()[0]
    np_ver = np.__version__
    os_name = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"

    print(f"OS: {os_name}")
    print(f"Python: {py_ver}")
    print(f"NumPy: {np_ver}")
    print(f"CPU: {cpu_info}")

    # Generate synthetic stations across Delhi bounding box
    np.random.seed(42)
    max_stations = 90
    lats = np.random.uniform(28.40, 28.90, max_stations)
    lons = np.random.uniform(76.85, 77.40, max_stations)
    p25 = np.random.uniform(20.0, 250.0, max_stations)
    p10 = p25 * np.random.uniform(1.4, 2.2, max_stations)

    all_records = [
        StationForecastRecord(
            station_id=f"ST_{i:03d}",
            latitude=lats[i],
            longitude=lons[i],
            predicted_pm25=float(p25[i]),
            predicted_pm10=float(p10[i]),
            target_time="2026-11-15 12:00:00",
            issue_time="2026-11-15 11:00:00",
            forecast_horizon="Next 1 Hour (t + 1)",
            model_source="Benchmark_Synthetic",
        )
        for i in range(max_stations)
    ]

    service = SpatialInterpolationService()

    # 2. Benchmark Grid Resolutions (at Delhi cluster size N=40)
    delhi_records = all_records[:40]
    ctx_delhi = ForecastContext.from_records(delhi_records)

    resolutions = [25, 50, 75, 100]
    grid_benchmarks = []

    print("\n--- Benchmarking Grid Resolution Scaling (N=40 Stations) ---")
    for res in resolutions:
        total_pts = res * res
        # Warmup
        service.generate_city_grid(city_id="delhi_ncr", forecast_context=ctx_delhi, grid_resolution=res)

        timings = []
        for _ in range(25):
            t0 = time.perf_counter()
            service.generate_city_grid(city_id="delhi_ncr", forecast_context=ctx_delhi, grid_resolution=res)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        median_ms = float(np.median(timings))
        p95_ms = float(np.percentile(timings, 95))
        min_ms = float(np.min(timings))

        print(f"Grid {res:3d}x{res:<3d} ({total_pts:6,d} points) | Median: {median_ms:6.2f} ms | Min: {min_ms:6.2f} ms | P95: {p95_ms:6.2f} ms")
        grid_benchmarks.append({
            "grid_resolution": f"{res}x{res}",
            "grid_points": total_pts,
            "station_count": 40,
            "median_runtime_ms": round(median_ms, 2),
            "min_runtime_ms": round(min_ms, 2),
            "p95_runtime_ms": round(p95_ms, 2),
        })

    # 3. Benchmark Arbitrary Point Query
    print("\n--- Benchmarking Arbitrary Point Inference (1 Coordinate) ---")
    point_timings = []
    # Warmup
    service.predict_at_location(28.65, 77.20, ctx_delhi)
    for _ in range(100):
        t0 = time.perf_counter()
        service.predict_at_location(28.65, 77.20, ctx_delhi)
        t1 = time.perf_counter()
        point_timings.append((t1 - t0) * 1000.0)

    pt_median = float(np.median(point_timings))
    pt_p95 = float(np.percentile(point_timings, 95))
    print(f"Point Query (1 coordinate) | Median: {pt_median:6.3f} ms | P95: {pt_p95:6.3f} ms")

    # 4. Benchmark Station Count Scaling (at 50x50 resolution)
    print("\n--- Benchmarking Station Count Scaling (50x50 Grid = 2,500 points) ---")
    st_counts = [10, 25, 40, 90]
    st_benchmarks = []
    for count in st_counts:
        sub_records = all_records[:count]
        ctx_sub = ForecastContext.from_records(sub_records)

        timings = []
        for _ in range(20):
            t0 = time.perf_counter()
            service.generate_city_grid(city_id="delhi_ncr", forecast_context=ctx_sub, grid_resolution=50)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        med = float(np.median(timings))
        print(f"Stations: {count:2d} monitors | 50x50 Grid Median: {med:6.2f} ms")
        st_benchmarks.append({
            "station_count": count,
            "grid_points": 2500,
            "median_runtime_ms": round(med, 2),
        })

    output_data = {
        "benchmark_metadata": {
            "os": os_name,
            "python_version": py_ver,
            "numpy_version": np_ver,
            "cpu_processor": cpu_info,
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "grid_resolution_benchmarks": grid_benchmarks,
        "point_query_benchmark": {
            "median_ms": round(pt_median, 3),
            "p95_ms": round(pt_p95, 3),
            "queries_per_second": round(1000.0 / pt_median, 1) if pt_median > 0 else 0,
        },
        "station_scaling_benchmarks": st_benchmarks,
    }

    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nBenchmark results saved to {BENCHMARK_JSON}")


if __name__ == "__main__":
    run_benchmarks()
