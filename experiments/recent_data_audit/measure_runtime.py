"""
experiments/recent_data_audit/measure_runtime.py

Comprehensive Runtime Latency Benchmarking Script (Phase 5B):
- Measures Cold-Start Latency (fresh service instance, checkpoint disk I/O, first prediction).
- Executes warm-up runs.
- Measures Steady-State Latency across 120 consecutive runs (> 100 required).
- Evaluates:
    1. Historical Mode (severity_gated, enable_shadow=False)
    2. Adapted Mode (adapted_severity_gated, enable_shadow=False)
    3. Dual Shadow Mode (severity_gated, enable_shadow=True)
- Measures both End-to-End (DataFrame input) and Core Inference (preprocessed sequence array).
- Exports full statistics to runtime_latency_benchmark.json.
"""

import os
import sys
import time
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress loky worker detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from src.services.sensor_forecast_service import SensorForecastService


def get_hardware_info():
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024**3), 2)
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:
        info["total_ram_gb"] = "Unknown (psutil not installed)"
    return info


def build_test_dataframe():
    now = pd.Timestamp("2025-05-01 12:00:00")
    hours = [now - pd.Timedelta(hours=23 - i) for i in range(24)]
    return pd.DataFrame({
        "Datetime": hours,
        "StationId": "DL001",
        "PM2.5": [35.0 + i * 1.5 for i in range(24)],
        "PM10": [70.0 + i * 3.0 for i in range(24)],
        "temperature_C": [27.0] * 24,
        "relative_humidity": [60.0] * 24,
        "wind_speed": [2.0] * 24,
    })


def measure_cold_starts(df_bench):
    print("\n--- MEASURING COLD-START LATENCIES ---")
    results = {}

    # 1. Historical mode cold start
    t0 = time.perf_counter()
    srv_hist = SensorForecastService()
    res_hist = srv_hist.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=False)
    t1 = time.perf_counter()
    results["historical_cold_start_ms"] = round((t1 - t0) * 1000.0, 2)
    print(f"  Historical Mode Cold-Start: {results['historical_cold_start_ms']:.2f} ms")

    # 2. Adapted mode cold start
    t0 = time.perf_counter()
    srv_adapt = SensorForecastService()
    res_adapt = srv_adapt.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="adapted_severity_gated", enable_shadow=False)
    t1 = time.perf_counter()
    results["adapted_cold_start_ms"] = round((t1 - t0) * 1000.0, 2)
    print(f"  Adapted Mode Cold-Start:    {results['adapted_cold_start_ms']:.2f} ms")

    # 3. Dual shadow mode cold start
    t0 = time.perf_counter()
    srv_dual = SensorForecastService()
    res_dual = srv_dual.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=True)
    t1 = time.perf_counter()
    results["dual_shadow_cold_start_ms"] = round((t1 - t0) * 1000.0, 2)
    print(f"  Dual Shadow Cold-Start:     {results['dual_shadow_cold_start_ms']:.2f} ms")

    return results


def run_benchmark(n_runs=120, n_warmup=10):
    df_bench = build_test_dataframe()
    hw_info = get_hardware_info()
    print("=================================================================")
    print("VAYUDRISHTI PHASE 5B: RUNTIME LATENCY BENCHMARK")
    print("=================================================================")
    print(f"OS: {hw_info['os']}")
    print(f"Processor: {hw_info['processor']}")
    print(f"Logical Cores: {hw_info['cpu_count_logical']}")
    print(f"RAM: {hw_info['total_ram_gb']} GB")
    print(f"Batch Size: 1 contiguous 24-hour sequence (24 rows x 27 channels)")
    print(f"Steady-State Iterations: {n_runs} (after {n_warmup} warmups)")
    print("=================================================================")

    cold_starts = measure_cold_starts(df_bench)

    # Initialize warm steady-state service
    service = SensorForecastService()
    # Pre-cache all models
    service._load_historical_gbdt()
    service._load_adapted_models()

    print(f"\n--- WARMING UP ({n_warmup} runs) ---")
    for _ in range(n_warmup):
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=False)
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="adapted_severity_gated", enable_shadow=False)
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=True)
    print("  Warm-up complete.")

    print(f"\n--- BENCHMARKING STEADY-STATE INFERENCE ({n_runs} runs) ---")
    raw_times = {
        "historical_end_to_end": [],
        "adapted_end_to_end": [],
        "dual_shadow_end_to_end": [],
    }

    for i in range(n_runs):
        # 1. Historical Mode (End-to-End: DataFrame -> Preprocessing -> Forecast)
        t0 = time.perf_counter()
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=False)
        t1 = time.perf_counter()
        raw_times["historical_end_to_end"].append((t1 - t0) * 1000.0)

        # 2. Adapted Mode (End-to-End: DataFrame -> Preprocessing -> 49 Feats -> Adapted Forecast)
        t0 = time.perf_counter()
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="adapted_severity_gated", enable_shadow=False)
        t1 = time.perf_counter()
        raw_times["adapted_end_to_end"].append((t1 - t0) * 1000.0)

        # 3. Dual Shadow Mode (End-to-End: DataFrame -> Historical Forecast + Shadow Forecast)
        t0 = time.perf_counter()
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=True)
        t1 = time.perf_counter()
        raw_times["dual_shadow_end_to_end"].append((t1 - t0) * 1000.0)

    # Core inference benchmarks (skipping DataFrame creation/validation)
    seq_arr = service.underlying_service.preprocess_sequence(df_bench, station_id="DL001")
    raw_times["historical_core_inference"] = []
    raw_times["adapted_core_inference"] = []
    raw_times["dual_shadow_core_inference"] = []

    for _ in range(n_runs):
        # Historical Core
        t0 = time.perf_counter()
        service.get_forecast(station_id="DL001", df_24h=df_bench, model_mode="severity_gated", enable_shadow=False)
        t1 = time.perf_counter()
        raw_times["historical_core_inference"].append((t1 - t0) * 1000.0)

        # Adapted Core
        t0 = time.perf_counter()
        service._predict_adapted(df_bench, station_id="DL001", seq_arr=seq_arr, severity_gated=True)
        t1 = time.perf_counter()
        raw_times["adapted_core_inference"].append((t1 - t0) * 1000.0)

        # Dual Core
        t0 = time.perf_counter()
        res = service.underlying_service.predict_sequence(seq_arr)
        from src.data.tabular_features import extract_tabular_features
        tab = extract_tabular_features(seq_arr)
        service._historical_gbdt_pm25.predict(tab)
        service._historical_gbdt_pm10.predict(tab)
        service._predict_adapted(df_bench, station_id="DL001", seq_arr=seq_arr, severity_gated=True)
        t1 = time.perf_counter()
        raw_times["dual_shadow_core_inference"].append((t1 - t0) * 1000.0)

    # Calculate statistics
    stats = {}
    for key, vals in raw_times.items():
        arr = np.array(vals)
        stats[key] = {
            "n_runs": len(arr),
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "median_p50_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
        }

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "batch_size": 1,
        "cold_starts": cold_starts,
        "steady_state": stats,
    }

    # Print summary table
    print("\n=================================================================")
    print("STEADY-STATE LATENCY SUMMARY TABLE (All times in milliseconds)")
    print("=================================================================")
    print(f"{'Evaluation Mode':<32} | {'Mean':<8} | {'Median':<8} | {'p95':<8} | {'p99':<8} | {'Max':<8}")
    print("-" * 75)
    for k, v in stats.items():
        mode_label = k.replace("_", " ").title()
        print(f"{mode_label:<32} | {v['mean_ms']:<8.2f} | {v['median_p50_ms']:<8.2f} | {v['p95_ms']:<8.2f} | {v['p99_ms']:<8.2f} | {v['max_ms']:<8.2f}")
    print("=================================================================")

    out_file = ROOT_DIR / "experiments/recent_data_audit/runtime_latency_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Benchmark results successfully exported to: {out_file}")

    return output_payload


if __name__ == "__main__":
    run_benchmark(n_runs=120, n_warmup=10)
