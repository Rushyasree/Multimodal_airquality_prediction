"""
experiments/spatial_prediction/validate_spatial_prediction.py

Leave-One-Station-Out (LOSO) Spatial Validation Experiment for VayuDrishti Phase 12.
Evaluates Modified IDW spatial interpolation against the Nearest-Station baseline
using historical synchronous CAAQMS observations across urban monitor clusters.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.services.spatial_interpolation_service import (
    SpatialInterpolationService,
    StationForecastRecord,
    ForecastContext,
)
from src.config.spatial_config import SpatialInterpolationConfig

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "spatial_validation_results.csv"
METRICS_JSON = OUTPUT_DIR / "spatial_validation_metrics.json"


def run_loso_spatial_validation():
    print("=" * 78)
    print("VAYUDRISHTI PHASE 12: LEAVE-ONE-STATION-OUT (LOSO) SPATIAL VALIDATION")
    print("=" * 78)

    coords_path = ROOT_DIR / "data/processed/station_coordinates_90.csv"
    data_path = ROOT_DIR / "data/processed/air_quality_hourly_clean.csv"

    if not coords_path.exists():
        raise FileNotFoundError(f"Missing station coordinates: {coords_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing air quality data: {data_path}")

    df_coords = pd.read_csv(coords_path)
    df_coords["StationId"] = df_coords["StationId"].str.strip().str.upper()
    coord_map = {
        row["StationId"]: (float(row["Latitude"]), float(row["Longitude"]))
        for _, row in df_coords.iterrows()
        if not np.isnan(row["Latitude"]) and not np.isnan(row["Longitude"])
    }

    # Focus on Delhi cluster (DL* prefix or City == 'Delhi') where station network is densest
    delhi_stations = set(df_coords[df_coords["City"].str.lower() == "delhi"]["StationId"].tolist())
    if not delhi_stations:
        delhi_stations = {sid for sid in coord_map if sid.startswith("DL")}

    print(f"Loaded {len(coord_map)} total station coordinates. Identified {len(delhi_stations)} dense cluster stations.")

    # Read slice of clean hourly data for Delhi cluster
    print("Scanning hourly dataset for multi-station synchronous slices...")
    # Read relevant columns only
    cols = ["StationId", "Datetime", "PM2.5", "PM10"]
    chunks = []
    # Read chunked to conserve memory and find dense multi-station timestamps
    for chunk in pd.read_csv(data_path, usecols=cols, chunksize=100000, low_memory=False):
        chunk["StationId"] = chunk["StationId"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["StationId"].isin(delhi_stations)].dropna(subset=["PM2.5", "PM10"])
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No matching records found for dense cluster stations.")

    df_dense = pd.concat(chunks, ignore_index=True)
    df_dense["Datetime"] = pd.to_datetime(df_dense["Datetime"])

    # Count stations per timestamp
    st_counts = df_dense.groupby("Datetime")["StationId"].nunique()
    # Select timestamps where at least 10 stations reported simultaneously
    rich_timestamps = st_counts[st_counts >= 10].index.tolist()
    print(f"Found {len(rich_timestamps)} synchronous hourly timestamps with >= 10 reporting monitors.")

    # Sample 60 diverse chronological timestamps across the period (seasons / diurnal)
    np.random.seed(42)
    sample_size = min(60, len(rich_timestamps))
    step = max(1, len(rich_timestamps) // sample_size)
    sampled_timestamps = rich_timestamps[::step][:sample_size]

    print(f"Evaluating Leave-One-Station-Out interpolation across {len(sampled_timestamps)} timestamps...")

    service = SpatialInterpolationService(
        config=SpatialInterpolationConfig(
            power=2.0,
            min_stations=3,
            max_distance_km=45.0,
            max_neighbors=12,
            exact_match_threshold_km=0.05,
        )
    )

    evaluation_rows = []
    t_start = time.time()

    for t_idx, dt in enumerate(sampled_timestamps, 1):
        dt_str = str(dt)
        slice_df = df_dense[df_dense["Datetime"] == dt].drop_duplicates("StationId")
        slice_stations = slice_df["StationId"].tolist()

        if len(slice_stations) < 5:
            continue

        # Iterate over each station as the hold-out target
        for target_sid in slice_stations:
            target_row = slice_df[slice_df["StationId"] == target_sid].iloc[0]
            true_pm25 = float(target_row["PM2.5"])
            true_pm10 = float(target_row["PM10"])
            t_lat, t_lon = coord_map[target_sid]

            # Pool of candidate input stations (strictly EXCLUDING target_sid)
            input_df = slice_df[slice_df["StationId"] != target_sid]
            records = [
                StationForecastRecord(
                    station_id=row["StationId"],
                    latitude=coord_map[row["StationId"]][0],
                    longitude=coord_map[row["StationId"]][1],
                    predicted_pm25=float(row["PM2.5"]),
                    predicted_pm10=float(row["PM10"]),
                    target_time=dt_str,
                    issue_time=dt_str,
                    forecast_horizon="Next 1 Hour (t + 1)",
                    model_source="Ground_Truth_Sensors",
                )
                for _, row in input_df.iterrows()
                if row["StationId"] in coord_map
            ]

            if len(records) < 3:
                continue

            ctx = ForecastContext.from_records(records)

            # 1. IDW Prediction at target coordinates
            pred = service.predict_at_location(
                latitude=t_lat,
                longitude=t_lon,
                forecast_context=ctx,
            )

            if pred.get("predicted_pm25") is None:
                continue

            idw_pm25 = pred["predicted_pm25"]
            idw_pm10 = pred["predicted_pm10"]
            nearest_sid = pred["nearest_station_id"]
            nearest_dist = pred["nearest_station_distance_km"]
            n_contrib = pred["num_contributing_stations"]

            # 2. Nearest-Station Baseline
            nearest_row = input_df[input_df["StationId"] == nearest_sid].iloc[0]
            nearest_pm25 = float(nearest_row["PM2.5"])
            nearest_pm10 = float(nearest_row["PM10"])

            # Errors
            err_idw_25 = abs(idw_pm25 - true_pm25)
            err_nn_25 = abs(nearest_pm25 - true_pm25)
            err_idw_10 = abs(idw_pm10 - true_pm10)
            err_nn_10 = abs(nearest_pm10 - true_pm10)

            evaluation_rows.append({
                "timestamp": dt_str,
                "target_station": target_sid,
                "latitude": t_lat,
                "longitude": t_lon,
                "true_pm25": true_pm25,
                "true_pm10": true_pm10,
                "idw_pm25": idw_pm25,
                "idw_pm10": idw_pm10,
                "nearest_station": nearest_sid,
                "nearest_dist_km": nearest_dist,
                "num_contributing": n_contrib,
                "nearest_pm25": nearest_pm25,
                "nearest_pm10": nearest_pm10,
                "ae_idw_pm25": err_idw_25,
                "ae_nn_pm25": err_nn_25,
                "ae_idw_pm10": err_idw_10,
                "ae_nn_pm10": err_nn_10,
                "se_idw_pm25": (idw_pm25 - true_pm25) ** 2,
                "se_nn_pm25": (nearest_pm25 - true_pm25) ** 2,
                "se_idw_pm10": (idw_pm10 - true_pm10) ** 2,
                "se_nn_pm10": (nearest_pm10 - true_pm10) ** 2,
            })

    elapsed = time.time() - t_start
    print(f"LOSO Validation complete in {elapsed:.2f}s. Evaluated {len(evaluation_rows)} held-out station instances.")

    results_df = pd.DataFrame(evaluation_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Exported granular validation cases to {RESULTS_CSV}")

    # Compute Summary Statistics
    mae_idw_25 = float(results_df["ae_idw_pm25"].mean())
    mae_nn_25 = float(results_df["ae_nn_pm25"].mean())
    rmse_idw_25 = float(np.sqrt(results_df["se_idw_pm25"].mean()))
    rmse_nn_25 = float(np.sqrt(results_df["se_nn_pm25"].mean()))

    mae_idw_10 = float(results_df["ae_idw_pm10"].mean())
    mae_nn_10 = float(results_df["ae_nn_pm10"].mean())
    rmse_idw_10 = float(np.sqrt(results_df["se_idw_pm10"].mean()))
    rmse_nn_10 = float(np.sqrt(results_df["se_nn_pm10"].mean()))

    # R2 scores
    ss_tot_25 = float(np.sum((results_df["true_pm25"] - results_df["true_pm25"].mean()) ** 2))
    r2_idw_25 = float(1.0 - (results_df["se_idw_pm25"].sum() / ss_tot_25)) if ss_tot_25 > 0 else 0.0
    r2_nn_25 = float(1.0 - (results_df["se_nn_pm25"].sum() / ss_tot_25)) if ss_tot_25 > 0 else 0.0

    ss_tot_10 = float(np.sum((results_df["true_pm10"] - results_df["true_pm10"].mean()) ** 2))
    r2_idw_10 = float(1.0 - (results_df["se_idw_pm10"].sum() / ss_tot_10)) if ss_tot_10 > 0 else 0.0
    r2_nn_10 = float(1.0 - (results_df["se_nn_pm10"].sum() / ss_tot_10)) if ss_tot_10 > 0 else 0.0

    # Error versus distance bins
    dist_bins = [0.0, 5.0, 10.0, 20.0, 50.0]
    bin_labels = ["0-5 km", "5-10 km", "10-20 km", "20-50 km"]
    results_df["dist_bin"] = pd.cut(results_df["nearest_dist_km"], bins=dist_bins, labels=bin_labels)

    distance_breakdown = {}
    for b in bin_labels:
        sub = results_df[results_df["dist_bin"] == b]
        if not sub.empty:
            distance_breakdown[b] = {
                "sample_count": len(sub),
                "mean_distance_km": round(float(sub["nearest_dist_km"].mean()), 2),
                "idw_pm25_mae": round(float(sub["ae_idw_pm25"].mean()), 2),
                "nn_pm25_mae": round(float(sub["ae_nn_pm25"].mean()), 2),
                "idw_pm10_mae": round(float(sub["ae_idw_pm10"].mean()), 2),
                "nn_pm10_mae": round(float(sub["ae_nn_pm10"].mean()), 2),
            }

    # Station-wise errors
    st_perf = results_df.groupby("target_station").agg(
        sample_count=("ae_idw_pm25", "count"),
        mean_nn_dist=("nearest_dist_km", "mean"),
        idw_pm25_mae=("ae_idw_pm25", "mean"),
        nn_pm25_mae=("ae_nn_pm25", "mean"),
    ).reset_index()

    station_summary = [
        {
            "station_id": row["target_station"],
            "sample_count": int(row["sample_count"]),
            "mean_nearest_dist_km": round(float(row["mean_nn_dist"]), 2),
            "idw_pm25_mae": round(float(row["idw_pm25_mae"]), 2),
            "nn_pm25_mae": round(float(row["nn_pm25_mae"]), 2),
        }
        for _, row in st_perf.iterrows()
    ]

    metrics = {
        "evaluation_name": "VayuDrishti_Phase12_LOSO_Spatial_Validation",
        "sample_count": len(results_df),
        "timestamps_evaluated": len(sampled_timestamps),
        "target_stations_evaluated": results_df["target_station"].nunique(),
        "execution_time_seconds": round(elapsed, 2),
        "pm25": {
            "idw_mae": round(mae_idw_25, 2),
            "nn_mae": round(mae_nn_25, 2),
            "mae_reduction_pct": round(((mae_nn_25 - mae_idw_25) / mae_nn_25) * 100.0, 2),
            "idw_rmse": round(rmse_idw_25, 2),
            "nn_rmse": round(rmse_nn_25, 2),
            "rmse_reduction_pct": round(((rmse_nn_25 - rmse_idw_25) / rmse_nn_25) * 100.0, 2),
            "idw_r2": round(r2_idw_25, 4),
            "nn_r2": round(r2_nn_25, 4),
        },
        "pm10": {
            "idw_mae": round(mae_idw_10, 2),
            "nn_mae": round(mae_nn_10, 2),
            "mae_reduction_pct": round(((mae_nn_10 - mae_idw_10) / mae_nn_10) * 100.0, 2),
            "idw_rmse": round(rmse_idw_10, 2),
            "nn_rmse": round(rmse_nn_10, 2),
            "rmse_reduction_pct": round(((rmse_nn_10 - rmse_idw_10) / rmse_nn_10) * 100.0, 2),
            "idw_r2": round(r2_idw_10, 4),
            "nn_r2": round(r2_nn_10, 4),
        },
        "distance_breakdown": distance_breakdown,
        "station_summary": station_summary,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n--- SPATIAL VALIDATION BENCHMARK RESULTS ---")
    print(f"PM2.5 MAE: IDW = {mae_idw_25:.2f} µg/m³ vs Nearest = {mae_nn_25:.2f} µg/m³ (Improvement: {metrics['pm25']['mae_reduction_pct']}%)")
    print(f"PM2.5 RMSE: IDW = {rmse_idw_25:.2f} µg/m³ vs Nearest = {rmse_nn_25:.2f} µg/m³ (Improvement: {metrics['pm25']['rmse_reduction_pct']}%)")
    print(f"PM2.5 R²: IDW = {r2_idw_25:.4f} vs Nearest = {r2_nn_25:.4f}")
    print(f"PM10 MAE: IDW = {mae_idw_10:.2f} µg/m³ vs Nearest = {mae_nn_10:.2f} µg/m³ (Improvement: {metrics['pm10']['mae_reduction_pct']}%)")
    print(f"Metrics saved to {METRICS_JSON}")


if __name__ == "__main__":
    run_loso_spatial_validation()
