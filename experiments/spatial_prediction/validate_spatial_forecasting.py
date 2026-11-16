"""
experiments/spatial_prediction/validate_spatial_forecasting.py

VayuDrishti Phase 13: Spatial Forecast Validation Experiment.
Evaluates spatial interpolation of genuine forward-looking model forecasts (t+1)
rather than spatial interpolation of simultaneous ground-truth observations.

Compares:
1. Direct Target-Station Temporal Forecast (LSTM)
2. Spatial Interpolation of Neighboring Station Forecasts (IDW-Forecast)
3. Nearest-Neighbor Station Forecast (NN-Forecast)
Against ground-truth observed concentration at target time t+1.
"""

import sys
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
OUTPUT_CSV = OUTPUT_DIR / "spatial_forecast_validation_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "spatial_forecast_validation_metrics.json"


def run_spatial_forecast_validation():
    print("=" * 78, flush=True)
    print("VAYUDRISHTI PHASE 13: SPATIAL FORECAST INTERPOLATION VALIDATION", flush=True)
    print("=" * 78, flush=True)

    seq_path = ROOT_DIR / "data/processed/sequences/air_quality_24h_sequence_index.csv"
    preds_path = ROOT_DIR / "models/lstm_baseline/test_predictions.npy"
    coords_path = ROOT_DIR / "data/processed/station_coordinates_90.csv"

    if not seq_path.exists() or not preds_path.exists() or not coords_path.exists():
        raise FileNotFoundError("Missing test sequence index, predictions, or station coordinates.")

    print("Loading test predictions and sequence index...", flush=True)
    preds = np.load(preds_path)
    seq = pd.read_csv(seq_path)

    test_mask = (seq["split"] == "test").values
    test_seq = seq.loc[test_mask].copy().reset_index(drop=True)

    if len(preds) != len(test_seq):
        raise ValueError(f"Mismatch: preds length {len(preds)} != test_seq length {len(test_seq)}")

    test_seq["pred_lstm_pm25"] = preds[:, 0]
    test_seq["pred_lstm_pm10"] = preds[:, 1]

    # Load coordinates
    df_coords = pd.read_csv(coords_path)
    df_coords["StationId"] = df_coords["StationId"].str.strip().str.upper()
    coord_map = {
        row["StationId"]: (float(row["Latitude"]), float(row["Longitude"]))
        for _, row in df_coords.iterrows()
        if not np.isnan(row["Latitude"]) and not np.isnan(row["Longitude"])
    }

    delhi_sids = set(df_coords[df_coords["City"].str.lower() == "delhi"]["StationId"])
    delhi_test = test_seq[test_seq["StationId"].isin(delhi_sids)].copy()
    print(f"Loaded {len(delhi_test):,} Delhi test sequences across {delhi_test['StationId'].nunique()} stations.", flush=True)

    # Find timestamps with at least 10 reporting stations
    ts_counts = delhi_test.groupby("target_time")["StationId"].nunique()
    rich_ts = ts_counts[ts_counts >= 10].index.tolist()
    print(f"Found {len(rich_ts)} test timestamps with >= 10 synchronous station forecasts.", flush=True)

    # Sample 60 diverse timestamps across the test period (Fall 2019 to Summer 2020)
    np.random.seed(42)
    sample_size = min(60, len(rich_ts))
    step = max(1, len(rich_ts) // sample_size)
    sampled_ts = rich_ts[::step][:sample_size]

    service = SpatialInterpolationService(
        config=SpatialInterpolationConfig(
            power=2.0,
            min_stations=3,
            max_distance_km=45.0,
            max_neighbors=12,
            exact_match_threshold_km=0.05,
        )
    )

    results = []
    t0 = time.time()

    for dt_str in sampled_ts:
        slice_df = delhi_test[delhi_test["target_time"] == dt_str].drop_duplicates("StationId")
        slice_stations = slice_df["StationId"].tolist()

        if len(slice_stations) < 5:
            continue

        for target_sid in slice_stations:
            target_row = slice_df[slice_df["StationId"] == target_sid].iloc[0]
            true_pm25 = float(target_row["target_PM2.5"])
            true_pm10 = float(target_row["target_PM10"])
            direct_temporal_pm25 = float(target_row["pred_lstm_pm25"])
            direct_temporal_pm10 = float(target_row["pred_lstm_pm10"])

            t_lat, t_lon = coord_map[target_sid]

            # Input stations: strictly EXCLUDE target_sid
            input_df = slice_df[slice_df["StationId"] != target_sid]
            # NOTICE: Ingesting FORECASTED values from neighboring stations!
            records = [
                StationForecastRecord(
                    station_id=row["StationId"],
                    latitude=coord_map[row["StationId"]][0],
                    longitude=coord_map[row["StationId"]][1],
                    predicted_pm25=float(row["pred_lstm_pm25"]),
                    predicted_pm10=float(row["pred_lstm_pm10"]),
                    target_time=dt_str,
                    issue_time=row["history_end"],
                    forecast_horizon="Next 1 Hour (t + 1)",
                    model_source="LSTM_Sensor_Forecast",
                )
                for _, row in input_df.iterrows()
                if row["StationId"] in coord_map
            ]

            if len(records) < 3:
                continue

            ctx = ForecastContext.from_records(records)

            # Spatial interpolation of neighboring forecasts
            pred = service.predict_at_location(
                latitude=t_lat,
                longitude=t_lon,
                forecast_context=ctx,
            )

            if pred.get("predicted_pm25") is None:
                continue

            idw_forecast_pm25 = pred["predicted_pm25"]
            idw_forecast_pm10 = pred["predicted_pm10"]
            nearest_sid = pred["nearest_station_id"]
            nearest_dist = pred["nearest_station_distance_km"]

            # Nearest-station forecast
            nearest_row = input_df[input_df["StationId"] == nearest_sid].iloc[0]
            nn_forecast_pm25 = float(nearest_row["pred_lstm_pm25"])
            nn_forecast_pm10 = float(nearest_row["pred_lstm_pm10"])

            results.append({
                "target_time": dt_str,
                "target_station": target_sid,
                "latitude": t_lat,
                "longitude": t_lon,
                "nearest_station": nearest_sid,
                "nearest_dist_km": nearest_dist,
                "true_pm25": true_pm25,
                "true_pm10": true_pm10,
                "direct_temporal_pm25": direct_temporal_pm25,
                "direct_temporal_pm10": direct_temporal_pm10,
                "idw_forecast_pm25": idw_forecast_pm25,
                "idw_forecast_pm10": idw_forecast_pm10,
                "nn_forecast_pm25": nn_forecast_pm25,
                "nn_forecast_pm10": nn_forecast_pm10,
                "ae_direct_temporal_pm25": abs(direct_temporal_pm25 - true_pm25),
                "ae_idw_forecast_pm25": abs(idw_forecast_pm25 - true_pm25),
                "ae_nn_forecast_pm25": abs(nn_forecast_pm25 - true_pm25),
                "ae_direct_temporal_pm10": abs(direct_temporal_pm10 - true_pm10),
                "ae_idw_forecast_pm10": abs(idw_forecast_pm10 - true_pm10),
                "ae_nn_forecast_pm10": abs(nn_forecast_pm10 - true_pm10),
                "se_direct_temporal_pm25": (direct_temporal_pm25 - true_pm25) ** 2,
                "se_idw_forecast_pm25": (idw_forecast_pm25 - true_pm25) ** 2,
                "se_nn_forecast_pm25": (nn_forecast_pm25 - true_pm25) ** 2,
                "se_direct_temporal_pm10": (direct_temporal_pm10 - true_pm10) ** 2,
                "se_idw_forecast_pm10": (idw_forecast_pm10 - true_pm10) ** 2,
                "se_nn_forecast_pm10": (nn_forecast_pm10 - true_pm10) ** 2,
            })

    elapsed = time.time() - t0
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Evaluated {len(res_df):,} spatial forecast instances across {len(sampled_ts)} timestamps in {elapsed:.2f}s.", flush=True)

    # Compute comparative metrics
    mae_direct_25 = float(res_df["ae_direct_temporal_pm25"].mean())
    mae_idw_fc_25 = float(res_df["ae_idw_forecast_pm25"].mean())
    mae_nn_fc_25 = float(res_df["ae_nn_forecast_pm25"].mean())

    rmse_direct_25 = float(np.sqrt(res_df["se_direct_temporal_pm25"].mean()))
    rmse_idw_fc_25 = float(np.sqrt(res_df["se_idw_forecast_pm25"].mean()))
    rmse_nn_fc_25 = float(np.sqrt(res_df["se_nn_forecast_pm25"].mean()))

    mae_direct_10 = float(res_df["ae_direct_temporal_pm10"].mean())
    mae_idw_fc_10 = float(res_df["ae_idw_forecast_pm10"].mean())
    mae_nn_fc_10 = float(res_df["ae_nn_forecast_pm10"].mean())

    rmse_direct_10 = float(np.sqrt(res_df["se_direct_temporal_pm10"].mean()))
    rmse_idw_fc_10 = float(np.sqrt(res_df["se_idw_forecast_pm10"].mean()))
    rmse_nn_fc_10 = float(np.sqrt(res_df["se_nn_forecast_pm10"].mean()))

    metrics = {
        "evaluation_name": "Phase13_Spatial_Forecast_Validation",
        "sample_count": len(res_df),
        "timestamps_evaluated": len(sampled_ts),
        "target_stations_evaluated": res_df["target_station"].nunique(),
        "execution_time_seconds": round(elapsed, 2),
        "pm25": {
            "direct_temporal_lstm_mae": round(mae_direct_25, 2),
            "idw_spatial_forecast_mae": round(mae_idw_fc_25, 2),
            "nearest_neighbor_forecast_mae": round(mae_nn_fc_25, 2),
            "direct_temporal_lstm_rmse": round(rmse_direct_25, 2),
            "idw_spatial_forecast_rmse": round(rmse_idw_fc_25, 2),
            "nearest_neighbor_forecast_rmse": round(rmse_nn_fc_25, 2),
            "idw_vs_nn_forecast_mae_gain_pct": round(((mae_nn_fc_25 - mae_idw_fc_25) / mae_nn_fc_25) * 100.0, 2),
        },
        "pm10": {
            "direct_temporal_lstm_mae": round(mae_direct_10, 2),
            "idw_spatial_forecast_mae": round(mae_idw_fc_10, 2),
            "nearest_neighbor_forecast_mae": round(mae_nn_fc_10, 2),
            "direct_temporal_lstm_rmse": round(rmse_direct_10, 2),
            "idw_spatial_forecast_rmse": round(rmse_idw_fc_10, 2),
            "nearest_neighbor_forecast_rmse": round(rmse_nn_fc_10, 2),
            "idw_vs_nn_forecast_mae_gain_pct": round(((mae_nn_fc_10 - mae_idw_fc_10) / mae_nn_fc_10) * 100.0, 2),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- SPATIAL FORECAST VALIDATION RESULTS ---", flush=True)
    print(f"PM2.5 MAE: Direct Target-Station LSTM = {mae_direct_25:.2f} µg/m³", flush=True)
    print(f"PM2.5 MAE: IDW Spatial Interpolation of Neighbor Forecasts = {mae_idw_fc_25:.2f} µg/m³", flush=True)
    print(f"PM2.5 MAE: Nearest-Station Neighbor Forecast = {mae_nn_fc_25:.2f} µg/m³", flush=True)
    print(f"IDW Forecast vs Nearest Forecast Gain: {metrics['pm25']['idw_vs_nn_forecast_mae_gain_pct']}%", flush=True)
    print(f"\nPM10 MAE: Direct Target-Station LSTM = {mae_direct_10:.2f} µg/m³", flush=True)
    print(f"PM10 MAE: IDW Spatial Interpolation of Neighbor Forecasts = {mae_idw_fc_10:.2f} µg/m³", flush=True)
    print(f"PM10 MAE: Nearest-Station Neighbor Forecast = {mae_nn_fc_10:.2f} µg/m³", flush=True)
    print(f"IDW Forecast vs Nearest Forecast Gain: {metrics['pm10']['idw_vs_nn_forecast_mae_gain_pct']}%", flush=True)
    print(f"Results saved to {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    run_spatial_forecast_validation()
