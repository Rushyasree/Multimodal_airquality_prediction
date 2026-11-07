"""
scripts/run_phase13_spatial_advection.py

Executes comprehensive Spatial-Temporal Cross-Station Advection & Multi-Node Modeling for Phase 13.
Analyzes:
1. Intra-urban cluster geometry (distance, forward/backward azimuth).
2. Pairwise Cross-Correlation Functions (CCF) for PM2.5 and PM10 at lags [-12, +12] hours.
3. Wind vector projection, physical advection transit times, and empirical lag alignment.
4. Causal predictive value of neighboring spatial lags on next-hour concentration forecasts.

Outputs:
- experiments/phase13/spatial_clusters.csv
- experiments/phase13/cross_correlation_analysis.csv
- experiments/phase13/advection_wind_alignment.csv
- experiments/phase13/spatial_forecast_comparison.csv
"""

import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.services.spatial_advection_service import (
    SpatialAdvectionService,
    compute_cross_correlation,
    compute_along_transect_wind,
    haversine_distance,
    calculate_bearing,
)

PHASE13_DIR = ROOT_DIR / "experiments" / "phase13"
RECENT_AQ_CSV = ROOT_DIR / "data" / "processed" / "recent_air_quality_cleaned.csv"
WEATHER_DIR = ROOT_DIR / "data" / "raw" / "era5_land" / "station_weather"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-6 else 0.0
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def main():
    print("=" * 78)
    print("VAYUDRISHTI PHASE 13: SPATIAL-TEMPORAL CROSS-STATION ADVECTION ANALYSIS")
    print("=" * 78)

    PHASE13_DIR.mkdir(parents=True, exist_ok=True)
    service = SpatialAdvectionService()

    # 1. Output spatial clusters metadata
    print("\n--- 1. Intra-Urban Spatial Clusters ---")
    df_clusters = pd.DataFrame(service.clusters)
    df_clusters.to_csv(PHASE13_DIR / "spatial_clusters.csv", index=False)
    print(df_clusters[["cluster_name", "city", "station_a", "station_b", "distance_km", "bearing_a_to_b", "bearing_b_to_a"]])

    # 2. Load cleaned hourly air quality data
    print(f"\nLoading cleaned hourly telemetry from {RECENT_AQ_CSV.name}...")
    df_aq = pd.read_csv(RECENT_AQ_CSV, usecols=["station_id", "Datetime", "PM2.5", "PM10"])
    df_aq["dt"] = pd.to_datetime(df_aq["Datetime"], utc=True)

    # Cross-Correlation & Advection alignment results
    ccf_records = []
    advection_records = []
    forecast_comparison_records = []

    for cluster in service.clusters:
        c_name = cluster["cluster_name"]
        city = cluster["city"]
        sa, sb = cluster["station_a"], cluster["station_b"]
        dist_km = cluster["distance_km"]
        bearing_ab = cluster["bearing_a_to_b"]
        bearing_ba = cluster["bearing_b_to_a"]

        print(f"\nAnalyzing Cluster: {c_name} ({sa} <-> {sb}, distance={dist_km} km)...")

        # Extract aligned observations
        df_a = df_aq[df_aq["station_id"] == sa].set_index("dt").sort_index()
        df_b = df_aq[df_aq["station_id"] == sb].set_index("dt").sort_index()

        joint = df_a[["PM2.5", "PM10"]].join(df_b[["PM2.5", "PM10"]], lsuffix="_A", rsuffix="_B", how="inner").dropna()
        print(f"  Overlapping continuous hourly observations: {len(joint):,}")

        # Cross-Correlation Analysis for PM2.5 and PM10
        for param in ["PM2.5", "PM10"]:
            ccf_res = service.analyze_cross_correlation(joint, param=param, max_lag=12)
            for lag, corr in ccf_res["ccf_curve"].items():
                ccf_records.append({
                    "cluster_name": c_name,
                    "city": city,
                    "station_a": sa,
                    "station_b": sb,
                    "parameter": param,
                    "distance_km": dist_km,
                    "lag_hours": lag,
                    "cross_correlation": corr,
                    "is_peak_lag": lag == ccf_res["peak_correlation_lag_hours"],
                })

            print(f"  [{param}] Lag 0 Corr: {ccf_res['lag_zero_correlation']:.4f} | Peak Corr: {ccf_res['peak_correlation_value']:.4f} at Lag {ccf_res['peak_correlation_lag_hours']:+d}h")

        # Load Weather Data for Advection Modeling
        wf_a = WEATHER_DIR / f"{sa}_weather.csv"
        wf_b = WEATHER_DIR / f"{sb}_weather.csv"
        target_wf = wf_a if wf_a.exists() else wf_b

        if target_wf.exists():
            df_w = pd.read_csv(target_wf)
            time_col = "timestamp" if "timestamp" in df_w.columns else "Datetime"
            df_w["dt"] = pd.to_datetime(df_w[time_col], utc=True)
            df_w = df_w.set_index("dt").sort_index()

            joint_w = joint.join(df_w[["wind_speed", "wind_direction"]], how="inner").dropna()

            if not joint_w.empty:
                adv_res = service.evaluate_wind_advection_alignment(joint_w, dist_km, bearing_ab, param="PM2.5")
                advection_records.append({
                    "cluster_name": c_name,
                    "city": city,
                    "transect": f"{sa} -> {sb}",
                    "distance_km": dist_km,
                    "transect_bearing_deg": bearing_ab,
                    "total_aligned_hours": len(joint_w),
                    "favorable_wind_hours": adv_res.get("favorable_wind_hours", 0),
                    "mean_along_wind_velocity_ms": adv_res.get("mean_favorable_velocity_ms", 0.0),
                    "theoretical_advection_transit_hours": adv_res.get("theoretical_advection_hours", 0.0),
                    "empirical_peak_lag_under_wind": adv_res.get("empirical_peak_lag_under_wind", 0),
                    "empirical_peak_corr_under_wind": adv_res.get("empirical_peak_corr_under_wind", 0.0),
                })
                print(f"  [Wind Advection {sa}->{sb}] Theoretical transit time: {adv_res.get('theoretical_advection_hours', 0.0):.1f}h | Empirical Peak Lag: {adv_res.get('empirical_peak_lag_under_wind', 0)}h (r={adv_res.get('empirical_peak_corr_under_wind', 0.0):.4f})")

        # -------------------------------------------------------------
        # Causal Spatial-Temporal Next-Hour Forecast Value
        # Predict Station B next hour (t+1) using Station B history vs (Station B + Station A history <= t)
        # -------------------------------------------------------------
        print(f"  Evaluating predictive value of spatial neighbor lags (predicting {sb} at t+1)...")
        # Build features strictly using past observations <= t
        # Target: PM2.5_B at t+1
        df_model = joint.copy()
        y_target = df_model["PM2.5_B"].shift(-1).iloc[:-1].values  # target at t+1

        # Base Autoregressive Features of Station B (up to t, zero lookahead)
        feat_single = []
        feat_spatial = []

        pm_b = df_model["PM2.5_B"].iloc[:-1].values
        pm_a = df_model["PM2.5_A"].iloc[:-1].values

        n_pts = len(y_target)
        # Lags 0, 1, 2, 3, 5, 11 (relative to t)
        lags = [0, 1, 2, 3, 5, 11]

        # Assemble matrices
        X_single = np.column_stack([df_model["PM2.5_B"].shift(k).iloc[:-1].bfill().values for k in lags])
        X_neighbor_lags = np.column_stack([df_model["PM2.5_A"].shift(k).iloc[:-1].bfill().values for k in lags])
        X_spatial = np.hstack([X_single, X_neighbor_lags])

        # 80/20 Chronological train/test split (no random shuffling, strictly preserving temporal causality)
        split_idx = int(0.80 * n_pts)

        X_train_s, X_test_s = X_single[:split_idx], X_single[split_idx:]
        X_train_m, X_test_m = X_spatial[:split_idx], X_spatial[split_idx:]
        y_train, y_test = y_target[:split_idx], y_target[split_idx:]

        # Persistence baseline at t: y_pred = PM2.5_B(t)
        y_pred_persist = X_test_s[:, 0]

        # Single-Station Model
        model_s = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model_s.fit(X_train_s, y_train)
        y_pred_s = model_s.predict(X_test_s)

        # Multi-Node Spatial Model (adds Station A lags <= t)
        model_m = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model_m.fit(X_train_m, y_train)
        y_pred_m = model_m.predict(X_test_m)

        m_persist = compute_metrics(y_test, y_pred_persist)
        m_s = compute_metrics(y_test, y_pred_s)
        m_m = compute_metrics(y_test, y_pred_m)

        imp_s_vs_p = ((m_persist["mae"] - m_s["mae"]) / m_persist["mae"]) * 100.0
        imp_m_vs_s = ((m_s["mae"] - m_m["mae"]) / m_s["mae"]) * 100.0
        imp_m_vs_p = ((m_persist["mae"] - m_m["mae"]) / m_persist["mae"]) * 100.0

        forecast_comparison_records.append({
            "cluster_name": c_name,
            "city": city,
            "target_station": sb,
            "neighbor_station": sa,
            "distance_km": dist_km,
            "test_samples": len(y_test),
            "persistence_mae": m_persist["mae"],
            "persistence_rmse": m_persist["rmse"],
            "persistence_r2": m_persist["r2"],
            "single_station_gbdt_mae": m_s["mae"],
            "single_station_gbdt_rmse": m_s["rmse"],
            "single_station_gbdt_r2": m_s["r2"],
            "multinode_spatial_gbdt_mae": m_m["mae"],
            "multinode_spatial_gbdt_rmse": m_m["rmse"],
            "multinode_spatial_gbdt_r2": m_m["r2"],
            "spatial_mae_reduction_vs_single": round(m_s["mae"] - m_m["mae"], 4),
            "pct_gain_from_spatial_lags": round(imp_m_vs_s, 2),
            "pct_gain_vs_persistence": round(imp_m_vs_p, 2),
        })
        print(f"  Results for predicting {sb}:")
        print(f"    Persistence MAE: {m_persist['mae']:.2f}")
        print(f"    Single-Station GBDT MAE: {m_s['mae']:.2f} ({imp_s_vs_p:+.2f}% vs persist)")
        print(f"    Multi-Node Spatial GBDT MAE: {m_m['mae']:.2f} ({imp_m_vs_s:+.2f}% incremental gain from neighbor)")

    # Save CSV outputs
    df_ccf = pd.DataFrame(ccf_records)
    df_ccf.to_csv(PHASE13_DIR / "cross_correlation_analysis.csv", index=False)
    print(f"\nSaved {PHASE13_DIR / 'cross_correlation_analysis.csv'}")

    df_adv = pd.DataFrame(advection_records)
    df_adv.to_csv(PHASE13_DIR / "advection_wind_alignment.csv", index=False)
    print(f"Saved {PHASE13_DIR / 'advection_wind_alignment.csv'}")

    df_fc = pd.DataFrame(forecast_comparison_records)
    df_fc.to_csv(PHASE13_DIR / "spatial_forecast_comparison.csv", index=False)
    print(f"Saved {PHASE13_DIR / 'spatial_forecast_comparison.csv'}")

    print("\nSummary of Multi-Node Spatial Value:")
    print(df_fc[["city", "target_station", "neighbor_station", "persistence_mae", "single_station_gbdt_mae", "multinode_spatial_gbdt_mae", "pct_gain_from_spatial_lags"]])


if __name__ == "__main__":
    main()
