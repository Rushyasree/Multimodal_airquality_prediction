"""
scripts/audit_stage5_live_trial.py

Independent statistical audit of the live shadow trial:
- 418 matched prediction-observation pairs from OpenAQ/CPCB (experiments/recent_data_audit/live_trial_evaluation.csv)
- Stratified by station (AP001, AP003, AP004, AP005)
- Station-stratified block bootstrap (24h, 48h, 72h blocks) for paired error difference (GBDT vs Persistence)
- Recomputes MAE, RMSE, MBE, MedAE, and saves LIVE_TRIAL_RECOMPUTED_METRICS.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    mbe = float(np.mean(diff))
    medae = float(np.median(abs_diff))
    return {
        "count": int(len(y_true)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mbe": round(mbe, 4),
        "medae": round(medae, 4)
    }

def block_bootstrap_paired_diff(df: pd.DataFrame, block_hours: int = 24, n_boot: int = 1000, seed: int = 42):
    """Performs station-stratified block bootstrap of mean paired error difference:
       diff = |GBDT error| - |Persistence error|.
       Negative value indicates GBDT is superior to Persistence.
    """
    rng = np.random.RandomState(seed)
    stations = df["station_id"].unique()
    
    boot_diffs_pm25 = []
    boot_diffs_pm10 = []
    
    # Pre-segment into station blocks
    station_blocks = {}
    for stn in stations:
        df_stn = df[df["station_id"] == stn].sort_values("target_time_utc").copy()
        df_stn["target_dt"] = pd.to_datetime(df_stn["target_time_utc"])
        
        # Group into contiguous blocks based on timestamps
        blocks = []
        if len(df_stn) > 0:
            start_t = df_stn["target_dt"].min()
            end_t = df_stn["target_dt"].max()
            curr = start_t
            while curr <= end_t:
                next_t = curr + pd.Timedelta(hours=block_hours)
                blk = df_stn[(df_stn["target_dt"] >= curr) & (df_stn["target_dt"] < next_t)]
                if len(blk) > 0:
                    blocks.append(blk)
                curr = next_t
        station_blocks[stn] = blocks

    for b in range(n_boot):
        sample_dfs = []
        for stn in stations:
            blks = station_blocks[stn]
            if len(blks) == 0:
                continue
            sampled_idx = rng.choice(len(blks), size=len(blks), replace=True)
            for idx in sampled_idx:
                sample_dfs.append(blks[idx])
        
        df_sample = pd.concat(sample_dfs, ignore_index=True)
        # diff = GBDT abs err - Pers abs err
        d25 = np.mean(df_sample["gbdt_abs_err_pm25"] - df_sample["pers_abs_err_pm25"])
        d10 = np.mean(df_sample["gbdt_abs_err_pm10"] - df_sample["pers_abs_err_pm10"])
        boot_diffs_pm25.append(d25)
        boot_diffs_pm10.append(d10)
        
    diff_arr_25 = np.array(boot_diffs_pm25)
    diff_arr_10 = np.array(boot_diffs_pm10)
    
    ci_25 = (float(np.percentile(diff_arr_25, 2.5)), float(np.percentile(diff_arr_25, 97.5)))
    ci_10 = (float(np.percentile(diff_arr_10, 2.5)), float(np.percentile(diff_arr_10, 97.5)))
    p_val_25 = float(np.mean(diff_arr_25 >= 0.0))
    p_val_10 = float(np.mean(diff_arr_10 >= 0.0))
    
    return {
        "block_hours": block_hours,
        "pm25_mean_diff": round(float(np.mean(diff_arr_25)), 4),
        "pm25_ci_lower": round(ci_25[0], 4),
        "pm25_ci_upper": round(ci_25[1], 4),
        "pm25_p_val": round(p_val_25, 4),
        "pm10_mean_diff": round(float(np.mean(diff_arr_10)), 4),
        "pm10_ci_lower": round(ci_10[0], 4),
        "pm10_ci_upper": round(ci_10[1], 4),
        "pm10_p_val": round(p_val_10, 4),
    }

def main():
    print("--- Stage 5: Live Shadow Trial Independent Audit ---")
    live_eval_path = ROOT_DIR / "experiments/recent_data_audit/live_trial_evaluation.csv"
    df_eval = pd.read_csv(live_eval_path)
    print(f"Loaded {len(df_eval)} matched evaluation pairs from {live_eval_path}")

    y_true_pm25 = df_eval["actual_pm25"].values
    y_true_pm10 = df_eval["actual_pm10"].values
    y_pred_gbdt_pm25 = df_eval["pred_pm25"].values
    y_pred_gbdt_pm10 = df_eval["pred_pm10"].values
    y_pred_pers_pm25 = df_eval["persistence_pm25"].values
    y_pred_pers_pm10 = df_eval["persistence_pm10"].values

    audit_rows = []

    # Overall Pooled Metrics
    gbdt_m25 = compute_metrics(y_true_pm25, y_pred_gbdt_pm25)
    gbdt_m10 = compute_metrics(y_true_pm10, y_pred_gbdt_pm10)
    pers_m25 = compute_metrics(y_true_pm25, y_pred_pers_pm25)
    pers_m10 = compute_metrics(y_true_pm10, y_pred_pers_pm10)

    print("\n--- Pooled Live Trial Results (N = 418) ---")
    print(f"Persistence PM2.5 | MAE: {pers_m25['mae']:6.2f}, RMSE: {pers_m25['rmse']:6.2f}, R2: {pers_m25['r2']:6.4f}, MBE: {pers_m25['mbe']:6.2f}")
    print(f"GBDT        PM2.5 | MAE: {gbdt_m25['mae']:6.2f}, RMSE: {gbdt_m25['rmse']:6.2f}, R2: {gbdt_m25['r2']:6.4f}, MBE: {gbdt_m25['mbe']:6.2f}")
    print(f"Persistence PM10  | MAE: {pers_m10['mae']:6.2f}, RMSE: {pers_m10['rmse']:6.2f}, R2: {pers_m10['r2']:6.4f}, MBE: {pers_m10['mbe']:6.2f}")
    print(f"GBDT        PM10  | MAE: {gbdt_m10['mae']:6.2f}, RMSE: {gbdt_m10['rmse']:6.2f}, R2: {gbdt_m10['r2']:6.4f}, MBE: {gbdt_m10['mbe']:6.2f}")

    audit_rows.append({"station": "POOLED_ALL", "model": "Persistence", "pollutant": "PM2.5", **pers_m25})
    audit_rows.append({"station": "POOLED_ALL", "model": "Adapted GBDT", "pollutant": "PM2.5", **gbdt_m25})
    audit_rows.append({"station": "POOLED_ALL", "model": "Persistence", "pollutant": "PM10", **pers_m10})
    audit_rows.append({"station": "POOLED_ALL", "model": "Adapted GBDT", "pollutant": "PM10", **gbdt_m10})

    # Station-wise Stratification
    stations = sorted(df_eval["station_id"].unique())
    print(f"\n--- Station Stratification ({len(stations)} Stations) ---")
    for stn in stations:
        sub = df_eval[df_eval["station_id"] == stn]
        s_true_25 = sub["actual_pm25"].values
        s_true_10 = sub["actual_pm10"].values
        
        s_gbdt_25 = compute_metrics(s_true_25, sub["pred_pm25"].values)
        s_pers_25 = compute_metrics(s_true_25, sub["persistence_pm25"].values)
        s_gbdt_10 = compute_metrics(s_true_10, sub["pred_pm10"].values)
        s_pers_10 = compute_metrics(s_true_10, sub["persistence_pm10"].values)
        
        print(f"\n[Station {stn}] (N = {len(sub)})")
        print(f"  PM2.5 | GBDT MAE: {s_gbdt_25['mae']:6.2f} vs Pers MAE: {s_pers_25['mae']:6.2f} (diff: {s_gbdt_25['mae']-s_pers_25['mae']:+.2f})")
        print(f"  PM10  | GBDT MAE: {s_gbdt_10['mae']:6.2f} vs Pers MAE: {s_pers_10['mae']:6.2f} (diff: {s_gbdt_10['mae']-s_pers_10['mae']:+.2f})")
        
        audit_rows.append({"station": stn, "model": "Persistence", "pollutant": "PM2.5", **s_pers_25})
        audit_rows.append({"station": stn, "model": "Adapted GBDT", "pollutant": "PM2.5", **s_gbdt_25})
        audit_rows.append({"station": stn, "model": "Persistence", "pollutant": "PM10", **s_pers_10})
        audit_rows.append({"station": stn, "model": "Adapted GBDT", "pollutant": "PM10", **s_gbdt_10})

    # Block Bootstrap Sensitivity Analysis
    print("\n--- Block Bootstrap Paired Difference Sensitivity Analysis (GBDT - Persistence) ---")
    boot_results = []
    for bh in [24, 48, 72]:
        res = block_bootstrap_paired_diff(df_eval, block_hours=bh, n_boot=1000, seed=42)
        boot_results.append(res)
        print(f"Block: {bh:2d}h | PM2.5 Mean Diff: {res['pm25_mean_diff']:+.3f} (95% CI: [{res['pm25_ci_lower']:+.3f}, {res['pm25_ci_upper']:+.3f}], p={res['pm25_p_val']:.3f}) | PM10 Mean Diff: {res['pm10_mean_diff']:+.3f} (95% CI: [{res['pm10_ci_lower']:+.3f}, {res['pm10_ci_upper']:+.3f}], p={res['pm10_p_val']:.3f})")

    df_out = pd.DataFrame(audit_rows)
    out_csv = ROOT_DIR / "LIVE_TRIAL_RECOMPUTED_METRICS.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved recomputed live trial metrics to {out_csv}")

if __name__ == "__main__":
    main()
