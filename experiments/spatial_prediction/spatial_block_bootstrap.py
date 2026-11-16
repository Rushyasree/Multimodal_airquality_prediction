"""
experiments/spatial_prediction/spatial_block_bootstrap.py

VayuDrishti Phase 13: Temporal Block Bootstrap for Spatial Evaluation.
Properly accounts for contemporaneous spatial cross-correlation across stations
and serial dependence across timestamps.

Optimized with precomputed per-timestamp error arrays for instantaneous sampling.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = OUTPUT_DIR / "spatial_validation_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "spatial_block_bootstrap_results.json"
OUTPUT_CSV = OUTPUT_DIR / "spatial_block_bootstrap_results.csv"

SEED = 42
N_BOOTSTRAP = 2000


def run_temporal_block_bootstrap():
    t0_start = time.time()
    print("=" * 78, flush=True)
    print("VAYUDRISHTI PHASE 13: TEMPORAL BLOCK BOOTSTRAP SIGNIFICANCE EVALUATION", flush=True)
    print("=" * 78, flush=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input results: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    unique_timestamps = sorted(df["timestamp"].unique())
    T = len(unique_timestamps)
    print(f"Total evaluated observations: {len(df):,}", flush=True)
    print(f"Total distinct chronological timestamps: {T}", flush=True)

    # Precompute per-timestamp NumPy arrays to avoid pandas concat overhead
    # Each timestamp maps to its arrays of absolute errors and squared errors
    ts_map = {t: idx for idx, t in enumerate(unique_timestamps)}
    df["ts_idx"] = df["timestamp"].map(ts_map)

    ae_idw_25_by_t = [df.loc[df["ts_idx"] == i, "ae_idw_pm25"].values for i in range(T)]
    ae_nn_25_by_t = [df.loc[df["ts_idx"] == i, "ae_nn_pm25"].values for i in range(T)]
    se_idw_25_by_t = [df.loc[df["ts_idx"] == i, "se_idw_pm25"].values for i in range(T)]
    se_nn_25_by_t = [df.loc[df["ts_idx"] == i, "se_nn_pm25"].values for i in range(T)]

    ae_idw_10_by_t = [df.loc[df["ts_idx"] == i, "ae_idw_pm10"].values for i in range(T)]
    ae_nn_10_by_t = [df.loc[df["ts_idx"] == i, "ae_nn_pm10"].values for i in range(T)]
    se_idw_10_by_t = [df.loc[df["ts_idx"] == i, "se_idw_pm10"].values for i in range(T)]
    se_nn_10_by_t = [df.loc[df["ts_idx"] == i, "se_nn_pm10"].values for i in range(T)]

    # Pre-concatenate per-timestamp statistics
    sum_ae_idw_25 = np.array([np.sum(x) for x in ae_idw_25_by_t])
    sum_ae_nn_25 = np.array([np.sum(x) for x in ae_nn_25_by_t])
    sum_se_idw_25 = np.array([np.sum(x) for x in se_idw_25_by_t])
    sum_se_nn_25 = np.array([np.sum(x) for x in se_nn_25_by_t])

    sum_ae_idw_10 = np.array([np.sum(x) for x in ae_idw_10_by_t])
    sum_ae_nn_10 = np.array([np.sum(x) for x in ae_nn_10_by_t])
    sum_se_idw_10 = np.array([np.sum(x) for x in se_idw_10_by_t])
    sum_se_nn_10 = np.array([np.sum(x) for x in se_nn_10_by_t])

    counts_by_t = np.array([len(x) for x in ae_idw_25_by_t])

    block_sizes = [1, 3, 6, 12]

    np.random.seed(SEED)
    bootstrap_results = {}
    summary_rows = []

    for B in block_sizes:
        print(f"\n--- Running Block Bootstrap: Block Length B = {B} timestamps ({T // B} effective blocks) ---", flush=True)

        n_blocks_possible = T - B + 1
        # Contiguous index blocks
        blocks = [list(range(i, i + B)) for i in range(n_blocks_possible)]
        blocks_to_draw = int(np.ceil(T / B))

        # Sample matrix of block indices: (N_BOOTSTRAP, blocks_to_draw)
        sampled_block_ids = np.random.choice(len(blocks), size=(N_BOOTSTRAP, blocks_to_draw), replace=True)

        # Collect results
        diff_mae_pm25 = np.empty(N_BOOTSTRAP, dtype=np.float64)
        diff_mae_pm10 = np.empty(N_BOOTSTRAP, dtype=np.float64)
        diff_rmse_pm25 = np.empty(N_BOOTSTRAP, dtype=np.float64)
        diff_rmse_pm10 = np.empty(N_BOOTSTRAP, dtype=np.float64)
        rel_gain_pm25 = np.empty(N_BOOTSTRAP, dtype=np.float64)
        rel_gain_pm10 = np.empty(N_BOOTSTRAP, dtype=np.float64)

        for rep in range(N_BOOTSTRAP):
            rep_block_indices = sampled_block_ids[rep]
            # Flatten sampled timestamp indices, taking first T
            rep_t_indices = [idx for b_idx in rep_block_indices for idx in blocks[b_idx]][:T]

            # Vectorized sum across sampled timestamps
            total_n = np.sum(counts_by_t[rep_t_indices])
            m_idw_25 = np.sum(sum_ae_idw_25[rep_t_indices]) / total_n
            m_nn_25 = np.sum(sum_ae_nn_25[rep_t_indices]) / total_n
            r_idw_25 = np.sqrt(np.sum(sum_se_idw_25[rep_t_indices]) / total_n)
            r_nn_25 = np.sqrt(np.sum(sum_se_nn_25[rep_t_indices]) / total_n)

            m_idw_10 = np.sum(sum_ae_idw_10[rep_t_indices]) / total_n
            m_nn_10 = np.sum(sum_ae_nn_10[rep_t_indices]) / total_n
            r_idw_10 = np.sqrt(np.sum(sum_se_idw_10[rep_t_indices]) / total_n)
            r_nn_10 = np.sqrt(np.sum(sum_se_nn_10[rep_t_indices]) / total_n)

            d_mae_25 = m_nn_25 - m_idw_25
            d_mae_10 = m_nn_10 - m_idw_10
            d_rmse_25 = r_nn_25 - r_idw_25
            d_rmse_10 = r_nn_10 - r_idw_10

            g_25 = (d_mae_25 / m_nn_25) * 100.0
            g_10 = (d_mae_10 / m_nn_10) * 100.0

            diff_mae_pm25[rep] = d_mae_25
            diff_mae_pm10[rep] = d_mae_10
            diff_rmse_pm25[rep] = d_rmse_25
            diff_rmse_pm10[rep] = d_rmse_10
            rel_gain_pm25[rep] = g_25
            rel_gain_pm10[rep] = g_10

        # Percentile CIs
        ci_d_mae_25 = (float(np.percentile(diff_mae_pm25, 2.5)), float(np.percentile(diff_mae_pm25, 97.5)))
        ci_d_mae_10 = (float(np.percentile(diff_mae_pm10, 2.5)), float(np.percentile(diff_mae_pm10, 97.5)))
        ci_gain_25 = (float(np.percentile(rel_gain_pm25, 2.5)), float(np.percentile(rel_gain_pm25, 97.5)))
        ci_gain_10 = (float(np.percentile(rel_gain_pm10, 2.5)), float(np.percentile(rel_gain_pm10, 97.5)))

        ci_d_rmse_25 = (float(np.percentile(diff_rmse_pm25, 2.5)), float(np.percentile(diff_rmse_pm25, 97.5)))
        ci_d_rmse_10 = (float(np.percentile(diff_rmse_pm10, 2.5)), float(np.percentile(diff_rmse_pm10, 97.5)))

        p_val_mae_25 = float(np.mean(diff_mae_pm25 <= 0.0))
        p_val_mae_10 = float(np.mean(diff_mae_pm10 <= 0.0))

        print(f"PM2.5 Paired MAE Improvement (NN - IDW): {np.mean(diff_mae_pm25):.2f} µg/m³ (95% CI: [{ci_d_mae_25[0]:.2f}, {ci_d_mae_25[1]:.2f}])", flush=True)
        print(f"PM2.5 Relative Gain: {np.mean(rel_gain_pm25):.2f}% (95% CI: [{ci_gain_25[0]:.2f}%, {ci_gain_25[1]:.2f}%]), p = {p_val_mae_25:.4f}", flush=True)
        print(f"PM10 Paired MAE Improvement (NN - IDW): {np.mean(diff_mae_pm10):.2f} µg/m³ (95% CI: [{ci_d_mae_10[0]:.2f}, {ci_d_mae_10[1]:.2f}])", flush=True)
        print(f"PM10 Relative Gain: {np.mean(rel_gain_pm10):.2f}% (95% CI: [{ci_gain_10[0]:.2f}%, {ci_gain_10[1]:.2f}%]), p = {p_val_mae_10:.4f}", flush=True)

        block_key = f"block_size_{B}"
        bootstrap_results[block_key] = {
            "block_size_timestamps": B,
            "effective_blocks": T // B,
            "bootstrap_replicates": N_BOOTSTRAP,
            "pm25": {
                "mean_paired_mae_diff_ug_m3": round(float(np.mean(diff_mae_pm25)), 2),
                "ci_95_mae_diff": [round(ci_d_mae_25[0], 2), round(ci_d_mae_25[1], 2)],
                "mean_rmse_diff_ug_m3": round(float(np.mean(diff_rmse_pm25)), 2),
                "ci_95_rmse_diff": [round(ci_d_rmse_25[0], 2), round(ci_d_rmse_25[1], 2)],
                "mean_relative_gain_pct": round(float(np.mean(rel_gain_pm25)), 2),
                "ci_95_relative_gain": [round(ci_gain_25[0], 2), round(ci_gain_25[1], 2)],
                "bootstrap_p_value": round(p_val_mae_25, 5),
                "is_statistically_significant_at_005": bool(ci_d_mae_25[0] > 0.0),
            },
            "pm10": {
                "mean_paired_mae_diff_ug_m3": round(float(np.mean(diff_mae_pm10)), 2),
                "ci_95_mae_diff": [round(ci_d_mae_10[0], 2), round(ci_d_mae_10[1], 2)],
                "mean_rmse_diff_ug_m3": round(float(np.mean(diff_rmse_pm10)), 2),
                "ci_95_rmse_diff": [round(ci_d_rmse_10[0], 2), round(ci_d_rmse_10[1], 2)],
                "mean_relative_gain_pct": round(float(np.mean(rel_gain_pm10)), 2),
                "ci_95_relative_gain": [round(ci_gain_10[0], 2), round(ci_gain_10[1], 2)],
                "bootstrap_p_value": round(p_val_mae_10, 5),
                "is_statistically_significant_at_005": bool(ci_d_mae_10[0] > 0.0),
            },
        }

        summary_rows.append({
            "block_size": B,
            "effective_blocks": T // B,
            "pm25_paired_mae_diff": round(float(np.mean(diff_mae_pm25)), 2),
            "pm25_ci_95_lower": round(ci_d_mae_25[0], 2),
            "pm25_ci_95_upper": round(ci_d_mae_25[1], 2),
            "pm25_relative_gain_pct": round(float(np.mean(rel_gain_pm25)), 2),
            "pm25_ci_gain_lower": round(ci_gain_25[0], 2),
            "pm25_ci_gain_upper": round(ci_gain_25[1], 2),
            "pm25_p_value": round(p_val_mae_25, 5),
            "pm10_paired_mae_diff": round(float(np.mean(diff_mae_pm10)), 2),
            "pm10_ci_95_lower": round(ci_d_mae_10[0], 2),
            "pm10_ci_95_upper": round(ci_d_mae_10[1], 2),
            "pm10_relative_gain_pct": round(float(np.mean(rel_gain_pm10)), 2),
            "pm10_ci_gain_lower": round(ci_gain_10[0], 2),
            "pm10_ci_gain_upper": round(ci_gain_10[1], 2),
            "pm10_p_value": round(p_val_mae_10, 5),
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_CSV, index=False)
    elapsed = time.time() - t0_start
    print(f"\nExecution finished in {elapsed:.2f}s.", flush=True)
    print(f"Saved Block Bootstrap results to:\n- {OUTPUT_JSON}\n- {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    run_temporal_block_bootstrap()
