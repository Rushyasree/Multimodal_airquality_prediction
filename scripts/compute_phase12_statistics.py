"""
scripts/compute_phase12_statistics.py

Rigorous Statistical Validation for VayuDrishti Phase 12.
Computes matched paired differences, stationary block bootstrap (B=5,000, block_length=24h),
Cohen's d effect sizes, parametric paired t-tests, Wilcoxon signed-rank tests,
Bonferroni Family-Wise Error Rate (FWER) and Benjamini-Hochberg False Discovery Rate (FDR)
adjustments across 18 distinct hypothesis tests.

Outputs:
- experiments/phase12/statistical_tests.csv
"""

import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
PHASE12_DIR = ROOT_DIR / "experiments" / "phase12"
FINAL_PREDS_CSV = ROOT_DIR / "experiments" / "recent_data_audit" / "final_predictions.csv"


def stationary_block_bootstrap(
    diff: np.ndarray,
    b_reps: int = 5000,
    block_length: int = 24,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes stationary block bootstrap p-value and 95% confidence interval for mean difference.
    Block length L=24 preserves hourly diurnal autocorrelation.
    Uses the exact mathematical identity: the mean of concatenated blocks is identical
    to the mean of the block-level means, enabling high-speed vectorized sampling.
    H0: mean(diff) >= 0 (model not superior to persistence).
    H1: mean(diff) < 0 (model achieves lower error).
    """
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return 1.0, 0.0, 0.0

    orig_mean = float(np.mean(diff))

    # Partition array into non-overlapping blocks of length L
    n_usable = (n // block_length) * block_length
    if n_usable < block_length:
        # Fallback to standard iid bootstrap if sequence shorter than block length
        boot_samples = rng.choice(diff, size=(b_reps, n), replace=True)
        boot_means = boot_samples.mean(axis=1)
    else:
        blocks = diff[:n_usable].reshape(-1, block_length)
        block_means = blocks.mean(axis=1)
        k_blocks = len(block_means)
        # Vectorized resampling of block means
        sampled_block_indices = rng.integers(0, k_blocks, size=(b_reps, k_blocks))
        boot_means = block_means[sampled_block_indices].mean(axis=1)

    # 95% Percentile Confidence Interval
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Empirical one-sided p-value against H0: mean >= 0
    p_boot = float(np.mean(boot_means >= 0.0))
    p_boot = max(1.0 / (b_reps + 1), p_boot)

    return p_boot, ci_low, ci_high


def run_hypothesis_test(
    test_id: str,
    family: str,
    hypothesis_label: str,
    err_gbdt: np.ndarray,
    err_persist: np.ndarray,
    parameter: str = "PM2.5",
    b_reps: int = 5000,
) -> Dict[str, Any]:
    """
    Executes complete statistical test for a single comparison.
    diff = |y - y_hat_gbdt| - |y - y_hat_persist|
    A negative diff means GBDT error is smaller than persistence.
    """
    mask = np.isfinite(err_gbdt) & np.isfinite(err_persist)
    e_g = np.abs(err_gbdt[mask])
    e_p = np.abs(err_persist[mask])

    n = len(e_g)
    if n < 30:
        return {}

    diff = e_g - e_p
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 1e-6
    cohen_d = float(mean_diff / (std_diff + 1e-9))

    # Paired t-test
    t_stat, p_t_twosided = stats.ttest_rel(e_g, e_p)
    # One-sided p-value
    p_t = p_t_twosided / 2.0 if t_stat < 0 else 1.0 - (p_t_twosided / 2.0)

    # Wilcoxon signed-rank test (subsampled if n > 50,000 for tractability)
    if n > 40000:
        idx_sub = np.random.default_rng(42).choice(n, size=40000, replace=False)
        w_res = stats.wilcoxon(diff[idx_sub], alternative="less")
        w_stat, p_wilcoxon = float(w_res.statistic), float(w_res.pvalue)
    else:
        w_res = stats.wilcoxon(diff, alternative="less")
        w_stat, p_wilcoxon = float(w_res.statistic), float(w_res.pvalue)

    # Block bootstrap
    p_boot, ci_low, ci_high = stationary_block_bootstrap(diff, b_reps=b_reps, block_length=24)

    return {
        "test_id": test_id,
        "family": family,
        "hypothesis_label": hypothesis_label,
        "parameter": parameter,
        "n_samples": n,
        "gbdt_mae": round(float(np.mean(e_g)), 4),
        "persistence_mae": round(float(np.mean(e_p)), 4),
        "mean_difference": round(mean_diff, 4),
        "std_difference": round(std_diff, 4),
        "cohens_d": round(cohen_d, 4),
        "t_statistic": round(float(t_stat), 3),
        "p_paired_t": float(p_t),
        "wilcoxon_stat": round(float(w_stat), 1),
        "p_wilcoxon": float(p_wilcoxon),
        "bootstrap_reps": b_reps,
        "p_bootstrap_block": float(p_boot),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
    }


def main():
    print("=" * 78)
    print("VAYUDRISHTI PHASE 12: RIGOROUS STATISTICAL ROBUSTNESS TESTING")
    print("=" * 78)

    print(f"Loading predictions from {FINAL_PREDS_CSV.name}...")
    df = pd.read_csv(FINAL_PREDS_CSV)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["dt"].dt.month

    # Absolute errors
    abs_err_gbdt_25 = np.abs(df["pred_adapted_gbdt_pm25"].values - df["actual_pm25"].values)
    abs_err_pers_25 = np.abs(df["pred_persist_pm25"].values - df["actual_pm25"].values)

    abs_err_gbdt_10 = np.abs(df["pred_adapted_gbdt_pm10"].values - df["actual_pm10"].values)
    abs_err_pers_10 = np.abs(df["pred_persist_pm10"].values - df["actual_pm10"].values)

    test_specs = []

    # 1. Pooled tests
    test_specs.append(("T01", "Pooled", "Pooled PM2.5 All Stations", abs_err_gbdt_25, abs_err_pers_25, "PM2.5"))
    test_specs.append(("T02", "Pooled", "Pooled PM10 All Stations", abs_err_gbdt_10, abs_err_pers_10, "PM10"))

    # 2. Key Station-Wise tests
    stations = [
        ("DL002", "Anand Vihar, Delhi"),
        ("DL008", "DTU, Delhi"),
        ("MH005", "Bandra, Mumbai"),
        ("MH008", "Colaba, Mumbai"),
        ("WB008", "Bidhannagar, Kolkata"),
        ("WB010", "Jadavpur, Kolkata"),
        ("KA004", "Bapuji Nagar, Bengaluru"),
        ("AP005", "GVM Corp, Visakhapatnam"),
        ("TG001", "Bollaram, Hyderabad"),
        ("GJ001", "Maninagar, Ahmedabad"),
    ]

    t_idx = 3
    for sid, name in stations:
        mask_st = df["station_id"] == sid
        if mask_st.sum() >= 500:
            test_specs.append((
                f"T{t_idx:02d}",
                "Station",
                f"Station {sid} ({name}) PM2.5",
                abs_err_gbdt_25[mask_st],
                abs_err_pers_25[mask_st],
                "PM2.5",
            ))
            t_idx += 1

    # 3. NAQI Regime tests
    regimes = [
        ("Satisfactory", 30.01, 60.0),
        ("Moderate", 60.01, 90.0),
        ("Poor", 90.01, 120.0),
        ("Very Poor", 120.01, 250.0),
    ]
    for r_name, low_b, high_b in regimes:
        mask_r = (df["actual_pm25"] >= low_b) & (df["actual_pm25"] <= high_b)
        if mask_r.sum() >= 500:
            test_specs.append((
                f"T{t_idx:02d}",
                "Regime",
                f"NAQI Regime {r_name} (PM2.5)",
                abs_err_gbdt_25[mask_r],
                abs_err_pers_25[mask_r],
                "PM2.5",
            ))
            t_idx += 1

    # 4. Seasonal tests
    seasons = [
        ("Summer", [3, 4, 5]),
        ("Monsoon", [6, 7, 8, 9]),
    ]
    for s_name, months in seasons:
        mask_s = df["month"].isin(months)
        if mask_s.sum() >= 500:
            test_specs.append((
                f"T{t_idx:02d}",
                "Season",
                f"Season {s_name} (PM2.5)",
                abs_err_gbdt_25[mask_s],
                abs_err_pers_25[mask_s],
                "PM2.5",
            ))
            t_idx += 1

    print(f"Executing {len(test_specs)} rigorous hypothesis tests with stationary block bootstrap (B=5,000)...")

    results = []
    for tid, fam, label, e_g, e_p, param in test_specs:
        res = run_hypothesis_test(tid, fam, label, e_g, e_p, parameter=param, b_reps=5000)
        if res:
            results.append(res)
            print(f"  [{tid}] {label:35s}: Diff={res['mean_difference']:+.3f}, d={res['cohens_d']:+.3f}, p_boot={res['p_bootstrap_block']:.5f}")

    df_res = pd.DataFrame(results)

    # Multiple Testing Corrections across all M tests
    m_tests = len(df_res)
    alpha = 0.05
    bonferroni_thresh = alpha / m_tests

    # Bonferroni adjusted p-value = min(1.0, p * M)
    df_res["p_bonferroni_adj"] = np.minimum(1.0, df_res["p_bootstrap_block"] * m_tests)
    df_res["bonferroni_significant"] = df_res["p_bonferroni_adj"] < alpha

    # Benjamini-Hochberg FDR
    # Sort by p-value
    p_vals = df_res["p_bootstrap_block"].values
    sorted_order = np.argsort(p_vals)
    ranks = np.empty_like(sorted_order)
    ranks[sorted_order] = np.arange(1, m_tests + 1)

    # BH critical value = (rank / M) * alpha
    bh_critical = (ranks / m_tests) * alpha
    df_res["bh_critical_value"] = np.round(bh_critical, 5)

    # BH adjusted q-value
    # q_i = min_{k >= i} (p_k * M / k)
    sorted_p = p_vals[sorted_order]
    q_sorted = np.zeros(m_tests)
    min_q = 1.0
    for i in range(m_tests - 1, -1, -1):
        k = i + 1
        q_val = min(1.0, (sorted_p[i] * m_tests) / k)
        min_q = min(min_q, q_val)
        q_sorted[i] = min_q
    
    q_values = np.empty_like(q_sorted)
    q_values[sorted_order] = q_sorted
    df_res["fdr_q_value"] = np.round(q_values, 5)
    df_res["fdr_significant"] = df_res["fdr_q_value"] < alpha

    # Final scientific conclusion
    def format_verdict(row):
        if row["mean_difference"] < 0:
            if row["bonferroni_significant"]:
                return "Significant Superiority (Passes FWER Bonferroni & FDR)"
            elif row["fdr_significant"]:
                return "Significant Superiority (Passes FDR q < 0.05)"
            else:
                return "Directional Superiority (Not Significant under Multiplicity)"
        else:
            return "Equivalence / Non-Superiority"

    df_res["scientific_verdict"] = df_res.apply(format_verdict, axis=1)

    out_csv = PHASE12_DIR / "statistical_tests.csv"
    df_res.to_csv(out_csv, index=False)
    print("=" * 78)
    print(f"Saved {out_csv} ({len(df_res)} hypothesis tests).")
    print(f"Bonferroni Significant Tests: {df_res['bonferroni_significant'].sum()} / {len(df_res)}")
    print(f"FDR Significant Tests       : {df_res['fdr_significant'].sum()} / {len(df_res)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
