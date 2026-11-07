"""
Evaluation of Actual Classification Accuracy & AQI Category Metrics for VayuDrishti
===================================================================================

This script evaluates the operational accuracy of the VayuDrishti sequence forecasting
model (LSTM) by converting continuous PM2.5 and PM10 concentration predictions into
official CPCB National Air Quality Index (NAQI) concentration breakpoint categories.

Methodological Notes:
1. Target & Prediction Units: Both y_test and test_predictions are in raw physical
   concentration units (µg/m³).
2. Jurisdiction: India Central Pollution Control Board (CPCB) NAQI 2014 / NAAQS 2009.
3. Category Distinction: This is PM2.5 and PM10 concentration-to-AQI sub-category classification,
   not a composite multi-pollutant AQI (CPCB composite AQI requires >= 3 criteria pollutants).
4. Alignment: Evaluates all 404,431 held-out test sequences from y_test.npy and test_predictions.npy.
5. Separation of Concerns: Regression metrics (MAE, RMSE, R²) are computed and reported
   independently; R² is NOT conflated with accuracy.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "model_sequences" / "y_test.npy"
PREDS_PATH = BASE_DIR / "models" / "lstm_baseline" / "test_predictions.npy"
SAVED_METRICS_PATH = BASE_DIR / "models" / "lstm_baseline" / "test_metrics.json"
REPORTS_DIR = BASE_DIR / "reports"

# Documented CPCB NAQI Breakpoint Definitions (µg/m³)
# Good (0-50), Satisfactory (51-100), Moderate (101-200), Poor (201-300), Very Poor (301-400), Severe (401-500)
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

PM25_BREAKPOINTS = [
    (0.0, 30.0, "Good"),
    (30.0, 60.0, "Satisfactory"),
    (60.0, 90.0, "Moderate"),
    (90.0, 120.0, "Poor"),
    (120.0, 250.0, "Very Poor"),
    (250.0, float("inf"), "Severe"),
]

PM10_BREAKPOINTS = [
    (0.0, 50.0, "Good"),
    (50.0, 100.0, "Satisfactory"),
    (100.0, 250.0, "Moderate"),
    (250.0, 350.0, "Poor"),
    (350.0, 430.0, "Very Poor"),
    (430.0, float("inf"), "Severe"),
]


def map_concentration_to_category(conc: float, breakpoints: list) -> str:
    """Maps a continuous concentration (µg/m³) to a CPCB AQI category."""
    # Clamp negative unconstrained regression outputs to 0.0
    val = max(0.0, float(conc))
    for low, high, label in breakpoints:
        if low == 0.0:
            if val <= high:
                return label
        else:
            if low < val <= high:
                return label
    return breakpoints[-1][2]


def compute_adjacent_accuracy(cm: np.ndarray) -> float:
    """Computes adjacent-category accuracy (exact match or within +/- 1 category bin)."""
    n = cm.shape[0]
    adjacent_matches = 0
    total = np.sum(cm)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1:
                adjacent_matches += cm[i, j]
    return float(adjacent_matches / total) if total > 0 else 0.0


def main():
    print("=" * 80)
    print("VAYUDRISHTI: ACTUAL MODEL ACCURACY & CATEGORY EVALUATION")
    print("=" * 80)

    # 1. Load artifacts
    if not Y_TEST_PATH.exists() or not PREDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing test artifacts: {Y_TEST_PATH} or {PREDS_PATH}"
        )

    print(f"Loading true test targets from: {Y_TEST_PATH}")
    y_test = np.load(Y_TEST_PATH)
    print(f"Loading test predictions from:   {PREDS_PATH}")
    test_preds = np.load(PREDS_PATH)

    # 2. Inspect shapes and verify alignment
    assert y_test.shape == test_preds.shape, (
        f"Shape mismatch: y_test {y_test.shape} vs test_preds {test_preds.shape}"
    )
    total_samples, num_targets = y_test.shape
    print(f"Total evaluated test sequences: {total_samples:,}")
    print(f"Target count: {num_targets} (0: PM2.5, 1: PM10)")

    # Exclusions check
    y_test_nans = int(np.isnan(y_test).sum())
    y_test_infs = int(np.isinf(y_test).sum())
    preds_nans = int(np.isnan(test_preds).sum())
    preds_infs = int(np.isinf(test_preds).sum())
    excluded_samples = max(y_test_nans, preds_nans)
    print(f"Excluded / Invalid samples:    {excluded_samples} (0 NaNs, 0 Infs)")

    # 3. Separate Regression Metrics Verification
    print("\n" + "-" * 80)
    print("1. CONTINUOUS REGRESSION METRICS (Reported Separately)")
    print("-" * 80)
    regression_metrics = {}
    for idx, target_name in enumerate(["PM2.5", "PM10"]):
        y_t = y_test[:, idx].astype(np.float64)
        y_p = test_preds[:, idx].astype(np.float64)
        error = y_p - y_t
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        regression_metrics[target_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "units": "µg/m³",
        }
        print(f"  {target_name:6s} | MAE: {mae:7.4f} µg/m³ | RMSE: {rmse:7.4f} µg/m³ | R² Score: {r2:7.4f}")

    print("  *Note: R² represents proportion of variance explained and must NOT be labeled as accuracy.")

    # 4. Classification Metrics via CPCB Breakpoints
    print("\n" + "-" * 80)
    print("2. ACTUAL CLASSIFICATION ACCURACY & CATEGORY METRICS (CPCB NAQI)")
    print("-" * 80)

    classification_results = {}
    per_class_rows = []

    for idx, (target_name, bp) in enumerate([("PM2.5", PM25_BREAKPOINTS), ("PM10", PM10_BREAKPOINTS)]):
        y_t = y_test[:, idx]
        y_p = test_preds[:, idx]

        true_cats = [map_concentration_to_category(v, bp) for v in y_t]
        pred_cats = [map_concentration_to_category(v, bp) for v in y_p]

        acc = float(accuracy_score(true_cats, pred_cats))
        cm = confusion_matrix(true_cats, pred_cats, labels=CATEGORIES)
        adj_acc = compute_adjacent_accuracy(cm)

        # Macro and weighted averages
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            true_cats, pred_cats, labels=CATEGORIES, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_cats, pred_cats, labels=CATEGORIES, average="weighted", zero_division=0
        )

        # Per-class metrics
        p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
            true_cats, pred_cats, labels=CATEGORIES, average=None, zero_division=0
        )

        class_details = {}
        for c_idx, cat_name in enumerate(CATEGORIES):
            class_details[cat_name] = {
                "support": int(support_class[c_idx]),
                "precision": float(p_class[c_idx]),
                "recall": float(r_class[c_idx]),
                "f1_score": float(f1_class[c_idx]),
            }
            per_class_rows.append({
                "Target": target_name,
                "Category": cat_name,
                "Support": int(support_class[c_idx]),
                "Precision": float(p_class[c_idx]),
                "Recall": float(r_class[c_idx]),
                "F1-Score": float(f1_class[c_idx]),
            })

        classification_results[target_name] = {
            "exact_accuracy": acc,
            "adjacent_category_accuracy": adj_acc,
            "macro_avg": {
                "precision": float(p_macro),
                "recall": float(r_macro),
                "f1_score": float(f1_macro),
            },
            "weighted_avg": {
                "precision": float(p_weighted),
                "recall": float(r_weighted),
                "f1_score": float(f1_weighted),
            },
            "per_class": class_details,
            "confusion_matrix": cm.tolist(),
        }

        print(f"\nTarget: {target_name}")
        print(f"  Exact Classification Accuracy:   {acc * 100:6.2f}% ({acc:.4f})")
        print(f"  Within +/- 1 Category Accuracy:  {adj_acc * 100:6.2f}% ({adj_acc:.4f})")
        print(f"  Macro-Averaged F1-Score:         {f1_macro:6.4f}")
        print(f"  Weighted-Averaged F1-Score:      {f1_weighted:6.4f}")
        print("\n  Per-Class Metrics:")
        df_rep = pd.DataFrame([
            {
                "Category": cat,
                "Support": class_details[cat]["support"],
                "Precision": f"{class_details[cat]['precision']:.4f}",
                "Recall": f"{class_details[cat]['recall']:.4f}",
                "F1-Score": f"{class_details[cat]['f1_score']:.4f}",
            }
            for cat in CATEGORIES
        ])
        print(df_rep.to_string(index=False))

    # 5. Save Outputs
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Full JSON report
    report_data = {
        "dataset_name": "CPCB 90 CAAQMS Station Test Sequences",
        "total_test_samples": total_samples,
        "excluded_samples": excluded_samples,
        "jurisdiction": "Central Pollution Control Board (CPCB) NAQI Guidelines",
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_results,
    }
    json_path = REPORTS_DIR / "classification_metrics_report.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\n[Saved JSON Report]: {json_path}")

    # Confusion Matrix CSVs
    for target_name in ["PM2.5", "PM10"]:
        cm_arr = np.array(classification_results[target_name]["confusion_matrix"])
        df_cm = pd.DataFrame(
            cm_arr,
            index=[f"Actual_{c}" for c in CATEGORIES],
            columns=[f"Pred_{c}" for c in CATEGORIES],
        )
        safe_target = target_name.lower().replace(".", "")
        cm_path = REPORTS_DIR / f"confusion_matrix_{safe_target}.csv"
        df_cm.to_csv(cm_path)
        print(f"[Saved Confusion Matrix]: {cm_path}")

    # Per-Class Metrics CSV
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_path = REPORTS_DIR / "classification_per_class_metrics.csv"
    df_per_class.to_csv(per_class_path, index=False)
    print(f"[Saved Per-Class CSV]:   {per_class_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
