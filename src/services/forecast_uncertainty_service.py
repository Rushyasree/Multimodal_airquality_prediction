"""
src/services/forecast_uncertainty_service.py

Split-Conformal Prediction & Calibrated Uncertainty Estimation for VayuDrishti Phase 12.
Provides distribution-free prediction intervals (80%, 90%, 95%) with finite-sample coverage
guarantees calibrated on independent calibration observations.

Distinguishes:
- Prediction Intervals (PI): Bounds containing next-hour observation y_{t+1} with probability >= 1 - alpha.
- Confidence Intervals (CI): Parameter uncertainty of mean prediction estimator.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ConformalPredictionCalibrator:
    """Split-conformal prediction calibrator for pollutant concentrations."""

    NOMINAL_LEVELS = [0.80, 0.90, 0.95]

    def __init__(self):
        # Calibration quantiles: {parameter: {nominal_level: q_val}}
        self.quantiles: Dict[str, Dict[float, float]] = {
            "PM2.5": {},
            "PM10": {},
        }
        self.calibration_sample_counts: Dict[str, int] = {"PM2.5": 0, "PM10": 0}
        self.is_calibrated: bool = False

    def fit_calibration_set(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        parameter: str = "PM2.5",
    ) -> Dict[float, float]:
        """
        Computes conformal non-conformity quantiles from independent calibration residuals.
        Non-conformity score: R_i = |y_i - y_hat_i|

        Finite-sample corrected quantile:
            k = ceil((n + 1) * (1 - alpha))
            q_{1-alpha} = sorted_R[k - 1]
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_t) & np.isfinite(y_p)
        y_t = y_t[mask]
        y_p = y_p[mask]

        n = len(y_t)
        if n < 20:
            raise ValueError(f"Insufficient calibration instances ({n} < 20). Conformalization requires independent split.")

        abs_residuals = np.abs(y_t - y_p)
        sorted_residuals = np.sort(abs_residuals)

        self.calibration_sample_counts[parameter] = n
        self.quantiles[parameter] = {}

        for alpha_level in self.NOMINAL_LEVELS:
            # Finite-sample adjusted index
            rank = int(math.ceil((n + 1) * alpha_level))
            idx = min(n - 1, max(0, rank - 1))
            q_val = float(sorted_residuals[idx])
            self.quantiles[parameter][alpha_level] = round(q_val, 3)

        self.is_calibrated = True
        return dict(self.quantiles[parameter])

    def predict_intervals(
        self,
        point_prediction: float,
        parameter: str = "PM2.5",
    ) -> Dict[str, Any]:
        """
        Generates prediction intervals at 80%, 90%, and 95% nominal coverage for a point forecast.
        Physical lower bound is truncated at 0.0 ug/m3.
        """
        pred = max(0.0, float(point_prediction))
        if parameter not in self.quantiles or not self.quantiles[parameter]:
            # Uncalibrated fallback: empirical heuristic
            fallback_q = {"PM2.5": {0.80: 8.5, 0.90: 14.0, 0.95: 22.0}, "PM10": {0.80: 18.0, 0.90: 32.0, 0.95: 50.0}}
            q_dict = fallback_q.get(parameter, {0.80: 10.0, 0.90: 20.0, 0.95: 30.0})
        else:
            q_dict = self.quantiles[parameter]

        intervals = {}
        for level, q_val in q_dict.items():
            pct = int(level * 100)
            low = max(0.0, round(pred - q_val, 2))
            high = round(pred + q_val, 2)
            intervals[f"pi_{pct}"] = {
                "nominal_level": level,
                "lower": low,
                "upper": high,
                "width": round(high - low, 2),
                "margin": round(q_val, 2),
            }

        return {
            "parameter": parameter,
            "point_prediction": round(pred, 2),
            "intervals": intervals,
            "calibrated": self.is_calibrated,
            "calibration_samples": self.calibration_sample_counts.get(parameter, 0),
        }

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        parameter: str = "PM2.5",
    ) -> pd.DataFrame:
        """
        Evaluates empirical coverage, mean interval width, and Winkler score on test set.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_t) & np.isfinite(y_p)
        y_t = y_t[mask]
        y_p = y_p[mask]
        n_test = len(y_t)

        rows = []
        q_dict = self.quantiles.get(parameter, {})
        for level in self.NOMINAL_LEVELS:
            pct = int(level * 100)
            q_val = q_dict.get(level, 10.0)

            low = np.maximum(0.0, y_p - q_val)
            high = y_p + q_val

            covered = (y_t >= low) & (y_t <= high)
            emp_coverage = float(np.mean(covered))
            widths = high - low
            mean_width = float(np.mean(widths))
            median_width = float(np.median(widths))

            # Winkler Score: Width + (2/alpha)*(L - y)*I(y < L) + (2/alpha)*(y - U)*I(y > U)
            alpha = 1.0 - level
            penalty_low = np.where(y_t < low, (2.0 / alpha) * (low - y_t), 0.0)
            penalty_high = np.where(y_t > high, (2.0 / alpha) * (y_t - high), 0.0)
            winkler_score = float(np.mean(widths + penalty_low + penalty_high))

            rows.append({
                "parameter": parameter,
                "nominal_coverage": level,
                "nominal_pct": f"{pct}%",
                "empirical_coverage": round(emp_coverage, 4),
                "coverage_error": round(emp_coverage - level, 4),
                "mean_interval_width": round(mean_width, 2),
                "median_interval_width": round(median_width, 2),
                "winkler_score": round(winkler_score, 2),
                "test_instances": n_test,
                "calibration_instances": self.calibration_sample_counts.get(parameter, 0),
            })

        return pd.DataFrame(rows)
