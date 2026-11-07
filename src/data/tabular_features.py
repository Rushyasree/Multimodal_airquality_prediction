"""
src/data/tabular_features.py

Extracts leakage-free tabular time-series features from 24-hour sequence windows [N, 24, 27].
Every feature strictly uses past data up to forecast issue time t (timestep index 23).
No feature uses target data (t+1) or any future observation.
"""

from typing import Tuple, List, Optional
import numpy as np


FEATURE_NAMES_TABULAR = [
    # 1. Instantaneous at issue time t (27 original channels)
    "pm25_t", "pm10_t", "no_t", "no2_t", "nox_t", "nh3_t", "co_t", "so2_t", "o3_t",
    "benzene_t", "toluene_t", "xylene_t", "temperature_t", "dewpoint_t", "relative_humidity_t",
    "wind_speed_t", "precipitation_t", "wind_dir_sin_t", "wind_dir_cos_t", "hour_sin_t",
    "hour_cos_t", "month_sin_t", "month_cos_t", "day_of_week_sin_t", "day_of_week_cos_t",
    "latitude", "longitude",
    # 2. Historical Lags for PM2.5 and PM10 (preceding t)
    "pm25_lag1", "pm10_lag1",   # t-1 (k=22)
    "pm25_lag2", "pm10_lag2",   # t-2 (k=21)
    "pm25_lag3", "pm10_lag3",   # t-3 (k=20)
    "pm25_lag5", "pm10_lag5",   # t-5 (k=18)
    "pm25_lag11", "pm10_lag11", # t-11 (k=12)
    "pm25_lag23", "pm10_lag23", # t-23 (k=0, exactly 24h prior to target t+1)
    # 3. Rolling Statistics over Past Sub-Windows
    "pm25_roll3_mean", "pm10_roll3_mean", "pm25_roll3_std", "pm10_roll3_std",
    "pm25_roll6_mean", "pm10_roll6_mean", "pm25_roll6_std", "pm10_roll6_std",
    "pm25_roll12_mean", "pm10_roll12_mean", "pm25_roll12_std", "pm10_roll12_std",
    "pm25_roll24_mean", "pm10_roll24_mean", "pm25_roll24_std", "pm10_roll24_std",
    # 4. Differencing / Momentum
    "pm25_diff1h", "pm10_diff1h",   # x[23] - x[22]
    "pm25_diff3h", "pm10_diff3h",   # x[23] - x[20]
    "pm25_diff6h", "pm10_diff6h",   # x[23] - x[18]
    "pm25_diff24h", "pm10_diff24h", # x[23] - x[0]
    "pm25_accel", "pm10_accel",     # (x[23]-x[22]) - (x[22]-x[21])
    # 5. Physics and Domain Interactions
    "pm_ratio_t",                   # pm25_t / (pm10_t + eps)
    "dewpoint_depression_t",        # temperature_t - dewpoint_t
]

FEATURE_NAMES = FEATURE_NAMES_TABULAR

# Canonical indices for 49 reduced features (for adapted models without weather/trace gas channels)
REDUCED_FEATURE_INDICES: List[int] = [0, 1] + list(range(19, 27)) + list(range(27, 66))
REDUCED_FEATURE_NAMES: List[str] = [FEATURE_NAMES_TABULAR[i] for i in REDUCED_FEATURE_INDICES]

# Excluded unobserved channels (17 instantaneous weather/chemical channels + 1 dewpoint depression)
EXCLUDED_FEATURE_INDICES: List[int] = list(range(2, 19)) + [66]
EXCLUDED_FEATURE_NAMES: List[str] = [FEATURE_NAMES_TABULAR[i] for i in EXCLUDED_FEATURE_INDICES]


def get_tabular_feature_names() -> List[str]:
    """Returns the list of 67 engineered feature names."""
    return list(FEATURE_NAMES_TABULAR)


def get_reduced_tabular_feature_names() -> List[str]:
    """Returns the canonical list of 49 reduced feature names for adapted models."""
    return list(REDUCED_FEATURE_NAMES)


def extract_reduced_tabular_features(X_sequence: np.ndarray) -> np.ndarray:
    """Extracts the canonical 49 reduced tabular features from a [N, 24, 27] sequence array.

    Excludes unobserved trace gas and meteorological channels (channels 2-18 and dewpoint depression).
    Strictly causal at forecast issue time t with zero temporal leakage.

    Args:
        X_sequence: NumPy array of shape [N, 24, 27] or [24, 27].

    Returns:
        NumPy array of shape [N, 49] with canonical reduced tabular features.
    """
    full_features = extract_tabular_features(X_sequence)
    return full_features[:, REDUCED_FEATURE_INDICES]


def extract_tabular_features(X_sequence: np.ndarray) -> np.ndarray:
    """Extracts 67 tabular time-series features from a [N, 24, 27] sequence array.

    Args:
        X_sequence: NumPy array of shape [N, 24, 27] representing 24h historical telemetry.

    Returns:
        NumPy array of shape [N, 65] with engineered tabular features.
    """
    if X_sequence.ndim == 2:
        X_sequence = np.expand_dims(X_sequence, axis=0)

    N = X_sequence.shape[0]

    # 1. Instantaneous values at t (k=23)
    curr = X_sequence[:, 23, :]  # [N, 27]

    # 2. Short-term and seasonal lags for PM2.5 (col 0) and PM10 (col 1)
    l1 = X_sequence[:, 22, :2]
    l2 = X_sequence[:, 21, :2]
    l3 = X_sequence[:, 20, :2]
    l5 = X_sequence[:, 18, :2]
    l11 = X_sequence[:, 12, :2]
    l23 = X_sequence[:, 0, :2]  # Seasonal lag: exactly 24 hours before target t+1

    # 3. Rolling statistics
    r3_m = np.mean(X_sequence[:, 21:, :2], axis=1)
    r3_s = np.std(X_sequence[:, 21:, :2], axis=1)
    r6_m = np.mean(X_sequence[:, 18:, :2], axis=1)
    r6_s = np.std(X_sequence[:, 18:, :2], axis=1)
    r12_m = np.mean(X_sequence[:, 12:, :2], axis=1)
    r12_s = np.std(X_sequence[:, 12:, :2], axis=1)
    r24_m = np.mean(X_sequence[:, :, :2], axis=1)
    r24_s = np.std(X_sequence[:, :, :2], axis=1)

    # 4. Momentum / Differences
    d1 = curr[:, :2] - l1
    d3 = curr[:, :2] - l3
    d6 = curr[:, :2] - l5
    d24 = curr[:, :2] - l23
    accel = d1 - (l1 - l2)

    # 5. Domain interaction features
    # PM ratio
    pm_ratio = np.clip(curr[:, 0:1] / (curr[:, 1:2] + 1e-4), -5.0, 5.0)
    # Dewpoint depression: temperature (col 12) - dewpoint (col 13)
    dew_dep = curr[:, 12:13] - curr[:, 13:14]

    # Stack all features
    features = np.hstack([
        curr,
        l1, l2, l3, l5, l11, l23,
        r3_m, r3_s, r6_m, r6_s, r12_m, r12_s, r24_m, r24_s,
        d1, d3, d6, d24, accel,
        pm_ratio, dew_dep
    ]).astype(np.float32)

    return features
