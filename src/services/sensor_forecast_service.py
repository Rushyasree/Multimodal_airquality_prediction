"""
src/services/sensor_forecast_service.py

Sensor-based air quality sequence forecasting service.
Predicts next-hour PM2.5 and PM10 concentrations from 24-hour multivariate telemetry
using temporal LSTM and gradient boosted decision tree ensembles.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Suppress loky worker detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from src.prediction.lstm_forecaster import LSTMForecasterService
from src.prediction.data_loader import get_verified_24h_sequence_for_station

ROOT_DIR = Path(__file__).resolve().parents[2]


class SensorForecastService:
    """Service providing verified numerical sensor-only LSTM forecasts and ensembles."""

    def __init__(self):
        self.underlying_service = LSTMForecasterService()
        self.model_version = "LSTM-2Layer Baseline v1.2"
        self._load_metrics()

        # Cached models for low-latency CPU serving
        self._historical_gbdt_pm25 = None
        self._historical_gbdt_pm10 = None
        self._historical_models_loaded = False

        self._adapted_lstm_service = None
        self._adapted_gbdt_pm25 = None
        self._adapted_gbdt_pm10 = None
        self._adapted_checkpoints_loaded = False

    def _load_metrics(self):
        metrics_file = ROOT_DIR / "models/lstm_baseline/test_metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r") as f:
                self.saved_metrics = json.load(f)
        else:
            self.saved_metrics = {
                "pm25": {"MAE": 10.5719, "RMSE": 18.5789, "R2": 0.9492},
                "pm10": {"MAE": 18.8643, "RMSE": 32.2027, "R2": 0.9301},
            }

    def _load_historical_gbdt(self) -> bool:
        """Loads historical GBDT models into memory cache."""
        if self._historical_models_loaded:
            return True
        try:
            import joblib
            ckpt_dir = ROOT_DIR / "experiments/accuracy_optimization_v2/checkpoints"
            pm25_path = ckpt_dir / "best_gbdt_pm25.joblib"
            pm10_path = ckpt_dir / "best_gbdt_pm10.joblib"
            if pm25_path.exists() and pm10_path.exists():
                self._historical_gbdt_pm25 = joblib.load(pm25_path)
                self._historical_gbdt_pm10 = joblib.load(pm10_path)
                self._historical_models_loaded = True
                return True
            return False
        except Exception:
            return False

    def _load_adapted_models(self) -> bool:
        """Loads post-2020 adapted models into memory cache with graceful fallback."""
        if self._adapted_checkpoints_loaded:
            return True
        try:
            import joblib
            adapted_dir = ROOT_DIR / "experiments/recent_data_audit/checkpoints"
            lstm_path = adapted_dir / "best_finetuned_lstm.pt"
            gbdt_25_path = adapted_dir / "adapted_gbdt_pm25.joblib"
            gbdt_10_path = adapted_dir / "adapted_gbdt_pm10.joblib"

            if not (lstm_path.exists() and gbdt_25_path.exists() and gbdt_10_path.exists()):
                return False

            if self._adapted_lstm_service is None:
                self._adapted_lstm_service = LSTMForecasterService(model_path=lstm_path)
            if self._adapted_gbdt_pm25 is None:
                self._adapted_gbdt_pm25 = joblib.load(gbdt_25_path)
            if self._adapted_gbdt_pm10 is None:
                self._adapted_gbdt_pm10 = joblib.load(gbdt_10_path)

            self._adapted_checkpoints_loaded = True
            return True
        except Exception:
            return False

    def _predict_adapted(
        self,
        df_to_use: pd.DataFrame,
        station_id: str = "AP001",
        seq_arr: Optional[np.ndarray] = None,
        severity_gated: bool = True,
    ) -> Dict[str, Any]:
        """Runs the post-2020 adapted model pipeline on 24h telemetry.

        Uses the verified 49 canonical reduced tabular features and fine-tuned LSTM.
        Strictly causal at forecast issue time t with zero temporal leakage.
        """
        if not self._load_adapted_models():
            raise RuntimeError("Adapted model checkpoints unavailable in experiments/recent_data_audit/checkpoints")

        from src.data.tabular_features import extract_reduced_tabular_features

        # 1. Reuse or preprocess sequence
        if seq_arr is None:
            seq_arr = self.underlying_service.preprocess_sequence(df_to_use, station_id=station_id)

        # 2. Fine-tuned LSTM inference (scaled z-scores with internal inverse transform)
        lstm_preds = self._adapted_lstm_service.predict_sequence(seq_arr)
        lstm_pm25 = float(lstm_preds.get("predicted_PM25", 0.0))
        lstm_pm10 = float(lstm_preds.get("predicted_PM10", 0.0))

        # 3. Canonical 49 reduced tabular features extraction
        tab_feats_49 = extract_reduced_tabular_features(seq_arr)

        # 4. Reduced GBDT inference (direct unscaled ug/m3)
        gbdt_pm25 = float(self._adapted_gbdt_pm25.predict(tab_feats_49)[0])
        gbdt_pm10 = float(self._adapted_gbdt_pm10.predict(tab_feats_49)[0])

        # 5. Severity gating / dynamic decoupling (robust to non-numeric strings or NaNs)
        def _to_float_or_fallback(series_val, fallback):
            try:
                num = pd.to_numeric(series_val, errors="coerce")
                return fallback if pd.isna(num) else float(num)
            except Exception:
                return fallback

        pm25_curr = _to_float_or_fallback(df_to_use["PM2.5"].iloc[-1], lstm_pm25) if "PM2.5" in df_to_use.columns else lstm_pm25
        pm10_curr = _to_float_or_fallback(df_to_use["PM10"].iloc[-1], lstm_pm10) if "PM10" in df_to_use.columns else lstm_pm10

        if severity_gated:
            w25 = 1.0 if pm25_curr >= 250.0 else 0.40
            w10 = 1.0 if pm10_curr >= 400.0 else 0.45
            version_str = "Adapted Severity-Gated Ensemble (Fine-Tuned LSTM + 49-Feat GBDT)"
            metrics_dict = {
                "pm25_mae": 6.62,
                "pm25_rmse": 13.04,
                "pm25_r2": 0.8254,
                "pm10_mae": 16.14,
                "pm10_rmse": 34.53,
                "pm10_r2": 0.8479,
                "evaluation_cohort": "236,289 Sequences (Apr-Aug 2025 Holdout across 75 Stations)",
            }
            note_str = "Post-2020 adapted model using 2021-2023 fine-tuning and 49 reduced features with severity gating (PM2.5 >= 250, PM10 >= 400 ug/m3)."
        else:
            w25 = 0.40
            w10 = 0.45
            version_str = "Adapted Static Ensemble (Fine-Tuned LSTM + 49-Feat GBDT)"
            metrics_dict = {
                "pm25_mae": 6.62,
                "pm25_rmse": 13.03,
                "pm25_r2": 0.8256,
                "pm10_mae": 16.17,
                "pm10_rmse": 34.69,
                "pm10_r2": 0.8465,
                "evaluation_cohort": "236,289 Sequences (Apr-Aug 2025 Holdout across 75 Stations)",
            }
            note_str = "Post-2020 static ensemble (0.40 LSTM / 0.60 GBDT for PM2.5, 0.45 / 0.55 for PM10)."

        final_pm25 = max(0.0, w25 * lstm_pm25 + (1.0 - w25) * gbdt_pm25)
        final_pm10 = max(0.0, w10 * lstm_pm10 + (1.0 - w10) * gbdt_pm10)

        return {
            "status": "success",
            "pm25": round(final_pm25, 2),
            "pm10": round(final_pm10, 2),
            "unit": "ug/m3",
            "model_version": version_str,
            "components": {
                "lstm_pm25": round(lstm_pm25, 2),
                "lstm_pm10": round(lstm_pm10, 2),
                "gbdt_pm25": round(gbdt_pm25, 2),
                "gbdt_pm10": round(gbdt_pm10, 2),
                "weights": {"w_pm25_lstm": w25, "w_pm10_lstm": w10},
                "n_features_gbdt": 49,
            },
            "metrics": metrics_dict,
            "scientific_note": note_str,
            "experimental": True,
        }

    def _predict_adapted_gbdt_only(
        self,
        df_to_use: pd.DataFrame,
        station_id: str = "AP001",
        seq_arr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Runs the standalone post-2020 adapted GBDT models independently of any LSTM.

        Uses strictly the canonical 49 reduced tabular features.
        Zero dependence on trace gas, VOC, or meteorological channels.
        Strictly causal at forecast issue time t.
        """
        if not self._load_adapted_models():
            raise RuntimeError("Adapted model checkpoints unavailable in experiments/recent_data_audit/checkpoints")

        from src.data.tabular_features import extract_reduced_tabular_features

        if seq_arr is None:
            seq_arr = self.underlying_service.preprocess_sequence(df_to_use, station_id=station_id)

        tab_feats_49 = extract_reduced_tabular_features(seq_arr)

        gbdt_pm25 = max(0.0, float(self._adapted_gbdt_pm25.predict(tab_feats_49)[0]))
        gbdt_pm10 = max(0.0, float(self._adapted_gbdt_pm10.predict(tab_feats_49)[0]))

        version_str = "Standalone Adapted GBDT Forecaster (49 Reduced Features)"
        metrics_dict = {
            "pm25_mae": 6.67,
            "pm25_rmse": 13.17,
            "pm25_r2": 0.8219,
            "pm10_mae": 16.29,
            "pm10_rmse": 35.24,
            "pm10_r2": 0.8415,
            "evaluation_cohort": "236,289 Sequences (Apr-Aug 2025 Holdout across 75 Stations)",
        }
        note_str = (
            "Standalone post-2020 adapted GBDT forecaster predicting next-hour PM2.5 and PM10 "
            "exclusively from 24h particulate telemetry, cyclical temporal transforms, and station coordinates. "
            "Operates 100% independently of the LSTM and requires zero trace gas or meteorological observations."
        )

        return {
            "status": "success",
            "pm25": round(gbdt_pm25, 2),
            "pm10": round(gbdt_pm10, 2),
            "unit": "ug/m3",
            "model_version": version_str,
            "components": {
                "gbdt_pm25": round(gbdt_pm25, 2),
                "gbdt_pm10": round(gbdt_pm10, 2),
                "n_features_gbdt": 49,
            },
            "metrics": metrics_dict,
            "scientific_note": note_str,
            "experimental": True,
            "is_standalone_gbdt": True,
        }

    def get_forecast(
        self,
        station_id: str = "AP001",
        df_24h: Optional[pd.DataFrame] = None,
        model_mode: str = "lstm",
        enable_shadow: bool = False,
    ) -> Dict[str, Any]:
        """Predicts next-hour PM2.5 and PM10 from a 24-hour sensor sequence.

        Args:
            station_id: CPCB monitoring station identifier.
            df_24h: 24 contiguous hourly observations. If None, uses simulated sequence.
            model_mode: Model selection:
                - "lstm": Historical 2-layer LSTM baseline
                - "ensemble": Historical static ensemble (0.90 / 0.80)
                - "severity_gated": Historical severity-gated ensemble (Production Default)
                - "adapted_severity_gated": Opt-in post-2020 adapted severity-gated ensemble
                - "adapted_ensemble": Opt-in post-2020 adapted static ensemble
                - "adapted_gbdt_only": Opt-in post-2020 standalone adapted GBDT (49 features, zero LSTM dependency)
            enable_shadow: If True, executes adapted model in non-blocking shadow mode
                and returns payload under 'shadow_forecast' without altering primary forecast.
        """
        now = datetime.now()
        target_time = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:00:00")

        if df_24h is None:
            df_to_use, input_freshness, is_synth = get_verified_24h_sequence_for_station(station_id)
        else:
            df_to_use = df_24h
            input_freshness = f"Continuous 24h Telemetry ({station_id} Verified Ground Sensors)"
            is_synth = False

        # Preprocess 24h sequence array once for all downstream models
        seq_arr = self.underlying_service.preprocess_sequence(df_to_use, station_id=station_id)

        fallback_note = None

        # Check for explicit opt-in adapted mode
        if model_mode == "adapted_gbdt_only":
            try:
                gbdt_res = self._predict_adapted_gbdt_only(
                    df_to_use,
                    station_id=station_id,
                    seq_arr=seq_arr,
                )
                return {
                    "result_type": "Independent Forecast",
                    "station_id": station_id,
                    "forecast_horizon": "Next 1 Hour (t + 1)",
                    "target_time": target_time,
                    "input_freshness": input_freshness,
                    "model_version": gbdt_res["model_version"],
                    "pm25": gbdt_res["pm25"],
                    "pm10": gbdt_res["pm10"],
                    "unit": "ug/m3",
                    "metrics": gbdt_res["metrics"],
                    "scientific_note": gbdt_res["scientific_note"],
                    "shadow_mode": False,
                    "is_adapted": True,
                    "is_standalone_gbdt": True,
                    "components": gbdt_res.get("components"),
                }
            except Exception as e:
                fallback_note = f"Standalone adapted GBDT checkpoint unavailable ({e}); served production model safely."

        elif model_mode in ("adapted_severity_gated", "adapted_ensemble"):
            try:
                adapted_res = self._predict_adapted(
                    df_to_use,
                    station_id=station_id,
                    seq_arr=seq_arr,
                    severity_gated=(model_mode == "adapted_severity_gated"),
                )
                return {
                    "result_type": "Independent Forecast",
                    "station_id": station_id,
                    "forecast_horizon": "Next 1 Hour (t + 1)",
                    "target_time": target_time,
                    "input_freshness": input_freshness,
                    "model_version": adapted_res["model_version"],
                    "pm25": adapted_res["pm25"],
                    "pm10": adapted_res["pm10"],
                    "unit": "ug/m3",
                    "metrics": adapted_res["metrics"],
                    "scientific_note": adapted_res["scientific_note"],
                    "shadow_mode": False,
                    "is_adapted": True,
                    "components": adapted_res.get("components"),
                }
            except Exception as e:
                # Graceful fallback: serve production baseline safely without crashing
                fallback_note = f"Adapted model checkpoint unavailable ({e}); served production model safely."

        # Production forecast path (Historical LSTM / Historical GBDT / Historical Gating)
        preds_res = self.underlying_service.predict_sequence(seq_arr)
        pm25_pred = float(preds_res.get("predicted_PM25", 45.0))
        pm10_pred = float(preds_res.get("predicted_PM10", 90.0))

        model_version = self.model_version
        metrics_dict = {
            "pm25_mae": (self.saved_metrics.get("PM2.5") or self.saved_metrics.get("pm25") or {}).get("MAE", 10.57),
            "pm25_rmse": (self.saved_metrics.get("PM2.5") or self.saved_metrics.get("pm25") or {}).get("RMSE", 18.58),
            "pm25_r2": (self.saved_metrics.get("PM2.5") or self.saved_metrics.get("pm25") or {}).get("R2", 0.9492),
            "pm10_mae": (self.saved_metrics.get("PM10") or self.saved_metrics.get("pm10") or {}).get("MAE", 18.86),
            "pm10_rmse": (self.saved_metrics.get("PM10") or self.saved_metrics.get("pm10") or {}).get("RMSE", 32.20),
            "pm10_r2": (self.saved_metrics.get("PM10") or self.saved_metrics.get("pm10") or {}).get("R2", 0.9301),
            "evaluation_cohort": "404,431 Test Sequences (2015-2020 CPCB Benchmark)",
        }
        scientific_note = "Sensor-only LSTM forecast using 24 preceding hours of 27 continuous telemetry channels."

        if model_mode in ("ensemble", "severity_gated"):
            try:
                from src.data.tabular_features import extract_tabular_features

                if self._load_historical_gbdt():
                    gbt_pm25 = self._historical_gbdt_pm25
                    gbt_pm10 = self._historical_gbdt_pm10

                    tab_feats = extract_tabular_features(seq_arr)
                    gbdt_pm25_val = float(gbt_pm25.predict(tab_feats)[0])
                    gbdt_pm10_val = float(gbt_pm10.predict(tab_feats)[0])

                    if model_mode == "severity_gated":
                        def _to_float_or_fallback_hist(series_val, fallback):
                            try:
                                num = pd.to_numeric(series_val, errors="coerce")
                                return fallback if pd.isna(num) else float(num)
                            except Exception:
                                return fallback

                        pm25_curr = _to_float_or_fallback_hist(df_to_use["PM2.5"].iloc[-1], pm25_pred) if "PM2.5" in df_to_use.columns else pm25_pred
                        pm10_curr = _to_float_or_fallback_hist(df_to_use["PM10"].iloc[-1], pm10_pred) if "PM10" in df_to_use.columns else pm10_pred
                        gate_25 = 1.0 / (1.0 + np.exp(-(pm25_curr - 250.0) / 25.0))
                        gate_10 = 1.0 / (1.0 + np.exp(-(pm10_curr - 400.0) / 40.0))
                        w25 = 0.90 + 0.10 * gate_25
                        w10 = 0.80 + 0.20 * gate_10

                        pm25_pred = w25 * pm25_pred + (1.0 - w25) * gbdt_pm25_val
                        pm10_pred = w10 * pm10_pred + (1.0 - w10) * gbdt_pm10_val
                        model_version = "Severity-Gated Ensemble Forecaster (LSTM + GBDT)"
                        metrics_dict["pm25_mae"] = 10.35
                        metrics_dict["pm25_rmse"] = 18.38
                        metrics_dict["pm25_r2"] = 0.9503
                        metrics_dict["pm10_mae"] = 18.14
                        metrics_dict["pm10_rmse"] = 31.64
                        metrics_dict["pm10_r2"] = 0.9325
                        scientific_note = "Severity-gated multi-modal sensor ensemble with dynamic GBDT decoupling above severe episode thresholds (PM2.5 >= 250 ug/m3, PM10 >= 400 ug/m3)."
                    else:
                        pm25_pred = 0.90 * pm25_pred + 0.10 * gbdt_pm25_val
                        pm10_pred = 0.80 * pm10_pred + 0.20 * gbdt_pm10_val
                        model_version = "Validated Ensemble Forecaster (LSTM + GBDT) v2"
                        metrics_dict["pm25_mae"] = 10.40
                        metrics_dict["pm25_rmse"] = 18.55
                        metrics_dict["pm25_r2"] = 0.9494
                        metrics_dict["pm10_mae"] = 18.28
                        metrics_dict["pm10_rmse"] = 31.97
                        metrics_dict["pm10_r2"] = 0.9311
                        scientific_note = "Validated multi-modal sensor ensemble combining LSTM temporal representations with gradient-boosted tabular lag dynamics."
            except Exception:
                pass

        result = {
            "result_type": "Independent Forecast",
            "station_id": station_id,
            "forecast_horizon": "Next 1 Hour (t + 1)",
            "target_time": target_time,
            "input_freshness": input_freshness,
            "model_version": model_version,
            "pm25": round(pm25_pred, 2),
            "pm10": round(pm10_pred, 2),
            "unit": "ug/m3",
            "metrics": metrics_dict,
            "scientific_note": scientific_note,
        }

        if fallback_note:
            result["fallback_note"] = fallback_note

        # Shadow execution: run adapted model in parallel without exposing it as official forecast
        if enable_shadow:
            try:
                result["shadow_forecast"] = self._predict_adapted(
                    df_to_use,
                    station_id=station_id,
                    seq_arr=seq_arr,
                    severity_gated=True,
                )
            except Exception as e:
                result["shadow_forecast"] = {
                    "status": "error",
                    "error": str(e),
                    "message": "Shadow inference failed; production forecast unaffected.",
                }

        return result

    def predict_next_hour(
        self,
        scenario_id: Optional[str] = None,
        station_id: str = "AP001",
        df_24h: Optional[pd.DataFrame] = None,
        model_mode: str = "lstm",
        enable_shadow: bool = False,
    ) -> Dict[str, Any]:
        """Runs operational forecast for next hour (t+1) with uncertainty bounds."""
        from src.prediction.data_loader import load_demo_scenarios

        if df_24h is not None:
            fc_data = df_24h
        elif scenario_id is not None:
            demo_df = load_demo_scenarios()
            matching = demo_df[demo_df["scenario_id"] == scenario_id]
            if matching.empty:
                matching = demo_df[demo_df["scenario_id"].str.contains(scenario_id, case=False, na=False)]
            if matching.empty:
                matching = demo_df[demo_df["scenario_id"] == demo_df["scenario_id"].iloc[0]]
            fc_data = matching.sort_values("Datetime").reset_index(drop=True)
        else:
            demo_df = load_demo_scenarios()
            fc_data = demo_df[demo_df["scenario_id"] == demo_df["scenario_id"].iloc[0]].sort_values("Datetime").reset_index(drop=True)

        if model_mode == "adapted_gbdt_only":
            try:
                gbdt_res = self._predict_adapted_gbdt_only(
                    fc_data,
                    station_id=station_id,
                )
                res = self.underlying_service.predict_from_dataframe(fc_data, station_id=station_id)
                res["predicted_PM25"] = gbdt_res["pm25"]
                res["predicted_PM10"] = gbdt_res["pm10"]
                res["model"] = gbdt_res["model_version"]
                res["is_adapted"] = True
                res["is_standalone_gbdt"] = True
                res["test_benchmarks"] = {
                    "PM25_test_mae": gbdt_res["metrics"]["pm25_mae"],
                    "PM25_test_r2": gbdt_res["metrics"]["pm25_r2"],
                    "PM10_test_mae": gbdt_res["metrics"]["pm10_mae"],
                    "PM10_test_r2": gbdt_res["metrics"]["pm10_r2"],
                }
                return res
            except Exception as e:
                res = self.underlying_service.predict_from_dataframe(fc_data, station_id=station_id)
                res["fallback_reason"] = f"Adapted GBDT checkpoint unavailable ({e}); served production model."
                return res

        if model_mode in ("adapted_severity_gated", "adapted_ensemble"):
            try:
                adapted_res = self._predict_adapted(
                    fc_data,
                    station_id=station_id,
                    severity_gated=(model_mode == "adapted_severity_gated"),
                )
                res = self.underlying_service.predict_from_dataframe(fc_data, station_id=station_id)
                res["predicted_PM25"] = adapted_res["pm25"]
                res["predicted_PM10"] = adapted_res["pm10"]
                res["model"] = adapted_res["model_version"]
                res["is_adapted"] = True
                res["test_benchmarks"] = {
                    "PM25_test_mae": adapted_res["metrics"]["pm25_mae"],
                    "PM25_test_r2": adapted_res["metrics"]["pm25_r2"],
                    "PM10_test_mae": adapted_res["metrics"]["pm10_mae"],
                    "PM10_test_r2": adapted_res["metrics"]["pm10_r2"],
                }
                return res
            except Exception as e:
                # Fallback to production
                res = self.underlying_service.predict_from_dataframe(fc_data, station_id=station_id)
                res["fallback_reason"] = f"Adapted model checkpoint unavailable ({e}); served production model."
                return res

        # Production model path
        res = self.underlying_service.predict_from_dataframe(fc_data, station_id=station_id)
        if enable_shadow:
            try:
                res["shadow_forecast"] = self._predict_adapted(fc_data, station_id=station_id, severity_gated=True)
            except Exception as e:
                res["shadow_forecast"] = {
                    "status": "error",
                    "error": str(e),
                    "message": "Shadow inference failed; production forecast unaffected.",
                }
        return res
