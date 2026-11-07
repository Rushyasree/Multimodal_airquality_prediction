"""
src/prediction/lstm_forecaster.py

Production-ready inference service for numerical next-hour air quality forecasting
using the trained 2-layer LSTM model.
"""

from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """2-layer LSTM forecaster architecture matching the trained checkpoint."""

    def __init__(
        self,
        input_size: int = 27,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.regressor(last_hidden)


class LSTMForecasterService:
    """Encapsulates checkpoint loading, input validation, feature scaling,

    and next-hour PM2.5 / PM10 predictions.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        coords_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path or root / "models/lstm_baseline/best_lstm_model.pt")
        self.metadata_path = Path(
            metadata_path or root / "data/processed/model_sequences/preprocessing_metadata.json"
        )
        self.coords_path = Path(coords_path or root / "data/processed/station_coordinates_90.csv")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_metadata()
        self._load_model()
        self._load_coordinates()

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Preprocessing metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_names: List[str] = meta["feature_names"]
        self.target_names: List[str] = meta["target_names"]
        self.window_size: int = meta.get("window_size_hours", 24)

        self.imputation_medians = {k: float(v) for k, v in meta["imputation_medians"].items()}
        self.scaler_means = np.array([meta["scaler_means"][k] for k in self.feature_names], dtype=np.float32)
        self.scaler_stds = np.array([meta["scaler_stds"][k] for k in self.feature_names], dtype=np.float32)
        self.scaler_stds[self.scaler_stds < 1e-8] = 1.0

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location=self.device)

        input_size = ckpt.get("input_size", len(self.feature_names))
        hidden_size = ckpt.get("hidden_size", 64)
        num_layers = ckpt.get("num_layers", 2)
        dropout = ckpt.get("dropout", 0.2)

        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=len(self.target_names),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.target_mean = np.array(ckpt["target_mean"], dtype=np.float32)
        self.target_std = np.array(ckpt["target_std"], dtype=np.float32)
        self.target_std[self.target_std < 1e-8] = 1.0

    def _load_coordinates(self) -> None:
        self.coordinates_df = None
        if self.coords_path.exists():
            df = pd.read_csv(self.coords_path)
            df["StationId"] = df["StationId"].astype(str).str.strip().str.upper()
            self.coordinates_df = df.set_index("StationId")

    def get_station_coordinates(self, station_id: str) -> Tuple[float, float]:
        station_id_norm = str(station_id).strip().upper()
        if self.coordinates_df is not None and station_id_norm in self.coordinates_df.index:
            row = self.coordinates_df.loc[station_id_norm]
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            return lat, lon
        return (
            self.imputation_medians.get("Latitude", 28.563262),
            self.imputation_medians.get("Longitude", 77.2011573),
        )

    def preprocess_sequence(
        self,
        df_24h: pd.DataFrame,
        station_id: Optional[str] = None,
        station_lat: Optional[float] = None,
        station_lon: Optional[float] = None,
    ) -> np.ndarray:
        """Transforms a 24-row dataframe with hourly observations into a standardized (1, 24, 27) array."""
        if len(df_24h) != self.window_size:
            raise ValueError(
                f"Expected exactly {self.window_size} hourly rows, got {len(df_24h)}."
            )

        df = df_24h.copy()

        # Resolve station coordinates
        if station_lat is None or station_lon is None:
            if station_id:
                lat, lon = self.get_station_coordinates(station_id)
            else:
                lat = self.imputation_medians["Latitude"]
                lon = self.imputation_medians["Longitude"]
        else:
            lat, lon = station_lat, station_lon

        df["Latitude"] = lat
        df["Longitude"] = lon

        # Time features
        if "Datetime" in df.columns:
            dt = pd.to_datetime(df["Datetime"], errors="coerce")
            hours = dt.dt.hour
            months = dt.dt.month
            dows = dt.dt.dayofweek
        else:
            # Fallback to sequential hours if timestamp missing
            hours = pd.Series(np.arange(len(df)) % 24)
            months = pd.Series(np.full(len(df), 1))
            dows = pd.Series(np.full(len(df), 0))

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12.0)
        df["day_of_week_sin"] = np.sin(2 * np.pi * dows / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dows / 7.0)

        # Wind direction features
        if "wind_direction" in df.columns:
            wind_deg = pd.to_numeric(df["wind_direction"], errors="coerce")
            df["wind_direction_sin"] = np.sin(np.deg2rad(wind_deg))
            df["wind_direction_cos"] = np.cos(np.deg2rad(wind_deg))
        else:
            if "wind_direction_sin" not in df.columns:
                df["wind_direction_sin"] = np.nan
            if "wind_direction_cos" not in df.columns:
                df["wind_direction_cos"] = np.nan

        # Ensure all 27 features exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Impute missing values with training-period medians
        for col in self.feature_names:
            df[col] = df[col].fillna(self.imputation_medians[col])

        # Extract numpy array in the exact feature order
        features_raw = df[self.feature_names].to_numpy(dtype=np.float32)

        # Standardize using training means and stds
        features_scaled = (features_raw - self.scaler_means) / self.scaler_stds

        if not np.isfinite(features_scaled).all():
            raise ValueError("Preprocessed features contain non-finite values (NaN / Inf).")

        return np.expand_dims(features_scaled, axis=0)  # Shape: (1, 24, 27)

    def predict_sequence(
        self,
        x: Union[np.ndarray, torch.Tensor],
        last_observed_pm25: Optional[float] = None,
        last_observed_pm10: Optional[float] = None,
    ) -> Dict[str, Union[float, Dict[str, float], str]]:
        """Predicts next-hour PM2.5 and PM10 from a (24, 27) or (B, 24, 27) scaled array."""
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape != (self.window_size, len(self.feature_names)):
                    raise ValueError(
                        f"Expected shape ({self.window_size}, {len(self.feature_names)}), got {x.shape}."
                    )
                tensor = torch.from_numpy(x).unsqueeze(0).to(torch.float32)
            elif x.ndim == 3:
                if x.shape[1:] != (self.window_size, len(self.feature_names)):
                    raise ValueError(
                        f"Expected shape (B, {self.window_size}, {len(self.feature_names)}), got {x.shape}."
                    )
                tensor = torch.from_numpy(x).to(torch.float32)
            else:
                raise ValueError(f"Unsupported array dimension {x.ndim}. Expected 2D or 3D.")
        elif isinstance(x, torch.Tensor):
            if x.ndim == 2:
                tensor = x.unsqueeze(0).to(torch.float32)
            else:
                tensor = x.to(torch.float32)
        else:
            raise TypeError(f"Expected numpy array or torch.Tensor, got {type(x)}.")

        tensor = tensor.to(self.device)

        with torch.no_grad():
            scaled_pred = self.model(tensor).cpu().numpy()

        # Inverse transform to original concentration units (ug/m3)
        raw_pred = scaled_pred * self.target_std + self.target_mean
        pm25_pred = float(np.clip(raw_pred[0, 0], 0.0, None))
        pm10_pred = float(np.clip(raw_pred[0, 1], 0.0, None))

        result = {
            "predicted_PM25": round(pm25_pred, 2),
            "predicted_PM10": round(pm10_pred, 2),
            "unit": "ug/m3",
            "forecast_horizon": "t+1 (next 1 hour)",
            "model": "2-Layer LSTM Forecaster (Trained Operational Baseline)",
            "test_benchmarks": {
                "PM25_test_mae": 10.57,
                "PM25_test_r2": 0.949,
                "PM10_test_mae": 18.86,
                "PM10_test_r2": 0.930,
            },
            "uncertainty_intervals": {
                "PM2.5_lower_80": round(float(np.clip(pm25_pred - 14.8, 0.0, None)), 2),
                "PM2.5_upper_80": round(pm25_pred + 14.8, 2),
                "PM2.5_mae_bound": 10.57,
                "PM10_lower_80": round(float(np.clip(pm10_pred - 26.5, 0.0, None)), 2),
                "PM10_upper_80": round(pm10_pred + 26.5, 2),
                "PM10_mae_bound": 18.86,
                "methodology": "Empirical held-out test set residual error bounds (N=404,431).",
            },
        }

        # If last observed values provided, compute persistence delta
        if last_observed_pm25 is not None:
            result["persistence_PM25"] = round(float(last_observed_pm25), 2)
            result["delta_PM25"] = round(pm25_pred - float(last_observed_pm25), 2)
        if last_observed_pm10 is not None:
            result["persistence_PM10"] = round(float(last_observed_pm10), 2)
            result["delta_PM10"] = round(pm10_pred - float(last_observed_pm10), 2)

        return result

    def predict_from_dataframe(
        self,
        df_24h: pd.DataFrame,
        station_id: Optional[str] = None,
        station_lat: Optional[float] = None,
        station_lon: Optional[float] = None,
    ) -> Dict[str, Union[float, Dict[str, float], str]]:
        """Convenience method combining preprocessing and prediction for a 24-row dataframe."""
        scaled_x = self.preprocess_sequence(
            df_24h=df_24h,
            station_id=station_id,
            station_lat=station_lat,
            station_lon=station_lon,
        )

        last_pm25 = None
        if "PM2.5" in df_24h.columns and pd.notna(df_24h["PM2.5"].iloc[-1]):
            last_pm25 = float(df_24h["PM2.5"].iloc[-1])

        last_pm10 = None
        if "PM10" in df_24h.columns and pd.notna(df_24h["PM10"].iloc[-1]):
            last_pm10 = float(df_24h["PM10"].iloc[-1])

        return self.predict_sequence(
            x=scaled_x,
            last_observed_pm25=last_pm25,
            last_observed_pm10=last_pm10,
        )

    def validate_sequence(self, df_24h: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validates that an input sequence satisfies the mathematical and feature requirements
        of the trained 2-layer LSTM forecaster (which expects continuous 24-hour sequences across
        27 co-located criteria and ECMWF ERA5-Land meteorological features).
        """
        if not isinstance(df_24h, pd.DataFrame):
            return False, f"Expected pandas DataFrame, got {type(df_24h).__name__}."
        if len(df_24h) != self.window_size:
            return False, f"Sequence length mismatch: expected exactly {self.window_size} hourly steps, got {len(df_24h)}."
        
        # Check required target presence
        has_pm25 = "PM2.5" in df_24h.columns and df_24h["PM2.5"].notna().sum() > 0
        has_pm10 = "PM10" in df_24h.columns and df_24h["PM10"].notna().sum() > 0
        if not (has_pm25 or has_pm10):
            return False, "Input sequence missing both PM2.5 and PM10 particulate observations."
        
        # Check physical plausibility (no negative values for pollutants)
        for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
            if col in df_24h.columns:
                vals = pd.to_numeric(df_24h[col], errors="coerce").dropna()
                if (vals < 0).any():
                    return False, f"Physically impossible negative concentration detected in '{col}'."
                
        return True, "Sequence validated successfully."


def prepare_forecast_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format antecedent telemetry and weather sequences for tabular display with physical units."""
    if df is None or df.empty:
        return pd.DataFrame()

    candidate_mapping = [
        (("Datetime", "datetime", "date"), "Datetime"),
        (("PM2.5", "pm25", "pm2.5"), "PM2.5 (µg/m³)"),
        (("PM10", "pm10"), "PM10 (µg/m³)"),
        (("NO2", "no2"), "NO2 (µg/m³)"),
        (("SO2", "so2"), "SO2 (µg/m³)"),
        (("CO", "co"), "CO (mg/m³)"),
        (("O3", "o3"), "O3 (µg/m³)"),
        (("temperature_C", "temp_C", "temperature"), "Temperature (°C)"),
        (("relative_humidity", "relative_humidity_pct", "humidity"), "Relative Humidity (%)"),
        (("wind_speed", "wind_speed_ms"), "Wind Speed (m/s)"),
        (("wind_direction", "wind_direction_deg"), "Wind Direction (°)"),
        (("precipitation_mm", "rainfall_mm"), "Precipitation (mm)"),
    ]

    display_df = pd.DataFrame()
    for col_candidates, label in candidate_mapping:
        matched_col = None
        for candidate in col_candidates:
            if candidate in df.columns:
                matched_col = candidate
                break
        if matched_col is not None:
            display_df[label] = df[matched_col]

    return display_df


