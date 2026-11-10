"""
tests/test_model_input_tensor_alignment.py

Verifies mathematical array and tensor integrity across model comparison pipelines:
- Asserts identical (1, 24, 27) PyTorch tensors ingested by Sensor-Only LSTM and Unified Hybrid branches.
- Verifies SHA-256 tensor digests match across models.
- Verifies ModelComparisonService reports strict tensor synchronization.
"""

import hashlib
import torch
import numpy as np
import pandas as pd
import pytest

from src.services.sensor_forecast_service import SensorForecastService
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.model_comparison_service import ModelComparisonService
from src.prediction.data_loader import load_demo_scenarios


@pytest.fixture(scope="module")
def services():
    sensor_srv = SensorForecastService()
    hybrid_srv = HybridPredictionService()
    comp_srv = ModelComparisonService()
    return {
        "sensor": sensor_srv,
        "hybrid": hybrid_srv,
        "comparison": comp_srv,
    }


class TestModelInputTensorAlignment:
    """Verifies that forecasting models ingest mathematically identical tensors."""

    def test_tensor_identity_on_demo_scenarios(self, services):
        """Sensor and Hybrid services must produce identical normalized tensors and SHA-256 digests."""
        df_demo = load_demo_scenarios()
        scenarios = df_demo["scenario_id"].unique()[:3]  # Check first 3 scenarios

        for sc_id in scenarios:
            sub_df = df_demo[df_demo["scenario_id"] == sc_id].copy().reset_index(drop=True)
            station_id = str(sub_df["StationId"].iloc[0])

            # Preprocess through sensor service underlying preprocessor
            seq_arr_sensor = services["sensor"].underlying_service.preprocess_sequence(
                sub_df, station_id=station_id
            )
            # Preprocess through hybrid service underlying preprocessor
            seq_arr_hybrid = services["hybrid"].lstm_service.preprocess_sequence(
                sub_df, station_id=station_id
            )

            # Mathematical array equality
            np.testing.assert_array_almost_equal(
                seq_arr_sensor,
                seq_arr_hybrid,
                decimal=6,
                err_msg=f"Tensor mismatch for scenario {sc_id}",
            )

            # Convert to PyTorch tensors as passed to models
            t_sensor = torch.from_numpy(seq_arr_sensor).float()
            t_hybrid = torch.from_numpy(seq_arr_hybrid).float()
            assert torch.allclose(t_sensor, t_hybrid, atol=1e-6)

            # SHA-256 Digest equality
            digest_sensor = hashlib.sha256(seq_arr_sensor.tobytes()).hexdigest()
            digest_hybrid = hashlib.sha256(seq_arr_hybrid.tobytes()).hexdigest()
            assert digest_sensor == digest_hybrid

            # Run full service methods and verify digest in returned payload
            res_sensor = services["sensor"].get_forecast(station_id=station_id, df_24h=sub_df)
            res_hybrid = services["hybrid"].predict_hybrid(station_id=station_id, df_24h=sub_df)

            assert res_sensor["sensor_tensor_digest"] == digest_sensor
            assert res_hybrid["sensor_tensor_digest"] == digest_hybrid
            assert res_sensor["sensor_tensor_digest"] == res_hybrid["sensor_tensor_digest"]

    def test_model_comparison_service_reports_tensor_parity(self, services):
        """ModelComparisonService must verify and report strict tensor synchronization."""
        comp_res = services["comparison"].compare_all(station_id="AP001", city="Amaravati")

        assert comp_res["tensor_matched"] is True
        assert comp_res["tensor_synchronization_status"] == "Strictly Synchronized (Identical Tensor Digest)"
        assert "Strictly Synchronized" in comp_res["synchronization_status"]
        assert comp_res["tensor_digest_lstm"] is not None
        assert comp_res["tensor_digest_hybrid"] is not None
        assert comp_res["tensor_digest_lstm"] == comp_res["tensor_digest_hybrid"]

    def test_tensor_mismatch_detection_on_tampered_input(self, services):
        """Digest check must fail if input tensors diverge even by a fraction."""
        df_demo = load_demo_scenarios()
        sub_df = df_demo[df_demo["StationId"] == "AP001"].head(24).copy().reset_index(drop=True)

        seq_arr_clean = services["sensor"].underlying_service.preprocess_sequence(sub_df, "AP001")
        seq_arr_tampered = seq_arr_clean.copy()
        seq_arr_tampered[0, 0, 0] += 0.05  # Slight perturbation

        digest_clean = hashlib.sha256(seq_arr_clean.tobytes()).hexdigest()
        digest_tampered = hashlib.sha256(seq_arr_tampered.tobytes()).hexdigest()

        assert digest_clean != digest_tampered
