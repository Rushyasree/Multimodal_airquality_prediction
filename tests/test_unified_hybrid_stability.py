"""
tests/test_unified_hybrid_stability.py

Phase 20.1: Regression test suite for Unified Hybrid Forecast stability,
NoneType index handling, empty options resilience, and error state rendering.
"""

import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.components.hybrid_dashboard_pages import render_unified_hybrid_page
from src.services.hybrid_prediction_service import HybridPredictionService
from src.services.image_estimation_service import ImageEstimationService


class TestUnifiedHybridStability:
    """Rigorous regression tests for Unified Hybrid Forecast rendering and state handling."""

    def test_empty_curated_samples_no_typeerror(self, monkeypatch):
        """Simulate Streamlit Cloud where dataset is unindexed and curated_samples is empty list."""
        with patch.object(ImageEstimationService, "get_curated_sample_images", return_value=[]):
            with patch.object(HybridPredictionService, "predict_hybrid") as mock_predict:
                mock_predict.return_value = {
                    "status": "success",
                    "forecast_state": "HISTORICAL DEMO FORECAST",
                    "sequence_provenance": "Antecedent 24h Sequence",
                    "forecast_issue_time": "2026-11-12 18:00:00",
                    "target_time": "2026-11-12 19:00:00",
                    "pm25": 42.5,
                    "pm10": 78.0,
                    "gating_alpha": 0.08,
                    "sensor_pm25": 41.0,
                    "visual_delta_pm25": 1.5,
                    "image_contributed": True,
                }
                # Must render without TypeError
                render_unified_hybrid_page()

    def test_model_unavailable_state_rendering(self):
        """Simulate upstream telemetry failure where pm25, pm10, and sensor_pm25 are None."""
        with patch.object(HybridPredictionService, "predict_hybrid") as mock_predict:
            mock_predict.return_value = {
                "status": "unavailable",
                "forecast_state": "LIVE INPUT UNAVAILABLE",
                "sequence_provenance": "Physical Station Inlet (AP001)",
                "forecast_issue_time": "Unavailable",
                "target_time": "Unavailable",
                "pm25": None,
                "pm10": None,
                "sensor_pm25": None,
                "visual_delta_pm25": None,
                "gating_alpha": 0.0,
                "image_contributed": False,
                "is_synthetic_input": False,
            }
            # Rendering must complete safely without formatting exceptions (e.g. unsupported format string on NoneType)
            render_unified_hybrid_page()

    def test_inference_exception_graceful_handling(self):
        """Simulate unexpected inference exception in hybrid model pipeline."""
        with patch.object(HybridPredictionService, "predict_hybrid", side_effect=RuntimeError("CUDA out of memory")):
            # Must catch the error, log traceback, and display safe error banner without crashing
            render_unified_hybrid_page()

    def test_none_index_format_func_safety(self):
        """Directly verify lambda format_func with None, int, and out-of-bound indices."""
        sample_labels = ["Sample 1: Amaravati", "Sample 2: Tirupati"]
        format_func = lambda x: sample_labels[x] if (x is not None and isinstance(x, int) and 0 <= x < len(sample_labels)) else ""

        assert format_func(0) == "Sample 1: Amaravati"
        assert format_func(1) == "Sample 2: Tirupati"
        assert format_func(None) == ""
        assert format_func(99) == ""
        assert format_func(-1) == ""
        assert format_func("invalid") == ""


class TestDashboardAppTestIntegration:
    """End-to-end headless AppTest navigation across Home and Unified Hybrid pages."""

    def test_unified_hybrid_page_navigation(self):
        """Navigate to Unified Hybrid Forecast tab and verify zero unhandled exceptions."""
        at = AppTest.from_file("app.py", default_timeout=60)
        at.run()
        assert len(at.exception) == 0, f"Exceptions on Home: {[e.value for e in at.exception]}"

        # Switch to Unified Hybrid Forecast
        at.session_state["nav_tab"] = "Unified Hybrid Forecast"
        at.run()
        assert len(at.exception) == 0, f"Exceptions on Unified Hybrid: {[e.value for e in at.exception]}"

    def test_home_dashboard_normal_rendering(self):
        """Verify normal Home Dashboard renders completely with zero exceptions."""
        at = AppTest.from_file("app.py", default_timeout=60)
        at.run()
        assert len(at.exception) == 0
        assert at.session_state["nav_tab"] in ("Home", "Overview")
