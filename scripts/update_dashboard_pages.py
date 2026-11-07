"""
scripts/update_dashboard_pages.py
"""
from pathlib import Path

def update_dashboard():
    p = Path("src/components/hybrid_dashboard_pages.py")
    content = p.read_text(encoding="utf-8")
    
    old_block = """    fusion_benchmark_data = [
        {"Model Architecture": "Sensor-Only LSTM (Sequence)", "PM2.5 MAE (µg/m³)": "27.51", "PM2.5 RMSE (µg/m³)": "51.82", "PM2.5 R²": "0.5850", "PM10 MAE (µg/m³)": "63.40", "PM10 R²": "0.4120", "Modality": "Sensor Telemetry"},
        {"Model Architecture": "Image-Only ResNet-50", "PM2.5 MAE (µg/m³)": "60.99", "PM2.5 RMSE (µg/m³)": "88.20", "PM2.5 R²": "-0.2023", "PM10 MAE (µg/m³)": "N/A", "PM10 R²": "N/A", "Modality": "Photographic"},
        {"Model Architecture": "Unified Hybrid (Full Multimodal)", "PM2.5 MAE (µg/m³)": "56.59", "PM2.5 RMSE (µg/m³)": "82.72", "PM2.5 R²": "-0.0576", "PM10 MAE (µg/m³)": "140.91", "PM10 R²": "-0.0114", "Modality": "Sensor + Camera"},
        {"Model Architecture": "Unified Hybrid (Sensor Fallback, m=0)", "PM2.5 MAE (µg/m³)": "54.06", "PM2.5 RMSE (µg/m³)": "72.56", "PM2.5 R²": "+0.1863", "PM10 MAE (µg/m³)": "147.43", "PM10 R²": "+0.1009", "Modality": "Gated Sensor Fallback"},
    ]
    st.table(pd.DataFrame(fusion_benchmark_data))"""

    new_block = """    fusion_benchmark_data = [
        {"Model Architecture": "Sensor-Only LSTM (validation_v2 baseline)", "PM2.5 MAE (µg/m³)": "40.94", "PM2.5 RMSE (µg/m³)": "64.81", "PM2.5 R²": "+0.3508", "PM10 MAE (µg/m³)": "N/A", "PM10 R²": "N/A", "Modality": "Sensor Sequence", "Experiment Source": "experiments/multimodal_fusion/validation_v2/metrics.json"},
        {"Model Architecture": "Image-Only ResNet-50 (validation_v2 baseline)", "PM2.5 MAE (µg/m³)": "64.99", "PM2.5 RMSE (µg/m³)": "94.59", "PM2.5 R²": "-0.3828", "PM10 MAE (µg/m³)": "N/A", "PM10 R²": "N/A", "Modality": "Outdoor Image Haze", "Experiment Source": "experiments/multimodal_fusion/validation_v2/metrics.json"},
        {"Model Architecture": "Multimodal Fusion (validation_v2 concat MLP)", "PM2.5 MAE (µg/m³)": "58.94", "PM2.5 RMSE (µg/m³)": "85.92", "PM2.5 R²": "-0.1410", "PM10 MAE (µg/m³)": "N/A", "PM10 R²": "N/A", "Modality": "Sensor + Camera (Concat)", "Experiment Source": "experiments/multimodal_fusion/validation_v2/metrics.json"},
        {"Model Architecture": "Unified Hybrid Model (Full Multimodal)", "PM2.5 MAE (µg/m³)": "56.59", "PM2.5 RMSE (µg/m³)": "82.72", "PM2.5 R²": "-0.0576", "PM10 MAE (µg/m³)": "140.91", "PM10 R²": "-0.0114", "Modality": "Sensor + Camera (Gated)", "Experiment Source": "experiments/unified_hybrid/metrics.json"},
        {"Model Architecture": "Unified Hybrid Model (Sensor Fallback, m=0)", "PM2.5 MAE (µg/m³)": "54.06", "PM2.5 RMSE (µg/m³)": "72.56", "PM2.5 R²": "+0.1863", "PM10 MAE (µg/m³)": "147.43", "PM10 R²": "+0.1009", "Modality": "Gated Pure Telemetry", "Experiment Source": "experiments/unified_hybrid/metrics.json"},
    ]
    st.table(pd.DataFrame(fusion_benchmark_data))

    st.caption("All models above are evaluated on the exact same 97 test samples (9 station-date groups) with zero test-loss early stopping leakage. Preprocessing statistics fitted strictly on training split (18 groups, 81 samples).")"""

    assert old_block in content, "old_block not found in hybrid_dashboard_pages.py"
    p.write_text(content.replace(old_block, new_block), encoding="utf-8")
    print("Updated hybrid_dashboard_pages.py Section 3 benchmark table successfully!")

if __name__ == "__main__":
    update_dashboard()
