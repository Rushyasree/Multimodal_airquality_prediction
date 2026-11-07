"""
scripts/patch_dashboard_section4.py
Adds Section 4 (Accuracy Optimization v2) to render_model_performance_page in src/components/hybrid_dashboard_pages.py.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
dashboard_path = ROOT_DIR / "src/components/hybrid_dashboard_pages.py"
text = dashboard_path.read_text(encoding="utf-8")

old_closing = """        • <strong>Deterministic Fallback:</strong> When camera photos are absent ($m = 0$), the model guarantees exact mathematical equivalence to the pure sensor baseline (MAE = <strong>9.04 ug/m3</strong>), ensuring 100% operational reliability.
    </div>
    \"\"\")"""

section_4_code = """        • <strong>Deterministic Fallback:</strong> When camera photos are absent ($m = 0$), the model guarantees exact mathematical equivalence to the pure sensor baseline (MAE = <strong>9.04 ug/m3</strong>), ensuring 100% operational reliability.
    </div>
    \"\"\")

    st.markdown(\"### ⚡ Section 4: Accuracy Optimization v2 — Large Benchmark & Validated Ensemble\")
    st.markdown(
        \"Rigorous out-of-sample benchmark across **404,431 test sequences** comparing 1-hour persistence, \"
        \"24-hour and 168-hour seasonal baselines, pretrained 2-layer LSTM, tuned gradient boosting regressor (GBDT), \"
        \"residual boosting, and the **Validated Multi-Modal Sensor Ensemble Blend**.\"
    )

    v2_dir = ROOT_DIR / \"experiments\" / \"accuracy_optimization_v2\"
    bench_csv = v2_dir / \"baseline_comparison.csv\"
    if bench_csv.exists():
        st.markdown(\"#### 📋 Full Historical Benchmark Comparison (N = 404,431 Test Sequences)\")
        st.dataframe(pd.read_csv(bench_csv), use_container_width=True)

    tier_csv = v2_dir / \"tier_breakdown_pm25.csv\"
    if tier_csv.exists():
        st.markdown(\"#### 🎯 Performance by CPCB National Air Quality Index (NAQI) Tier\")
        st.dataframe(pd.read_csv(tier_csv), use_container_width=True)

    v2_fig1 = v2_dir / \"figures\" / \"actual_vs_predicted_benchmark.png\"
    v2_fig2 = v2_dir / \"figures\" / \"residual_distribution_benchmark.png\"
    v2_fig3 = v2_dir / \"figures\" / \"category_error_breakdown.png\"
    v2_fig4 = v2_dir / \"figures\" / \"underprediction_bias_by_tier.png\"

    if v2_fig1.exists() and v2_fig2.exists():
        st.markdown(\"#### 📊 Benchmark Regression & Error Distribution (404,431 Sequences)\")
        c_a, c_b = st.columns(2)
        with c_a:
            st.image(str(v2_fig1), caption=\"Ensemble Actual vs Predicted Hexbin Density (N=404,431)\", use_container_width=True)
        with c_b:
            st.image(str(v2_fig2), caption=\"PM2.5 & PM10 Residual Error Distribution\", use_container_width=True)

    if v2_fig3.exists() and v2_fig4.exists():
        st.markdown(\"#### 🔍 Tier Breakdown & Extreme Event Underprediction Bias\")
        c_c, c_d = st.columns(2)
        with c_c:
            st.image(str(v2_fig3), caption=\"PM2.5 MAE by Pollution Tier\", use_container_width=True)
        with c_d:
            st.image(str(v2_fig4), caption=\"Systematic Underprediction Bias in Severe Pollution\", use_container_width=True)

    render_html(\"\"\"
    <div style=\"background: #122238; border-left: 3px solid #00E676; padding: 1.1rem; border-radius: 8px; margin-top: 1rem; font-size: 0.85rem; color: #8DA4BE; line-height: 1.6;\">
        <strong style=\"color: #F0F4F8;\">Accuracy Optimization v2 Key Research Findings:</strong><br>
        • <strong>Winning Multi-Modal Ensemble:</strong> Blending LSTM sequential representations (0.90 PM2.5 / 0.80 PM10) with tree-based tabular gradient boosting (0.10 PM2.5 / 0.20 PM10) achieves the best overall performance: PM2.5 MAE = <strong>10.40 µg/m³</strong> (RMSE 18.55, R² 0.9494) and PM10 MAE = <strong>18.28 µg/m³</strong> (RMSE 31.97, R² 0.9311) on all 404,431 test sequences.<br>
        • <strong>Short-term Inertia Dominance:</strong> Next-hour forecasting is heavily driven by 1-hour physical inertia (Persistence MAE = 10.68 µg/m³). Diurnal cycles (24h lag MAE = 28.99 µg/m³) and weekly cycles (168h lag MAE = 45.12 µg/m³) have vastly weaker short-term predictive power.<br>
        • <strong>Severe Pollution Underprediction:</strong> In the 'Severe' category (>250 µg/m³), regression models exhibit mean-reversion bias (-16 to -22 µg/m³), while outperforming persistence in the 'Very Poor' category (120-250 µg/m³, Ensemble MAE 17.46 vs Persistence 21.10).<br>
        • <strong>Ablation Insights:</strong> Weather features (temperature, dewpoint depression, relative humidity, wind speed) contribute measurably to reducing validation error compared to pollutant-only tabular baselines.
    </div>
    \"\"\")"""

text = text.replace("\r\n", "\n")
old_closing_norm = old_closing.replace("\r\n", "\n")
section_4_code_norm = section_4_code.replace("\r\n", "\n")

assert old_closing_norm in text, "Could not find old_closing in text"
text = text.replace(old_closing_norm, section_4_code_norm)

dashboard_path.write_text(text, encoding="utf-8")
print("Successfully patched hybrid_dashboard_pages.py with Section 4!")
