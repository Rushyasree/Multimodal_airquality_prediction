# VayuDrishti: Production Deployment & Verification Guide

This guide provides instructions for launching the interactive VayuDrishti dashboard, operating the autonomous live multi-station monitoring pipeline, and compiling the peer-reviewed conference publication package.

---

## 1. Interactive Streamlit Dashboard Launch

### 1.1 Local Launch Command
To launch the VayuDrishti dashboard in your web browser:
```bash
streamlit run app.py
```
By default, the dashboard will bind to `http://localhost:8501`.

### 1.2 Dashboard Architecture: 8 Interactive Sections
The application features eight modules accessible via the navigation sidebar:
1. **Overview (Dual-Mode Summary)**: Real-time AQI hero card, current weather, health advisories, dynamic greeting based on IST timezone, and quick dual-mode navigation.
2. **Live Air Quality**: Station-level telemetry inspection across CAAQMS stations, displaying particulate ($\text{PM}_{2.5}$, $\text{PM}_{10}$) and trace gas levels with data freshness indicators.
3. **Individual Forecasting Methods (Mode A)**:
   - *Method 1 (Sensor-Only LSTM)*: 24-hour sequence autoregression predicting $t+1$ hour particulate concentrations.
   - *Method 2 (Sensor + Meteorology)*: Numerical forecasting augmented by ERA5-Land weather reanalysis.
   - *Method 3 (Environmental Vision)*: Visual haze estimation from outdoor camera imagery using ResNet-50.
   - *Method 4 (Baselines)*: 1-hour persistence ($y_{t+1} = y_t$) and historical rolling statistics.
4. **Unified Hybrid Forecast (Mode B)**: Automated severity-gated ensemble combining deep recurrent sequences and LightGBM decision trees with missing-modality fallback.
5. **Side-by-Side Method Comparison**: Comprehensive comparison matrix evaluating all forecasting methods simultaneously for any selected station.
6. **Interactive 3D Station Map**: Geospatial mapping of 90 CPCB continuous monitoring stations across India with zoom, pan, and real-time AQI status pins.
7. **Model Performance & Benchmark Matrices**:
   - Classification confusion matrices across CPCB NAQI severity bins ($>99.3\%$ adjacent-category accuracy).
   - 168-hour continuous forecasting timeseries trajectory and residual scatter distributions.
   - Multi-station and seasonal evaluation charts across 15 monitoring stations and 4 IMD seasons.
8. **Data Sources & System Health**:
   - Physical model checkpoint verification (`best_lstm_model.pt`, `adapted_gbdt_pm25.joblib`, `best_unified_hybrid_model.pt`).
   - Longitudinal live monitoring registry and data provenance audit log.

---

## 2. Autonomous Hourly Live-Data Collection & Monitoring

The continuous multi-station monitoring pipeline operates via `src/services/longitudinal_collector_service.py`.

### 2.1 Manual or On-Demand Execution
To run a single collection and forecast cycle across all 15 stations:
```bash
python -m src.services.longitudinal_collector_service
```

### 2.2 Automated Scheduling
- **Windows Task Scheduler**: Register the task to trigger every hour at minute 15:
  ```powershell
  schtasks /create /tn "VayuDrishti_Hourly_Collector" /tr "python.exe E:\Documents\multimodal_air_quality_vscode_starter\src\services\longitudinal_collector_service.py" /sc hourly /st 00:15
  ```
- **Linux / macOS Crontab**:
  ```cron
  15 * * * * cd /path/to/multimodal_air_quality_vscode_starter && python3 -m src.services.longitudinal_collector_service >> logs/collector.log 2>&1
  ```

---

## 3. Academic Research Publication Package (`paper/`)

The academic research manuscript is ready in the `paper/` directory:
- **`paper/main.tex`**: Master LaTeX manuscript formatted for ACM / IEEE / NeurIPS conferences.
- **`paper/manuscript.md`**: Complete Markdown document with all tables and citations.
- **`paper/references.bib`**: BibTeX file with 18 verified peer-reviewed citations.
- **`paper/tables/`**:
  - `table_overall_benchmarks.tex`: Overall model comparison on $N=236,289$ sequences.
  - `table_negative_result_ablation.tex`: Multimodal alignment ablation and negative result.
  - `table_multistation_performance.tex`: Multi-station 15-station evaluation across 11 States/UTs.
  - `table_seasonal_metrics.tex`: Seasonal breakdown across IMD seasons (506,848 evaluations).
  - `table_conformal_calibration.tex`: Split-conformal distribution-free uncertainty calibration.
  - `table_statistical_tests.tex`: Stationary Block Bootstrap ($B=5,000$), paired t, Wilcoxon, Bonferroni, FDR.
  - `table_spatial_advection.tex`: Cross-station advection, lag correlations, and multi-node forecasting gains.
- **`paper/figures/`**: High-resolution figures (`fig1_conformal_calibration.png`, `fig2_seasonal_model_comparison.png`, `fig3_multistation_performance.png`, `fig4_regime_error_curve.png`, `fig_spatial_cross_correlation.png`, `fig_spatial_forecast_comparison.png`).

### 3.1 Compilation via Overleaf or Local TeX Live
1. Zip the `paper/` directory and upload to [Overleaf](https://www.overleaf.com).
2. Set compiler to **pdfLaTeX** and root file to **`main.tex`**.
3. Or compile locally on any machine with TeX Live:
   ```bash
   cd paper/
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

---

## 4. Verification & Governance Summary
- **Model Checkpoints**: All checkpoints (`best_lstm_model.pt`, `adapted_gbdt_pm25.joblib`, `adapted_gbdt_pm10.joblib`, `resnet50_weights.pth`) remain completely untouched and verified with immutable SHA-256 hashes.
- **Temporal Integrity**: All features and buffers are strictly causal ($t \le \text{issue\_time}$). Zero lookahead leakage, zero synthetic imputation, and strict 8-gate sequence validation.
- **Comprehensive QA Suite**: 100% of tests pass across `test_operational_monitoring_and_reliability.py`, `test_live_data_provenance_and_validation.py`, `test_forecast_time_alignment.py`, `test_eight_sections_smoke.py`, and `test_dashboard_navigation.py`. See [docs/qa/](qa/README.md) for full test strategy and execution details.
