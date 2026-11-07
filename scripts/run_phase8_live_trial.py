"""
scripts/run_phase8_live_trial.py

CLI runner for executing Phase 8 Real Live Shadow Trial with Standalone Adapted GBDT.
Connects to OpenAQ API v3, ingests genuine PM2.5 and PM10 observations across target CAAQMS stations,
generates causal 1-hour-ahead forecasts and persistence baselines, matches against actual observations,
and writes live trial evaluation artifacts.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import json
import logging
from src.prediction.current_air_quality_service import load_env_file
from src.services.standalone_gbdt_live_trial import StandaloneGBDTLiveTrialRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase8Runner")

def main():
    load_env_file()
    print("=" * 70)
    print("VayuDrishti Phase 8: Real Live Shadow Trial — Standalone Adapted GBDT")
    print("=" * 70)

    runner = StandaloneGBDTLiveTrialRunner()

    # Verify OpenAQ connectivity
    if not runner.openaq_service.has_valid_api_key():
        print("ERROR: OpenAQ API key is not configured or invalid.")
        sys.exit(1)

    print("OpenAQ API Key Status: Configured and active.")
    print("Enabling trial opt-in for Phase 8 execution...")
    runner.enable_trial()

    print("\nFetching real particulate telemetry (past 72h) & running rolling evaluation...")
    metrics = runner.execute_live_trial(hours_back=72)

    print("\n" + "=" * 70)
    print("LIVE SHADOW TRIAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Observations Ingested: {metrics.get('total_observations_ingested')}")
    print(f"Total Matched Forecast-Target Pairs: {metrics.get('total_matched_pairs')}")

    overall = metrics.get("overall", {})
    if overall:
        gbdt = overall.get("gbdt", {})
        pers = overall.get("persistence", {})
        comp = overall.get("comparison", {})

        print("\n--- Model Performance Comparison (MAE / RMSE / Bias) ---")
        print(f"PM2.5 Standalone GBDT : MAE = {gbdt.get('pm25', {}).get('mae')} µg/m³, RMSE = {gbdt.get('pm25', {}).get('rmse')} µg/m³, Bias = {gbdt.get('pm25', {}).get('mean_bias')} µg/m³")
        print(f"PM2.5 Persistence     : MAE = {pers.get('pm25', {}).get('mae')} µg/m³, RMSE = {pers.get('pm25', {}).get('rmse')} µg/m³, Bias = {pers.get('pm25', {}).get('mean_bias')} µg/m³")
        print(f"PM2.5 Improvement     : {comp.get('pm25_mae_improvement_pct')}% (Wins: GBDT {comp.get('gbdt_wins_pm25')} vs Pers {comp.get('pers_wins_pm25')})")

        print(f"\nPM10 Standalone GBDT  : MAE = {gbdt.get('pm10', {}).get('mae')} µg/m³, RMSE = {gbdt.get('pm10', {}).get('rmse')} µg/m³, Bias = {gbdt.get('pm10', {}).get('mean_bias')} µg/m³")
        print(f"PM10 Persistence      : MAE = {pers.get('pm10', {}).get('mae')} µg/m³, RMSE = {pers.get('pm10', {}).get('rmse')} µg/m³, Bias = {pers.get('pm10', {}).get('mean_bias')} µg/m³")
        print(f"PM10 Improvement      : {comp.get('pm10_mae_improvement_pct')}% (Wins: GBDT {comp.get('gbdt_wins_pm10')} vs Pers {comp.get('pers_wins_pm10')})")

        print("\n--- Per-Station Matched Breakdown ---")
        for sid, s_m in metrics.get("per_station", {}).items():
            print(f"Station {sid} (n={s_m.get('matched_count')} pairs):")
            print(f"  PM2.5 MAE: GBDT = {s_m.get('gbdt_pm25_mae')} vs Pers = {s_m.get('pers_pm25_mae')}")
            print(f"  PM10  MAE: GBDT = {s_m.get('gbdt_pm10_mae')} vs Pers = {s_m.get('pers_pm10_mae')}")

    print("\nArtifacts saved:")
    print(f"  - Observations CSV : {runner.observations_csv}")
    print(f"  - Predictions CSV  : {runner.predictions_csv}")
    print(f"  - Evaluation CSV   : {runner.evaluation_csv}")
    print(f"  - Metrics JSON     : {runner.metrics_json}")
    print("=" * 70)

if __name__ == "__main__":
    main()
