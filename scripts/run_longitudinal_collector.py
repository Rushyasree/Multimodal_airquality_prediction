"""
scripts/run_longitudinal_collector.py

CLI entrypoint for Phase 10 Longitudinal Live Monitoring and Dataset Preservation.
Usage:
    python scripts/run_longitudinal_collector.py              # Execute standard incremental cycle
    python scripts/run_longitudinal_collector.py --status     # Display current monitoring status and telemetry counts
    python scripts/run_longitudinal_collector.py --hours-back 72 # Query past 72 hours
"""

import sys
import argparse
import json
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.prediction.current_air_quality_service import load_env_file
from src.services.longitudinal_live_collector import LongitudinalCollector
from src.services.sensor_forecast_service import SensorForecastService


def print_status(collector: LongitudinalCollector):
    state = collector.state
    print("=" * 75)
    print("VAYUDRISHTI LONGITUDINAL MONITORING DATASET STATUS")
    print("=" * 75)
    print(f"Collector Version           : {state.get('collector_version')}")
    print(f"Trial Phase                 : {state.get('trial_phase')}")
    print(f"Total Collector Executions  : {state.get('total_runs')}")
    print(f"Last Execution Window (UTC) : {state.get('last_run_start_utc')} -> {state.get('last_run_end_utc')}")
    print(f"Last Status                 : {state.get('last_status', 'never_run')}")
    print(f"Total Observations (CSV)    : {state.get('total_observations_ingested')}")
    print(f"Total Predictions (CSV)     : {state.get('total_predictions_generated')}")
    print(f"Total Matched Pairs (CSV)   : {state.get('total_targets_matched')}")
    print(f"Unmatched / Pending Targets : {state.get('unmatched_predictions_count')}")

    if collector.obs_csv.exists() and not collector.df_obs.empty:
        min_dt = collector.df_obs["observation_time_utc"].min()
        max_dt = collector.df_obs["observation_time_utc"].max()
        print(f"Observation Calendar Window : {min_dt} to {max_dt}")

    print("\n--- Per-Station Telemetry Status ---")
    st_status = state.get("station_telemetry_status", {})
    for sid, s_dict in st_status.items():
        print(f"Station {sid}:")
        print(f"  Raw Observations : {s_dict.get('raw_observation_count')}")
        print(f"  Hourly Bins      : {s_dict.get('hourly_bins_count')}")
        print(f"  Predictions      : {s_dict.get('predictions_count')}")
        print(f"  Matched Targets  : {s_dict.get('evaluations_count')}")

    outages = state.get("api_outages_logged", [])
    print(f"\nAPI Outages Logged: {len(outages)}")
    if outages:
        for out in outages[-3:]:
            print(f"  - [{out.get('logged_at_utc')}] Station {out.get('station_id')} {out.get('parameter')}: {out.get('error')}")

    print("=" * 75)


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="VayuDrishti Longitudinal Live Monitoring Collector")
    parser.add_argument("--status", action="store_true", help="Display current collector status and dataset counts")
    parser.add_argument("--hours-back", type=int, default=48, help="Hours of historical telemetry to fetch (default: 48)")
    parser.add_argument("--sync-only", action="store_true", help="Synchronize predictions and matches from existing observations CSV without calling API")
    args = parser.parse_args()

    collector = LongitudinalCollector()

    if args.status:
        print_status(collector)
        return

    if args.sync_only:
        print("Synchronizing predictions and evaluations from local live_trial_observations.csv...")
        fc_svc = SensorForecastService()
        n_p, n_m = collector.build_provenance_forecasts_and_matches(fc_svc)
        collector.save_state()
        print(f"Sync complete: {n_p} new predictions, {n_m} new matches.")
        print_status(collector)
        return

    print("Executing incremental longitudinal live monitoring cycle...")
    fc_svc = SensorForecastService()
    res = collector.execute_collection_cycle(hours_back=args.hours_back, forecast_service=fc_svc)
    print("\nExecution Summary:")
    print(json.dumps(res, indent=2))
    print("\nUpdated Status:")
    print_status(collector)


if __name__ == "__main__":
    main()
