"""
scripts/run_autonomous_collector.py

CLI entrypoint for Phase 12 Autonomous Live Air Quality Collector.
Usage:
    python scripts/run_autonomous_collector.py                     # Run single live collection cycle
    python scripts/run_autonomous_collector.py --status            # Display collector status and health
    python scripts/run_autonomous_collector.py --dry-run           # Test ingestion without writing files
    python scripts/run_autonomous_collector.py --hours-back 12     # Fetch trailing 12 hours
    python scripts/run_autonomous_collector.py --sync-from-phase10 # Bootstrap from Phase 10 dataset
    python scripts/run_autonomous_collector.py --scheduler-info   # Print OS scheduler command
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.services.autonomous_live_collector import (
    AutonomousLiveCollector,
    generate_scheduler_script,
    PHASE12_DIR,
)
from src.services.sensor_forecast_service import SensorForecastService


def print_status(collector: AutonomousLiveCollector):
    state = collector.state
    config = collector.config
    print("=" * 78)
    print("VAYUDRISHTI PHASE 12: AUTONOMOUS LIVE MONITORING COLLECTOR STATUS")
    print("=" * 78)
    print(f"Collector Version          : {config.get('collector_version')}")
    print(f"Trial Phase                : {state.get('trial_phase')}")
    print(f"Operational Mode           : {config.get('operational_mode')}")
    print(f"Collector Enabled          : {config.get('collector_enabled')}")
    print(f"Kill Switch Active         : {collector.is_kill_switch_active()}")
    print(f"Model Retraining Lock      : {'PROTECTED (Retraining Blocked)' if config.get('prevent_model_retraining') else 'UNPROTECTED'}")
    print(f"Total Cycles Completed     : {state.get('total_execution_cycles', 0)}")
    print(f"Last Successful Cycle (UTC): {state.get('last_successful_run_utc', 'Never')}")
    print(f"Last Cycle Duration        : {state.get('last_run_duration_seconds', 0.0)} seconds")
    print(f"Total Raw Observations     : {len(collector.df_obs):,}")
    print(f"Total Causal Predictions   : {len(collector.df_preds):,}")
    print(f"Total Matched Targets      : {len(collector.df_eval):,}")
    print(f"Unmatched Predictions      : {state.get('unmatched_predictions_count', 0)}")
    print(f"Monitored CAAQMS Stations  : {len(collector.station_registry)}")

    print("\n--- Monitored Station Cohorts ---")
    if not collector.station_registry.empty:
        cohorts = collector.station_registry.groupby("cohort")
        for cohort_name, grp in cohorts:
            print(f"Cohort '{cohort_name}' ({len(grp)} stations):")
            for _, r in grp.iterrows():
                print(f"  [{r['station_id']}] {r['station_name']} ({r['city']}, {r['state']}) - {r['data_quality_status']}")

    print("\n--- Storage Locations ---")
    print(f"Observations CSV  : {collector.obs_csv}")
    print(f"Predictions CSV   : {collector.preds_csv}")
    print(f"Evaluations CSV   : {collector.eval_csv}")
    print(f"Health Log CSV    : {collector.health_csv}")
    print(f"State JSON        : {collector.state_json}")
    print("=" * 78)


def bootstrap_from_phase10(collector: AutonomousLiveCollector):
    """Safely synchronizes Phase 10 observations, predictions, and evaluations into Phase 12."""
    p10_obs = ROOT_DIR / "experiments/recent_data_audit/live_trial_observations.csv"
    p10_preds = ROOT_DIR / "experiments/recent_data_audit/live_trial_predictions.csv"
    p10_eval = ROOT_DIR / "experiments/recent_data_audit/live_trial_evaluation.csv"

    print("Bootstrapping Phase 12 storage from verified Phase 10 dataset...")
    if p10_obs.exists():
        df_o = pd.read_csv(p10_obs)
        recs = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for _, r in df_o.iterrows():
            recs.append({
                "station_id": r["station_id"],
                "parameter": r["parameter"],
                "value": float(r["value"]),
                "unit": "µg/m³",
                "observation_time_utc": r["observation_time_utc"],
                "collection_time_utc": now_iso,
                "observation_key": f"{r['station_id']}::{r['parameter']}::{r['observation_time_utc']}",
                "provenance_provider": "OpenAQ API v3 (Phase 10 Historical Cohort)",
                "provenance_sensor_id": 0,
            })
        n_obs = collector.append_observations(recs)
        print(f"Imported {n_obs} observations from Phase 10.")

    fc_svc = SensorForecastService()
    n_p, n_m = collector.generate_forecasts_and_matches(fc_svc)
    collector.save_state()
    print(f"Generated {n_p} predictions and matched {n_m} targets.")
    print_status(collector)


def main():
    parser = argparse.ArgumentParser(description="VayuDrishti Phase 12 Autonomous Live Collector CLI")
    parser.add_argument("--status", action="store_true", help="Display current collector status and telemetry counts")
    parser.add_argument("--dry-run", action="store_true", help="Execute trial fetch without persisting records to disk")
    parser.add_argument("--hours-back", type=int, default=6, help="Hours of trailing telemetry to query (default: 6)")
    parser.add_argument("--sync-from-phase10", action="store_true", help="Bootstrap from Phase 10 dataset")
    parser.add_argument("--scheduler-info", action="store_true", help="Display exact host OS scheduling commands")
    args = parser.parse_args()

    collector = AutonomousLiveCollector()

    if args.status:
        print_status(collector)
        return

    if args.scheduler_info:
        print("=" * 78)
        print("HOST OPERATING SYSTEM SCHEDULER INTEGRATION INSTRUCTIONS")
        print("=" * 78)
        print("Windows Task Scheduler registration command (Administrator / Command Prompt):")
        print(generate_scheduler_script(platform="windows"))
        print("\nLinux / POSIX crontab entry (crontab -e):")
        print(generate_scheduler_script(platform="linux"))
        print("=" * 78)
        return

    if args.sync_from_phase10:
        import pandas as pd
        from datetime import datetime, timezone
        bootstrap_from_phase10(collector)
        return

    # Normal execution cycle
    fc_svc = SensorForecastService()
    res = collector.execute_collection_cycle(hours_back=args.hours_back, dry_run=args.dry_run, forecast_service=fc_svc)
    print("\nExecution Cycle Result:")
    print(json.dumps(res, indent=2))
    print("\nUpdated Status:")
    print_status(collector)


if __name__ == "__main__":
    main()
