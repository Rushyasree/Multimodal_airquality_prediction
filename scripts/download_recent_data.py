"""
scripts/download_recent_data.py

Concurrent, resilient, and resumable download pipeline for post-2020 air-quality
telemetry from the XKDR India Database across all 85 matched CPCB monitoring stations.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

RAW_DIR = ROOT_DIR / "data" / "raw" / "recent_air_quality"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DEMO_KEY = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
DEFAULT_BASE_URL = "https://airquality.xkdr.org/v1"
HISTORICAL_CUTOFF = "2020-07-01"

YEAR_BLOCKS = [
    ("2020", "2020-07-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-09-01"),
]


def fetch_chunk_with_retry(
    station_id: str,
    vayu_id: str,
    param: str,
    year_label: str,
    start_date: str,
    end_date: str,
    api_key: str,
    max_retries: int = 3,
) -> dict:
    """Fetches a single station-parameter-year chunk with disk caching and retry."""
    cache_file = RAW_DIR / f"{vayu_id}_{station_id}_{param}_{year_label}.json"

    # Resume capability: check if already downloaded
    if cache_file.exists() and cache_file.stat().st_size > 10:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return {
                    "vayu_id": vayu_id,
                    "station_id": station_id,
                    "param": param,
                    "year": year_label,
                    "status": "cached",
                    "rows": len(cached.get("data", [])),
                }
        except Exception:
            pass

    url = (
        f"{DEFAULT_BASE_URL}/measurements"
        f"?station={station_id}&parameter={param}"
        f"&start={start_date}&end={end_date}"
        f"&agg=hourly&format=json"
    )

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "VayuDrishti/2.0",
            "Accept": "application/json",
        },
    )

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                content = resp.read().decode("utf-8")
                data = json.loads(content)
                # Atomically write to cache
                tmp_file = cache_file.with_suffix(".tmp")
                with open(tmp_file, "w", encoding="utf-8") as f:
                    f.write(content)
                tmp_file.replace(cache_file)

                rows = len(data.get("data", []))
                return {
                    "vayu_id": vayu_id,
                    "station_id": station_id,
                    "param": param,
                    "year": year_label,
                    "status": "downloaded",
                    "rows": rows,
                }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(attempt * 3.0)
            elif attempt == max_retries:
                return {
                    "vayu_id": vayu_id,
                    "station_id": station_id,
                    "param": param,
                    "year": year_label,
                    "status": "http_error",
                    "code": e.code,
                }
            time.sleep(attempt * 1.5)
        except Exception as e:
            if attempt == max_retries:
                return {
                    "vayu_id": vayu_id,
                    "station_id": station_id,
                    "param": param,
                    "year": year_label,
                    "status": "error",
                    "error": str(e),
                }
            time.sleep(attempt * 1.5)

    return {
        "vayu_id": vayu_id,
        "station_id": station_id,
        "param": param,
        "year": year_label,
        "status": "failed",
    }


def main():
    print("=" * 80)
    print("VAYUDRISHTI — POST-2020 AIR QUALITY DATA INGESTION ENGINE")
    print("=" * 80)

    matched_csv = ROOT_DIR / "experiments" / "recent_data_audit" / "matched_stations.csv"
    if not matched_csv.exists():
        raise FileNotFoundError(f"Missing matched stations catalogue: {matched_csv}")

    df_matched = pd.read_csv(matched_csv)
    df_matched = df_matched[df_matched["matched"] == True].copy()
    print(f"Loaded {len(df_matched)} matched stations for recent ingestion.")

    api_key = os.environ.get("XKDR_API_KEY", DEFAULT_DEMO_KEY)

    # Build work items list
    tasks = []
    for _, row in df_matched.iterrows():
        vayu_id = row["vayu_station_id"]
        xkdr_id = row["xkdr_station_id"]
        first_seen = str(row["xkdr_first_seen"])[:10]
        last_seen = str(row["xkdr_last_seen"])[:10]

        for year_label, start_dt, end_dt in YEAR_BLOCKS:
            # Skip if this year block is entirely after last_seen or before first_seen
            if start_dt > last_seen or end_dt < first_seen:
                continue

            for param in ["PM2.5", "PM10"]:
                tasks.append((xkdr_id, vayu_id, param, year_label, start_dt, end_dt, api_key))

    print(f"Constructed {len(tasks)} station-parameter-year tasks to execute.")

    # Execute with ThreadPoolExecutor
    start_time = time.time()
    results = []
    completed = 0
    cached = 0
    downloaded = 0
    errors = 0
    total_records = 0

    max_workers = 6
    print(f"Starting execution with {max_workers} worker threads...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_chunk_with_retry, *t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1

            st = res.get("status")
            if st == "cached":
                cached += 1
                total_records += res.get("rows", 0)
            elif st == "downloaded":
                downloaded += 1
                total_records += res.get("rows", 0)
            else:
                errors += 1

            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - start_time
                pct = (completed / len(tasks)) * 100
                print(
                    f"[{completed}/{len(tasks)} - {pct:5.1f}%] "
                    f"Cached: {cached} | Downloaded: {downloaded} | Errors: {errors} | "
                    f"Total records: {total_records:,} | Elapsed: {elapsed:.1f}s"
                )

    print("\n" + "=" * 80)
    print("DOWNLOAD PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Total Tasks:          {len(tasks)}")
    print(f"Cached from previous: {cached}")
    print(f"Newly Downloaded:     {downloaded}")
    print(f"Failed / Errors:      {errors}")
    print(f"Total Valid Records:  {total_records:,}")
    print(f"Total Time Taken:     {time.time() - start_time:.1f}s")
    print(f"Raw cache directory:  {RAW_DIR}")


if __name__ == "__main__":
    main()
