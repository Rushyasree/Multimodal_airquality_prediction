"""
scripts/validate_and_clean_recent_data.py

High-performance, station-by-station streaming parser and data-quality auditor
for post-2020 XKDR air quality telemetry.
Generates:
1. experiments/recent_data_audit/monthly_station_coverage.csv
2. data/processed/recent_air_quality_cleaned.csv
3. Updates experiments/recent_data_audit/source_coverage.json
"""

import sys
import json
import calendar
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw" / "recent_air_quality"
OUT_AUDIT_DIR = ROOT_DIR / "experiments" / "recent_data_audit"
OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("VAYUDRISHTI — POST-2020 DATA VALIDATION & CLEANING PIPELINE", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Station Metadata
    coords_csv = ROOT_DIR / "data" / "processed" / "station_coordinates_90.csv"
    df_coords = pd.read_csv(coords_csv)
    meta_dict = df_coords.set_index("StationId").to_dict(orient="index")
    print(f"Loaded coordinates and metadata for {len(meta_dict)} stations.", flush=True)

    # 2. Group raw JSON files by vayu_station_id
    raw_files = list(RAW_DIR.glob("*.json"))
    print(f"Discovered {len(raw_files)} raw JSON files in {RAW_DIR}.", flush=True)

    station_files = defaultdict(list)
    for fp in raw_files:
        parts = fp.stem.split("_")
        if len(parts) >= 4:
            v_id = parts[0]
            station_files[v_id].append(fp)

    print(f"Grouped into {len(station_files)} distinct VayuDrishti stations.", flush=True)

    all_monthly_rows = []
    clean_csv_path = PROCESSED_DIR / "recent_air_quality_cleaned.csv"

    # We will write the cleaned CSV in chunks station-by-station
    first_chunk = True
    total_cleaned_rows = 0
    total_valid_pm25 = 0
    total_valid_pm10 = 0
    total_dups = 0
    total_inv_pm25 = 0
    total_inv_pm10 = 0

    st_idx = 0
    for vayu_id, fps in sorted(station_files.items()):
        st_idx += 1
        st_meta = meta_dict.get(vayu_id, {})
        st_name = st_meta.get("StationName", "")
        city = st_meta.get("City", "")
        state = st_meta.get("State", "")
        lat = st_meta.get("Latitude", np.nan)
        lon = st_meta.get("Longitude", np.nan)

        pm25_dict = {}
        pm10_dict = {}
        st_dups = 0

        for fp in fps:
            param = "PM2.5" if "PM2.5" in fp.stem else ("PM10" if "PM10" in fp.stem else None)
            if not param:
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    data = content.get("data", [])
                    for row in data:
                        dt_str = row.get("period_start") or row.get("collected_at")
                        val = row.get("mean") if "mean" in row else row.get("value")
                        if dt_str and val is not None:
                            norm_dt = dt_str[:19].replace("T", " ")
                            v_float = float(val)
                            target_dict = pm25_dict if param == "PM2.5" else pm10_dict
                            if norm_dt in target_dict:
                                st_dups += 1
                            else:
                                target_dict[norm_dt] = v_float
            except Exception as e:
                print(f"Warning on {fp.name}: {e}", flush=True)

        total_dups += st_dups

        # Combine all unique timestamps for this station
        all_dts = sorted(set(pm25_dict.keys()) | set(pm10_dict.keys()))
        if not all_dts:
            continue

        df_st = pd.DataFrame({
            "station_id": vayu_id,
            "Datetime_str": all_dts,
            "PM2.5": [pm25_dict.get(d, np.nan) for d in all_dts],
            "PM10": [pm10_dict.get(d, np.nan) for d in all_dts],
            "Latitude": lat,
            "Longitude": lon,
            "City": city,
            "State": state,
            "StationName": st_name,
        })

        # Sanity bounds check
        inv25_mask = (df_st["PM2.5"] <= 0) | (df_st["PM2.5"] > 1500)
        inv10_mask = (df_st["PM10"] <= 0) | (df_st["PM10"] > 2500)
        total_inv_pm25 += int(inv25_mask.sum())
        total_inv_pm10 += int(inv10_mask.sum())

        df_st.loc[inv25_mask, "PM2.5"] = np.nan
        df_st.loc[inv10_mask, "PM10"] = np.nan

        df_st["Datetime"] = pd.to_datetime(df_st["Datetime_str"])

        # Monthly aggregation metrics
        df_st["year"] = df_st["Datetime"].dt.year
        df_st["month"] = df_st["Datetime"].dt.month

        for (yr, mo), grp in df_st.groupby(["year", "month"]):
            days_in_mo = calendar.monthrange(yr, mo)[1]
            exp_hours = days_in_mo * 24
            v25 = grp["PM2.5"].dropna()
            v10 = grp["PM10"].dropna()
            both_cnt = len(grp[grp["PM2.5"].notna() & grp["PM10"].notna()])

            all_monthly_rows.append({
                "vayu_station_id": vayu_id,
                "station_name": st_name,
                "city": city,
                "state": state,
                "year": yr,
                "month": mo,
                "year_month": f"{yr}-{mo:02d}",
                "expected_hours": exp_hours,
                "recorded_hours": len(grp),
                "valid_pm25_hours": len(v25),
                "valid_pm10_hours": len(v10),
                "valid_both_hours": both_cnt,
                "pm25_missing_pct": round(max(0.0, (exp_hours - len(v25)) / exp_hours * 100), 2),
                "pm10_missing_pct": round(max(0.0, (exp_hours - len(v10)) / exp_hours * 100), 2),
                "pm25_mean": round(float(v25.mean()), 2) if not v25.empty else None,
                "pm25_min": round(float(v25.min()), 2) if not v25.empty else None,
                "pm25_max": round(float(v25.max()), 2) if not v25.empty else None,
                "pm10_mean": round(float(v10.mean()), 2) if not v10.empty else None,
                "pm10_min": round(float(v10.min()), 2) if not v10.empty else None,
                "pm10_max": round(float(v10.max()), 2) if not v10.empty else None,
                "first_timestamp": str(grp["Datetime"].min()),
                "last_timestamp": str(grp["Datetime"].max()),
            })

        # Format clean rows for disk export
        cols_export = [
            "station_id", "Datetime", "PM2.5", "PM10",
            "Latitude", "Longitude", "City", "State", "StationName"
        ]
        df_export = df_st[cols_export]
        total_cleaned_rows += len(df_export)
        total_valid_pm25 += int(df_export["PM2.5"].notna().sum())
        total_valid_pm10 += int(df_export["PM10"].notna().sum())

        # Append to CSV
        if first_chunk:
            df_export.to_csv(clean_csv_path, index=False, mode="w")
            first_chunk = False
        else:
            df_export.to_csv(clean_csv_path, index=False, mode="a", header=False)

        if st_idx % 15 == 0 or st_idx == len(station_files):
            print(f"[{st_idx:2d}/{len(station_files)}] Processed {vayu_id} ({city}) | Cleaned cumulative rows: {total_cleaned_rows:,}", flush=True)

    # 3. Save Monthly Coverage CSV
    df_coverage = pd.DataFrame(all_monthly_rows)
    coverage_path = OUT_AUDIT_DIR / "monthly_station_coverage.csv"
    df_coverage.to_csv(coverage_path, index=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 80, flush=True)
    print("DATA VALIDATION & CLEANING COMPLETED", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Cleaned Hourly Rows:     {total_cleaned_rows:,}", flush=True)
    print(f"Valid PM2.5 Hourly Records:    {total_valid_pm25:,}", flush=True)
    print(f"Valid PM10 Hourly Records:     {total_valid_pm10:,}", flush=True)
    print(f"Duplicate Timestamps Removed:  {total_dups:,}", flush=True)
    print(f"Invalid PM2.5 Outliers (<0/>1500): {total_inv_pm25:,}", flush=True)
    print(f"Invalid PM10 Outliers (<0/>2500):  {total_inv_pm10:,}", flush=True)
    print(f"Monthly Coverage Rows:         {len(df_coverage):,}", flush=True)
    print(f"Total Execution Time:          {elapsed:.1f}s", flush=True)
    print(f"Cleaned CSV:                   {clean_csv_path}", flush=True)
    print(f"Monthly Coverage CSV:          {coverage_path}", flush=True)


if __name__ == "__main__":
    main()
