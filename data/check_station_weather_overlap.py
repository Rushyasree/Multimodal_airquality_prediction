
from pathlib import Path
import pandas as pd

AQ_PATH = Path("data/processed/air_quality_hourly_clean.csv")
WEATHER_DIR = Path("data/raw/era5_land/station_weather")

aq = pd.read_csv(AQ_PATH, usecols=["StationId", "Datetime"])
aq["StationId"] = aq["StationId"].astype(str).str.strip()
aq["Datetime"] = pd.to_datetime(aq["Datetime"], errors="coerce")
aq = aq.dropna(subset=["StationId", "Datetime"])

required_weather = [
    "Datetime", "t2m", "d2m", "u10", "v10", "tp",
    "temperature_C", "dewpoint_C", "relative_humidity",
    "wind_speed", "wind_direction", "precipitation_mm"
]

print("=" * 88)
print("STATION-BY-STATION WEATHER ALIGNMENT AUDIT")
print("=" * 88)

results = []

for station, group in aq.groupby("StationId", sort=True):
    path = WEATHER_DIR / f"{station}.csv"

    if not path.exists():
        results.append({
            "StationId": station,
            "AQ_rows": len(group),
            "Matched_rows": 0,
            "Match_pct": 0.0,
            "Status": "WEATHER FILE MISSING"
        })
        continue

    weather = pd.read_csv(path)

    missing_cols = set(required_weather) - set(weather.columns)
    if missing_cols:
        results.append({
            "StationId": station,
            "AQ_rows": len(group),
            "Matched_rows": 0,
            "Match_pct": 0.0,
            "Status": f"MISSING COLUMNS: {sorted(missing_cols)}"
        })
        continue

    weather["Datetime"] = pd.to_datetime(
        weather["Datetime"], errors="coerce"
    )
    weather = weather.dropna(subset=["Datetime"])

    # Compare distinct station-hour keys, avoiding duplicate inflation.
    aq_keys = group[["StationId", "Datetime"]].drop_duplicates()
    weather_keys = weather[["Datetime"]].drop_duplicates()

    matched = aq_keys["Datetime"].isin(weather_keys["Datetime"]).sum()
    match_pct = 100 * matched / len(aq_keys) if len(aq_keys) else 0

    numeric_cols = [
        "t2m", "d2m", "u10", "v10", "tp",
        "temperature_C", "dewpoint_C", "relative_humidity",
        "wind_speed", "wind_direction", "precipitation_mm"
    ]
    missing_weather_values = int(
        weather[numeric_cols].isna().sum().sum()
    )

    status = "PASS" if matched == len(aq_keys) and missing_weather_values == 0 else "CHECK"

    results.append({
        "StationId": station,
        "AQ_rows": len(group),
        "Matched_rows": int(matched),
        "Match_pct": round(match_pct, 3),
        "Missing_weather_values": missing_weather_values,
        "Status": status
    })

report = pd.DataFrame(results)
print(report.to_string(index=False))

print("\nSUMMARY")
print("Air-quality stations:", len(report))
print("Stations passing exact-time/value checks:",
      int((report["Status"] == "PASS").sum()))
print("Stations needing review:",
      int((report["Status"] != "PASS").sum()))
print("Air-quality rows:", len(aq))
print("Matched station-hour keys:", int(report["Matched_rows"].sum()))

output = Path("data/processed/station_weather_overlap_audit.csv")
report.to_csv(output, index=False)
print("\nAudit report saved to:", output)
print("No source data was modified.")
