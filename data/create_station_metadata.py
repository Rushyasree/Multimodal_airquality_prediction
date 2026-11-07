import pandas as pd
from pathlib import Path

# ============================================================
# CREATE STATION METADATA FOR THE 90 ACTIVE PROJECT STATIONS
# ============================================================

AIR_FILE = Path("data/processed/air_quality_hourly_clean.csv")
STATIONS_FILE = Path("data/raw/air_quality/stations.csv")
OUTPUT_FILE = Path("data/processed/station_metadata_90.csv")

print("=" * 70)
print("CREATING STATION METADATA")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load processed air-quality data
# ------------------------------------------------------------

print("\nLoading processed air-quality data...")

air = pd.read_csv(
    AIR_FILE,
    usecols=["StationId"]
)

active_stations = sorted(air["StationId"].dropna().unique())

print(f"Stations in processed dataset: {len(active_stations)}")

# ------------------------------------------------------------
# 2. Load original station metadata
# ------------------------------------------------------------

print("\nLoading stations.csv...")

stations = pd.read_csv(STATIONS_FILE)

print(f"Total stations in original metadata: {len(stations)}")

# ------------------------------------------------------------
# 3. Keep only stations used by our project
# ------------------------------------------------------------

metadata = stations[
    stations["StationId"].isin(active_stations)
].copy()

# ------------------------------------------------------------
# 4. Sort by StationId
# ------------------------------------------------------------

metadata = metadata.sort_values("StationId").reset_index(drop=True)

# ------------------------------------------------------------
# 5. Add empty coordinate columns
# ------------------------------------------------------------

metadata["Latitude"] = pd.NA
metadata["Longitude"] = pd.NA
metadata["CoordinateSource"] = pd.NA
metadata["CoordinateConfidence"] = pd.NA

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

metadata.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("STATION METADATA CREATED")
print("=" * 70)

print(f"\nStations included: {len(metadata)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nColumns:")
print(metadata.columns.tolist())

print("\nFirst 20 stations:")
print(
    metadata[
        [
            "StationId",
            "StationName",
            "City",
            "State"
        ]
    ].head(20).to_string(index=False)
)

print("\nState distribution:")
print(metadata["State"].value_counts())

print("\nNext step:")
print("Populate Latitude/Longitude only from a verified source.")