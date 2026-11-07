import pandas as pd
from pathlib import Path

# ============================================================
# MATCH PROJECT STATIONS WITH VERIFIED COORDINATES
# ============================================================
#
# Input:
#   data/processed/station_metadata_90.csv
#   data/raw/station_coordinates/station_data.csv
#
# Output:
#   data/processed/station_coordinates_90.csv
#   data/processed/stations_unmatched.csv
#
# Special verified coordinate:
#   AS001 - Railway Colony, Guwahati - APCB
#   Latitude  : 26.181742
#   Longitude : 91.780630
#
# ============================================================


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

PROJECT_FILE = Path(
    "data/processed/station_metadata_90.csv"
)

COORD_FILE = Path(
    "data/raw/station_coordinates/station_data.csv"
)

OUTPUT_FILE = Path(
    "data/processed/station_coordinates_90.csv"
)

UNMATCHED_FILE = Path(
    "data/processed/stations_unmatched.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MATCHING STATION COORDINATES")
print("=" * 70)


# ============================================================
# 1. LOAD PROJECT STATION METADATA
# ============================================================

print("\nLoading project station metadata...")

project = pd.read_csv(PROJECT_FILE)

print(f"Project stations: {len(project)}")


# ------------------------------------------------------------
# Remove any old coordinate columns
# ------------------------------------------------------------

project = project.drop(
    columns=[
        "Latitude",
        "Longitude",
        "CoordinateSource",
        "CoordinateConfidence"
    ],
    errors="ignore"
)


# ============================================================
# 2. LOAD COORDINATE REFERENCE
# ============================================================

print("\nLoading coordinate reference...")

coords = pd.read_csv(COORD_FILE)

print(f"Coordinate records: {len(coords)}")

print("\nCoordinate columns:")
print(coords.columns.tolist())


# ============================================================
# 3. CLEAN STATION NAMES
# ============================================================

project["StationName_clean"] = (
    project["StationName"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

coords["station_clean"] = (
    coords["station"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)


# ============================================================
# 4. KEEP ONLY RECORDS WITH VALID COORDINATES
# ============================================================

coords = coords.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
).copy()


# ------------------------------------------------------------
# Remove duplicate station names
# ------------------------------------------------------------

coords = coords.drop_duplicates(
    subset=["station_clean"],
    keep="first"
)


# ============================================================
# 5. MATCH BY EXACT STATION NAME
# ============================================================

matched = project.merge(
    coords[
        [
            "station_clean",
            "latitude",
            "longitude"
        ]
    ],
    left_on="StationName_clean",
    right_on="station_clean",
    how="left"
)


# ============================================================
# 6. RENAME COORDINATE COLUMNS
# ============================================================

matched = matched.rename(
    columns={
        "latitude": "Latitude",
        "longitude": "Longitude"
    }
)


# ============================================================
# 7. SPECIAL CASE: AS001
# ============================================================
#
# Station:
#   AS001
#   Railway Colony, Guwahati - APCB
#
# Verified research-source coordinates:
#   Latitude  = 26.181742
#   Longitude = 91.780630
#
# ============================================================

as001_mask = (
    matched["StationId"] == "AS001"
)

matched.loc[
    as001_mask,
    "Latitude"
] = 26.181742

matched.loc[
    as001_mask,
    "Longitude"
] = 91.780630


# ============================================================
# 8. ADD COORDINATE SOURCE
# ============================================================

matched["CoordinateSource"] = (
    "Zeel/C1 station_data.csv"
)


# ------------------------------------------------------------
# Override source for AS001
# ------------------------------------------------------------

matched.loc[
    as001_mask,
    "CoordinateSource"
] = (
    "ACP 2025 supplementary station metadata "
    "(AS001 Railway Colony, Guwahati)"
)


# ============================================================
# 9. ADD COORDINATE CONFIDENCE
# ============================================================

matched["CoordinateConfidence"] = "unmatched"


# ------------------------------------------------------------
# Normal exact station-name matches
# ------------------------------------------------------------

matched.loc[
    matched["Latitude"].notna()
    & matched["Longitude"].notna(),
    "CoordinateConfidence"
] = "verified_name_match"


# ------------------------------------------------------------
# AS001 research-source match
# ------------------------------------------------------------

matched.loc[
    as001_mask,
    "CoordinateConfidence"
] = "verified_research_source"


# ============================================================
# 10. REMOVE TEMPORARY COLUMNS
# ============================================================

matched = matched.drop(
    columns=[
        "StationName_clean",
        "station_clean"
    ],
    errors="ignore"
)


# ============================================================
# 11. FIND UNMATCHED STATIONS
# ============================================================

unmatched = matched[
    matched["Latitude"].isna()
    | matched["Longitude"].isna()
].copy()


# ============================================================
# 12. SAVE COMPLETE COORDINATE FILE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

matched.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. SAVE UNMATCHED STATIONS
# ============================================================

unmatched.to_csv(
    UNMATCHED_FILE,
    index=False
)


# ============================================================
# 14. CALCULATE MATCH STATISTICS
# ============================================================

total = len(matched)

matched_count = (
    total - len(unmatched)
)

match_rate = (
    matched_count / total * 100
)


# ============================================================
# 15. PRINT MATCHING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING COMPLETE")
print("=" * 70)

print(
    f"\nTotal project stations : {total}"
)

print(
    f"Matched stations       : {matched_count}"
)

print(
    f"Unmatched stations     : {len(unmatched)}"
)

print(
    f"Match rate             : {match_rate:.2f}%"
)

print("\nSaved complete file:")
print(OUTPUT_FILE)

print("\nSaved unmatched stations:")
print(UNMATCHED_FILE)


# ============================================================
# 16. DISPLAY FIRST 20 STATIONS
# ============================================================

print("\nFirst 20 stations:")

print(
    matched[
        [
            "StationId",
            "StationName",
            "City",
            "State",
            "Latitude",
            "Longitude",
            "CoordinateSource",
            "CoordinateConfidence"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 17. DISPLAY UNMATCHED STATIONS
# ============================================================

if len(unmatched) > 0:

    print("\n" + "=" * 70)
    print("UNMATCHED STATIONS")
    print("=" * 70)

    print(
        unmatched[
            [
                "StationId",
                "StationName",
                "City",
                "State"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\n🎉 ALL 90 STATIONS MATCHED!"
    )


# ============================================================
# 18. VALIDATE LATITUDE / LONGITUDE RANGES
# ============================================================

valid = matched.dropna(
    subset=[
        "Latitude",
        "Longitude"
    ]
).copy()


invalid = valid[
    (valid["Latitude"] < -90)
    | (valid["Latitude"] > 90)
    | (valid["Longitude"] < -180)
    | (valid["Longitude"] > 180)
]


print("\n" + "=" * 70)
print("COORDINATE VALIDATION")
print("=" * 70)

print(
    f"\nValid coordinate rows   : {len(valid)}"
)

print(
    f"Invalid coordinate rows : {len(invalid)}"
)


if len(invalid) > 0:

    print(
        "\nWARNING: Invalid coordinates found:"
    )

    print(
        invalid[
            [
                "StationId",
                "StationName",
                "Latitude",
                "Longitude"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "All matched coordinates are within valid ranges."
    )


# ============================================================
# 19. CHECK DUPLICATE COORDINATES
# ============================================================

duplicate_coordinates = (
    valid
    .duplicated(
        subset=[
            "Latitude",
            "Longitude"
        ],
        keep=False
    )
)

duplicate_count = duplicate_coordinates.sum()


print("\n" + "=" * 70)
print("DUPLICATE COORDINATE CHECK")
print("=" * 70)

print(
    f"\nStations sharing identical coordinates: "
    f"{duplicate_count}"
)


if duplicate_count > 0:

    print(
        "\nPotential duplicate locations:"
    )

    print(
        valid.loc[
            duplicate_coordinates,
            [
                "StationId",
                "StationName",
                "Latitude",
                "Longitude"
            ]
        ].sort_values(
            [
                "Latitude",
                "Longitude"
            ]
        ).to_string(index=False)
    )

else:

    print(
        "No duplicate coordinates found."
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"""
Project stations       : {total}
Stations with coords   : {len(valid)}
Stations without coords: {len(unmatched)}
Match rate             : {match_rate:.2f}%
Invalid coordinates    : {len(invalid)}
Duplicate coordinates  : {duplicate_count}

Output:
{OUTPUT_FILE}
"""
)

if (
    len(unmatched) == 0
    and len(invalid) == 0
):

    print(
        "✅ STATION COORDINATE LAYER READY"
    )

else:

    print(
        "⚠️ Coordinate layer requires additional review."
    )

print("=" * 70)