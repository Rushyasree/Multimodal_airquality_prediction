
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

AQ_FILE = Path("data/processed/air_quality_hourly_clean.csv")
WEATHER_DIR = Path("data/raw/era5_land/station_weather")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "air_quality_weather_merged.csv"
REPORT_FILE = OUTPUT_DIR / "air_quality_weather_merge_report.txt"

AQ_STATION_COL = "StationId"
AQ_TIME_COL = "Datetime"

WEATHER_STATION_COL = "station_id"
WEATHER_TIME_COL = "timestamp"

TARGET_COLUMNS = ["PM2.5", "PM10"]

WEATHER_FEATURES = [
    "temperature_C",
    "dewpoint_C",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "precipitation_mm",
]

# ============================================================
# LOAD AIR-QUALITY DATA
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AIR QUALITY + ERA5-LAND WEATHER MERGE")
    print("=" * 70)

    if not AQ_FILE.exists():
        raise FileNotFoundError(f"Air-quality file not found: {AQ_FILE}")

    aq = pd.read_csv(AQ_FILE, low_memory=False)

    required_aq = [
        AQ_STATION_COL,
        AQ_TIME_COL,
        *TARGET_COLUMNS,
    ]

    missing = [c for c in required_aq if c not in aq.columns]
    if missing:
        raise ValueError(f"Missing air-quality columns: {missing}")

    aq[AQ_STATION_COL] = (
        aq[AQ_STATION_COL].astype("string").str.strip().str.upper()
    )
    aq[AQ_TIME_COL] = pd.to_datetime(
        aq[AQ_TIME_COL], errors="coerce"
    )

    for col in TARGET_COLUMNS:
        aq[col] = pd.to_numeric(aq[col], errors="coerce")

    print(f"\nAir-quality rows: {len(aq):,}")
    print(f"Air-quality stations: {aq[AQ_STATION_COL].nunique()}")
    print(
        f"Air-quality date range: {aq[AQ_TIME_COL].min()} "
        f"to {aq[AQ_TIME_COL].max()}"
    )

    invalid_aq_keys = aq[
        aq[AQ_STATION_COL].isna() | aq[AQ_TIME_COL].isna()
    ]

    print(f"Rows with invalid station/time: {len(invalid_aq_keys):,}")

    aq = aq.dropna(subset=[AQ_STATION_COL, AQ_TIME_COL]).copy()

    duplicate_aq = aq.duplicated(
        [AQ_STATION_COL, AQ_TIME_COL], keep=False
    )
    print(f"Rows involved in duplicate AQ keys: {duplicate_aq.sum():,}")

    if duplicate_aq.any():
        examples = aq.loc[
            duplicate_aq, [AQ_STATION_COL, AQ_TIME_COL]
        ].head(10)
        raise ValueError(
            "Duplicate station/timestamp keys found in air-quality data. "
            "Resolve them before merging.\n"
            f"{examples.to_string(index=False)}"
        )

    # ========================================================
    # LOAD AND COMBINE WEATHER FILES
    # ========================================================

    weather_files = sorted(WEATHER_DIR.glob("*_weather.csv"))

    if not weather_files:
        raise FileNotFoundError(
            f"No weather files found in {WEATHER_DIR}"
        )

    print(f"\nWeather files found: {len(weather_files)}")

    weather_frames = []

    required_weather = [
        WEATHER_STATION_COL,
        WEATHER_TIME_COL,
        *WEATHER_FEATURES,
    ]

    for file in weather_files:
        df = pd.read_csv(file, low_memory=False)

        # Fallback to filename only if station_id is absent.
        if WEATHER_STATION_COL not in df.columns:
            station_from_filename = file.name.replace(
                "_weather.csv", ""
            )
            df[WEATHER_STATION_COL] = station_from_filename

        missing_weather = [
            c for c in required_weather if c not in df.columns
        ]

        if missing_weather:
            raise ValueError(
                f"{file.name} is missing columns: {missing_weather}"
            )

        df[WEATHER_STATION_COL] = (
            df[WEATHER_STATION_COL]
            .astype("string").str.strip().str.upper()
        )
        df[WEATHER_TIME_COL] = pd.to_datetime(
            df[WEATHER_TIME_COL], errors="coerce"
        )

        for col in WEATHER_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        weather_frames.append(df[required_weather])

    weather = pd.concat(weather_frames, ignore_index=True)

    print(f"Combined weather rows: {len(weather):,}")
    print(
        f"Weather stations: "
        f"{weather[WEATHER_STATION_COL].nunique()}"
    )

    invalid_weather_keys = weather[
        weather[WEATHER_STATION_COL].isna()
        | weather[WEATHER_TIME_COL].isna()
    ]

    if not invalid_weather_keys.empty:
        raise ValueError(
            f"Weather has {len(invalid_weather_keys)} invalid station/time keys."
        )

    duplicate_weather = weather.duplicated(
        [WEATHER_STATION_COL, WEATHER_TIME_COL], keep=False
    )

    if duplicate_weather.any():
        examples = weather.loc[
            duplicate_weather,
            [WEATHER_STATION_COL, WEATHER_TIME_COL],
        ].head(10)

        raise ValueError(
            "Duplicate station/timestamp keys found in weather data.\n"
            f"{examples.to_string(index=False)}"
        )

    # Rename weather join keys to match the air-quality dataset.
    weather = weather.rename(
        columns={
            WEATHER_STATION_COL: AQ_STATION_COL,
            WEATHER_TIME_COL: AQ_TIME_COL,
        }
    )

    # ========================================================
    # CHECK STATION COVERAGE
    # ========================================================

    aq_stations = set(aq[AQ_STATION_COL].dropna().unique())
    weather_stations = set(weather[AQ_STATION_COL].dropna().unique())

    print("\nStation coverage")
    print(f"Air-quality stations: {len(aq_stations)}")
    print(f"Weather stations: {len(weather_stations)}")
    print(f"Stations in both: {len(aq_stations & weather_stations)}")

    print(
        "Air-quality stations without weather:",
        sorted(aq_stations - weather_stations),
    )
    print(
        "Weather stations without air-quality observations:",
        sorted(weather_stations - aq_stations),
    )

    # ========================================================
    # MERGE
    # ========================================================

    merged = aq.merge(
        weather,
        on=[AQ_STATION_COL, AQ_TIME_COL],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = int((merged["_merge"] == "both").sum())
    unmatched = int((merged["_merge"] == "left_only").sum())

    print("\nMerge results")
    print(f"Air-quality rows: {len(aq):,}")
    print(f"Matched rows: {matched:,}")
    print(f"Unmatched air-quality rows: {unmatched:,}")
    print(f"Match rate: {matched / len(aq) * 100:.2f}%")

    unmatched_by_station = (
        merged.loc[merged["_merge"] == "left_only"]
        .groupby(AQ_STATION_COL)
        .size()
        .sort_values(ascending=False)
    )

    # The indicator is useful for validation, but is not a model feature.
    merged = merged.drop(columns=["_merge"])

    # ========================================================
    # MISSINGNESS AND TARGET CHECKS
    # ========================================================

    print("\nMissing weather features after merge:")
    weather_missing = merged[WEATHER_FEATURES].isna().sum()
    print(weather_missing.to_string())

    print("\nTarget missingness:")
    print(merged[TARGET_COLUMNS].isna().sum().to_string())

    finite_weather = np.isfinite(
        merged[WEATHER_FEATURES].to_numpy(dtype=float)
    ).all(axis=1)

    print(
        "\nRows with all six weather features finite: "
        f"{finite_weather.sum():,} / {len(merged):,}"
    )

    # Preserve every original air-quality row. Do not silently discard
    # unmatched rows or rows with missing targets.
    merged = merged.sort_values(
        [AQ_STATION_COL, AQ_TIME_COL]
    ).reset_index(drop=True)

    merged.to_csv(OUTPUT_FILE, index=False)

    # ========================================================
    # WRITE AUDIT REPORT
    # ========================================================

    report_lines = [
        "AIR QUALITY + ERA5-LAND MERGE REPORT",
        "=" * 55,
        f"Air-quality source: {AQ_FILE}",
        f"Weather directory: {WEATHER_DIR}",
        f"Output file: {OUTPUT_FILE}",
        f"Weather files: {len(weather_files)}",
        f"Air-quality rows: {len(aq)}",
        f"Weather rows: {len(weather)}",
        f"Output rows: {len(merged)}",
        f"Air-quality stations: {len(aq_stations)}",
        f"Weather stations: {len(weather_stations)}",
        f"Stations in both: {len(aq_stations & weather_stations)}",
        f"Matched rows: {matched}",
        f"Unmatched rows: {unmatched}",
        f"Match rate (%): {matched / len(aq) * 100:.4f}",
        "",
        "Missing weather values after merge:",
        weather_missing.to_string(),
        "",
        "Missing target values:",
        merged[TARGET_COLUMNS].isna().sum().to_string(),
        "",
        "Unmatched rows by station:",
        (
            unmatched_by_station.to_string()
            if not unmatched_by_station.empty
            else "None"
        ),
        "",
        "NOTE: Rows were preserved; no unmatched rows were silently removed.",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"\nMerged dataset saved to: {OUTPUT_FILE}")
    print(f"Audit report saved to: {REPORT_FILE}")
    print("\nMerge script completed.")


if __name__ == "__main__":
    main()
