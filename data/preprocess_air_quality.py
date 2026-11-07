from pathlib import Path

import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw/air_quality")
PROCESSED_DIR = Path("data/processed")

INPUT_FILE = RAW_DIR / "station_hour.csv"
OUTPUT_FILE = PROCESSED_DIR / "air_quality_hourly_clean.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET_COLUMNS = [
    "PM2.5",
    "PM10",
]

POLLUTANT_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    print("Loading station_hour.csv...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Loaded rows: {len(df):,}")

    return df


# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------

def basic_cleaning(df):

    print("\nCleaning data...")

    # Convert timestamp
    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="coerce"
    )

    # Remove invalid timestamps
    invalid_dates = df["Datetime"].isna().sum()

    if invalid_dates:
        print(f"Removing {invalid_dates:,} invalid timestamps.")

    df = df.dropna(
        subset=["StationId", "Datetime"]
    )

    # Sort chronologically
    df = df.sort_values(
        ["StationId", "Datetime"]
    )

    # Remove duplicate station-time records
    before = len(df)

    df = df.drop_duplicates(
        subset=["StationId", "Datetime"],
        keep="first"
    )

    duplicates_removed = before - len(df)

    print(
        f"Duplicate station-time records removed: "
        f"{duplicates_removed:,}"
    )

    return df


# ---------------------------------------------------------
# Target filtering
# ---------------------------------------------------------

def filter_targets(df):

    print("\nFiltering target availability...")

    before = len(df)

    # We need both targets for multi-output regression
    df = df.dropna(
        subset=TARGET_COLUMNS
    )

    removed = before - len(df)

    print(
        f"Rows removed because PM2.5/PM10 "
        f"were missing: {removed:,}"
    )

    print(
        f"Remaining rows: {len(df):,}"
    )

    return df


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def add_time_features(df):

    print("\nCreating temporal features...")

    df["hour"] = df["Datetime"].dt.hour

    df["day_of_week"] = df["Datetime"].dt.dayofweek

    df["day"] = df["Datetime"].dt.day

    df["month"] = df["Datetime"].dt.month

    df["year"] = df["Datetime"].dt.year

    # Cyclic encoding
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


# ---------------------------------------------------------
# Missing-value report
# ---------------------------------------------------------

def missing_report(df):

    print("\nMissing-value report:")
    print("-" * 60)

    for column in POLLUTANT_COLUMNS:

        if column not in df.columns:
            continue

        missing = df[column].isna().sum()

        percentage = (
            missing / len(df) * 100
        )

        print(
            f"{column:8s} "
            f"{missing:10,} "
            f"({percentage:6.2f}%)"
        )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

def save_data(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved processed dataset to:"
        f"\n{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("AIR QUALITY PREPROCESSING")
    print("=" * 70)

    df = load_data()

    missing_report(df)

    df = basic_cleaning(df)

    df = filter_targets(df)

    df = add_time_features(df)

    print("\nFinal dataset:")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nStations:")
    print(
        df["StationId"].nunique()
    )

    print("\nDate range:")

    print(
        df["Datetime"].min()
    )

    print(
        df["Datetime"].max()
    )

    save_data(df)

    print("\nPREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()