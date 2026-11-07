from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/air_quality")


def inspect_csv(filename):
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not path.exists():
        print("File not found.")
        return

    df = pd.read_csv(path)

    print(f"\nRows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Memory     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("  No missing values.")
    else:
        for col, count in missing.items():
            percentage = count / len(df) * 100
            print(f"  {col}: {count:,} ({percentage:.2f}%)")

    if "Date" in df.columns:
        print("\nDate information:")

        dates = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        print(f"  Valid dates : {dates.notna().sum():,}")
        print(f"  Invalid dates: {dates.isna().sum():,}")

        if dates.notna().any():
            print(f"  Start       : {dates.min()}")
            print(f"  End         : {dates.max()}")

    if "City" in df.columns:
        print("\nCities:")
        print(f"  Number of cities: {df['City'].nunique()}")

        print("\n  Top cities:")
        print(df["City"].value_counts().head(15))

    if "StationId" in df.columns:
        print("\nStations:")
        print(f"  Number of stations: {df['StationId'].nunique()}")

    print()


files = [
    "city_day.csv",
    "city_hour.csv",
    "station_day.csv",
    "station_hour.csv",
    "stations.csv",
]

for filename in files:
    inspect_csv(filename)