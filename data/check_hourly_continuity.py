from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/air_quality_hourly_clean.csv"
)


def main():

    print("=" * 70)
    print("HOURLY CONTINUITY ANALYSIS")
    print("=" * 70)

    print("\nLoading processed dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "StationId",
            "Datetime",
            "PM2.5",
            "PM10",
        ]
    )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "StationId",
            "Datetime",
            "PM2.5",
            "PM10",
        ]
    )

    df = df.sort_values(
        ["StationId", "Datetime"]
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Stations: {df['StationId'].nunique()}"
    )

    # -----------------------------------------------------
    # Calculate time difference between consecutive rows
    # for each station
    # -----------------------------------------------------

    df["time_diff"] = (
        df.groupby("StationId")["Datetime"]
        .diff()
    )

    # Expected interval
    expected = pd.Timedelta(hours=1)

    continuous = (
        df["time_diff"] == expected
    )

    gaps = df.loc[
        df["time_diff"] > expected,
        ["StationId", "Datetime", "time_diff"]
    ].copy()

    duplicates_or_backwards = df.loc[
        df["time_diff"] <= pd.Timedelta(0),
        ["StationId", "Datetime", "time_diff"]
    ].copy()

    print("\n" + "-" * 70)
    print("CONTINUITY SUMMARY")
    print("-" * 70)

    print(
        f"Continuous 1-hour transitions : "
        f"{continuous.sum():,}"
    )

    print(
        f"Gaps greater than 1 hour       : "
        f"{len(gaps):,}"
    )

    print(
        f"Non-positive time transitions   : "
        f"{len(duplicates_or_backwards):,}"
    )

    # -----------------------------------------------------
    # Gap distribution
    # -----------------------------------------------------

    if len(gaps) > 0:

        gaps["gap_hours"] = (
            gaps["time_diff"]
            / pd.Timedelta(hours=1)
        )

        print("\nGap distribution:")

        print(
            gaps["gap_hours"]
            .value_counts()
            .sort_index()
            .head(20)
        )

        print("\nLargest gaps:")

        print(
            gaps.sort_values(
                "gap_hours",
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )

    # -----------------------------------------------------
    # Station-level statistics
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("STATION-LEVEL STATISTICS")
    print("-" * 70)

    station_stats = (
        df.groupby("StationId")
        .agg(
            records=("Datetime", "size"),
            start=("Datetime", "min"),
            end=("Datetime", "max"),
            transitions=("time_diff", "count"),
            gaps=("time_diff",
                  lambda x:
                  (x > expected).sum()),
        )
        .reset_index()
    )

    station_stats["gap_percentage"] = (
        station_stats["gaps"]
        / station_stats["transitions"]
        * 100
    )

    print(
        station_stats
        .sort_values(
            "records",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # 24-hour sequence eligibility
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("24-HOUR SEQUENCE ELIGIBILITY")
    print("-" * 70)

    sequence_length = 24

    eligible_sequences = 0

    for station_id, group in df.groupby(
        "StationId"
    ):

        times = (
            group["Datetime"]
            .sort_values()
            .reset_index(drop=True)
        )

        if len(times) < sequence_length:
            continue

        differences = (
            times.diff()
            .eq(expected)
        )

        # Find runs of continuous hourly records
        run_id = (
            (~differences)
            .cumsum()
        )

        run_lengths = (
            times.groupby(run_id)
            .size()
        )

        # Every continuous run of L hours
        # can produce L - sequence_length + 1
        # windows.

        eligible_sequences += (
            run_lengths[
                run_lengths >= sequence_length
            ]
            - sequence_length
            + 1
        ).sum()

    print(
        f"Potential 24-hour windows: "
        f"{int(eligible_sequences):,}"
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()