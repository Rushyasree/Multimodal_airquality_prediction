from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/air_quality_hourly_clean.csv"
)

OUTPUT_DIR = Path(
    "data/processed/sequences"
)


LOOKBACK_HOURS = 24


def load_data():

    print("Loading processed air-quality data...")

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
    ).reset_index(drop=True)

    return df


def calculate_split_boundaries(df):

    start = df["Datetime"].min()
    end = df["Datetime"].max()

    total_seconds = (
        end - start
    ).total_seconds()

    train_end = (
        start
        + pd.to_timedelta(
            total_seconds * 0.70,
            unit="s"
        )
    )

    validation_end = (
        start
        + pd.to_timedelta(
            total_seconds * 0.85,
            unit="s"
        )
    )

    return (
        start,
        train_end,
        validation_end,
        end,
    )


def generate_sequences_for_group(
    group,
    split_name,
    split_start,
    split_end,
):

    group = group.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    sequences = []

    timestamps = group["Datetime"]

    for i in range(
        LOOKBACK_HOURS,
        len(group)
    ):

        history_start = i - LOOKBACK_HOURS
        history_end = i

        history = group.iloc[
            history_start:history_end
        ]

        target = group.iloc[i]

        # -------------------------------------------------
        # Check that all history observations are exactly
        # one hour apart.
        # -------------------------------------------------

        diffs = (
            history["Datetime"]
            .diff()
            .dropna()
        )

        if not (
            diffs
            == pd.Timedelta(hours=1)
        ).all():

            continue

        # -------------------------------------------------
        # Check that target immediately follows history.
        # -------------------------------------------------

        last_history_time = (
            history["Datetime"].iloc[-1]
        )

        target_time = target["Datetime"]

        if (
            target_time
            - last_history_time
            != pd.Timedelta(hours=1)
        ):
            continue

        # -------------------------------------------------
        # Prevent train/validation/test leakage.
        #
        # Entire sequence and target must belong to
        # the same chronological split.
        # -------------------------------------------------

        sequence_start = (
            history["Datetime"].iloc[0]
        )

        sequence_end = target_time

        if (
            sequence_start < split_start
            or sequence_end >= split_end
        ):
            continue

        sequences.append(
            {
                "StationId":
                    target["StationId"],

                "history_start":
                    sequence_start,

                "history_end":
                    last_history_time,

                "target_time":
                    target_time,

                "target_PM2.5":
                    target["PM2.5"],

                "target_PM10":
                    target["PM10"],

                "split":
                    split_name,
            }
        )

    return sequences


def main():

    print("=" * 70)
    print("24-HOUR LSTM SEQUENCE INDEX GENERATION")
    print("=" * 70)

    df = load_data()

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Stations: "
        f"{df['StationId'].nunique()}"
    )

    (
        global_start,
        train_end,
        validation_end,
        global_end,
    ) = calculate_split_boundaries(df)

    print("\nChronological boundaries:")
    print(
        f"Start      : {global_start}"
    )
    print(
        f"Train end  : {train_end}"
    )
    print(
        f"Val end    : {validation_end}"
    )
    print(
        f"End        : {global_end}"
    )

    all_sequences = []

    # -----------------------------------------------------
    # Process station by station
    # -----------------------------------------------------

    for station_id, group in df.groupby(
        "StationId",
        sort=False
    ):

        # ---------------------------------------------
        # Training
        # ---------------------------------------------

        train_sequences = (
            generate_sequences_for_group(
                group,
                "train",
                global_start,
                train_end,
            )
        )

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        validation_sequences = (
            generate_sequences_for_group(
                group,
                "validation",
                train_end,
                validation_end,
            )
        )

        # ---------------------------------------------
        # Test
        # ---------------------------------------------

        test_sequences = (
            generate_sequences_for_group(
                group,
                "test",
                validation_end,
                global_end
                + pd.Timedelta(hours=1),
            )
        )

        all_sequences.extend(
            train_sequences
        )

        all_sequences.extend(
            validation_sequences
        )

        all_sequences.extend(
            test_sequences
        )

    sequence_df = pd.DataFrame(
        all_sequences
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "air_quality_24h_sequence_index.csv"
    )

    sequence_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("SEQUENCE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal valid sequences: "
        f"{len(sequence_df):,}"
    )

    print("\nSequences by split:")

    print(
        sequence_df["split"]
        .value_counts()
        .sort_index()
    )

    print("\nSequences by station:")

    print(
        sequence_df["StationId"]
        .value_counts()
        .head(15)
    )

    print(
        f"\nSaved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()