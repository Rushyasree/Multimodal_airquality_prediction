
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/air_quality_weather_merged.csv")
SEQUENCE_FILE = Path(
    "data/processed/sequences/air_quality_24h_sequence_index.csv"
)
COORDINATE_FILE = Path("data/processed/station_coordinates_90.csv")

OUTPUT_DIR = Path("data/processed/model_sequences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 24
HOUR_NS = 3_600_000_000_000  # One hour in nanoseconds

POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

WEATHER_BASE_FEATURES = [
    "temperature_C",
    "dewpoint_C",
    "relative_humidity",
    "wind_speed",
    "precipitation_mm",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
]

TARGETS = ["target_PM2.5", "target_PM10"]

SPLITS = ["train", "validation", "test"]


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def main():
    print("=" * 72)
    print("PREPARING MODEL SEQUENCES")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. LOAD INPUT FILES
    # --------------------------------------------------------

    for path in [DATA_FILE, SEQUENCE_FILE, COORDINATE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    seq = pd.read_csv(SEQUENCE_FILE, low_memory=False)
    coords = pd.read_csv(COORDINATE_FILE, low_memory=False)

    require_columns(
        df,
        ["StationId", "Datetime", *POLLUTANT_FEATURES,
         *WEATHER_BASE_FEATURES, "wind_direction",
         "hour_sin", "hour_cos", "month_sin", "month_cos"],
        "Merged dataset",
    )
    require_columns(
        seq,
        ["StationId", "history_start", "history_end",
         "target_time", *TARGETS, "split"],
        "Sequence index",
    )
    require_columns(
        coords,
        ["StationId", "Latitude", "Longitude"],
        "Station coordinates",
    )

    # Normalize identifiers and timestamps.
    for frame in [df, seq, coords]:
        frame["StationId"] = (
            frame["StationId"].astype("string").str.strip().str.upper()
        )

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    for col in ["history_start", "history_end", "target_time"]:
        seq[col] = pd.to_datetime(seq[col], errors="coerce")

    for col in ["Latitude", "Longitude"]:
        coords[col] = pd.to_numeric(coords[col], errors="coerce")

    if df["Datetime"].isna().any():
        raise ValueError("Merged data contains invalid timestamps.")

    if seq[["history_start", "history_end", "target_time"]].isna().any().any():
        raise ValueError("Sequence index contains invalid timestamps.")

    if coords["StationId"].duplicated().any():
        raise ValueError("Duplicate station IDs in coordinate metadata.")

    if coords[["Latitude", "Longitude"]].isna().any().any():
        raise ValueError("Missing station coordinates.")

    print(f"Merged rows: {len(df):,}")
    print(f"Sequence rows: {len(seq):,}")
    print(f"Coordinate stations: {coords['StationId'].nunique()}")

    # --------------------------------------------------------
    # 2. CREATE FEATURES
    # --------------------------------------------------------

    # Wind direction is circular: 359 degrees is close to 0 degrees.
    wind_deg = pd.to_numeric(df["wind_direction"], errors="coerce")
    df["wind_direction_sin"] = np.sin(np.deg2rad(wind_deg))
    df["wind_direction_cos"] = np.cos(np.deg2rad(wind_deg))

    # Cyclical day-of-week features.
    dow = pd.to_numeric(df["day_of_week"], errors="coerce")
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)

    feature_names = (
        POLLUTANT_FEATURES
        + WEATHER_BASE_FEATURES
        + ["wind_direction_sin", "wind_direction_cos"]
        + TIME_FEATURES
        + ["Latitude", "Longitude"]
    )

    # Join the original station coordinates, not the adjusted ERA5
    # extraction coordinates, as the station's spatial features.
    df = df.merge(
        coords[["StationId", "Latitude", "Longitude"]],
        on="StationId",
        how="left",
        validate="many_to_one",
    )

    if df[["Latitude", "Longitude"]].isna().any().any():
        missing_stations = sorted(
            df.loc[df["Latitude"].isna(), "StationId"].unique().tolist()
        )
        raise ValueError(
            f"Stations lack coordinate matches: {missing_stations}"
        )

    # Convert all features to numeric and replace infinities with NaN.
    for col in feature_names + ["PM2.5", "PM10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_names] = df[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )

    # Target labels must never be imputed.
    if df[["PM2.5", "PM10"]].isna().any().any():
        raise ValueError("PM2.5/PM10 contain missing values.")

    # --------------------------------------------------------
    # 3. VALIDATE THE EXISTING CHRONOLOGICAL SPLITS
    # --------------------------------------------------------

    seq["split"] = seq["split"].astype(str).str.strip().str.lower()

    unknown_splits = set(seq["split"].unique()) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"Unexpected split labels: {unknown_splits}")

    if seq.duplicated(["StationId", "target_time"]).any():
        raise ValueError("Duplicate station/target-time sequence entries.")

    if not (
        seq["history_end"] + pd.Timedelta(hours=1)
        == seq["target_time"]
    ).all():
        raise ValueError("Some targets are not exactly one hour after history.")

    if not (
        seq["history_end"] - seq["history_start"]
        == pd.Timedelta(hours=WINDOW_SIZE - 1)
    ).all():
        raise ValueError("Some sequence windows have incorrect time bounds.")

    split_counts = seq["split"].value_counts().to_dict()
    print("\nExisting split counts:")
    for split in SPLITS:
        print(f"  {split}: {split_counts.get(split, 0):,}")

    # --------------------------------------------------------
    # 4. FIT IMPUTATION AND SCALING ON TRAINING PERIOD ONLY
    # --------------------------------------------------------

    train_target_end = seq.loc[
        seq["split"] == "train", "target_time"
    ].max()

    if pd.isna(train_target_end):
        raise ValueError("No training sequences found.")

    # Use unique source rows up to the last training target time.
    # No validation/test period is used to fit preprocessing.
    train_rows = df.loc[df["Datetime"] <= train_target_end, feature_names]

    if train_rows.empty:
        raise ValueError("No training-period feature rows found.")

    medians = train_rows.median(axis=0).fillna(0.0)

    # Impute missing feature values with training-period medians.
    df[feature_names] = df[feature_names].fillna(medians)

    # Fail instead of silently retaining invalid features.
    feature_matrix = df[feature_names].to_numpy(dtype=np.float64)

    if not np.isfinite(feature_matrix).all():
        raise ValueError("Non-finite feature values remain after imputation.")

    train_matrix = df.loc[
        df["Datetime"] <= train_target_end, feature_names
    ].to_numpy(dtype=np.float64)

    means = train_matrix.mean(axis=0)
    stds = train_matrix.std(axis=0)
    stds[stds < 1e-8] = 1.0

    print(f"\nTraining preprocessing cutoff: {train_target_end}")
    print(f"Number of input features: {len(feature_names)}")
    print(f"Window length: {WINDOW_SIZE} hours")

    # --------------------------------------------------------
    # 5. BUILD PER-STATION ARRAYS
    # --------------------------------------------------------

    station_arrays = {}

    for station_id, group in df.groupby("StationId", sort=False):
        group = group.sort_values("Datetime")

        times = group["Datetime"].to_numpy(dtype="datetime64[ns]").astype(
            np.int64
        )
        x = group[feature_names].to_numpy(dtype=np.float32)
        y = group[["PM2.5", "PM10"]].to_numpy(dtype=np.float32)

        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(
                f"{station_id}: timestamps are duplicated or unordered."
            )

        station_arrays[str(station_id)] = (times, x, y)

    # --------------------------------------------------------
    # 6. ALLOCATE OUTPUT ARRAYS
    # --------------------------------------------------------

    output_arrays = {}
    positions = {}

    for split in SPLITS:
        n = int((seq["split"] == split).sum())

        if n == 0:
            continue

        x_path = OUTPUT_DIR / f"X_{split}.npy"
        y_path = OUTPUT_DIR / f"y_{split}.npy"

        output_arrays[split] = (
            np.lib.format.open_memmap(
                x_path,
                mode="w+",
                dtype=np.float32,
                shape=(n, WINDOW_SIZE, len(feature_names)),
            ),
            np.lib.format.open_memmap(
                y_path,
                mode="w+",
                dtype=np.float32,
                shape=(n, 2),
            ),
        )
        positions[split] = 0

    # --------------------------------------------------------
    # 7. EXTRACT WINDOWS WITHOUT CROSSING GAPS
    # --------------------------------------------------------

    seq = seq.sort_values(
        ["StationId", "target_time"]
    ).reset_index(drop=True)

    for i, row in enumerate(seq.itertuples(index=False), start=1):
        station_id = str(row.StationId)
        split = row.split

        if station_id not in station_arrays:
            raise ValueError(f"No merged observations for {station_id}")

        times, x_station, y_station = station_arrays[station_id]

        start_ns = pd.Timestamp(row.history_start).value
        end_ns = pd.Timestamp(row.history_end).value
        target_ns = pd.Timestamp(row.target_time).value

        start_idx = np.searchsorted(times, start_ns)
        end_idx = start_idx + WINDOW_SIZE
        target_idx = np.searchsorted(times, target_ns)

        if end_idx > len(times):
            raise ValueError(f"{station_id}: incomplete history window.")

        window_times = times[start_idx:end_idx]

        if len(window_times) != WINDOW_SIZE:
            raise ValueError(f"{station_id}: incorrect history length.")

        if window_times[0] != start_ns or window_times[-1] != end_ns:
            raise ValueError(
                f"{station_id}: history boundaries do not match source data."
            )

        if np.any(np.diff(window_times) != HOUR_NS):
            raise ValueError(
                f"{station_id}: history window crosses an hourly gap."
            )

        if target_idx >= len(times) or times[target_idx] != target_ns:
            raise ValueError(
                f"{station_id}: target timestamp missing from merged data."
            )

        # Verify the sequence index labels agree with the source dataset.
        actual_target = y_station[target_idx]
        expected_target = np.array(
    [row[seq.columns.get_loc("target_PM2.5")],
     row[seq.columns.get_loc("target_PM10")]],
    dtype=np.float32
)

        if not np.allclose(
            actual_target, expected_target, rtol=1e-5, atol=1e-4
        ):
            raise ValueError(
                f"{station_id}: target labels disagree with merged data "
                f"at {row.target_time}."
            )

        X_out, y_out = output_arrays[split]
        pos = positions[split]

        window = x_station[start_idx:end_idx].astype(np.float64)
        window = (window - means) / stds

        X_out[pos] = window.astype(np.float32)
        y_out[pos] = actual_target
        positions[split] += 1

        if i % 100_000 == 0:
            print(f"Processed {i:,} / {len(seq):,} sequences")

    # --------------------------------------------------------
    # 8. FLUSH AND VERIFY OUTPUTS
    # --------------------------------------------------------

    shapes = {}

    for split, (X_out, y_out) in output_arrays.items():
        X_out.flush()
        y_out.flush()

        expected_count = split_counts[split]

        if positions[split] != expected_count:
            raise ValueError(
                f"{split}: wrote {positions[split]} rows, "
                f"expected {expected_count}."
            )

        if not np.isfinite(X_out).all():
            raise ValueError(f"{split}: non-finite values in X array.")

        if not np.isfinite(y_out).all():
            raise ValueError(f"{split}: non-finite values in y array.")

        shapes[split] = {
            "X_shape": list(X_out.shape),
            "y_shape": list(y_out.shape),
        }

        print(f"\n{split.upper()}")
        print(f"  X shape: {X_out.shape}")
        print(f"  y shape: {y_out.shape}")
        print(f"  Input finite: {np.isfinite(X_out).all()}")
        print(f"  Target finite: {np.isfinite(y_out).all()}")

        del X_out, y_out

    # Store preprocessing parameters for reproducibility/inference.
    metadata = {
        "window_size_hours": WINDOW_SIZE,
        "forecast_horizon_hours": 1,
        "feature_names": feature_names,
        "target_names": ["PM2.5", "PM10"],
        "split_counts": {
            split: int(split_counts.get(split, 0))
            for split in SPLITS
        },
        "array_shapes": shapes,
        "training_preprocessing_cutoff": str(train_target_end),
        "imputation_medians": {
            k: float(v) for k, v in medians.items()
        },
        "scaler_means": {
            k: float(v) for k, v in zip(feature_names, means)
        },
        "scaler_stds": {
            k: float(v) for k, v in zip(feature_names, stds)
        },
        "coordinate_source": str(COORDINATE_FILE),
        "data_source": str(DATA_FILE),
        "sequence_index_source": str(SEQUENCE_FILE),
    }

    with open(OUTPUT_DIR / "preprocessing_metadata.json", "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 72)
    print("SUCCESS: MODEL SEQUENCES CREATED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Input arrays: X_train.npy, X_validation.npy, X_test.npy")
    print("Target arrays: y_train.npy, y_validation.npy, y_test.npy")
    print("Metadata: preprocessing_metadata.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
