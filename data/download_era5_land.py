
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import zipfile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COORDINATES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "station_coordinates_90.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "era5_land"
    / "station_weather"
)

DATASET = "reanalysis-era5-land-timeseries"

START_DATE = "2015-01-01"
END_DATE = "2020-07-01"

# First validate AP005. Change to False only after the
# affected stations have been individually checked.

TEST_MODE = False

TEST_STATIONS = [
    "AP005",
    "MH005",
    "MH008",
    "MH013",
    "MH014",
    "TN002",
]


MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 5

# AP005's original ERA5 grid point (17.7, 83.3) returned
# all-NaN values. The tested alternative (17.8, 83.3)
# returned valid measurements for all five variables.
#
# These are ERA5 request coordinates, not replacements
# for the original station metadata coordinates.

ERA5_COORDINATE_OVERRIDES = {
    "AP005": (17.8, 83.3),
    "MH005": (19.1, 72.9),
    "MH008": (18.8, 72.9),
    "MH013": (19.1, 72.9),
    "MH014": (19.0, 72.9),
    "TN002": (13.2, 80.2),
}


VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

RAW_COLUMNS = ["t2m", "d2m", "u10", "v10", "tp"]

VARIABLE_MAP = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp": "total_precipitation",
}

# IMPORTANT:
# The CDS request includes every hour of END_DATE.
# Therefore, the expected final timestamp is 23:00 on
# 2020-07-01, not 00:00 on that date.
EXPECTED_START = pd.Timestamp(f"{START_DATE} 00:00:00")
EXPECTED_END = pd.Timestamp(f"{END_DATE} 23:00:00")

EXPECTED_INDEX = pd.date_range(
    start=EXPECTED_START,
    end=EXPECTED_END,
    freq="h",
)

EXPECTED_HOURS = len(EXPECTED_INDEX)

OUTPUT_COLUMNS = [
    "timestamp",
    "station_id",
    "latitude",
    "longitude",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "tp",
    "temperature_C",
    "dewpoint_C",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "precipitation_mm",
]


# ============================================================
# 2. STATION COORDINATES
# ============================================================

def normalize_station_code(value):
    return str(value).strip().upper()


def find_column(df, candidates, description):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Cannot find the {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Accepted names: {candidates}"
    )


def load_station_coordinates():
    if not COORDINATES_FILE.exists():
        raise FileNotFoundError(
            f"Station coordinate file not found:\n"
            f"{COORDINATES_FILE}"
        )

    source = pd.read_csv(COORDINATES_FILE)

    station_col = find_column(
        source,
        [
            "station_id",
            "station",
            "station_code",
            "stationid",
            "StationId",
            "id",
        ],
        "station identifier",
    )

    latitude_col = find_column(
        source,
        ["latitude", "lat", "station_latitude"],
        "latitude",
    )

    longitude_col = find_column(
        source,
        ["longitude", "lon", "lng", "station_longitude"],
        "longitude",
    )

    stations = pd.DataFrame({
        "station_id": source[station_col].map(
            normalize_station_code
        ),
        "latitude": pd.to_numeric(
            source[latitude_col],
            errors="coerce",
        ),
        "longitude": pd.to_numeric(
            source[longitude_col],
            errors="coerce",
        ),
    })

    stations = stations.dropna(
        subset=["station_id", "latitude", "longitude"]
    )

    stations = stations[
        stations["station_id"].ne("")
    ]

    stations = stations.drop_duplicates(
        subset=["station_id"],
        keep="first",
    ).reset_index(drop=True)

    if stations.empty:
        raise ValueError(
            "No valid station coordinates were found."
        )

    print("\nStation coordinate file loaded successfully.")
    print(f"Coordinate rows: {len(stations)}")
    print(
        f"Columns used: {station_col}, "
        f"{latitude_col}, {longitude_col}"
    )

    return stations


# ============================================================
# 3. CDS REQUEST AND DOWNLOAD
# ============================================================

def build_request(latitude, longitude):
    return {
        "variable": VARIABLES,
        "location": [
            {
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        ],
        "date": [START_DATE, END_DATE],
        "data_format": "netcdf",
        "download_format": "zip",
    }


def download_archive(
    client,
    latitude,
    longitude,
    archive_path,
):
    request = build_request(latitude, longitude)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"CDS request attempt "
                f"{attempt}/{MAX_RETRIES}"
            )
            print(
                f"Request coordinates: "
                f"{latitude:.4f}, {longitude:.4f}"
            )

            result = client.retrieve(DATASET, request)
            result.download(str(archive_path))

            if (
                not archive_path.exists()
                or archive_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Downloaded ZIP archive is empty."
                )

            if not zipfile.is_zipfile(archive_path):
                raise RuntimeError(
                    "Downloaded file is not a valid ZIP archive."
                )

            print("Download completed successfully.")
            return

        except Exception as exc:
            last_error = exc

            print(
                f"Attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"ERA5-Land download failed after "
        f"{MAX_RETRIES} attempts."
    ) from last_error


# ============================================================
# 4. NETCDF PROCESSING
# ============================================================

def find_time_coordinate(ds):
    for name in ["valid_time", "time"]:
        if name in ds.coords or name in ds.variables:
            return name

    raise ValueError(
        "Neither 'valid_time' nor 'time' exists in the NetCDF. "
        f"Coordinates: {list(ds.coords)}"
    )


def read_netcdf_variable(ds, variable_name, time_name):
    if variable_name not in ds.data_vars:
        return None

    series = ds[variable_name].squeeze(drop=True)

    if time_name not in series.dims:
        raise ValueError(
            f"{variable_name} has unexpected dimensions: "
            f"{series.dims}"
        )

    extra_dimensions = [
        dimension
        for dimension in series.dims
        if dimension != time_name
    ]

    if extra_dimensions:
        raise ValueError(
            f"{variable_name} has unexpected extra dimensions: "
            f"{series.dims}"
        )

    timestamps = pd.to_datetime(
        np.asarray(ds[time_name].values).reshape(-1),
        errors="coerce",
    )

    values = np.asarray(series.values).reshape(-1)

    if len(timestamps) != len(values):
        raise ValueError(
            f"Timestamp/value count mismatch for "
            f"{variable_name}."
        )

    return pd.DataFrame({
        "timestamp": timestamps,
        variable_name: values,
    })


def read_archive(archive_path):
    frames = []
    discovered_variables = set()
    grid_latitudes = []
    grid_longitudes = []

    with TemporaryDirectory(
        prefix="era5_extract_"
    ) as temporary_directory:

        temporary_path = Path(temporary_directory)

        with zipfile.ZipFile(archive_path, "r") as archive:
            netcdf_members = [
                member
                for member in archive.namelist()
                if member.lower().endswith(".nc")
            ]

            if not netcdf_members:
                raise ValueError(
                    "No NetCDF files were found in the ZIP archive."
                )

            print(
                f"\nNetCDF files found: "
                f"{len(netcdf_members)}"
            )

            for member in netcdf_members:
                print("\n" + "=" * 65)
                print(f"Inspecting: {member}")

                extracted_path = (
                    temporary_path / Path(member).name
                )

                with archive.open(member) as source:
                    extracted_path.write_bytes(source.read())

                with xr.open_dataset(extracted_path) as ds:
                    time_name = find_time_coordinate(ds)

                    print("Dimensions:", dict(ds.sizes))
                    print("Coordinates:", list(ds.coords))
                    print("Data variables:", list(ds.data_vars))

                    for coordinate_name, target_list in [
                        ("latitude", grid_latitudes),
                        ("longitude", grid_longitudes),
                    ]:
                        if coordinate_name in ds.coords:
                            try:
                                coordinate_value = float(
                                    np.asarray(
                                        ds[coordinate_name].values
                                    ).squeeze().item()
                                )
                                target_list.append(coordinate_value)
                            except (ValueError, TypeError):
                                pass

                    for variable_name in RAW_COLUMNS:
                        if variable_name not in ds.data_vars:
                            continue

                        discovered_variables.add(variable_name)

                        values = np.asarray(
                            ds[variable_name].values
                        )

                        finite_count = int(
                            np.isfinite(values).sum()
                        )

                        print(
                            f"{variable_name}: "
                            f"{finite_count}/{values.size} "
                            f"finite values"
                        )

                        if finite_count == 0:
                            print(
                                f"WARNING: {variable_name} "
                                "contains only missing values."
                            )

                        frame = read_netcdf_variable(
                            ds,
                            variable_name,
                            time_name,
                        )

                        if frame is not None:
                            frames.append(frame)

    missing_variables = (
        set(RAW_COLUMNS) - discovered_variables
    )

    if missing_variables:
        raise ValueError(
            "The archive is missing required variables: "
            f"{sorted(missing_variables)}"
        )

    if not frames:
        raise ValueError(
            "No usable variable dataframes were created."
        )

    # Merge using timestamps only. Never merge using floating-
    # point latitude/longitude keys.
    merged = frames[0]

    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on="timestamp",
            how="outer",
            validate="one_to_one",
        )

    merged = merged.dropna(
        subset=["timestamp"]
    )

    merged = merged.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("MERGED RAW DATA SUMMARY")
    print("=" * 65)
    print("Rows:", len(merged))
    print("Columns:", list(merged.columns))

    print("\nMissing values:")
    print(merged[RAW_COLUMNS].isna().sum())

    print("\nFinite values:")
    for column in RAW_COLUMNS:
        values = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).to_numpy()

        print(
            f"{column}: "
            f"{np.isfinite(values).sum()}/{len(values)}"
        )

    grid_latitude = (
        float(np.median(grid_latitudes))
        if grid_latitudes
        else None
    )

    grid_longitude = (
        float(np.median(grid_longitudes))
        if grid_longitudes
        else None
    )

    return merged, grid_latitude, grid_longitude


# ============================================================
# 5. STRICT HOURLY VALIDATION
# ============================================================

def validate_raw_weather(weather):
    if weather.empty:
        raise ValueError(
            "The downloaded weather dataframe is empty."
        )

    missing_columns = [
        column
        for column in RAW_COLUMNS
        if column not in weather.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required raw columns are missing: "
            f"{missing_columns}"
        )

    weather = weather.copy()

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce",
    )

    weather = weather.dropna(
        subset=["timestamp"]
    )

    for column in RAW_COLUMNS:
        weather[column] = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

    weather = weather.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if weather["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps were found."
        )

    missing_counts = weather[RAW_COLUMNS].isna().sum()

    print("\nRaw weather validation")
    print("----------------------")
    print("Rows:", len(weather))
    print("Expected rows:", EXPECTED_HOURS)
    print("Start:", weather["timestamp"].min())
    print("End:", weather["timestamp"].max())
    print("\nMissing values:")
    print(missing_counts)

    if missing_counts.any():
        raise ValueError(
            "One or more raw weather variables contain "
            "missing values. Output will not be saved."
        )

    for column in RAW_COLUMNS:
        values = weather[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains infinite or non-finite values."
            )

    actual_index = pd.DatetimeIndex(
        weather["timestamp"]
    )

    if len(weather) != EXPECTED_HOURS:
        raise ValueError(
            f"Expected {EXPECTED_HOURS} hourly rows but received "
            f"{len(weather)}."
        )

    if not actual_index.equals(EXPECTED_INDEX):
        missing_timestamps = EXPECTED_INDEX.difference(
            actual_index
        )

        unexpected_timestamps = actual_index.difference(
            EXPECTED_INDEX
        )

        raise ValueError(
            "Hourly timeline validation failed.\n"
            f"Missing timestamps: {len(missing_timestamps)}\n"
            f"Unexpected timestamps: {len(unexpected_timestamps)}\n"
            f"First missing: {list(missing_timestamps[:5])}\n"
            f"First unexpected: "
            f"{list(unexpected_timestamps[:5])}"
        )

    print("\nAll raw variables are finite.")
    print("Hourly timeline is continuous.")
    print("Row count and date range are correct.")

    return weather


# ============================================================
# 6. DERIVED WEATHER FEATURES
# ============================================================


def add_derived_features(weather):
    result = weather.copy()

    # ERA5 temperature variables are in Kelvin.
    result["temperature_C"] = result["t2m"] - 273.15
    result["dewpoint_C"] = result["d2m"] - 273.15

    # Relative humidity using the Magnus approximation.
    temperature = result["temperature_C"]
    dewpoint = result["dewpoint_C"]

    temperature_denominator = 243.12 + temperature
    dewpoint_denominator = 243.12 + dewpoint

    if (
        (temperature_denominator == 0).any()
        or (dewpoint_denominator == 0).any()
    ):
        raise ValueError(
            "Relative humidity could not be calculated: "
            "a formula denominator is zero."
        )

    exponent = (
        17.625 * dewpoint / dewpoint_denominator
        - 17.625 * temperature / temperature_denominator
    )

    result["relative_humidity"] = (
        100.0 * np.exp(exponent)
    ).clip(lower=0, upper=100)

    # Wind speed in metres per second.
    result["wind_speed"] = np.sqrt(
        result["u10"] ** 2 + result["v10"] ** 2
    )

    # Meteorological wind direction in degrees.
    # 0 = north, 90 = east.
    result["wind_direction"] = (
        np.degrees(
            np.arctan2(
                -result["u10"],
                -result["v10"],
            )
        ) + 360.0
    ) % 360.0

    # ERA5 total precipitation is expressed in metres.
    # Inspect negative values before converting to millimetres.
    precipitation_m = result["tp"].astype(float)

    negative_mask = precipitation_m < 0

    if negative_mask.any():
        negative_values = precipitation_m[negative_mask]

        print("\nWARNING: Negative ERA5 precipitation values found.")
        print("Negative value count:", int(negative_mask.sum()))
        print(
            "Minimum negative value (metres):",
            float(negative_values.min()),
        )
        print(
            "Minimum negative value (millimetres):",
            float(negative_values.min() * 1000.0),
        )
        print(
            "Most negative timestamps:",
            list(result.loc[negative_mask, "timestamp"].head(10)),
        )

        # Tolerance: 0.1 mm of precipitation.
        # Clip only small negative numerical noise.
        tolerance_m = 0.0001

        materially_negative = precipitation_m < -tolerance_m

        if materially_negative.any():
            bad_values = precipitation_m[materially_negative]

            raise ValueError(
                "Materially negative precipitation detected. "
                f"Count: {int(materially_negative.sum())}; "
                f"minimum: {float(bad_values.min())} metres "
                f"({float(bad_values.min() * 1000.0)} mm). "
                "Inspect the source data before continuing."
            )

        print(
            "All negative precipitation values are within "
            "the 0.1 mm tolerance; clipping them to zero."
        )

        precipitation_m = precipitation_m.clip(lower=0)

    result["precipitation_mm"] = precipitation_m * 1000.0

    derived_columns = [
        "temperature_C",
        "dewpoint_C",
        "relative_humidity",
        "wind_speed",
        "wind_direction",
        "precipitation_mm",
    ]

    for column in derived_columns:
        values = result[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Derived feature {column} contains "
                "non-finite values."
            )

    if (result["wind_speed"] < 0).any():
        raise ValueError("Wind speed cannot be negative.")

    if (result["precipitation_mm"] < 0).any():
        raise ValueError(
            "Negative precipitation remains after validation."
        )

    return result


# ============================================================
# 7. EXISTING OUTPUT VALIDATION
# ============================================================

def existing_file_is_valid(output_file):
    if not output_file.exists():
        return False

    try:
        existing = pd.read_csv(output_file)

        if not set(OUTPUT_COLUMNS).issubset(
            existing.columns
        ):
            print(
                "Existing file has an outdated schema; "
                "it will be downloaded again."
            )
            return False

        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"],
            errors="coerce",
        )

        if existing["timestamp"].isna().any():
            return False

        if existing["timestamp"].duplicated().any():
            return False

        if len(existing) != EXPECTED_HOURS:
            print(
                f"Existing file contains {len(existing)} rows; "
                f"expected {EXPECTED_HOURS}. Redownloading."
            )
            return False

        if not pd.DatetimeIndex(
            existing["timestamp"]
        ).equals(EXPECTED_INDEX):
            return False

        for column in RAW_COLUMNS:
            values = pd.to_numeric(
                existing[column],
                errors="coerce",
            ).to_numpy()

            if not np.isfinite(values).all():
                print(
                    f"Existing file has invalid values in "
                    f"{column}. Redownloading."
                )
                return False

        print(
            f"Existing validated file found: "
            f"{output_file.name}"
        )
        return True

    except Exception as exc:
        print(
            f"Could not validate existing file: {exc}"
        )
        return False


# ============================================================
# 8. DOWNLOAD AND SAVE ONE STATION
# ============================================================

def download_station(client, station):
    station_id = normalize_station_code(
        station["station_id"]
    )

    station_latitude = float(station["latitude"])
    station_longitude = float(station["longitude"])

    output_file = OUTPUT_DIR / (
        f"{station_id}_weather.csv"
    )

    print("\n" + "#" * 75)
    print(f"STATION: {station_id}")
    print("#" * 75)

    print(
        f"Original station coordinates: "
        f"{station_latitude:.4f}, "
        f"{station_longitude:.4f}"
    )

    request_latitude, request_longitude = (
        ERA5_COORDINATE_OVERRIDES.get(
            station_id,
            (station_latitude, station_longitude),
        )
    )

    if station_id in ERA5_COORDINATE_OVERRIDES:
        print(
            "Using validated ERA5 coordinate override: "
            f"{request_latitude:.4f}, "
            f"{request_longitude:.4f}"
        )
    else:
        print(
            "Using original station coordinates for "
            "the ERA5 request."
        )

    if existing_file_is_valid(output_file):
        print("Skipping download.")
        return True

    try:
        with TemporaryDirectory(
            prefix=f"era5_{station_id}_"
        ) as temporary_directory:

            archive_path = (
                Path(temporary_directory) / "weather.zip"
            )

            download_archive(
                client,
                request_latitude,
                request_longitude,
                archive_path,
            )

            weather, grid_latitude, grid_longitude = (
                read_archive(archive_path)
            )

            print("\nActual ERA5 grid coordinates:")
            print("Latitude:", grid_latitude)
            print("Longitude:", grid_longitude)

            if (
                grid_latitude is not None
                and grid_longitude is not None
            ):
                print(
                    "Offset from original station: "
                    f"{abs(grid_latitude - station_latitude):.4f}° "
                    "latitude, "
                    f"{abs(grid_longitude - station_longitude):.4f}° "
                    "longitude"
                )

            weather = validate_raw_weather(weather)
            weather = add_derived_features(weather)

            # Preserve the original station coordinates in the
            # station metadata columns. The request-grid
            # override is configured separately above.
            weather["station_id"] = station_id
            weather["latitude"] = station_latitude
            weather["longitude"] = station_longitude

            weather = weather[OUTPUT_COLUMNS]

            if weather.isna().any().any():
                raise ValueError(
                    "Final output contains missing values."
                )

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_output = output_file.with_suffix(
                ".csv.tmp"
            )

            weather.to_csv(
                temporary_output,
                index=False,
                float_format="%.6f",
            )

            # Verify the actual CSV before replacing the final file.
            saved = pd.read_csv(temporary_output)

            if len(saved) != EXPECTED_HOURS:
                temporary_output.unlink(missing_ok=True)
                raise ValueError(
                    "Saved CSV failed row-count validation."
                )

            saved_timestamps = pd.to_datetime(
                saved["timestamp"],
                errors="coerce",
            )

            if not pd.DatetimeIndex(
                saved_timestamps
            ).equals(EXPECTED_INDEX):
                temporary_output.unlink(missing_ok=True)
                raise ValueError(
                    "Saved CSV failed timestamp validation."
                )

            for column in RAW_COLUMNS:
                saved_values = pd.to_numeric(
                    saved[column],
                    errors="coerce",
                ).to_numpy()

                if not np.isfinite(saved_values).all():
                    temporary_output.unlink(missing_ok=True)
                    raise ValueError(
                        f"Saved CSV has invalid values in "
                        f"{column}."
                    )

            # Replace the final output only after all checks pass.
            temporary_output.replace(output_file)

            print("\n" + "=" * 65)
            print(f"SUCCESS: {station_id}")
            print("=" * 65)
            print(f"Output file: {output_file}")
            print(f"Rows: {len(weather)}")
            print(f"Columns: {len(weather.columns)}")
            print(
                f"Start: {weather['timestamp'].min()}"
            )
            print(
                f"End: {weather['timestamp'].max()}"
            )
            print(
                "Missing values:",
                int(weather.isna().sum().sum()),
            )
            print("\nPreview:")
            print(
                weather.head(3).to_string(index=False)
            )

            return True

    except Exception as exc:
        print("\n" + "!" * 70)
        print(f"FAILED: {station_id}")
        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )
        print(
            "The new output was not accepted for this station."
        )
        print("!" * 70)

        return False


# ============================================================
# 9. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("ERA5-LAND WEATHER DOWNLOADER")
    print("=" * 75)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Coordinates: {COORDINATES_FILE}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Start date: {START_DATE}")
    print(f"End date: {END_DATE}")
    print(f"Expected hourly rows: {EXPECTED_HOURS}")
    print(f"Expected first timestamp: {EXPECTED_START}")
    print(f"Expected last timestamp: {EXPECTED_END}")
    print(f"Test mode: {TEST_MODE}")

    # The correct inclusive hourly count is 48,216:
    # 2015-01-01 00:00 through 2020-07-01 23:00.
    if EXPECTED_HOURS != 48216:
        raise RuntimeError(
            f"Expected 48,216 hourly timestamps, "
            f"but calculated {EXPECTED_HOURS}. "
            "Check EXPECTED_START and EXPECTED_END."
        )

    stations = load_station_coordinates()

    if TEST_MODE:
        target_stations = [
            normalize_station_code(station_id)
            for station_id in TEST_STATIONS
        ]

        stations = stations[
            stations["station_id"].isin(target_stations)
        ]

        found_stations = set(stations["station_id"])
        missing_stations = (
            set(target_stations) - found_stations
        )

        if missing_stations:
            raise ValueError(
                "These test stations were not found in the "
                f"coordinate file: {sorted(missing_stations)}"
            )

        print(
            f"\nTEST MODE: processing {len(stations)} stations: "
            f"{stations['station_id'].tolist()}"
        )
    else:
        print(
            f"\nProduction mode: {len(stations)} stations."
        )

    client = cdsapi.Client()

    successful = []
    failed = []

    for _, station in stations.iterrows():
        station_id = normalize_station_code(
            station["station_id"]
        )

        if download_station(client, station):
            successful.append(station_id)
        else:
            failed.append(station_id)

    print("\n" + "=" * 75)
    print("DOWNLOAD SUMMARY")
    print("=" * 75)
    print(
        f"Successful stations ({len(successful)}): "
        f"{successful}"
    )
    print(
        f"Failed stations ({len(failed)}): {failed}"
    )

    if failed:
        print(
            "\nSome stations failed validation. "
            "Do not use their weather data until corrected."
        )
    else:
        print(
            "\nAll selected stations passed validation."
        )


if __name__ == "__main__":
    main()
