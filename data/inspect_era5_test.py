import zipfile
from pathlib import Path

import xarray as xr


# ============================================================
# ERA5-LAND TEST FILE INSPECTION
# ============================================================

ARCHIVE = Path(
    "data/raw/era5_land/test_era5.nc"
)

EXTRACT_DIR = Path(
    "data/raw/era5_land/test_extracted"
)


print("=" * 70)
print("ERA5-LAND TEST FILE INSPECTION")
print("=" * 70)


# ============================================================
# 1. CHECK ARCHIVE
# ============================================================

print(f"\nArchive: {ARCHIVE}")
print(f"Exists : {ARCHIVE.exists()}")

if not ARCHIVE.exists():
    raise FileNotFoundError(
        f"ERA5 test archive not found: {ARCHIVE}"
    )


# ============================================================
# 2. VERIFY ZIP
# ============================================================

if not zipfile.is_zipfile(ARCHIVE):

    raise ValueError(
        "The downloaded file is not a valid ZIP archive."
    )

print("\nZIP archive detected.")


# ============================================================
# 3. EXTRACT FILES
# ============================================================

EXTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with zipfile.ZipFile(ARCHIVE, "r") as z:

    files = z.namelist()

    print("\nFiles inside archive:")

    for file in files:
        print(f" - {file}")

    z.extractall(EXTRACT_DIR)


# ============================================================
# 4. FIND NETCDF FILES
# ============================================================

nc_files = sorted(
    EXTRACT_DIR.glob("*.nc")
)


if not nc_files:

    raise FileNotFoundError(
        "No NetCDF files were found inside the ERA5 archive."
    )


print("\n" + "=" * 70)
print("NETCDF FILES FOUND")
print("=" * 70)

for file in nc_files:
    print(f"\n{file.name}")


# ============================================================
# 5. INSPECT EACH NETCDF
# ============================================================

for nc_file in nc_files:

    print("\n\n")
    print("=" * 70)
    print(f"INSPECTING: {nc_file.name}")
    print("=" * 70)

    ds = xr.open_dataset(
        nc_file,
        engine="netcdf4"
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print("\nDimensions:")
    print(ds.sizes)

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    print("\nCoordinates:")
    print(list(ds.coords))

    for coord in ds.coords:

        print(
            f"\n{coord}:"
        )

        print(
            ds[coord].values
        )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA VARIABLES")
    print("-" * 70)

    for variable in ds.data_vars:

        da = ds[variable]

        print(
            f"\nVariable: {variable}"
        )

        print(
            f"Dimensions: {da.dims}"
        )

        print(
            f"Shape: {da.shape}"
        )

        print(
            f"Units: {da.attrs.get('units', 'N/A')}"
        )

        print(
            f"Long name: "
            f"{da.attrs.get('long_name', 'N/A')}"
        )

        try:

            print(
                f"Minimum: {float(da.min().values)}"
            )

            print(
                f"Maximum: {float(da.max().values)}"
            )

        except Exception:

            print(
                "Minimum/Maximum: unavailable"
            )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    if "time" in ds.coords:

        print("\n" + "-" * 70)
        print("TIME INFORMATION")
        print("-" * 70)

        times = ds["time"].values

        print(
            f"Number of timestamps: {len(times)}"
        )

        print(
            f"First timestamp: {times[0]}"
        )

        print(
            f"Last timestamp: {times[-1]}"
        )

    # --------------------------------------------------------
    # Global attributes
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GLOBAL ATTRIBUTES")
    print("-" * 70)

    for key, value in ds.attrs.items():

        print(
            f"{key}: {value}"
        )

    ds.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ERA5-LAND INSPECTION COMPLETE")
print("=" * 70)

print(
    f"\nExtracted files are available at:"
)

print(
    EXTRACT_DIR
)