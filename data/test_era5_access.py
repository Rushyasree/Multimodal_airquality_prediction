import cdsapi
from pathlib import Path


# ============================================================
# ERA5-LAND API ACCESS TEST
# ============================================================

OUTPUT = Path(
    "data/raw/era5_land/test_era5.nc"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("ERA5-LAND API ACCESS TEST")
print("=" * 70)

print("\nConnecting to Copernicus CDS...")

client = cdsapi.Client()

print("Connection successful.")


# ------------------------------------------------------------
# Small test request
# ------------------------------------------------------------

request = {
    "variable": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
    ],
    "location": {
        "latitude": 16.515083,
        "longitude": 80.518167,
    },
    "date": [
        "2019-01-01/2019-01-01"
    ],
    "time": [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
    "data_format": "netcdf",
}

print("\nRequest:")
print("Location : AP001 / Amaravati")
print("Latitude : 16.515083")
print("Longitude: 80.518167")
print("Date     : 2019-01-01")
print("Variables: 5")
print("\nSubmitting request...")

client.retrieve(
    "reanalysis-era5-land-timeseries",
    request,
    str(OUTPUT)
)

print("\n" + "=" * 70)
print("ERA5-LAND TEST COMPLETE")
print("=" * 70)

print(f"\nSaved file:")
print(OUTPUT)

print("\nIf this file exists, API access is working correctly.")