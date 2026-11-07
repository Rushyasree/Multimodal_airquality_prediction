
from pathlib import Path
import pandas as pd

aq_path = Path("data/processed/air_quality_hourly_clean.csv")
weather_path = Path("data/raw/era5_land/station_weather/AP001.csv")

aq = pd.read_csv(aq_path, usecols=["StationId", "Datetime"])
weather = pd.read_csv(weather_path, usecols=["StationId", "Datetime"])

aq["Datetime"] = pd.to_datetime(aq["Datetime"], errors="coerce")
weather["Datetime"] = pd.to_datetime(
    weather["Datetime"], errors="coerce"
)

aq = aq.dropna(subset=["Datetime"])
weather = weather.dropna(subset=["Datetime"])

print("\n--- AIR-QUALITY TIMESTAMPS ---")
print("First records:")
print(aq.head(5).to_string(index=False))
print("Date range:", aq["Datetime"].min(), "to", aq["Datetime"].max())
print("Minute distribution:")
print(aq["Datetime"].dt.minute.value_counts().sort_index().to_dict())

print("\n--- ERA5-LAND TIMESTAMPS (AP001) ---")
print("First records:")
print(weather.head(5).to_string(index=False))
print(
    "Date range:",
    weather["Datetime"].min(),
    "to",
    weather["Datetime"].max()
)
print("Minute distribution:")
print(
    weather["Datetime"].dt.minute.value_counts()
    .sort_index().to_dict()
)

print("\n--- DIRECT TIMESTAMP OVERLAP ---")
aq_times = set(aq["Datetime"].drop_duplicates())
weather_times = set(weather["Datetime"].drop_duplicates())
direct = aq_times & weather_times

print("Unique air-quality timestamps:", len(aq_times))
print("Unique weather timestamps:", len(weather_times))
print("Exact matching timestamps:", len(direct))

print("\nDiagnostic only: no timestamps or files were modified.")
