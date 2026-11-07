"""Create the feature schema used by the alignment stage.

Do not assume that all three source datasets share the same timestamps,
locations, or target semantics. Inspect them first and then map their columns
into this schema.
"""

AIR_QUALITY_COLUMNS = [
    "timestamp", "station_id", "latitude", "longitude",
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"
]

WEATHER_COLUMNS = [
    "timestamp", "latitude", "longitude",
    "temperature_2m", "wind_u", "wind_v",
    "surface_pressure", "precipitation"
]

IMAGE_COLUMNS = [
    "image_path", "timestamp", "latitude", "longitude", "PM2.5"
]

TARGET_COLUMNS = ["target_PM2.5", "target_PM10"]
