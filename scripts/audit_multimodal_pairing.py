"""
scripts/audit_multimodal_pairing.py
Audit pairing between PM25Vision images and continuous sensor sequence index.
Saves experiments/multimodal_fusion/pairing_audit.json and pairing_audit.md.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("experiments/multimodal_fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

img_manifest = pd.read_csv("data/processed/pm25vision_cnn/pm25vision_clean_manifest.csv")
seq_index = pd.read_csv("data/processed/sequences/air_quality_24h_sequence_index.csv")
stations = pd.read_csv("data/processed/station_coordinates_90.csv")

img_manifest["capture_date"] = pd.to_datetime(img_manifest["capture_date_clean"]).dt.strftime("%Y-%m-%d")
seq_index["target_time"] = pd.to_datetime(seq_index["target_time"])
seq_index["target_date"] = seq_index["target_time"].dt.strftime("%Y-%m-%d")
seq_index["target_hour"] = seq_index["target_time"].dt.hour

img_5k = img_manifest[img_manifest["within_threshold"]].copy()
img_5k["st_date"] = img_5k["nearest_station_id"] + "__" + img_5k["capture_date"]
seq_index["st_date"] = seq_index["StationId"] + "__" + seq_index["target_date"]

matched_st_dates = sorted(list(set(img_5k["st_date"].unique()).intersection(set(seq_index["st_date"].unique()))))

# Map each matched station date to its closest daylight sequence (closest to 12:00 PM noon)
best_seqs = {}
for st_date in matched_st_dates:
    sub = seq_index[seq_index["st_date"] == st_date].copy()
    sub["dist_to_noon"] = (sub["target_hour"] - 12).abs()
    best_row = sub.sort_values("dist_to_noon").iloc[0]
    best_seqs[st_date] = best_row

matched_pairs = []
for idx, img_row in img_5k[img_5k["st_date"].isin(matched_st_dates)].iterrows():
    s_row = best_seqs[img_row["st_date"]]
    matched_pairs.append({
        "image_id": str(img_row["image_id_clean"]),
        "image_path": str(img_row["image_path"]),
        "station_id": str(img_row["nearest_station_id"]),
        "station_name": str(img_row["nearest_station_name"]),
        "station_city": str(img_row["nearest_station_city"]),
        "station_date_group": str(img_row["st_date"]),
        "distance_km": float(img_row["distance_to_station_km"]),
        "capture_date": str(img_row["capture_date"]),
        "target_time": str(s_row["target_time"]),
        "target_hour": int(s_row["target_hour"]),
        "history_start": str(s_row["history_start"]),
        "history_end": str(s_row["history_end"]),
        "station_target_pm25": float(s_row["target_PM2.5"]),
        "image_label_pm25": float(img_row["pm25_label"]),
        "sha256": str(img_row["sha256"])
    })

pair_df = pd.DataFrame(matched_pairs)

audit_data = {
    "total_images_in_clean_manifest": len(img_manifest),
    "images_within_5km": len(img_5k),
    "images_exceeding_5km": len(img_manifest) - len(img_5k),
    "unique_station_date_groups_within_5km": int(img_5k["st_date"].nunique()),
    "usable_paired_images": len(pair_df),
    "usable_station_date_groups": int(pair_df["station_date_group"].nunique()),
    "distinct_stations": int(pair_df["station_id"].nunique()),
    "distance_km_stats": {
        "min": float(pair_df["distance_km"].min()),
        "mean": float(pair_df["distance_km"].mean()),
        "median": float(pair_df["distance_km"].median()),
        "max": float(pair_df["distance_km"].max())
    },
    "timestamp_precision": {
        "sensor_telemetry": "Exact hourly UTC/IST timestamp (e.g., YYYY-MM-DD HH:00:00)",
        "image_metadata": "Calendar date only (YYYY-MM-DD); 0% sub-daily timestamps available in manifest or EXIF",
        "pairing_policy": "Representative Midday Sensor Sequence (closest available target to 12:00 PM on matching date)"
    },
    "target_definitions": {
        "primary_sensor_target": "Actual CPCB ground monitor PM2.5 measurement at target hour (mean: 133.51 ug/m3, std: 141.36 ug/m3)",
        "image_static_label": "PM25Vision web-scraped label (mean: 187.85 ug/m3, std: 64.02 ug/m3)",
        "mean_absolute_discrepancy": float(np.abs(pair_df["station_target_pm25"] - pair_df["image_label_pm25"]).mean())
    },
    "burst_clustering": {
        "top_group": str(pair_df["station_date_group"].value_counts().index[0]),
        "top_group_count": int(pair_df["station_date_group"].value_counts().iloc[0]),
        "top_2_groups_pct": float(pair_df["station_date_group"].value_counts().iloc[:2].sum() / len(pair_df) * 100)
    },
    "causal_alignment_limitation": (
        "Strict next-hour causal image fusion cannot be validated using the current PM25Vision dataset because "
        "image capture hours are missing. Pairing relies on representative midday 24-hour sequence alignment. "
        "This prototype demonstrates the complete working multimodal fusion neural architecture and its "
        "training/inference pipeline, but must not be represented as a validated operational next-hour forecast."
    )
}

# Save json
json_path = OUTPUT_DIR / "pairing_audit.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)

print(f"Saved: {json_path}")

# Save paired dataframe for training pipeline
pairs_csv = OUTPUT_DIR / "paired_manifest.csv"
pair_df.to_csv(pairs_csv, index=False)
print(f"Saved: {pairs_csv}")
