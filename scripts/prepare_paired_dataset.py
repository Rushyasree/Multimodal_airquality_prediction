"""
scripts/prepare_paired_dataset.py
Extracts and caches the 24-hour sensor sequences for the 33 matched groups
and assigns leakage-safe GroupShuffleSplit partitions.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.prediction.lstm_forecaster import LSTMForecasterService

OUTPUT_DIR = Path("experiments/multimodal_fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading paired manifest and merged telemetry...")
manifest = pd.read_csv(OUTPUT_DIR / "paired_manifest.csv")
df_merged = pd.read_csv("data/processed/air_quality_weather_merged.csv")
df_merged["Datetime"] = pd.to_datetime(df_merged["Datetime"])

service = LSTMForecasterService()

# 1. Extract and cache 24h sequence for each unique group
unique_groups = manifest["station_date_group"].unique()
print(f"Total unique station-date groups to process: {len(unique_groups)}")

cached_sequences = {}
missing_groups = []

for g in unique_groups:
    group_rows = manifest[manifest["station_date_group"] == g].iloc[0]
    st_id = group_rows["station_id"]
    t_start = pd.to_datetime(group_rows["history_start"])
    t_end = pd.to_datetime(group_rows["history_end"])
    
    sub = df_merged[
        (df_merged["StationId"] == st_id) & 
        (df_merged["Datetime"] >= t_start) & 
        (df_merged["Datetime"] <= t_end)
    ].sort_values("Datetime")
    
    if len(sub) == 24:
        seq_arr = service.preprocess_sequence(sub, station_id=st_id) # Shape: (1, 24, 27)
        cached_sequences[g] = seq_arr.squeeze(0) # Shape: (24, 27)
    else:
        print(f"Warning: Group {g} had {len(sub)} rows instead of 24!")
        missing_groups.append(g)

print(f"Successfully cached {len(cached_sequences)} / {len(unique_groups)} sequences.")

# Save sequences
np.savez_compressed(OUTPUT_DIR / "cached_sequences.npz", **cached_sequences)
print(f"Saved: {OUTPUT_DIR / 'cached_sequences.npz'}")

# 2. Partition manifest into Train (75%) and Test (25%) using GroupShuffleSplit on station_date_group
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(manifest, groups=manifest["station_date_group"]))

manifest["split"] = "train"
manifest.loc[test_idx, "split"] = "test"

train_df = manifest[manifest["split"] == "train"]
test_df = manifest[manifest["split"] == "test"]

print("\nPartition Summary:")
print(f"Train samples: {len(train_df)}, groups: {train_df['station_date_group'].nunique()}, stations: {train_df['station_id'].nunique()}")
print(f"Test samples:  {len(test_df)}, groups: {test_df['station_date_group'].nunique()}, stations: {test_df['station_id'].nunique()}")

# Confirm zero group overlap
train_groups = set(train_df["station_date_group"])
test_groups = set(test_df["station_date_group"])
overlap = train_groups.intersection(test_groups)
print(f"Group overlap between train and test: {len(overlap)}")
assert len(overlap) == 0, "Data leakage detected: groups cross split boundary!"

# Save paired dataset
manifest.to_csv(OUTPUT_DIR / "paired_dataset.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'paired_dataset.csv'}")
