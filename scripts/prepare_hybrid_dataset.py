"""
scripts/prepare_hybrid_dataset.py

Merges PM10 ground truth targets from data/processed/air_quality_weather_merged.csv
into experiments/multimodal_fusion/validation_v2/paired_dataset_v2.csv,
creating experiments/unified_hybrid/paired_dataset_hybrid.csv with identical
Train (81), Val (21), and Test (97) split assignments and zero group leakage.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
v2_manifest = pd.read_csv(ROOT / "experiments/multimodal_fusion/validation_v2/paired_dataset_v2.csv")
merged_weather = pd.read_csv(ROOT / "data/processed/air_quality_weather_merged.csv")

# Extract PM10 target
sub_pm10 = merged_weather[["StationId", "Datetime", "PM10"]].rename(
    columns={"StationId": "station_id", "Datetime": "target_time", "PM10": "station_target_pm10"}
)

df_hybrid = pd.merge(v2_manifest, sub_pm10, on=["station_id", "target_time"], how="left")

print(f"Total instances: {len(df_hybrid)}")
print("Missing PM10 count:", df_hybrid["station_target_pm10"].isna().sum())
assert df_hybrid["station_target_pm10"].isna().sum() == 0, "Missing PM10 targets!"

out_path = ROOT / "experiments/unified_hybrid/paired_dataset_hybrid.csv"
df_hybrid.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

for sp in ["train", "val", "test"]:
    sub = df_hybrid[df_hybrid["split"] == sp]
    print(f"  {sp.upper()}: {len(sub)} samples, {sub['station_date_group'].nunique()} groups")
    print(f"    PM2.5: mean={sub['station_target_pm25'].mean():.2f}")
    print(f"    PM10:  mean={sub['station_target_pm10'].mean():.2f}")
