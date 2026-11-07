"""
scripts/create_v2_manifest.py

Splits the 24 training groups from experiments/multimodal_fusion/paired_dataset.csv
into train (18 groups, 81 samples) and val (6 groups, 21 samples) with random_state=14,
while keeping the 9 test groups (97 samples) completely untouched.
Ensures zero group overlap across train, val, and test.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments/multimodal_fusion"
OUT_DIR = EXP_DIR / "validation_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(EXP_DIR / "paired_dataset.csv")

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

# Partition train_df into train (75% groups) and val (25% groups) using seed 14
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=14)
tr_idx, val_idx = next(gss.split(train_df, groups=train_df["station_date_group"]))

train_sub = train_df.iloc[tr_idx].copy()
val_sub = train_df.iloc[val_idx].copy()

train_sub["split"] = "train"
val_sub["split"] = "val"
test_df["split"] = "test"

df_v2 = pd.concat([train_sub, val_sub, test_df], ignore_index=True)
out_path = OUT_DIR / "paired_dataset_v2.csv"
df_v2.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

print("\nValidation v2 Split Summary:")
for sp in ["train", "val", "test"]:
    sub = df_v2[df_v2["split"] == sp]
    print(f"  {sp.upper()}: {len(sub)} samples, {sub['station_date_group'].nunique()} groups, {sub['station_id'].nunique()} stations")
    print(f"    PM2.5 Target Mean: {sub['station_target_pm25'].mean():.2f} ug/m3 (min: {sub['station_target_pm25'].min():.2f}, max: {sub['station_target_pm25'].max():.2f})")

# Check overlaps
tr_grp = set(df_v2[df_v2["split"] == "train"]["station_date_group"])
va_grp = set(df_v2[df_v2["split"] == "val"]["station_date_group"])
te_grp = set(df_v2[df_v2["split"] == "test"]["station_date_group"])

overlap_tv = len(tr_grp.intersection(va_grp))
overlap_tt = len(tr_grp.intersection(te_grp))
overlap_vt = len(va_grp.intersection(te_grp))

print(f"\nGroup Overlap train & val:  {overlap_tv}")
print(f"Group Overlap train & test: {overlap_tt}")
print(f"Group Overlap val & test:   {overlap_vt}")

assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, "Group leakage detected!"
print("ZERO GROUP OVERLAP CONFIRMED ACROSS ALL THREE PARTITIONS!")
