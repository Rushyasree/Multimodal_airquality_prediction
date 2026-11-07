"""
scripts/diagnose_timestamps.py
Inspects exact timestamps and splits across sequence index and processed merged dataset.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

# 1. Inspect sequence index
seq_csv = ROOT_DIR / "data/processed/sequences/air_quality_24h_sequence_index.csv"
df = pd.read_csv(seq_csv)

print("=" * 70)
print("SEQUENCE INDEX SPLIT AUDIT")
print("=" * 70)
for sp in ["train", "validation", "test"]:
    grp = df[df["split"] == sp]
    print(f"Split: {sp.upper()}")
    print(f"  Count: {len(grp)}")
    print(f"  Stations: {grp['StationId'].nunique()}")
    print(f"  history_start: {grp['history_start'].min()} to {grp['history_start'].max()}")
    print(f"  history_end:   {grp['history_end'].min()} to {grp['history_end'].max()}")
    print(f"  target_time:   {grp['target_time'].min()} to {grp['target_time'].max()}")

# 2. Inspect merged dataset
merged_csv = ROOT_DIR / "data/processed/air_quality_weather_merged.csv"
if merged_csv.exists():
    print("\n" + "=" * 70)
    print("AIR QUALITY WEATHER MERGED DATASET AUDIT")
    print("=" * 70)
    m_df = pd.read_csv(merged_csv, usecols=["StationId", "Datetime"])
    print(f"Total rows: {len(m_df)}")
    print(f"Unique stations: {m_df['StationId'].nunique()}")
    print(f"Datetime min: {m_df['Datetime'].min()} | max: {m_df['Datetime'].max()}")

# 3. Check where 2019-10-31 and 2020-12-31 came from
print("\n" + "=" * 70)
print("CHECKING SCRIPT SOURCES FOR SPLIT DEFINITIONS")
print("=" * 70)
for p in (ROOT_DIR / "src").rglob("*.py"):
    content = p.read_text(encoding="utf-8", errors="ignore")
    if "2019-10-31" in content or "2020-12-31" in content or "2019-09-04" in content or "2020-07-01" in content:
        print(f"Found in {p.relative_to(ROOT_DIR)}")

for p in (ROOT_DIR / "scripts").rglob("*.py"):
    content = p.read_text(encoding="utf-8", errors="ignore")
    if "2019-10-31" in content or "2020-12-31" in content or "2019-09-04" in content or "2020-07-01" in content:
        print(f"Found in {p.relative_to(ROOT_DIR)}")
