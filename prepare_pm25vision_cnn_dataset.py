"""
prepare_pm25vision_cnn_dataset.py

Prepares the PM25Vision image dataset for CNN regression modeling.
Handles:
- Manifest schema validation without assuming a pre-existing station_id.
- Haversine-distance calculation to match coordinates against 90 Indian CPCB stations.
- Explicit threshold analysis (<= 5 km vs > 5 km) with detailed logging.
- Image path resolution, PIL integrity checks, and SHA-256 duplicate detection.
- Exact duplicate removal while preserving raw image files on disk.
- Leakage-safe, group-aware splitting (GroupShuffleSplit by station_date_group) with fixed seed.
- Creation of full image-only splits and a station-matched subset (<= 5 km).
- Machine-readable audit reports in JSON and CSV format.
"""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent

MANIFEST_FILE = ROOT / "data" / "processed" / "pm25vision_image_manifest.csv"
STATION_FILE = ROOT / "data" / "processed" / "station_coordinates_90.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "pm25vision_cnn"
MATCHED_SUBSET_DIR = OUTPUT_DIR / "matched_subset"

IMAGE_DIR = ROOT / "data" / "raw" / "pm25vision" / "images"

MATCH_THRESHOLD_KM = 5.0
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # fraction of remaining training data (approx 16% of total)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHED_SUBSET_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def find_column(df, candidates, required=True):
    """Find a column using case-insensitive candidate names."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """
    Vectorized Haversine distance in kilometers from a single (lat1, lon1)
    to arrays of (lat2_arr, lon2_arr).
    """
    radius = 6371.0  # Earth mean radius in km
    p1_lat = math.radians(lat1)
    p1_lon = math.radians(lon1)
    p2_lat = np.radians(lat2_arr)
    p2_lon = np.radians(lon2_arr)

    dlat = p2_lat - p1_lat
    dlon = p2_lon - p1_lon

    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1_lat) * np.cos(p2_lat) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return radius * c


def resolve_image_path(row, path_col, filename_col, split_col):
    """Resolve a manifest image path to an existing file."""
    candidates = []

    if path_col:
        raw_path = str(row[path_col]).strip()
        if raw_path and raw_path.lower() not in ("nan", "none"):
            p = Path(raw_path)
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.extend([ROOT / p, Path.cwd() / p])

    if filename_col:
        filename = str(row[filename_col]).strip()
        if filename and filename.lower() not in ("nan", "none"):
            p = Path(filename)
            if p.is_absolute():
                candidates.append(p)
            else:
                if split_col:
                    split = str(row[split_col]).strip().lower()
                    if split in ("train", "test"):
                        candidates.append(IMAGE_DIR / split / p.name)
                candidates.extend([
                    IMAGE_DIR / "train" / p.name,
                    IMAGE_DIR / "test" / p.name,
                ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def sha256_file(path):
    """Compute SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------
# MAIN DATASET PREPARATION PIPELINE
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("PM25VISION CNN DATASET PREPARATION & SPATIAL ALIGNMENT AUDIT")
    print("=" * 70)

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")
    if not STATION_FILE.exists():
        raise FileNotFoundError(f"Station coordinates file not found: {STATION_FILE}")

    # 1. Load manifest and station coordinates
    df = pd.read_csv(MANIFEST_FILE)
    stations_df = pd.read_csv(STATION_FILE)

    initial_manifest_rows = len(df)
    print(f"Loaded raw image manifest: {initial_manifest_rows} rows")
    print(f"Manifest columns: {list(df.columns)}")
    print(f"Loaded station coordinates: {len(stations_df)} stations")

    # Detect manifest columns
    image_id_col = find_column(df, ["image_id", "id", "filename"])
    date_col = find_column(df, ["captured_at", "capture_date", "date", "timestamp"])
    pm25_col = find_column(df, ["pm25", "pm2.5", "pm2_5", "pm25_value"])
    lat_col = find_column(df, ["latitude", "lat"])
    lon_col = find_column(df, ["longitude", "lon", "long"])

    path_col = find_column(df, ["image_path", "filepath", "file_path", "path", "local_path"], required=False)
    filename_col = find_column(df, ["filename", "file_name", "image_filename"], required=False)
    split_col = find_column(df, ["source_split", "split", "dataset_split"], required=False)

    if filename_col is None:
        filename_col = image_id_col

    # Detect station coordinates columns
    st_id_col = find_column(stations_df, ["stationid", "station_id", "station"])
    st_name_col = find_column(stations_df, ["stationname", "station_name", "name"], required=False)
    st_city_col = find_column(stations_df, ["city"], required=False)
    st_lat_col = find_column(stations_df, ["latitude", "lat"])
    st_lon_col = find_column(stations_df, ["longitude", "lon", "long"])

    # Clean station dataframe
    valid_stations = stations_df.dropna(subset=[st_lat_col, st_lon_col]).copy()
    valid_stations[st_lat_col] = pd.to_numeric(valid_stations[st_lat_col], errors="coerce")
    valid_stations[st_lon_col] = pd.to_numeric(valid_stations[st_lon_col], errors="coerce")
    valid_stations = valid_stations.dropna(subset=[st_lat_col, st_lon_col]).drop_duplicates(subset=[st_id_col])

    st_lats = valid_stations[st_lat_col].values
    st_lons = valid_stations[st_lon_col].values
    st_ids = valid_stations[st_id_col].astype(str).str.strip().values
    st_names = (valid_stations[st_name_col].astype(str).str.strip().values if st_name_col else [""] * len(valid_stations))
    st_cities = (valid_stations[st_city_col].astype(str).str.strip().values if st_city_col else [""] * len(valid_stations))

    print(f"Active stations available for spatial matching: {len(valid_stations)}")

    # 2. Normalize image manifest fields
    df["image_id_clean"] = df[image_id_col].astype(str).str.strip()
    df["capture_date_clean"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    df["pm25_label"] = pd.to_numeric(df[pm25_col], errors="coerce")
    df["latitude_clean"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude_clean"] = pd.to_numeric(df[lon_col], errors="coerce")

    # Resolve local file paths
    df["image_path_clean"] = df.apply(
        lambda row: resolve_image_path(row, path_col, filename_col, split_col),
        axis=1
    )

    # 3. Filter valid metadata rows
    valid_meta = (
        df["image_id_clean"].notna()
        & (df["image_id_clean"] != "")
        & df["capture_date_clean"].notna()
        & df["pm25_label"].notna()
        & (df["pm25_label"] >= 0)
        & df["latitude_clean"].notna()
        & df["longitude_clean"].notna()
        & df["image_path_clean"].notna()
    )
    df_valid = df.loc[valid_meta].copy()
    excluded_meta_rows = initial_manifest_rows - len(df_valid)
    print(f"Rows after metadata & file presence validation: {len(df_valid)} (excluded {excluded_meta_rows})")

    # 4. Perform Haversine matching against CPCB stations
    nearest_station_ids = []
    nearest_station_names = []
    nearest_station_cities = []
    min_distances_km = []

    for _, row in df_valid.iterrows():
        img_lat = row["latitude_clean"]
        img_lon = row["longitude_clean"]
        dists = haversine_km(img_lat, img_lon, st_lats, st_lons)
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        nearest_station_ids.append(st_ids[min_idx])
        nearest_station_names.append(st_names[min_idx])
        nearest_station_cities.append(st_cities[min_idx])
        min_distances_km.append(round(min_dist, 4))

    df_valid["nearest_station_id"] = nearest_station_ids
    df_valid["nearest_station_name"] = nearest_station_names
    df_valid["nearest_station_city"] = nearest_station_cities
    df_valid["distance_to_station_km"] = min_distances_km
    df_valid["within_threshold"] = df_valid["distance_to_station_km"] <= MATCH_THRESHOLD_KM

    # Explicit threshold analysis
    pass_thresh = int(df_valid["within_threshold"].sum())
    fail_thresh = int((~df_valid["within_threshold"]).sum())
    print("\n--- Spatial Alignment Against 90 CPCB Stations ---")
    print(f"Haversine threshold: {MATCH_THRESHOLD_KM} km")
    print(f"Passing threshold (<= {MATCH_THRESHOLD_KM} km): {pass_thresh} rows ({pass_thresh / len(df_valid) * 100:.1f}%)")
    print(f"Failing threshold (> {MATCH_THRESHOLD_KM} km):  {fail_thresh} rows ({fail_thresh / len(df_valid) * 100:.1f}%)")
    print(f"Nearest distance summary: min={df_valid['distance_to_station_km'].min():.2f} km, "
          f"median={df_valid['distance_to_station_km'].median():.2f} km, "
          f"mean={df_valid['distance_to_station_km'].mean():.2f} km, "
          f"max={df_valid['distance_to_station_km'].max():.2f} km")

    # 5. Verify image file integrity and calculate SHA-256 hashes
    print("\n--- Image Integrity & Exact Duplicate Verification ---")
    valid_image_rows = []
    unreadable_count = 0

    for _, row in df_valid.iterrows():
        path = Path(row["image_path_clean"])
        try:
            with Image.open(path) as img:
                img.verify()
            # Calculate SHA-256 for bit-level deduplication
            file_hash = sha256_file(path)
            row_dict = row.to_dict()
            row_dict["image_path"] = str(path)
            row_dict["sha256"] = file_hash
            valid_image_rows.append(row_dict)
        except Exception as exc:
            unreadable_count += 1
            print(f"Corrupt/unreadable image excluded: {path.name} ({exc})")

    clean_df = pd.DataFrame(valid_image_rows)
    if clean_df.empty:
        raise RuntimeError("No valid images remain after file verification.")

    total_valid_images = len(clean_df)
    print(f"Successfully verified images: {total_valid_images}")
    print(f"Unreadable images excluded: {unreadable_count}")

    # Remove exact duplicate file contents (preserving original files on disk)
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["sha256"], keep="first").copy()
    exact_duplicates_removed = before_dedup - len(clean_df)
    print(f"Exact duplicate image contents removed from modeling manifest: {exact_duplicates_removed}")
    print(f"Unique physical image files retained: {len(clean_df)}")

    # Check image ID uniqueness
    conflicting_ids = clean_df.groupby("image_id_clean")["sha256"].nunique()
    conflicting_ids = conflicting_ids[conflicting_ids > 1]
    if len(conflicting_ids) > 0:
        print(f"WARNING: Image IDs mapped to different contents: {len(conflicting_ids)}")
    else:
        print("Image ID uniqueness check: PASSED (all retained image IDs map uniquely)")

    # Construct grouping key for leakage prevention
    # Images at the same nearest station / location area on the same day must not cross splits
    clean_df["station_date_group"] = clean_df["nearest_station_id"] + "__" + clean_df["capture_date_clean"]

    repeated_img_ids = int(clean_df["image_id_clean"].duplicated().sum())
    total_groups = int(clean_df["station_date_group"].nunique())
    groups_gt_1 = int((clean_df["station_date_group"].value_counts() > 1).sum())

    print(f"Total unique station-date groups: {total_groups}")
    print(f"Groups with multiple images: {groups_gt_1}")
    print(f"Largest single station-date group: {clean_df['station_date_group'].value_counts().max()} images")

    # 6. Save master clean manifest
    master_manifest_path = OUTPUT_DIR / "pm25vision_clean_manifest.csv"
    clean_cols = [
        "image_id_clean", "capture_date_clean", "pm25_label", "latitude_clean", "longitude_clean",
        "nearest_station_id", "nearest_station_name", "nearest_station_city", "distance_to_station_km",
        "within_threshold", "station_date_group", "image_path", "sha256"
    ]
    clean_df[clean_cols].to_csv(master_manifest_path, index=False)
    print(f"Saved master clean manifest -> {master_manifest_path}")

    # 7. Create Leakage-Safe Splits for Full Image-Only Dataset (640 images)
    print("\n" + "=" * 70)
    print("CREATING LEAKAGE-SAFE SPLITS: FULL IMAGE-ONLY DATASET")
    print("=" * 70)

    splits_full = perform_group_split(clean_df, group_col="station_date_group", seed=RANDOM_SEED)
    save_splits(splits_full, OUTPUT_DIR)

    # 8. Create Leakage-Safe Splits for Station-Matched Subset (<= 5 km)
    print("\n" + "=" * 70)
    print(f"CREATING LEAKAGE-SAFE SPLITS: STATION-MATCHED SUBSET (<= {MATCH_THRESHOLD_KM} KM)")
    print("=" * 70)

    matched_df = clean_df[clean_df["within_threshold"]].copy()
    splits_matched = perform_group_split(matched_df, group_col="station_date_group", seed=RANDOM_SEED)
    save_splits(splits_matched, MATCHED_SUBSET_DIR)

    # 9. Generate machine-readable JSON summary report
    report = {
        "source_manifest": str(MANIFEST_FILE),
        "station_coordinates_file": str(STATION_FILE),
        "initial_manifest_rows": initial_manifest_rows,
        "excluded_metadata_rows": excluded_meta_rows,
        "unreadable_images": unreadable_count,
        "exact_duplicates_removed": exact_duplicates_removed,
        "unique_images_retained": len(clean_df),
        "spatial_matching": {
            "distance_threshold_km": MATCH_THRESHOLD_KM,
            "images_within_threshold": int((clean_df["within_threshold"]).sum()),
            "images_exceeding_threshold": int((~clean_df["within_threshold"]).sum()),
            "distance_km_min": float(clean_df["distance_to_station_km"].min()),
            "distance_km_median": float(clean_df["distance_to_station_km"].median()),
            "distance_km_mean": float(clean_df["distance_to_station_km"].mean()),
            "distance_km_max": float(clean_df["distance_to_station_km"].max()),
            "unique_stations_matched_all": int(clean_df["nearest_station_id"].nunique()),
            "unique_stations_matched_within_5km": int(matched_df["nearest_station_id"].nunique()),
        },
        "grouping": {
            "strategy": "station_date_group (nearest_station_id + capture_date)",
            "total_groups_full": total_groups,
            "groups_with_multiple_images_full": groups_gt_1,
            "total_groups_matched": int(matched_df["station_date_group"].nunique()),
        },
        "full_dataset_splits": compile_split_stats(splits_full),
        "matched_subset_splits": compile_split_stats(splits_matched),
    }

    report_path = OUTPUT_DIR / "dataset_preparation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved comprehensive preparation report -> {report_path}")

    # Also save standard split_report.json for compatibility
    with open(OUTPUT_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report["full_dataset_splits"], f, indent=2)

    print("\nDataset preparation and validation completed successfully.")


def perform_group_split(df, group_col, seed):
    """Perform a strict 2-step GroupShuffleSplit to create Train, Val, and Test sets."""
    groups = df[group_col]

    first_split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    trainval_idx, test_idx = next(first_split.split(df, groups=groups))

    trainval = df.iloc[trainval_idx].copy()
    test = df.iloc[test_idx].copy()

    second_split = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=seed)
    train_idx, val_idx = next(second_split.split(trainval, groups=trainval[group_col]))

    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    # Leakage assertions
    train_groups = set(train[group_col])
    val_groups = set(val[group_col])
    test_groups = set(test[group_col])

    assert train_groups.isdisjoint(val_groups), "Leakage detected: train and val share groups!"
    assert train_groups.isdisjoint(test_groups), "Leakage detected: train and test share groups!"
    assert val_groups.isdisjoint(test_groups), "Leakage detected: val and test share groups!"

    # Content hash disjoint assertion
    train_hashes = set(train["sha256"])
    val_hashes = set(val["sha256"])
    test_hashes = set(test["sha256"])

    assert train_hashes.isdisjoint(val_hashes), "Leakage detected: train and val share file hashes!"
    assert train_hashes.isdisjoint(test_hashes), "Leakage detected: train and test share file hashes!"
    assert val_hashes.isdisjoint(test_hashes), "Leakage detected: val and test share file hashes!"

    return {"train": train, "validation": val, "test": test}


def compile_split_stats(splits):
    """Compile descriptive statistics for each split partition."""
    stats = {}
    for name, part in splits.items():
        stats[name] = {
            "images": int(len(part)),
            "unique_image_ids": int(part["image_id_clean"].nunique()),
            "station_date_groups": int(part["station_date_group"].nunique()),
            "unique_stations": int(part["nearest_station_id"].nunique()),
            "pm25_min": float(part["pm25_label"].min()) if len(part) > 0 else None,
            "pm25_max": float(part["pm25_label"].max()) if len(part) > 0 else None,
            "pm25_mean": float(round(part["pm25_label"].mean(), 2)) if len(part) > 0 else None,
            "pm25_std": float(round(part["pm25_label"].std(), 2)) if len(part) > 0 else None,
        }
    return stats


def save_splits(splits, target_dir):
    """Save split CSVs and display a summary table."""
    for name, part in splits.items():
        out_csv = target_dir / f"{name}.csv"
        part.to_csv(out_csv, index=False)
        # Also save val.csv alias if name is validation
        if name == "validation":
            part.to_csv(target_dir / "val.csv", index=False)

    print(f"Target directory: {target_dir}")
    print(f"{'Split':12s} | {'Images':8s} | {'Unique IDs':10s} | {'Groups':8s} | {'Stations':8s} | {'PM2.5 Mean (Std)':20s} | {'PM2.5 Range':14s}")
    print("-" * 92)
    for name, part in splits.items():
        pm_mean = part["pm25_label"].mean()
        pm_std = part["pm25_label"].std()
        pm_min = part["pm25_label"].min()
        pm_max = part["pm25_label"].max()
        print(f"{name:12s} | {len(part):8d} | {part['image_id_clean'].nunique():10d} | "
              f"{part['station_date_group'].nunique():8d} | {part['nearest_station_id'].nunique():8d} | "
              f"{pm_mean:6.1f} (+/- {pm_std:4.1f})   | [{pm_min:5.1f}, {pm_max:5.1f}]")
    print("-" * 92)
    print("Leakage check: PASSED (Disjoint groups & disjoint SHA-256 hashes across splits)")


if __name__ == "__main__":
    main()
