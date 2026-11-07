
from pathlib import Path
from io import BytesIO
import time

import pandas as pd
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError

DATASET_ID = "DeadCardassian/PM25Vision"

CANDIDATES = Path(
    "data/processed/pm25vision_historical_matches.csv"
)
IMAGE_ROOT = Path("data/raw/pm25vision/images")
MANIFEST = Path(
    "data/processed/pm25vision_image_manifest.csv"
)

IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)


def normalize_id(value):
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError, OverflowError):
        return str(value).strip()


def image_to_bytes(value):
    """Support byte data, image dictionaries and PIL images."""
    if isinstance(value, bytes):
        return value

    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw

        # Some image features provide a local path instead.
        path = value.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()

    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value.convert("RGB").save(buffer, format="JPEG")
        return buffer.getvalue()

    raise ValueError(
        f"Unsupported image representation: {type(value).__name__}"
    )


def validate_image(path):
    with Image.open(path) as img:
        img.verify()

    with Image.open(path) as img:
        return img.format, img.width, img.height


def main():
    if not CANDIDATES.exists():
        raise FileNotFoundError(CANDIDATES)

    candidates = pd.read_csv(CANDIDATES)
    candidates["image_id_key"] = (
        candidates["image_id"].map(normalize_id)
    )

    # Preserve the original source split and ID.
    wanted = {}
    metadata = {}

    for _, row in candidates.iterrows():
        split = str(row["split"])
        image_id = row["image_id_key"]

        wanted.setdefault(split, set()).add(image_id)
        metadata[(split, image_id)] = row.to_dict()

    # Load previous progress so a rerun can skip valid files.
    if MANIFEST.exists():
        old = pd.read_csv(MANIFEST).to_dict("records")
    else:
        old = []

    manifest = {
        (str(row["split"]), normalize_id(row["image_id"])): row
        for row in old
    }

    def save_manifest():
        pd.DataFrame(manifest.values()).to_csv(
            MANIFEST, index=False
        )

    for split in ["train", "test"]:
        remaining = wanted.get(split, set())

        if not remaining:
            print(f"\nNo candidates for {split}.")
            continue

        # Verify previously downloaded files before skipping them.
        for image_id in list(remaining):
            previous = manifest.get((split, image_id))

            if previous and previous.get("status") in (
                "downloaded", "already_valid"
            ):
                path_value = previous.get("local_path")
                if path_value and Path(path_value).exists():
                    try:
                        validate_image(Path(path_value))
                        remaining.remove(image_id)
                    except Exception:
                        pass

        print(f"\nScanning {split} source split")
        print("Candidate images still needed:", len(remaining))

        if not remaining:
            print("All candidate images already verified.")
            continue

        count = 0
        found = 0
        last_report = time.time()

        try:
            stream = load_dataset(
                DATASET_ID,
                split=split,
                streaming=True,
            )

            for row in stream:
                count += 1
                image_id = normalize_id(row.get("image_id"))

                if image_id in remaining:
                    key = (split, image_id)
                    info = metadata.get(key, {})

                    filename = Path(
                        str(info.get("filename") or f"{image_id}.jpg")
                    ).name

                    if Path(filename).suffix.lower() not in (
                        ".jpg", ".jpeg", ".png", ".webp"
                    ):
                        filename += ".jpg"

                    destination = (
                        IMAGE_ROOT / split / filename
                    )
                    destination.parent.mkdir(
                        parents=True, exist_ok=True
                    )

                    try:
                        raw = image_to_bytes(row.get("image"))
                        destination.write_bytes(raw)

                        fmt, width, height = validate_image(
                            destination
                        )

                        manifest[key] = {
                            "split": split,
                            "image_id": image_id,
                            "date": row.get("captured_at"),
                            "pm25": row.get("pm25"),
                            "latitude": row.get("latitude"),
                            "longitude": row.get("longitude"),
                            "local_path": str(destination),
                            "status": "downloaded",
                            "format": fmt,
                            "width": width,
                            "height": height,
                            "file_size_bytes": len(raw),
                            "error": "",
                        }

                        remaining.remove(image_id)
                        found += 1

                    except Exception as exc:
                        if destination.exists():
                            destination.unlink()

                        manifest[key] = {
                            "split": split,
                            "image_id": image_id,
                            "local_path": str(destination),
                            "status": "failed",
                            "error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }

                    save_manifest()

                    print(
                        f"{split}: {image_id}; "
                        f"verified={found}; "
                        f"remaining={len(remaining)}"
                    )

                    if not remaining:
                        print("All candidates found in this split.")
                        break

                if time.time() - last_report >= 30:
                    print(
                        f"Scanned {count} records in {split}; "
                        f"found {found}; "
                        f"remaining {len(remaining)}"
                    )
                    last_report = time.time()

        except Exception as exc:
            print(
                f"\nStreaming failed for {split}: "
                f"{type(exc).__name__}: {exc}"
            )
            print("Progress has been saved where possible.")

        save_manifest()

        if remaining:
            print(
                f"{len(remaining)} candidate IDs were not found "
                f"or could not be downloaded in {split}."
            )

    save_manifest()

    print("\n========== FINAL REPORT ==========")
    print("Manifest:", MANIFEST)
    print("Image directory:", IMAGE_ROOT)

    if manifest:
        report = pd.DataFrame(manifest.values())
        print("\nStatuses:")
        print(report["status"].value_counts().to_string())
        print(
            "\nVerified images:",
            int(report["status"].isin(
                ["downloaded", "already_valid"]
            ).sum()),
        )
    else:
        print("No image records have been saved.")

    print("==================================")


if __name__ == "__main__":
    main()
