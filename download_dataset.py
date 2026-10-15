from pathlib import Path
import json
import re

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET = "ytaek-oh/cc3m-subset-100k"

OUTPUT_DIR = Path("data")
IMAGE_DIR = OUTPUT_DIR / "images"
CAPTIONS_FILE = OUTPUT_DIR / "captions.jsonl"

TARGET_IMAGES = 100_000


# ============================================================
# CREATE DIRECTORIES
# ============================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("Loading CC3M 100K...")
print("=" * 60)

dataset = load_dataset(
    DATASET,
    split="train"
)

print(f"Dataset loaded: {len(dataset):,} rows")


# ============================================================
# RESUME SUPPORT
# ============================================================

existing = set()

if CAPTIONS_FILE.exists():

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:

        for line in f:

            try:
                row = json.loads(line)
                existing.add(row["file_name"])

            except Exception:
                pass


print(f"Already downloaded: {len(existing):,}")


# ============================================================
# DOWNLOAD
# ============================================================

count = len(existing)

with open(
    CAPTIONS_FILE,
    "a",
    encoding="utf-8"
) as captions_file:

    for index, row in enumerate(
        tqdm(dataset, desc="Downloading images")
    ):

        if count >= TARGET_IMAGES:
            break

        filename = f"{index:09d}.jpg"

        if filename in existing:
            continue

        try:

            image = row["jpg"]

            caption = row["txt"]

            if image is None:
                continue

            if caption is None:
                continue

            caption = str(caption).strip()

            if len(caption) < 3:
                continue


            # ------------------------------------------------
            # Convert image to RGB
            # ------------------------------------------------

            image = image.convert("RGB")


            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            image_path = IMAGE_DIR / filename

            image.save(
                image_path,
                format="JPEG",
                quality=95
            )


            # ------------------------------------------------
            # Save caption
            # ------------------------------------------------

            record = {
                "file_name": filename,
                "text": caption
            }

            captions_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            captions_file.flush()

            count += 1


        except Exception as e:

            print(
                f"\nSkipping {index}: {e}"
            )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Images:   {count:,}")
print(f"Captions: {CAPTIONS_FILE}")
print(f"Images:   {IMAGE_DIR}")