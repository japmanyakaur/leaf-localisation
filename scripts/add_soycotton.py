"""
Merge the SoyCotton leaf-detection dataset into data/processed.

Why this exists: PlantDoc (relabel.py) has ZERO cotton images and only 33 soy
images out of 2205 total (~1.5%). The detector has effectively never seen a
cotton leaf and has barely seen a soy leaflet cluster, so precision/recall on
those two crops is expected to be poor no matter how it's trained.

SoyCotton (Kellermann et al., Scientific Data 2026,
https://doi.org/10.6084/m9.figshare.28466636) is a purpose-built leaf-instance
dataset: 640 real field images, 7,221 soy + 5,190 cotton leaves annotated with
bounding boxes + segmentation masks, across growth stages / weed pressure /
lighting conditions. We only use the bounding boxes here. Both of its classes
('soy' and 'cotton') get collapsed to class id 0 ('leaf'), same as relabel.py
does for PlantDoc's 29 classes -- this project only ever localises "a leaf",
not the species.

The dataset is released alongside a peer-reviewed data descriptor; if you
redistribute anything derived from it, check the figshare page for the
current license/citation requirements.

Usage:
    pip install imagehash   # only needed by dedupe_splits.py, not this script
    python scripts/add_soycotton.py
    python scripts/verify.py          # confirm the new counts
    python scripts/dedupe_splits.py   # re-run after merging, see that script's docstring
"""
import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

FIGSHARE_ZIP_URL = "https://ndownloader.figshare.com/files/52692680"
RAW_DIR = Path("data/raw/soycotton")
ZIP_PATH = RAW_DIR / "SoyCotton.zip"
EXTRACT_DIR = RAW_DIR / "extracted"
OUT = Path("data/processed")

# Matches PlantDoc's existing ~70/20/10 split (1544/441/220 out of 2205).
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42
FILENAME_PREFIX = "soycotton_"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"Already downloaded: {ZIP_PATH}")
        return
    print("Downloading SoyCotton dataset (~260MB) from figshare...")
    urlretrieve(FIGSHARE_ZIP_URL, ZIP_PATH)
    print(f"Saved to {ZIP_PATH}")


def extract():
    coco_path = EXTRACT_DIR / "SoyCotton" / "annotations" / "coco.json"
    if coco_path.exists():
        print(f"Already extracted: {EXTRACT_DIR}")
        return
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO bbox is [x_min, y_min, width, height] in absolute pixels."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # clip in case any box straddles the image border
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def assign_splits(images):
    shuffled = list(images)
    random.Random(SEED).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    split_of = {}
    for i, img in enumerate(shuffled):
        if i < n_train:
            split_of[img["id"]] = "train"
        elif i < n_train + n_val:
            split_of[img["id"]] = "val"
        else:
            split_of[img["id"]] = "test"
    return split_of


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                         help="reuse an already-downloaded/extracted copy in data/raw/soycotton")
    args = parser.parse_args()

    if not args.skip_download:
        download()
    extract()

    coco_path = EXTRACT_DIR / "SoyCotton" / "annotations" / "coco.json"
    images_dir = EXTRACT_DIR / "SoyCotton" / "images"
    with open(coco_path) as f:
        coco = json.load(f)

    category_names = {c["id"]: c["name"] for c in coco["categories"]}
    boxes_by_image = {}
    species_counts_by_image = {}
    for ann in coco["annotations"]:
        boxes_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])
        species_counts_by_image.setdefault(ann["image_id"], []).append(category_names[ann["category_id"]])

    split_of = assign_splits(coco["images"])

    counts = {s: {"images": 0, "boxes": 0, "soy": 0, "cotton": 0} for s in ["train", "val", "test"]}
    for img in coco["images"]:
        split = split_of[img["id"]]
        boxes = boxes_by_image.get(img["id"], [])
        if not boxes:
            continue

        src_img = images_dir / img["file_name"]
        dst_name = FILENAME_PREFIX + img["file_name"]
        dst_img = OUT / split / "images" / dst_name
        dst_lbl = OUT / split / "labels" / (Path(dst_name).stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_img, dst_img)

        lines = []
        for bbox in boxes:
            cx, cy, w, h = coco_bbox_to_yolo(bbox, img["width"], img["height"])
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dst_lbl.write_text("\n".join(lines) + "\n")

        counts[split]["images"] += 1
        counts[split]["boxes"] += len(lines)
        for name in species_counts_by_image[img["id"]]:
            counts[split][name] += 1

    print("\nSoyCotton merged into data/processed:")
    total_images = total_boxes = 0
    for split in ["train", "val", "test"]:
        c = counts[split]
        print(f"  {split}: +{c['images']} images, +{c['boxes']} leaf boxes "
              f"(soy boxes={c['soy']}, cotton boxes={c['cotton']})")
        total_images += c["images"]
        total_boxes += c["boxes"]
    print(f"  total: +{total_images} images, +{total_boxes} leaf boxes")
    print("\nRun scripts/verify.py to see the combined dataset totals, then "
          "scripts/dedupe_splits.py before retraining.")


if __name__ == "__main__":
    main()
