"""
Find (and optionally remove) near-duplicate images that leaked across
train/val/test.

Why: a naive check based on filenames alone (e.g. same normalized stem in two
splits) produces mostly false positives here -- PlantDoc's Kaggle re-split
reused generic stock-photo names like "0.jpg" or "bacterial_leaf_spot.jpg" for
genuinely different photos. What actually matters is image *content*. This
script perceptual-hashes every image (average hash + phash) and flags
cross-split pairs that are near-identical by both metrics -- the same
combination/threshold used to manually verify real duplicates in this
dataset (e.g. train_irish-blight-symptom.jpg / test_irish-blight-symptom.jpg
came back as an exact hash match and are in fact the same photo).

If the same photo sits in both train and val/test, the model can effectively
"see" its eval examples during training, which inflates val/test precision,
recall and mAP without the model actually being better. That makes it
impossible to tell whether a change (like adding SoyCotton data, or upsizing
to yolo26m) genuinely helped.

Keep priority when a duplicate group spans multiple splits: test > val > train.
Held-out sets should stay pristine; train is the one it's cheapest to shrink.

Usage:
    pip install imagehash
    python scripts/dedupe_splits.py                 # dry run, just reports

"""
import argparse
from pathlib import Path

from PIL import Image
import imagehash

ROOT = Path("data/processed")
SPLITS = ["train", "val", "test"]
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = kept first

AHASH_MAX_DIFF = 2
PHASH_MAX_DIFF = 6


def load_all_images():
    records = []
    for split in SPLITS:
        for path in sorted((ROOT / split / "images").iterdir()):
            try:
                img = Image.open(path)
                ahash = imagehash.average_hash(img)
                phash = imagehash.phash(img)
            except Exception as e:
                print(f"  WARNING: could not hash {path}: {e}")
                continue
            records.append({"split": split, "path": path, "ahash": ahash, "phash": phash})
    return records


def find_dupe_groups(records):
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if records[i]["split"] == records[j]["split"]:
                continue
            da = records[i]["ahash"] - records[j]["ahash"]
            if da > AHASH_MAX_DIFF:
                continue
            dp = records[i]["phash"] - records[j]["phash"]
            if dp <= PHASH_MAX_DIFF:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(records[i])

    return [g for g in groups.values() if len({r["split"] for r in g}) > 1]


def label_path_for(image_path):
    split_dir = image_path.parent.parent
    return split_dir / "labels" / (image_path.stem + ".txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="actually delete the losing image/label pairs (default: dry run)")
    args = parser.parse_args()

    print("Hashing all images in data/processed (this takes a while)...")
    records = load_all_images()
    print(f"Hashed {len(records)} images.\n")

    groups = find_dupe_groups(records)
    print(f"Found {len(groups)} cross-split duplicate group(s).\n")

    to_delete = []
    for group in groups:
        group.sort(key=lambda r: SPLIT_PRIORITY[r["split"]])
        keep = group[0]
        losers = group[1:]
        print(f"KEEP {keep['split']}/{keep['path'].name}")
        for loser in losers:
            print(f"  DROP {loser['split']}/{loser['path'].name}")
            to_delete.append(loser["path"])
        print()

    print(f"Total images to remove: {len(to_delete)}")

    if not args.apply:
        print("\nDry run only -- nothing deleted. Re-run with --apply to remove the losers above.")
        return

    for img_path in to_delete:
        lbl_path = label_path_for(img_path)
        img_path.unlink(missing_ok=True)
        lbl_path.unlink(missing_ok=True)
    print(f"Deleted {len(to_delete)} image/label pairs.")


if __name__ == "__main__":
    main()
