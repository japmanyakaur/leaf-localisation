"""
Find near-duplicate images that leaked across
train/val/test.
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
