"""
Sanity-check the processed dataset before we ever touch a GPU:
  1. Every image has a matching label file, and vice versa.
  2. Count images/boxes per split.
  3. Draw one sample image's boxes and save it as a PNG so we can visually
     confirm the YOLO coordinates are being read/decoded correctly.
"""
import os
from PIL import Image, ImageDraw

ROOT = "data/processed"
SPLITS = ["train", "val", "test"]
SAMPLE_OUT = "data/processed/sample_check.png"


def check_split(split):
    images_dir = os.path.join(ROOT, split, "images")
    labels_dir = os.path.join(ROOT, split, "labels")

    image_names = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
    label_names = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    total_boxes = 0
    empty_labels = 0
    for name in label_names:
        with open(os.path.join(labels_dir, name + ".txt")) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        total_boxes += len(lines)
        if not lines:
            empty_labels += 1

    print(f"\n[{split}] images={len(image_names)} labels={len(label_names)} "
          f"boxes={total_boxes} empty_labels={empty_labels}")
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} images have no label file, e.g. {list(missing_labels)[:3]}")
    if missing_images:
        print(f"  WARNING: {len(missing_images)} labels have no matching image, e.g. {list(missing_images)[:3]}")

    return image_names, labels_dir, images_dir


def draw_sample(images_dir, labels_dir, image_names):
    for name in sorted(image_names):
        label_path = os.path.join(labels_dir, name + ".txt")
        with open(label_path) as f:
            lines = [l.strip().split() for l in f.read().splitlines() if l.strip()]
        if not lines:
            continue

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = os.path.join(images_dir, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for parts in lines:
            _, xc, yc, bw, bh = parts
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        img.save(SAMPLE_OUT)
        print(f"\nSample check image saved: {SAMPLE_OUT} (from {name}, {len(lines)} box(es) drawn)")
        return

    print("\nNo image with non-empty labels found to sample.")


if __name__ == "__main__":
    last_images_dir = last_labels_dir = last_image_names = None
    for split in SPLITS:
        image_names, labels_dir, images_dir = check_split(split)
        if split == "train":
            last_images_dir, last_labels_dir, last_image_names = images_dir, labels_dir, image_names

    draw_sample(last_images_dir, last_labels_dir, last_image_names)
