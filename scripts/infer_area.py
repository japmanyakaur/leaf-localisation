"""
Run the trained leaf detector over a folder of images and write out, for every
detected leaf: which image it's in, its box coordinates, confidence, and its
area in pixels (width_px * height_px of the box).

This gives box-approximated area, not pixel-exact area (a real leaf isn't a
perfect rectangle) -- see README for why that tradeoff was accepted.

Usage:
    python scripts/infer_area.py --source data/processed/test/images --weights runs/detect/train/weights/best.pt
"""
import argparse
import csv

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt",
                         help="path to trained model weights (best.pt)")
    parser.add_argument("--source", required=True,
                         help="folder of images to run detection on")
    parser.add_argument("--out", default="leaf_areas.csv",
                         help="path to write the results CSV")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="minimum confidence to count a detection")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(args.source, conf=args.conf, verbose=False, stream=True)

    rows = []
    for result in results:
        image_name = result.path
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width_px = x2 - x1
            height_px = y2 - y1
            rows.append({
                "image": image_name,
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "confidence": round(float(box.conf[0]), 4),
                "area_px": round(width_px * height_px, 1),
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "x1", "y1", "x2", "y2", "confidence", "area_px"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} detected leaf boxes to {args.out}")


if __name__ == "__main__":
    main()
