"""
Train the single-class leaf detector on data/processed (PlantDoc + SoyCotton,
see scripts/relabel.py and scripts/add_soycotton.py).

Defaults here match what's already been validated for this project (yolo26m,
imgsz=1280 for small/dense leaf instances, copy_paste + mixup for the crowded
multi-leaf images SoyCotton adds), plus:

  - seed=42: reproducible runs, so P/R changes between experiments reflect
    the data/hyperparameter change and not run-to-run noise.
  - Run scripts/add_soycotton.py and scripts/dedupe_splits.py BEFORE this --
    training on leaked eval data or on a dataset with 0 cotton / 33 soy
    images will not fix itself by changing model size or hyperparameters.

Deliberately NOT using multi_scale=True or autobatch (batch=-1): autobatch
profiles memory at 2x imgsz to size for multi_scale's largest possible input,
which on a 15GB GPU at imgsz=1280 tries to test a 2560px batch and OOMs before
training starts -- confirmed by actually hitting this on a T4. If you have
more VRAM to spare, raise --batch rather than turning either of those on.

Usage:
    python scripts/train.py
    python scripts/train.py --model yolo26m.pt --epochs 120 --imgsz 1280 --batch 6
    python scripts/train.py --resume runs/detect/train/weights/last.pt
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--copy-paste", type=float, default=0.3)
    parser.add_argument("--mixup", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None,
                         help="path to a last.pt to resume an interrupted run")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save_period=args.save_period,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        seed=args.seed,
    )

    metrics = model.val(data=args.data)
    print(f"\nPrecision  : {metrics.box.mp:.3f}")
    print(f"Recall     : {metrics.box.mr:.3f}")
    print(f"mAP50      : {metrics.box.map50:.3f}")
    print(f"mAP50-95   : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
