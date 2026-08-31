# Leaf Localisation

A single-class YOLO object detector that finds and boxes individual leaves in plant images tasks like leaf-area estimation.

The model answers **"where are the leaves?"**, producing one tight bounding box per visually distinct leaf.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Pipeline: From Raw Data to Trained Model](#pipeline-from-raw-data-to-trained-model)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Annotation Rules](#annotation-rules)

## Overview

This project trains a [YOLO26](https://github.com/ultralytics/ultralytics) detector on a merged, relabeled dataset (PlantDoc + SoyCotton) collapsed down to a **single class: `leaf`**. 

Typical use case: point a camera at plants, get back one bounding box per leaf, and use the box area as a proxy for leaf size or canopy coverage.

## Project Structure

```
leaf-localisation/
├── data.yaml                    # YOLO dataset config (paths, single class 'leaf')
├── ANNOTATION_GUIDELINES.md     # Rules used when labeling/reviewing leaf boxes
├── yolo26n.pt / yolo26m.pt      # Pretrained YOLO26 base weights (nano / medium)
│
├── data/
│   ├── raw/                     # Untouched source data (yolo_dataset, soycotton/)
│   └── processed/               # Final train/val/test splits used for training
│       ├── train/images|labels
│       ├── val/images|labels
│       └── test/images|labels
│
├── scripts/
│   ├── relabel.py                # PlantDoc: collapse 29 classes -> single 'leaf' class
│   ├── add_soycotton.py          # Merge SoyCotton dataset in (COCO -> YOLO, class -> 'leaf')
│   ├── dedupe_splits.py          # Detect/remove near-duplicate images leaked across splits
│   ├── verify.py                 # Sanity-check processed dataset before training
│   ├── train.py                  # Train the detector
│   ├── infer_area.py             # Batch inference: boxes + area (px²) -> CSV
│   └── webcam_demo.py            # Live webcam demo with on-screen leaf area
│
├── notebooks/
│   ├── train_colab.ipynb         # Google Colab training pipeline (GPU-in-the-cloud)
│   └── train_local(1).ipynb      # Local/Jupyter training pipeline
│
└── runs/detect/                  # Ultralytics training/validation run outputs
    ├── train/, train-2/, train-3/    # Training runs (weights, curves, results.csv)
    └── val/, val-2/ ... val-19/      # Standalone validation runs
```

> `data/`, `runs/`, `*.pt`, and `__pycache__/` are git-ignored — datasets and trained weights are not committed to source control 

## Dataset

The training set is built from 2 public sources, merged and relabeled to a **single class (`0 = leaf`)**:

| Source | Role | Notes |
|---|---|---|
| **PlantDoc** (Kaggle YOLO re-split, 29 species/disease classes) | Base dataset | Every class ID rewritten to `0` —  `scripts/relabel.py` |
| **SoyCotton** ([Kellermann et al., *Scientific Data*, 2026](https://doi.org/10.6084/m9.figshare.28466636)) | Fills a coverage gap | PlantDoc has 33 soy images out of 2,205 total — SoyCotton adds 640 field images with 7,221 soy + 5,190 cotton leaf boxes across growth stages, weed pressure, and lighting conditions |

Final split :

| Split | Images | Ratio |
|---|---|---|
| train | 1,544 | ~70% |
| val   | 441   | ~20% |
| test  | 220   | ~10% |

`data.yaml` points Ultralytics at `data/processed/{train,val,test}/images` with `nc: 1`, `names: ['leaf']`.



## Setup

```bash
git clone <this-repo>
cd leaf-localisation

pip install ultralytics opencv-python pillow imagehash "numpy<2"
```


## Pipeline: From Raw Data to Trained Model

```bash
# 1. Collapse PlantDoc's 29 classes into a single 'leaf' class, copy into data/processed/
python scripts/relabel.py

# 2. Merge in the SoyCotton dataset (downloads ~260MB from figshare, converts COCO -> YOLO)
python scripts/add_soycotton.py

# 3. Remove images that leaked across train/val/test as near-duplicates
python scripts/dedupe_splits.py --apply

# 4. Sanity-check the result before touching a GPU
python scripts/verify.py
```

`verify.py` checks that every image has a matching label file (and vice versa), reports box counts per split, and saves an annotated sample (`data/processed/sample_check.png`) so you can visually confirm the YOLO coordinates decode correctly.

## Training

| Param | Default | Why |
|---|---|---|
| `--model` | `yolo26m.pt` | Medium model — see [Results](#results) for size/accuracy tradeoff notes |
| `--imgsz` | `1280` | Leaves are small/dense; higher resolution helps recall |
| `--batch` | `6` | Fits a 15GB GPU (e.g. T4) at `imgsz=1280` |
| `--copy-paste` | `0.3` | Helps with SoyCotton's crowded, multi-leaf images |
| `--mixup` | `0.15` | Same reason |
| `--seed` | `42` | Reproducible runs, so metric changes reflect data/hyperparameter changes, not run-to-run noise |


Two notebooks cover cloud and local training end-to-end (mounting data, installing deps, training YOLO26m, validating, and optionally comparing against YOLO26s):

- **`notebooks/train_colab.ipynb`** — for Google Colab. Expects `data/processed` pre-zipped and uploaded to Drive 
- **`notebooks/train_local(1).ipynb`** — for a local/Jupyter GPU environment; auto-locates the repo root from wherever the notebook is opened.

## Inference

**Batch inference over a folder → CSV of boxes + area:**

```bash
python scripts/infer_area.py \
  --source data/processed/test/images \
  --weights runs/detect/train/weights/best.pt \
  --out leaf_areas.csv
```

Outputs, per detected leaf: image name, box coordinates, confidence, and **area in pixels** (`width_px × height_px` of the box — a box-approximated area, not a pixel-exact leaf outline).

**Live webcam demo:**

```bash
python scripts/webcam_demo.py --weights runs/detect/train/weights/best.pt --camera 0
```

Draws detected boxes and per-leaf pixel area on the live feed; press `q` to quit. Cross-platform (Windows/Linux/Raspberry Pi) — uses OpenCV's default camera backend; pass `--camera` to select a different camera index if `0` isn't your device.

## Results

Metrics from the most recent full training runs (`runs/detect/train/` and `runs/detect/train-3/`, both YOLO26m @ `imgsz=1280`), evaluated on the validation split:

| Model | Confidence | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| yolo26m | 0.25 (default) | 0.818| 0.752 | 0.845 | 0.685 |
| yolo26m 	 | 0.65 | 0.902 | 0.654 | - | - |

## Annotation Rules

Full rules in [`ANNOTATION_GUIDELINES.md`](./ANNOTATION_GUIDELINES.md).

Summary:

- One box per **visually distinct leaf**.
- Compound leaves (strawberry, blueberry, clover, etc.): **one box for the whole compound leaf**, not one per leaflet.
- Only box a leaf if **≥ ~20% is visible**; skip anything more occluded than that.
- Boxes should be **tight** around the visible leaf area — no background padding, no stem unless unavoidable.
- Dense/overlapping clusters: still **one box per individual leaf**, even where leaves touch — don't merge them.

