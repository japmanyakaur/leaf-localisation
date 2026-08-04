# Leaf Localisation

A single-class YOLO object detector that finds and boxes individual leaves in plant images — regardless of species — for downstream tasks like leaf-area estimation.

The model doesn't care what plant it's looking at (tomato, soy, cotton, strawberry, etc.); it only answers **"where are the leaves?"**, producing one tight bounding box per visually distinct leaf.

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
- [Notes, Caveats & Design Decisions](#notes-caveats--design-decisions)
- [Citation](#citation)
- [License](#license)

## Overview

This project trains a [YOLO26](https://github.com/ultralytics/ultralytics) detector on a merged, relabeled dataset (PlantDoc + SoyCotton) collapsed down to a **single class: `leaf`**. Species/disease labels from the source datasets are intentionally discarded — the model only localises leaf instances.

Typical use case: point a camera (or a folder of images) at plants, get back one bounding box per leaf, and use the box area as a proxy for leaf size or canopy coverage.

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

> `data/`, `runs/`, `*.pt`, and `__pycache__/` are git-ignored — datasets and trained weights are not committed to source control (see [Setup](#setup) for how to (re)generate them).

## Dataset

The training set is built from two public sources, merged and relabeled to a **single class (`0 = leaf`)**:

| Source | Role | Notes |
|---|---|---|
| **PlantDoc** (Kaggle YOLO re-split, 29 species/disease classes) | Base dataset | Every class ID rewritten to `0` — see `scripts/relabel.py` |
| **SoyCotton** ([Kellermann et al., *Scientific Data*, 2026](https://doi.org/10.6084/m9.figshare.28466636)) | Fills a coverage gap | PlantDoc has **0** cotton images and only 33 soy images out of 2,205 total — SoyCotton adds 640 field images with 7,221 soy + 5,190 cotton leaf boxes across growth stages, weed pressure, and lighting conditions |

Final split (after merge + dedup, current local snapshot):

| Split | Images | Ratio |
|---|---|---|
| train | 1,544 | ~70% |
| val   | 441   | ~20% |
| test  | 220   | ~10% |

`data.yaml` points Ultralytics at `data/processed/{train,val,test}/images` with `nc: 1`, `names: ['leaf']`.

**Cross-split duplicate protection:** `scripts/dedupe_splits.py` perceptual-hashes every image and removes near-identical images that leaked across train/val/test (a known issue in PlantDoc's Kaggle re-split, which reuses generic filenames like `0.jpg` for genuinely different photos). Without this, val/test metrics would be inflated by the model having effectively "seen" its own eval examples during training. Priority when trimming: `test > val > train` — held-out sets stay untouched, train is what shrinks.

## Setup

```bash
git clone <this-repo>
cd leaf-localisation

pip install ultralytics opencv-python pillow imagehash "numpy<2"
```

There's no committed `requirements.txt` — the notebooks pin `ultralytics` and `numpy<2` explicitly on install; the scripts share those same runtime deps (`ultralytics`, `opencv-python` for the webcam demo, `Pillow` for `verify.py`, `imagehash` only for `dedupe_splits.py`).

> `ultralytics` must be a recent build — versions before YOLO26 support landed don't recognize the `yolo26*.pt` model names and fail with an unhelpful `FileNotFoundError` instead of downloading them.

## Pipeline: From Raw Data to Trained Model

Run in this order whenever rebuilding the dataset from scratch:

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

```bash
python scripts/train.py
# or override defaults:
python scripts/train.py --model yolo26m.pt --epochs 120 --imgsz 1280 --batch 6

# resume an interrupted run:
python scripts/train.py --resume runs/detect/train/weights/last.pt
```

Key defaults (already validated for this project — see `scripts/train.py` for full rationale):

| Param | Default | Why |
|---|---|---|
| `--model` | `yolo26m.pt` | Medium model — see [Results](#results) for size/accuracy tradeoff notes |
| `--imgsz` | `1280` | Leaves are small/dense; higher resolution helps recall |
| `--batch` | `6` | Fits a 15GB GPU (e.g. T4) at `imgsz=1280` |
| `--copy-paste` | `0.3` | Helps with SoyCotton's crowded, multi-leaf images |
| `--mixup` | `0.15` | Same reason |
| `--seed` | `42` | Reproducible runs, so metric changes reflect data/hyperparameter changes, not run-to-run noise |

> **Don't** enable `multi_scale=True` or `autobatch` (`batch=-1`) together with `imgsz=1280` on a 15GB GPU — autobatch profiles memory assuming multi-scale's largest possible input (effectively 2560px), which OOMs before training even starts. If you have more VRAM, raise `--batch` instead of turning these on.

Two notebooks cover cloud and local training end-to-end (mounting data, installing deps, training YOLO26m, validating, and optionally comparing against YOLO26s):

- **`notebooks/train_colab.ipynb`** — for Google Colab. Expects `data/processed` pre-zipped and uploaded to Drive (build it locally first with the pipeline above).
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

Draws detected boxes and per-leaf pixel area on the live feed; press `q` to quit. Uses `cv2.CAP_DSHOW`, so this script currently targets **Windows**.

## Results

Metrics from the most recent full training runs (`runs/detect/train/` and `runs/detect/train-3/`, both YOLO26m @ `imgsz=1280`), evaluated on the validation split:

| Run | Epochs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `train` (120 ep) | 120 | 0.688 | 0.721 | 0.741 | 0.487 |
| `train-3` (100 ep) | 100 | 0.674 | 0.728 | 0.749 | 0.495 |

Full curves (`BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`, etc.) and per-epoch logs (`results.csv`) live inside each `runs/detect/<run>/` folder. The `runs/detect/val*/` folders hold standalone validation-only passes (e.g. re-checking a checkpoint against test data).

> Scripts default to `runs/detect/train/weights/best.pt`. If you're using weights from a different run (e.g. `train-3`), pass `--weights runs/detect/train-3/weights/best.pt` explicitly.

## Annotation Rules

Full rules in [`ANNOTATION_GUIDELINES.md`](./ANNOTATION_GUIDELINES.md). Summary:

- One box per **visually distinct leaf**.
- Compound leaves (strawberry, blueberry, clover, etc.): **one box for the whole compound leaf**, not one per leaflet.
- Only box a leaf if **≥ ~20% is visible**; skip anything more occluded than that.
- Boxes should be **tight** around the visible leaf area — no background padding, no stem unless unavoidable.
- Dense/overlapping clusters: still **one box per individual leaf**, even where leaves touch — don't merge them.

## Notes, Caveats & Design Decisions

- **Area is box-approximated, not pixel-exact.** A real leaf isn't a rectangle, so `width_px × height_px` overstates true leaf area. This tradeoff was accepted for simplicity and speed — see `scripts/infer_area.py`.
- **Species information is deliberately discarded.** Both PlantDoc's 29 classes and SoyCotton's `soy`/`cotton` classes collapse to a single `leaf` class — this project only ever localises leaves, not identifies them.
- **Duplicate-safe evaluation.** Always re-run `scripts/dedupe_splits.py` after adding new data sources, before retraining — otherwise val/test metrics can be silently inflated by train/eval leakage.
- **File copies over symlinks** in `relabel.py` — deliberate, since Windows restricts symlink creation to admin users by default, and this needs to work cross-platform without elevated permissions.
- **No `requirements.txt` / pinned environment file** currently exists in the repo — dependencies are declared ad hoc in each script's usage docstring and the notebooks' install cells.

## Citation

If you use the SoyCotton portion of this dataset, cite the original data descriptor:

> Kellermann et al., *Scientific Data* (2026). SoyCotton dataset. https://doi.org/10.6084/m9.figshare.28466636

Check the figshare page for current license/redistribution terms before sharing anything derived from it.

## License

No license file is currently included in this repository. Add one (e.g. MIT, Apache-2.0) before distributing or open-sourcing this project more broadly.
