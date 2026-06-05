# Pose-Aware Multi-View Fusion

Using pose information for enhanced 3D object reconstruction.

## Baseline Setup

Single/multi-view reconstruction baselines: **frozen DINOv2 + point cloud head**, trained with Chamfer distance on **ShapeNetCore mesh renders** (natural RGB, aligned with GT).

## Quick start

```bash
cd /Users/leonliu/Desktop/cs231n
python -m pip install -r requirements.txt
```

## ShapeNet pipeline

### 1. Download ShapeNetCore (Hugging Face)

1. Accept access at [ShapeNet/ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
2. Log in with a **classic Read** token (must allow gated repos):

```bash
hf auth login
python scripts/download_shapenet_hf.py --categories 03001627
```

Meshes extract to `data/ShapeNetCore.v2/03001627/`.

### 2. Pre-render views + GT point clouds

Uses CPU software rendering (no OpenGL/pyrender needed). Meshes are centered and scaled to unit radius; the camera is placed to fit the full object in frame (60° FOV).

**Important:** If you prepared data before the camera fix, delete `data/processed_shapenet_*` and re-run — older renders were heavily cropped.

```bash
# 1-view dataset
python scripts/prepare_dataset.py --num-views 1 --max-samples 500 \
  --processed-dir data/processed_shapenet_1view

# 2-view dataset
python scripts/prepare_dataset.py --num-views 2 --max-samples 500 \
  --processed-dir data/processed_shapenet_2view
```

Or run both via:

```bash
python scripts/setup_baseline.py
```

### 3. Train baselines

```bash
python scripts/train_baseline.py --config configs/baseline_single_view.yaml
python scripts/train_baseline.py --config configs/baseline_2view.yaml
```

### 4. Evaluate

```bash
python scripts/eval_baseline.py --config configs/baseline_single_view.yaml \
  --checkpoint runs/single_view_baseline/best.pt
```

### 5. Visualize predictions

```bash
python scripts/visualize_baseline.py --preset 1view --sample-idx 0 --layout compact
python scripts/visualize_baseline.py --preset 2view --sample-idx 0 --layout dual-angle
```

## Run on Modal (cloud GPU)

Run the full pipeline on [Modal](https://modal.com) with a persistent volume for data and checkpoints.

### One-time setup

```bash
pip install -r requirements-modal.txt
modal setup

# Hugging Face token for ShapeNet download (accept gated access first)
modal secret create huggingface HF_TOKEN=hf_...
```

### Commands

```bash
# Download ShapeNet chairs to Modal volume
modal run -m modal_app.pipeline --command download

# Prepare rendered views (CPU)
modal run -m modal_app.pipeline --command prepare --num-views 1 --max-samples 500

# Train 1-view baseline on A10G GPU
modal run -m modal_app.pipeline --command train --config single_view

# Train 2-view baseline
modal run -m modal_app.pipeline --command train --config 2view

# 2-view silhouette (recommended; stronger than RGB single-view)
modal run -m modal_app.pipeline --command all --max-samples 500 --config 2view_silhouette --skip-download

# Eval best checkpoint
modal run -m modal_app.pipeline --command eval --config single_view

# Full pipeline: download + prepare + train
modal run -m modal_app.pipeline --command all --max-samples 500 --config single_view

# Quick smoke test (synthetic demo data, 1 epoch)
modal run -m modal_app.pipeline --command demo --max-epochs 1
```

### Pull checkpoints locally

```bash
modal volume get cs231n-data runs/single_view_baseline/best.pt runs/single_view_baseline/best.pt
modal volume get cs231n-data runs/single_view_baseline/history.json runs/single_view_baseline/history.json
```

Modal configs live in `configs/modal_single_view.yaml` and `configs/modal_2view.yaml` (paths under `/vol/` on the volume).

## Project layout

```
configs/
  baseline_single_view.yaml      # local 1-view
  baseline_2view.yaml            # local 2-view
  modal_single_view.yaml         # Modal volume paths
  modal_2view.yaml
modal_app/
  pipeline.py                    # Modal entrypoint (modal run -m modal_app.pipeline)
scripts/
  download_shapenet_hf.py        # download from Hugging Face
  prepare_dataset.py             # mesh -> RGB views + GT points
  train_baseline.py
  eval_baseline.py
  visualize_baseline.py
src/
  data/render.py                 # CPU mesh rasterizer
  models/                        # DINOv2 + point cloud head
```

## Data directories

| Path | Purpose |
|------|---------|
| `data/ShapeNetCore.v2/` | Raw meshes from Hugging Face |
| `data/processed_shapenet_1view/` | 1 RGB view + GT points per object |
| `data/processed_shapenet_2view/` | 2 RGB views + GT points per object |
| `data/processed_shapenet_2view_silhouette/` | 2 silhouette views + GT points |
| `runs/` | Training checkpoints |

Old PartAnnotation data (`data/PartAnnotation/`, `data/processed/`) is no longer used and can be deleted.

## Baseline spec

| Component | Setting |
|-----------|---------|
| Encoder | Frozen DINOv2-ViT-B/14 |
| Decoder | FoldingNet-style head → 2048 points (tanh-bounded to unit ball) |
| Features | CLS token + mean patch tokens from frozen DINOv2 |
| Loss | Squared Chamfer L2 |
| Metrics | Chamfer + F-score @ 1% GT bbox diagonal |
