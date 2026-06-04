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

Uses CPU software rendering (no OpenGL/pyrender needed):

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

## Project layout

```
configs/
  baseline_single_view.yaml      # 1-view, ShapeNet renders
  baseline_2view.yaml            # 2-view concat fusion
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
| `runs/` | Training checkpoints |

Old PartAnnotation data (`data/PartAnnotation/`, `data/processed/`) is no longer used and can be deleted.

## Baseline spec

| Component | Setting |
|-----------|---------|
| Encoder | Frozen DINOv2-ViT-B/14 |
| Decoder | MLP point cloud head → 2048 points |
| Loss | Squared Chamfer L2 |
| Metrics | Chamfer + F-score @ 1% bbox |
