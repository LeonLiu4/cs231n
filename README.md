# Pose-Aware Multi-View Fusion

Using pose information for enhanced 3D object reconstruction.

## Baseline Setup

Single-view reconstruction baseline for the CS231n final project: **frozen DINOv2 + point cloud head**, trained with Chamfer distance on ShapeNet renders.

## Quick start (no ShapeNet yet)

```bash
cd /Users/leonliu/Desktop/cs231n
python -m pip install -r requirements.txt
python scripts/setup_baseline.py
```

This installs deps and runs a 1-epoch **demo** smoke test on synthetic meshes.

## Full ShapeNet pipeline

### 1. Get ShapeNet data

**Option A — you already have `archive.zip` (PartAnnotation)**

Extract to `data/PartAnnotation/` (already done if you ran setup):

```bash
unzip ~/Downloads/archive.zip -d data/
python scripts/prepare_partannotation.py --max-samples 200 --num-views 1
```

This uses `.pts` files as GT point clouds and `expert_verified/seg_img/` PNGs as single-view RGB inputs.

**Option B — ShapeNetCore.v2 (full mesh rendering)**

1. Register at [shapenet.org](https://shapenet.org/)
2. Download **ShapeNetCore.v2**
3. Extract to `data/ShapeNetCore.v2/` or pass your zip:

```bash
python scripts/download_shapenet.py --zip-path /path/to/ShapeNetCore.v2.zip
python scripts/download_shapenet.py   # verify installation
```

Default category in config: **chair** (`03001627`).

### 2. Pre-render views + GT point clouds

**PartAnnotation** (from archive.zip):

```bash
python scripts/prepare_partannotation.py --num-views 1 --max-samples 500
```

**ShapeNetCore** (mesh-based RGB rendering):

```bash
python scripts/prepare_dataset.py --num-views 8 --max-samples 200
```

Outputs go to `data/processed/` with train/val manifests. For the **single-view baseline**, training uses only view 0 (`num_views: 1` in config).

### 3. Train baseline

**Single view (original):**
```bash
python scripts/train_baseline.py --config configs/baseline_single_view.yaml
```

**2-view mean-pool baseline (improved training):**
```bash
# view 0 = seg_img, view 1 = second-angle point render (same 3D frame as GT)
python scripts/prepare_partannotation.py --num-views 2 --view-mode seg_first \
  --processed-dir data/processed_2view --max-samples 500
python scripts/train_baseline.py --config configs/baseline_2view.yaml
```

Improvements in `baseline_2view.yaml`:
- FoldingNet-style decoder (instead of flat MLP)
- CLS + mean patch token features
- Finetune last 2 DINOv2 blocks
- Color jitter augmentation
- ReduceLROnPlateau + gradient clipping

### 4. Evaluate

```bash
python scripts/eval_baseline.py --checkpoint runs/single_view_baseline/best.pt
```

### 5. Visualize predictions

```bash
python scripts/visualize_baseline.py \
  --config configs/baseline_2view.yaml \
  --checkpoint runs/baseline_2view/best.pt \
  --num-samples 5
```

PNG files go to `runs/visualizations/` (GT | Pred | Overlay side-by-side).

## Project layout

```
configs/baseline_single_view.yaml   # hyperparameters from your slides
scripts/
  setup_baseline.py                 # one-command setup
  download_shapenet.py              # ShapeNet instructions + verify
  prepare_dataset.py              # render RGB + sample GT points
  train_baseline.py                 # train single-view baseline
  eval_baseline.py                  # val Chamfer + F-score
  visualize_baseline.py             # save GT/pred point cloud PNGs
src/
  data/                             # camera, rendering, dataset
  models/                           # DINOv2 encoder + point cloud head
  losses/chamfer.py
  metrics/reconstruction.py
```

## Baseline spec (from slides)

| Component | Setting |
|-----------|---------|
| Input | 1 RGB view (224×224) |
| Encoder | Frozen DINOv2-ViT-B/14 |
| Decoder | MLP point cloud head → 2048 points |
| Loss | Chamfer L2 |
| Optimizer | AdamW, lr=1e-4, cosine schedule |
| Metrics | Chamfer + F-score @ 1% bbox |

## Notes

- **MPS/CUDA**: Training auto-detects Apple GPU or CUDA.
- **Multi-view later**: `prepare_dataset.py` renders 8 views; increase `data.num_views` in config for mean-pooling / transformer baselines.
- **First run** downloads DINOv2 weights via `torch.hub` (~330 MB).
