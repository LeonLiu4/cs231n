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

## Master training dataset (5000 objects)

Controlled ShapeNetCore subset for training and evaluation:

| Group | Objects | Split |
|-------|---------|-------|
| 7 main categories (500 each) | 3,500 | train 2,450 / val 525 / test_seen 525 (70/15/15 per category) |
| 8th generalization group (broad) | 1,500 | all `test_generalization` |

Categories: chair, table, sofa, car, airplane, lamp, cabinet. Generalization = mixed ShapeNet synsets outside those seven.

**Static on disk (built once from ShapeNet meshes):**

| Asset | Who gets it | Purpose |
|-------|-------------|---------|
| `gt_points.npy` (2048 pts) | all 5000 | supervision |
| `view_00..63.png` (bin centers) | all 5000 | **fixed eval** views |
| `train_views/az##_el##_j##.png` | **train split only** (2450) | jittered renders **per bin** (2 per bin × 64) |

**Nested view regions:** 1 ⊂ 2 ⊂ 4 ⊂ 8 ⊂ 16 ⊂ 64 (same bins, more views at each step).  
**Training:** nested K bins + random jitter pick from `train_views/` per bin.  
**Eval / val / test:** same nested K bins, **fixed** bin-center `view_XX.png`.  
**K must be** one of `{1, 2, 4, 8, 16, 64}`.

### Camera metadata at each step

**Pose vector** (every `*_pose.npy`, shape `(7,)`, `float32`):

| Index | Field | Meaning |
|------:|-------|---------|
| 0 | `elevation_deg` | Camera elevation in degrees (5–75°) |
| 1 | `azimuth_deg` | Camera azimuth in degrees (0–360°) |
| 2 | `distance` | Eye distance from origin (default 1.8, unit-sphere mesh) |
| 3–5 | `eye_x, eye_y, eye_z` | Camera position in normalized object coordinates |
| 6 | `1.0` | Reserved placeholder |

**View region** (which of the 64 bins a render belongs to) is stored separately from the pose vector:

| Step | What is recorded | Where |
|------|------------------|-------|
| **Build — eval bank** | Bin-center `(az_bin, el_bin)` → fixed `elevation`/`azimuth` at bin center | `view_XX.png` + `view_XX_pose.npy`; bin from `XX` via `view_index_to_bins` |
| **Build — train pool** | Jittered `(elevation, azimuth)` **within** each bin | `train_views/az##_el##_j##.png` + matching `*_pose.npy`; bin from filename |
| **Build — per object** | Full nested schedules, `jitter_per_bin`, build seed | `meta.json` |
| **Build — dataset** | Global split counts, nested K list | `master_manifest.json`, `splits/*.json` |
| **Dataset init** | Which K bins to use for this run | `nested_view_bins(K)` in code (must match schedules in `meta.json`) |
| **Train `__getitem__`** | One random jitter render per nested bin → its 7-float pose | Loaded from `train_views/`; **not** re-jittered online |
| **Eval `__getitem__`** | Fixed bin-center pose for each nested bin | `view_{eval_view_indices(K)[i]:02d}_pose.npy` |
| **Training batch** | `poses` tensor `[B, K, 7]` stacked with images | Passed to model; **baseline ignores poses** (mean-pool fusion only) |

**Bin grid:** 16 azimuth bins (22.5° wide, centers 0°, 22.5°, …) × 4 elevation bins (centers 15°, 30°, 45°, 60°). Train jitter samples uniformly inside each bin’s azimuth half-width and elevation bounds.

### Build on Modal volume (recommended if local disk is tight)

Stores meshes and the built dataset on Modal volume **`pose-aware-data`** at `/data/`:

| Path on volume | Contents |
|----------------|----------|
| `/data/ShapeNetCore.v2/` | Raw meshes |
| `/data/master_shapenet/` | GT points, 16 eval views (30° ring), train jitter pool, manifests |

**Volume:** builds use **`pose-aware-data-v2`** (Modal Volumes v2 — no inode cap).

```bash
pip install -r requirements-modal.txt
modal secret create huggingface HF_TOKEN=<your_hf_token>

# Build 2000 objects (7×200 main + 600 broad) — download meshes + render on v2
modal run scripts/modal_build_dataset.py --allow-partial

# Resume if interrupted (meshes already on volume)
modal run scripts/modal_build_dataset.py --skip-download --allow-partial

# Audit when finished
modal run scripts/modal_build_dataset.py --audit-only
```

Do **not** use `--migrate-legacy` unless you explicitly want to copy from the old v1 volume.

**Expected wall time (parallel):** ~3–6 h download + **~1–3 h render** for 5k objects (vs ~8–15 h on a single 8-CPU box).

Train on Modal with `configs/baseline_master_4view_modal.yaml` (`processed_dir: /data/master_shapenet`).

### Build master dataset (local)

```bash
# List required synset downloads
python scripts/build_master_dataset.py --download-categories

# Download all main + broad categories (batched hf commands)
hf auth login
python scripts/download_shapenet_hf.py --categories \
  03001627 04379243 04256520 02958343 02691156 03642806 02933112
# ... then broad synsets (see script output)

# Full build (~5000 objects, many hours)
python scripts/build_master_dataset.py \
  --shapenet-root data/ShapeNetCore.v2 \
  --output-dir data/master_shapenet \
  --jitter-per-bin 2 \
  --workers 8

# Smoke test on partial data
python scripts/build_master_dataset.py \
  --shapenet-root data/ShapeNetCore.v2 \
  --allow-partial --scale-counts 0.02 --max-render 20
```

### Visualize the dataset

Primary entry point (validate + category figure + per-sample gallery):

```bash
python scripts/visualize_master_dataset.py
python scripts/visualize_master_dataset.py --open

# Pull preview from Modal v2 volume, validate, visualize
python scripts/visualize_master_dataset.py --fetch-samples 12 --open

# Category overview only
python scripts/visualize_master_dataset.py --categories-only --per-category 6

# Validate without figures
python scripts/visualize_master_dataset.py --validate-only --dataset-dir data/modal_preview
```

Legacy wrappers: `scripts/visualize_dataset.py`, `scripts/visualize_categories.py`

Outputs under `figures/dataset/`:

| File | Content |
|------|---------|
| `00_overview.png` | Split counts + how many objects are rendered |
| `01_pointclouds_all_samples.png` | GT point cloud per object on disk |
| `02_pointclouds_by_category.png` | One sample per main category (when available) |
| `sample_{id}_panel.png` | Point cloud + 8-view render strip |
| `sample_{id}_renders_grid.png` | Full 16-view eval grid |
| `sample_{id}_input_views.png` | 1 / 2 / 4 / 8 / 16 input-view strips |
| `gallery/index.html` | Browse all samples in the browser |

Optional: `--sample-id 02691156_test_model` or `--max-samples 10`

Train on the master set by pointing `data.processed_dir` to `data/master_shapenet` and using split manifests under `splits/`.

### 2. Pre-render views + GT point clouds (legacy / single-category)

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
  build_master_dataset.py        # 5k master set: splits + 64 views + GT points
  visualize_master_dataset.py    # point cloud & render figures
  download_shapenet_hf.py        # download from Hugging Face
  prepare_dataset.py             # mesh -> RGB views + GT points (single category)
  train_baseline.py
  eval_baseline.py
  visualize_baseline.py
src/
  data/master_splits.py          # 5k object selection & splits
  data/camera.py                 # 64-view camera grid
  data/render.py                 # CPU mesh rasterizer
  data/dataset.py                # PyTorch loaders
  models/                        # DINOv2 + point cloud head
```

## Data directories

| Path | Purpose |
|------|---------|
| `data/ShapeNetCore.v2/` | Raw meshes from Hugging Face |
| `data/processed_shapenet_1view/` | 1 RGB view + GT points per object |
| `data/processed_shapenet_2view/` | 2 RGB views + GT points per object |
| `data/master_shapenet/` | Master 5k dataset (manifest + objects/) |
| `figures/master_dataset/` | Dataset visualization PNGs |
| `runs/` | Training checkpoints |

Old PartAnnotation data (`data/PartAnnotation/`, `data/processed/`) is no longer used and can be deleted.

## Baseline spec

| Component | Setting |
|-----------|---------|
| Encoder | Frozen DINOv2-ViT-B/14 |
| Decoder | MLP point cloud head → 2048 points |
| Loss | Squared Chamfer L2 |
| Metrics | Chamfer + F-score @ 1% bbox |
