# Baseline pipeline

Each baseline from the writeup is its own config + run script so they can be
trained and evaluated **one at a time**. All share the same DINOv2 encoder,
fusion, and point-cloud head; they differ only in the view/pose information
injected before fusion.

| # | Baseline | Script | Config | `model.type` / `pos_embedding` |
|---|----------|--------|--------|--------------------------------|
| 1 | 2D Positional Embedding | `run_2d_positional.sh` | `b1_2d_positional.yaml` | `multi_view` (ViT patch positions) |
| 2 | View-ID Embedding | `run_view_id.sh` | `b2_view_id.yaml` | `pose_aware` / `view_id` |
| 3 | Camera Pose Embedding | `run_camera_pose.sh` | `b3_camera_pose.yaml` | `pose_aware` / `camera_pose` |
| 4 | Single-View Model | `run_single_view.sh` | `b4_single_view.yaml` | `single_view` |
| 5 | Multi-View Model | `run_multi_view.sh` | `b5_multi_view.yaml` | `multi_view` |
| — | Geometry-Aware (proposed) | `run_geometry_aware.sh` | `main_geometry_aware.yaml` | `pose_aware` / `geometry_aware` |

## Run one baseline

```bash
scripts/baselines/run_view_id.sh
```

This trains the model (writes `runs/baselines/<name>/best.pt`, `last.pt`,
`history.json`) and then evaluates `best.pt` on `test_seen` and
`test_generalization`.

## Run all sequentially

```bash
scripts/baselines/run_all.sh
```

## Useful environment variables

| Var | Effect |
|-----|--------|
| `EPOCHS=N` | Override epoch count |
| `DEMO=1` | Use synthetic demo data (fast smoke test; evaluates on `val`) |
| `SKIP_TRAIN=1` | Only evaluate an existing checkpoint |
| `SKIP_EVAL=1` | Train only, no evaluation |
| `ONLY="b2 geometry"` | (run_all) substring filter over script names |

Quick end-to-end smoke test of the whole pipeline:

```bash
DEMO=1 EPOCHS=1 scripts/baselines/run_all.sh
```
