"""
Generate cross-baseline comparison figures from checkpoints on the Modal volume.

Creates:
  - chamfer_comparison.png   bar chart (test_seen)
  - fscore_comparison.png      bar chart (test_seen)
  - qualitative_{sample}.png side-by-side GT + predictions

Usage:
  modal run scripts/modal_visualize_baselines.py
  modal run scripts/modal_visualize_baselines.py --fetch-local
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "pose-aware-data-v2"
VOL_MOUNT = "/data"
RUNS = f"{VOL_MOUNT}/runs/baselines"
FIG_OUT = f"{RUNS}/figures"

# Checkpoints to compare (label -> run dir suffix on volume)
BASELINE_RUNS: dict[str, str] = {
    "2D positional": "b1_2d_positional_fast",
    "View-ID": "b2_view_id_fast",
    "Camera pose": "b3_camera_pose_fast",
    "Single-view": "b4_single_view_fast",
    "Multi-view": "b5_multi_view_fast",
    "Geometry-aware": "main_geometry_aware_fast",
}

CONFIG_FOR_RUN: dict[str, str] = {
    "b1_2d_positional_fast": "configs/baselines/b1_2d_positional.yaml",
    "b5_multi_view_fast": "configs/baselines/b5_multi_view.yaml",
    "main_geometry_aware_fast": "configs/baselines/main_geometry_aware.yaml",
    "b2_view_id_fast": "configs/baselines/b2_view_id.yaml",
    "b3_camera_pose_fast": "configs/baselines/b3_camera_pose.yaml",
    "b4_single_view_fast": "configs/baselines/b4_single_view.yaml",
    "b5_multi_view_compare_k8": "configs/baselines/b5_multi_view.yaml",
    "main_geometry_aware_compare_k8": "configs/baselines/main_geometry_aware.yaml",
}

app = modal.App("pose-aware-baseline-viz")
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

viz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
    )
    .env({"PYTHONPATH": "/root", "MPLCONFIGDIR": "/tmp/mpl", "TORCH_HOME": f"{VOL_MOUNT}/torch_cache"})
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/root/scripts")
    .add_local_dir(REPO_ROOT / "configs", remote_path="/root/configs")
)


# Base run name -> config, used to derive configs for category runs like
# "b4_single_view_cat_03001627".
BASE_CONFIG: dict[str, str] = {
    "b1_2d_positional": "configs/baselines/b1_2d_positional.yaml",
    "b2_view_id": "configs/baselines/b2_view_id.yaml",
    "b3_camera_pose": "configs/baselines/b3_camera_pose.yaml",
    "b4_single_view": "configs/baselines/b4_single_view.yaml",
    "b5_multi_view": "configs/baselines/b5_multi_view.yaml",
    "main_geometry_aware": "configs/baselines/main_geometry_aware.yaml",
}

# Baselines compared in the single-category diagnostic, in display order.
CATEGORY_BASELINES: list[tuple[str, str]] = [
    ("2D positional", "b1_2d_positional"),
    ("View-ID", "b2_view_id"),
    ("Camera pose", "b3_camera_pose"),
    ("Single-view", "b4_single_view"),
    ("Multi-view", "b5_multi_view"),
    ("Geometry-aware", "main_geometry_aware"),
]


def _config_for_run(run_suffix: str) -> tuple[str, Path]:
    if run_suffix in CONFIG_FOR_RUN:
        rel = CONFIG_FOR_RUN[run_suffix]
    else:
        # Derive from base name for category runs ("<base>_cat_<synset>").
        base = run_suffix.split("_cat_")[0]
        rel = BASE_CONFIG[base]
    return rel, Path(RUNS) / run_suffix


def _set_equal_3d(ax, points, pad: float = 0.08) -> None:
    import numpy as np

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3) * (1.0 + pad)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _scatter_cloud(ax, pts, color: str, size: float = 0.8, alpha: float = 0.85) -> None:
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=color, s=size, alpha=alpha, linewidths=0,
    )


def _load_models(
    device,
    run_suffixes: list[str] | None = None,
    labels: dict[str, str] | None = None,
):
    import copy

    import torch
    import yaml

    from scripts.train_baseline import build_model

    runs = run_suffixes or [s for s in BASELINE_RUNS.values()]
    label_map = labels or {v: k for k, v in BASELINE_RUNS.items()}
    models: dict[str, tuple] = {}

    for suffix in runs:
        ckpt_path = Path(RUNS) / suffix / "best.pt"
        if not ckpt_path.is_file():
            print(f"[skip] no checkpoint: {ckpt_path}")
            continue
        config_rel, out_dir = _config_for_run(suffix)
        with open(Path("/root") / config_rel) as f:
            cfg = yaml.safe_load(f)
        cfg = copy.deepcopy(cfg)
        cfg["data"]["processed_dir"] = f"{VOL_MOUNT}/master_shapenet"
        cfg["experiment"]["output_dir"] = str(out_dir)
        if suffix.endswith("_fast"):
            cfg["data"]["max_eval_samples"] = 80
        elif suffix.endswith("_compare_k8"):
            cfg["data"]["num_views"] = 8
            cfg["data"]["max_eval_samples"] = 210
        elif "_cat_" in suffix:
            # Category runs use the folding head (see SINGLE_CATEGORY_PRESET).
            cfg["model"]["head_type"] = "folding"
            cfg["data"]["category"] = suffix.split("_cat_")[1]

        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[suffix] = (label_map.get(suffix, suffix), model, cfg)
    return models


def _load_sample_by_id(
    sample_id: str,
    num_views: int,
    image_size: int = 224,
    category: str | None = None,
) -> dict:
    from src.data.dataset import ShapeNetReconstructionDataset

    ds = ShapeNetReconstructionDataset(
        processed_dir=f"{VOL_MOUNT}/master_shapenet",
        split="test_seen",
        num_views=num_views,
        image_size=image_size,
        augment=False,
        category=category,
    )
    for i in range(len(ds)):
        if ds.samples[i] == sample_id:
            return ds[i]
    raise ValueError(f"Sample {sample_id!r} not found in test_seen with K={num_views}")


@app.function(
    image=viz_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 20,
    memory=16384,
)
def visualize_single_test_subject(
    sample_id: str = "",
    elev: float = 22.0,
    azim: float = 35.0,
    category: str = "",
) -> dict:
    """Render GT + each baseline prediction for one test object."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    data_volume.reload()

    from scripts.train_baseline import get_device
    from src.data.dataset import ShapeNetReconstructionDataset
    from src.losses.chamfer import chamfer_distance
    from src.metrics.reconstruction import f_score

    device = get_device()

    if category:
        run_suffixes = [f"{base}_cat_{category}" for _, base in CATEGORY_BASELINES]
        labels = {
            f"{base}_cat_{category}": lbl for lbl, base in CATEGORY_BASELINES
        }
        ref_suffix = f"b5_multi_view_cat_{category}"
        overlay_pairs = [
            ("Multi-view", f"b5_multi_view_cat_{category}"),
            ("Geometry-aware", f"main_geometry_aware_cat_{category}"),
        ]
        models = _load_models(device, run_suffixes=run_suffixes, labels=labels)
    else:
        models = _load_models(device)
        ref_suffix = "b5_multi_view_fast"
        overlay_pairs = [
            ("Multi-view", "b5_multi_view_fast"),
            ("Geometry-aware", "main_geometry_aware_fast"),
        ]
    if not models:
        raise RuntimeError("No baseline checkpoints found on volume.")

    if ref_suffix not in models:
        ref_suffix = next(iter(models))
    _, _, ref_cfg = models[ref_suffix]
    num_views = ref_cfg["data"]["num_views"]
    cat_filter = category or None

    if not sample_id:
        ds = ShapeNetReconstructionDataset(
            processed_dir=f"{VOL_MOUNT}/master_shapenet",
            split="test_seen",
            num_views=num_views,
            image_size=ref_cfg["data"]["image_size"],
            augment=False,
            category=cat_filter,
        )
        sample_id = ds.samples[0]
        if not category:
            # Prefer an airplane from test_seen for a clear shape.
            for sid in ds.samples:
                if sid.startswith("02691156_"):
                    sample_id = sid
                    break

    sample = _load_sample_by_id(
        sample_id, num_views=num_views, image_size=ref_cfg["data"]["image_size"],
        category=cat_filter,
    )
    gt = sample["gt_points"].numpy()
    imgs = sample["images"][: min(4, sample["images"].shape[0])].permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def predict(model, cfg, sample_dict):
        images = sample_dict["images"].unsqueeze(0).to(device)
        poses = sample_dict["poses"].unsqueeze(0).to(device)
        return model(images, poses)[0].cpu().numpy()

    preds: list[tuple[str, np.ndarray, float, float]] = []
    gt_t = torch.from_numpy(gt).unsqueeze(0).to(device)
    f_threshold = ref_cfg["eval"]["f_score_threshold"]

    for suffix, (label, model, cfg) in models.items():
        k = cfg["data"]["num_views"]
        sample_k = _load_sample_by_id(
            sample_id, num_views=k, image_size=cfg["data"]["image_size"],
            category=cat_filter,
        )
        pred = predict(model, cfg, sample_k)
        pred_t = torch.from_numpy(pred).unsqueeze(0).to(device)
        chamfer, _, _ = chamfer_distance(pred_t, gt_t)
        f1 = f_score(pred_t, gt_t, threshold=f_threshold)
        preds.append((label, pred, chamfer.item(), f1.item()))

    # Layout: top = input strip; bottom row = GT + each baseline prediction
    n_pred = len(preds)
    fig = plt.figure(figsize=(2.8 * (n_pred + 1), 6.2))
    gs = fig.add_gridspec(2, n_pred + 1, height_ratios=[1.0, 2.2], hspace=0.28, wspace=0.08)

    ax_in = fig.add_subplot(gs[0, :])
    strip = np.concatenate([imgs[i] for i in range(len(imgs))], axis=1)
    ax_in.imshow(np.clip(strip, 0, 1))
    ax_in.set_title(f"Input views (K={num_views} eval schedule)", fontsize=12, fontweight="bold")
    ax_in.axis("off")

    all_pts = [gt] + [p[1] for p in preds]

    ax_gt = fig.add_subplot(gs[1, 0], projection="3d")
    _scatter_cloud(ax_gt, gt, "#2d8a46", size=1.2)
    _set_equal_3d(ax_gt, np.concatenate(all_pts, axis=0))
    ax_gt.view_init(elev=elev, azim=azim)
    ax_gt.set_title("Ground truth", fontsize=11, fontweight="bold")
    ax_gt.set_axis_off()

    colors = ["#4c6ef5", "#8a5cf6", "#e0892f", "#6b788c", "#3b82f6", "#e2553d"]
    for col, (label, pred, chamfer, f1), color in zip(range(1, n_pred + 1), preds, colors):
        ax = fig.add_subplot(gs[1, col], projection="3d")
        _scatter_cloud(ax, pred, color, size=1.2)
        _set_equal_3d(ax, np.concatenate(all_pts, axis=0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{label}\nC={chamfer:.4f}  F={f1:.3f}", fontsize=10)
        ax.set_axis_off()

    fig.suptitle(f"Baseline point clouds — {sample_id}", fontsize=13, fontweight="bold", y=0.98)
    out_dir = Path(FIG_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = sample_id.replace("/", "_")
    out_path = out_dir / f"single_subject_{safe}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Second figure: overlay GT (green) + pred (red) for geometry vs multi-view only
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={"projection": "3d"})
    for ax, (title, suffix) in zip(axes, overlay_pairs):
        if suffix not in models:
            continue
        _, model, cfg = models[suffix]
        sample_k = _load_sample_by_id(
            sample_id, num_views=cfg["data"]["num_views"], image_size=cfg["data"]["image_size"],
            category=cat_filter,
        )
        pred = predict(model, cfg, sample_k)
        _scatter_cloud(ax, gt, "#2d8a46", size=0.8, alpha=0.35)
        _scatter_cloud(ax, pred, "#e2553d", size=0.8, alpha=0.55)
        _set_equal_3d(ax, np.concatenate([gt, pred], axis=0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} vs GT", fontsize=11)
        ax.set_axis_off()
    overlay_path = out_dir / f"single_subject_{safe}_overlay.png"
    fig2.suptitle(f"{sample_id}: prediction (red) vs GT (green)", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    result = {
        "sample_id": sample_id,
        "main_figure": str(out_path),
        "overlay_figure": str(overlay_path),
        "predictions": [
            {"label": l, "chamfer": c, "f_score": f} for l, _, c, f in preds
        ],
    }
    data_volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=viz_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 30,
    memory=16384,
)
def generate_baseline_figures(
    run_suffixes: list[str] | None = None,
    num_qualitative: int = 3,
    labels_override: dict[str, str] | None = None,
    category: str = "",
) -> dict:
    import copy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    sys.path.insert(0, "/root")
    data_volume.reload()

    from scripts.train_baseline import build_model, evaluate, get_device
    from src.data.dataset import ShapeNetReconstructionDataset, collate_batch
    from torch.utils.data import DataLoader

    runs = run_suffixes or list(BASELINE_RUNS.values())
    labels = labels_override or {v: k for k, v in BASELINE_RUNS.items()}

    device = get_device()
    metrics_rows: list[dict] = []
    models: dict[str, tuple] = {}

    for suffix in runs:
        ckpt_path = Path(RUNS) / suffix / "best.pt"
        eval_path = Path(RUNS) / suffix / "eval_results.json"
        if not ckpt_path.is_file():
            print(f"[skip] no checkpoint: {ckpt_path}")
            continue

        config_rel, out_dir = _config_for_run(suffix)
        with open(Path("/root") / config_rel) as f:
            cfg = yaml.safe_load(f)

        cfg = copy.deepcopy(cfg)
        cfg["data"]["processed_dir"] = f"{VOL_MOUNT}/master_shapenet"
        cfg["experiment"]["output_dir"] = str(out_dir)
        if suffix.endswith("_fast"):
            cfg["data"]["max_eval_samples"] = 80
        elif suffix.endswith("_compare_k8"):
            cfg["data"]["num_views"] = 8
            cfg["data"]["max_eval_samples"] = 210
        elif "_cat_" in suffix:
            cfg["model"]["head_type"] = "folding"
            cfg["data"]["category"] = suffix.split("_cat_")[1]

        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        if eval_path.is_file():
            row = json.loads(eval_path.read_text())
        else:
            ds = ShapeNetReconstructionDataset(
                processed_dir=cfg["data"]["processed_dir"],
                split="test_seen",
                num_views=cfg["data"]["num_views"],
                image_size=cfg["data"]["image_size"],
                max_samples=cfg["data"].get("max_eval_samples"),
                augment=False,
            )
            loader = DataLoader(
                ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_batch
            )
            m = evaluate(model, loader, device, cfg["eval"]["f_score_threshold"])
            row = {"name": suffix, "splits": {"test_seen": m}}

        label = labels.get(suffix, suffix)
        metrics_rows.append(
            {
                "label": label,
                "suffix": suffix,
                "chamfer": row["splits"]["test_seen"]["chamfer"],
                "f_score": row["splits"]["test_seen"]["f_score"],
            }
        )
        models[suffix] = (label, model, cfg)

    out_dir = Path(FIG_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if metrics_rows:
        names = [r["label"] for r in metrics_rows]
        chamfers = [r["chamfer"] for r in metrics_rows]
        fscores = [r["f_score"] for r in metrics_rows]
        colors = ["#6b788c", "#e2553d", "#8a5cf6", "#e0892f"][: len(names)]

        for metric, values, fname, ylabel in [
            ("chamfer", chamfers, "chamfer_comparison.png", "Chamfer distance (lower is better)"),
            ("f_score", fscores, "fscore_comparison.png", "F-score @ 1% bbox (higher is better)"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.bar(range(len(names)), values, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Baseline comparison on test_seen ({metric})")
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Qualitative panels on shared test samples
    if len(models) >= 2:
        ref_suffix = next(iter(models))
        _, _, ref_cfg = models[ref_suffix]
        ds = ShapeNetReconstructionDataset(
            processed_dir=f"{VOL_MOUNT}/master_shapenet",
            split="test_seen",
            num_views=ref_cfg["data"]["num_views"],
            image_size=ref_cfg["data"]["image_size"],
            max_samples=num_qualitative,
            augment=False,
            category=category or None,
        )

        @torch.no_grad()
        def predict(model, sample):
            images = sample["images"].unsqueeze(0).to(device)
            poses = sample["poses"].unsqueeze(0).to(device)
            return model(images, poses)[0].cpu().numpy()

        for idx in range(len(ds)):
            sample = ds[idx]
            gt = sample["gt_points"].numpy()
            sample_id = sample["sample_id"]

            # Input view strip (first 4 eval views)
            imgs = sample["images"][:4].permute(0, 2, 3, 1).numpy()

            ncols = 2 + len(models)  # inputs | each model | GT
            fig = plt.figure(figsize=(3.2 * ncols, 3.5))
            gs = fig.add_gridspec(1, ncols, wspace=0.05)

            ax0 = fig.add_subplot(gs[0])
            strip = np.concatenate([imgs[i] for i in range(min(4, len(imgs)))], axis=1)
            ax0.imshow(np.clip(strip, 0, 1))
            ax0.set_title("Input views", fontsize=10)
            ax0.axis("off")

            col = 1
            for suffix, (label, model, mcfg) in models.items():
                if mcfg["data"]["num_views"] != ref_cfg["data"]["num_views"]:
                    continue
                pred = predict(model, sample)
                ax = fig.add_subplot(gs[col], projection="3d")
                ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=1, c="#3a7d44", alpha=0.3)
                ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=1, c="#e2553d", alpha=0.5)
                ax.set_title(label, fontsize=9)
                ax.set_axis_off()
                col += 1

            ax_gt = fig.add_subplot(gs[col], projection="3d")
            ax_gt.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=2, c="#3a7d44")
            ax_gt.set_title("Ground truth", fontsize=10)
            ax_gt.set_axis_off()

            safe = sample_id.replace("/", "_")
            fig.suptitle(sample_id, fontsize=11, fontweight="bold")
            fig.savefig(out_dir / f"qualitative_{safe}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    summary = {"metrics": metrics_rows, "figures_dir": str(out_dir)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(
    fetch_local: bool = False,
    sample_id: str = "",
    single_subject: bool = False,
    category: str = "",
) -> None:
    if single_subject:
        result = visualize_single_test_subject.remote(
            sample_id=sample_id, category=category
        )
    elif category:
        all_six = [
            "b1_2d_positional", "b2_view_id", "b3_camera_pose",
            "b4_single_view", "b5_multi_view", "main_geometry_aware",
        ]
        nice = {
            "b1_2d_positional": "2D positional", "b2_view_id": "View-ID",
            "b3_camera_pose": "Camera pose", "b4_single_view": "Single-view",
            "b5_multi_view": "Multi-view", "main_geometry_aware": "Geometry-aware",
        }
        run_suffixes = [f"{b}_cat_{category}" for b in all_six]
        labels_override = {f"{b}_cat_{category}": nice[b] for b in all_six}
        result = generate_baseline_figures.remote(
            run_suffixes=run_suffixes,
            labels_override=labels_override,
            category=category,
        )
    else:
        result = generate_baseline_figures.remote()
    print(json.dumps(result, indent=2))

    if fetch_local:
        dest = REPO_ROOT / "figures" / "baselines" / "figures"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "modal", "volume", "get", "--force", VOLUME_NAME,
                "runs/baselines/figures", str(REPO_ROOT / "figures" / "baselines"),
            ],
            check=False,
        )
        print(f"Figures in {dest}")
