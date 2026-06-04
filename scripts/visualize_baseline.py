#!/usr/bin/env python3
"""Visualize predicted vs GT point clouds from a trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_baseline import build_dataloaders, build_model, get_device
from src.data.camera import default_view_poses
from src.losses.chamfer import chamfer_distance
from src.metrics.reconstruction import f_score

PRESETS = {
    "1view": {
        "config": ROOT / "configs/baseline_single_view.yaml",
        "checkpoint": ROOT / "runs/single_view_baseline/best.pt",
    },
    "2view": {
        "config": ROOT / "configs/baseline_2view.yaml",
        "checkpoint": ROOT / "runs/baseline_2view/best.pt",
    },
}


def set_equal_3d_axes(ax, points: np.ndarray, pad: float = 0.05) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-3)
    radius *= 1.0 + pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_point_cloud(
    ax,
    points: np.ndarray,
    color: str,
    label: str,
    size: float = 1.0,
    alpha: float = 0.7,
    max_points: int = 2048,
) -> None:
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[idx]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        s=size,
        alpha=alpha,
        linewidths=0,
        label=label,
    )


def draw_overlay_panel(
    ax,
    gt: np.ndarray,
    pred: np.ndarray,
    elev: float,
    azim: float,
    title: str,
) -> None:
    combined = np.concatenate([gt, pred], axis=0)
    plot_point_cloud(ax, gt, color="#2563eb", label="GT", size=1.5, alpha=0.45)
    plot_point_cloud(ax, pred, color="#dc2626", label="Pred", size=1.5, alpha=0.45)
    set_equal_3d_axes(ax, combined)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()


def camera_angles_for_layout(layout: str, elev: float, azim: float) -> list[tuple[float, float, str]]:
    if layout == "dual-angle":
        poses = default_view_poses(2)
        return [
            (elev, float(poses[0].azimuth), f"angle 1 ({poses[0].azimuth:.0f}°)"),
            (elev, float(poses[1].azimuth), f"angle 2 ({poses[1].azimuth:.0f}°)"),
        ]
    return [(elev, azim, f"angle ({azim:.0f}°)")]


def save_visualization(
    gt: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    sample_id: str,
    chamfer: float,
    f1: float,
    layout: str = "full",
    elev: float = 20.0,
    azim: float = 30.0,
) -> None:
    angles = camera_angles_for_layout(layout, elev, azim)

    if layout == "compact":
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_overlay_panel(ax, gt, pred, angles[0][0], angles[0][1], "GT vs predicted")
    elif layout == "dual-angle":
        fig = plt.figure(figsize=(12, 5))
        for idx, (e, a, title) in enumerate(angles, start=1):
            ax = fig.add_subplot(1, 2, idx, projection="3d")
            draw_overlay_panel(ax, gt, pred, e, a, title)
    else:
        combined = np.concatenate([gt, pred], axis=0)
        fig = plt.figure(figsize=(14, 5))

        ax_gt = fig.add_subplot(131, projection="3d")
        plot_point_cloud(ax_gt, gt, color="#2563eb", label="GT", size=2.0)
        set_equal_3d_axes(ax_gt, combined)
        ax_gt.view_init(elev=angles[0][0], azim=angles[0][1])
        ax_gt.set_title("Ground truth")
        ax_gt.set_axis_off()

        ax_pred = fig.add_subplot(132, projection="3d")
        plot_point_cloud(ax_pred, pred, color="#dc2626", label="Predicted", size=2.0)
        set_equal_3d_axes(ax_pred, combined)
        ax_pred.view_init(elev=angles[0][0], azim=angles[0][1])
        ax_pred.set_title("Predicted")
        ax_pred.set_axis_off()

        ax_overlay = fig.add_subplot(133, projection="3d")
        draw_overlay_panel(ax_overlay, gt, pred, angles[0][0], angles[0][1], "Overlay")

    preset_tag = ""
    fig.suptitle(
        f"{sample_id}{preset_tag}\nchamfer={chamfer:.5f}  f1={f1:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is not None:
        preset = PRESETS[args.preset]
        if args.config is None:
            args.config = preset["config"]
        if args.checkpoint is None:
            args.checkpoint = preset["checkpoint"]
    if args.config is None:
        args.config = Path("configs/baseline_2view.yaml")
    if args.checkpoint is None:
        raise SystemExit("Pass --checkpoint or use --preset 1view / --preset 2view")
    return args


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize model point cloud predictions")
    parser.add_argument(
        "--preset",
        choices=["1view", "2view"],
        default=None,
        help="Use saved 1-view or 2-view config + best checkpoint (no retraining)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/visualizations"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--layout",
        choices=["compact", "full", "dual-angle"],
        default="compact",
        help="compact=1 camera angle; dual-angle=2 angles in one PNG; full=GT|Pred|Overlay",
    )
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=30.0)
    args = parser.parse_args()
    args = resolve_args(args)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["data"] = copy.deepcopy(cfg["data"])
    eval_cfg["data"]["augment"] = False

    device = get_device()
    train_loader, val_loader = build_dataloaders(eval_cfg, demo=args.demo)
    loader = train_loader if args.split == "train" else val_loader

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    f_threshold = cfg["eval"]["f_score_threshold"]
    num_views = cfg["data"]["num_views"]
    preset_suffix = f"_{args.preset}" if args.preset else ""
    layout_suffix = f"_{args.layout}"

    dataset = loader.dataset
    if args.num_samples == 1 and args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        indices = list(range(min(args.num_samples, len(dataset))))

    saved = 0
    for idx in indices:
        sample = dataset[idx]
        images = sample["images"].unsqueeze(0).to(device)
        gt_points = sample["gt_points"].unsqueeze(0).to(device)
        poses = sample["poses"].unsqueeze(0).to(device)

        pred_points = model(images, poses)
        loss, _, _ = chamfer_distance(pred_points, gt_points)
        f1 = f_score(pred_points, gt_points, threshold=f_threshold)

        gt_np = gt_points[0].cpu().numpy()
        pred_np = pred_points[0].cpu().numpy()
        sample_id = sample["sample_id"]
        safe_name = sample_id.replace("/", "_")
        out_path = args.output_dir / f"{safe_name}{preset_suffix}{layout_suffix}.png"

        save_visualization(
            gt_np,
            pred_np,
            out_path,
            f"{sample_id} ({num_views}-view model)",
            chamfer=loss.item(),
            f1=f1.item(),
            layout=args.layout,
            elev=args.elev,
            azim=args.azim,
        )
        print(
            f"Saved {out_path}  views={num_views}  "
            f"chamfer={loss.item():.5f}  f1={f1.item():.4f}"
        )
        saved += 1

    print(f"Wrote {saved} visualization(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
