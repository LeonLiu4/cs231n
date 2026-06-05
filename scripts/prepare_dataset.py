#!/usr/bin/env python3
"""Pre-render ShapeNet views and GT point clouds for training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.camera import default_view_poses
from src.data.render import RENDER_MODES, normalize_mesh, render_mesh_view, sample_surface_points


def find_models(shapenet_root: Path, category: str) -> list[Path]:
    category_dir = shapenet_root / category
    models = []
    for model_dir in sorted(category_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        obj_path = model_dir / "models" / "model_normalized.obj"
        if obj_path.exists():
            models.append(obj_path)
    return models


def process_model(
    obj_path: Path,
    output_dir: Path,
    num_views: int,
    num_gt_points: int,
    image_size: int,
    render_mode: str = "rgb",
) -> None:
    mesh = trimesh.load(obj_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = normalize_mesh(mesh)

    gt_points = sample_surface_points(mesh, num_gt_points)
    np.save(output_dir / "gt_points.npy", gt_points)

    for view_idx, pose in enumerate(default_view_poses(num_views)):
        rgb = render_mesh_view(
            mesh, pose, image_size=image_size, assume_normalized=True, render_mode=render_mode
        )
        from PIL import Image

        Image.fromarray((rgb * 255).astype(np.uint8)).save(output_dir / f"view_{view_idx:02d}.png")
        np.save(output_dir / f"view_{view_idx:02d}_pose.npy", pose.pose_vector())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ShapeNet renders")
    parser.add_argument("--shapenet-root", type=Path, default=Path("data/ShapeNetCore.v2"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_shapenet_1view"))
    parser.add_argument("--category", type=str, default="03001627")
    parser.add_argument("--num-views", type=int, default=1, help="Render up to N views per object")
    parser.add_argument("--num-gt-points", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-mode",
        choices=RENDER_MODES,
        default="rgb",
        help="rgb | silhouette (white on black) | normal (camera normals as RGB)",
    )
    args = parser.parse_args()

    model_paths = find_models(args.shapenet_root, args.category)
    if not model_paths:
        raise FileNotFoundError(
            f"No models found under {args.shapenet_root / args.category}. "
            "Run scripts/download_shapenet_hf.py first."
        )

    rng = random.Random(args.seed)
    rng.shuffle(model_paths)
    if args.max_samples is not None:
        model_paths = model_paths[: args.max_samples]

    split_idx = int(len(model_paths) * args.train_ratio)
    splits = {
        "train": model_paths[:split_idx],
        "val": model_paths[split_idx:],
    }

    args.processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, paths in splits.items():
        sample_ids = []
        for obj_path in tqdm(paths, desc=f"Processing {split_name}"):
            model_id = obj_path.parents[1].name
            sample_id = f"{args.category}_{model_id}"
            sample_dir = args.processed_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            process_model(
                obj_path,
                sample_dir,
                num_views=args.num_views,
                num_gt_points=args.num_gt_points,
                image_size=args.image_size,
                render_mode=args.render_mode,
            )
            sample_ids.append(sample_id)

        manifest = {
            "category": args.category,
            "num_views": args.num_views,
            "render_mode": args.render_mode,
            "samples": sample_ids,
        }
        manifest_path = args.processed_dir / f"{split_name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {len(sample_ids)} samples to {manifest_path}")


if __name__ == "__main__":
    main()
