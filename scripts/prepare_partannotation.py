#!/usr/bin/env python3
"""Prepare training data from ShapeNet PartAnnotation (archive.zip layout)."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.camera import default_view_poses
from src.data.render import (
    normalize_mesh,
    render_points_mesh_view,
    render_points_view,
    render_rgb_view,
    sample_surface_points,
)


def load_pts(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points[:, :3]


def subsample_points(points: np.ndarray, num_points: int, rng: random.Random) -> np.ndarray:
    if len(points) >= num_points:
        idx = rng.sample(range(len(points)), num_points)
        return points[idx]
    extra = rng.choices(range(len(points)), k=num_points - len(points))
    return np.concatenate([points, points[extra]], axis=0)


def find_partannotation_samples(
    partannotation_root: Path,
    category: str,
    use_expert_verified: bool = True,
) -> list[dict]:
    category_dir = partannotation_root / category
    points_dir = category_dir / "points"
    if not points_dir.exists():
        raise FileNotFoundError(f"Missing points dir: {points_dir}")

    if use_expert_verified:
        image_dir = category_dir / "expert_verified" / "seg_img"
        allowed_ids = {p.stem for p in image_dir.glob("*.png")} if image_dir.exists() else None
    else:
        image_dir = None
        allowed_ids = None

    samples = []
    for pts_path in sorted(points_dir.glob("*.pts")):
        model_id = pts_path.stem
        if allowed_ids is not None and model_id not in allowed_ids:
            continue
        samples.append(
            {
                "model_id": model_id,
                "pts_path": pts_path,
                "image_path": image_dir / f"{model_id}.png" if image_dir else None,
            }
        )
    return samples


def try_render_from_shapenet_core(
    shapenet_root: Path,
    category: str,
    model_id: str,
    pose,
    image_size: int,
) -> np.ndarray | None:
    import trimesh

    obj_path = shapenet_root / category / model_id / "models" / "model_normalized.obj"
    if not obj_path.exists():
        return None

    mesh = trimesh.load(obj_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = normalize_mesh(mesh)
    return render_rgb_view(mesh, pose, image_size=image_size)


def process_sample(
    sample: dict,
    output_dir: Path,
    category: str,
    num_views: int,
    num_gt_points: int,
    image_size: int,
    shapenet_root: Path | None,
    rng: random.Random,
    view_mode: str = "seg_first",
) -> bool:
    points = load_pts(sample["pts_path"])
    points = subsample_points(points, num_gt_points, rng)

    center = points.mean(axis=0)
    points -= center
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-8:
        points /= scale

    np.save(output_dir / "gt_points.npy", points.astype(np.float32))

    poses = default_view_poses(num_views)
    for view_idx, pose in enumerate(poses):
        rgb = None
        if shapenet_root is not None:
            rgb = try_render_from_shapenet_core(
                shapenet_root, category, sample["model_id"], pose, image_size
            )

        if rgb is None and view_mode == "points":
            rgb = render_points_view(points, pose, image_size=image_size)

        if rgb is None and view_mode == "seg_first" and view_idx == 0 and sample["image_path"] is not None and sample["image_path"].exists():
            img = Image.open(sample["image_path"]).convert("RGB").resize((image_size, image_size))
            rgb = np.asarray(img, dtype=np.float32) / 255.0

        if rgb is None:
            if view_idx == 0:
                rgb = render_points_view(points, pose, image_size=image_size)
            else:
                rgb = render_points_mesh_view(points, pose, image_size=image_size)

        Image.fromarray((rgb * 255).astype(np.uint8)).save(output_dir / f"view_{view_idx:02d}.png")
        np.save(output_dir / f"view_{view_idx:02d}_pose.npy", pose.pose_vector())

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PartAnnotation dataset")
    parser.add_argument(
        "--partannotation-root",
        type=Path,
        default=Path("data/PartAnnotation"),
    )
    parser.add_argument(
        "--shapenet-root",
        type=Path,
        default=Path("data/ShapeNetCore.v2"),
        help="Optional ShapeNetCore root for mesh-based RGB rendering",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--category", type=str, default="03001627")
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--num-gt-points", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-points",
        action="store_true",
        help="Use all point files instead of expert_verified subset only",
    )
    parser.add_argument(
        "--view-mode",
        choices=["seg_first", "points"],
        default="seg_first",
        help="seg_first: view 0 seg_img + extra point renders; points: all views from GT point cloud",
    )
    args = parser.parse_args()

    samples = find_partannotation_samples(
        args.partannotation_root,
        args.category,
        use_expert_verified=not args.all_points,
    )
    if not samples:
        raise FileNotFoundError(
            f"No samples found under {args.partannotation_root / args.category}. "
            "Extract archive.zip to data/PartAnnotation first."
        )

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    shapenet_root = args.shapenet_root if args.shapenet_root.exists() else None
    if shapenet_root is None:
        if args.view_mode == "points":
            print("ShapeNetCore.v2 not found — rendering all views from GT point clouds.")
        else:
            print(
                "ShapeNetCore.v2 not found — view 0 uses seg_img when available; "
                "additional views are rendered from the GT point cloud."
            )

    split_idx = int(len(samples) * args.train_ratio)
    splits = {"train": samples[:split_idx], "val": samples[split_idx:]}
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        sample_ids = []
        skipped = 0
        for sample in tqdm(split_samples, desc=f"Processing {split_name}"):
            sample_id = f"{args.category}_{sample['model_id']}"
            sample_dir = args.processed_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            ok = process_sample(
                sample,
                sample_dir,
                args.category,
                args.num_views,
                args.num_gt_points,
                args.image_size,
                shapenet_root,
                rng,
                view_mode=args.view_mode,
            )
            if not ok:
                skipped += 1
                shutil.rmtree(sample_dir, ignore_errors=True)
                continue
            sample_ids.append(sample_id)

        manifest = {
            "source": "PartAnnotation",
            "category": args.category,
            "num_views": args.num_views,
            "samples": sample_ids,
        }
        manifest_path = args.processed_dir / f"{split_name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {len(sample_ids)} samples to {manifest_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
