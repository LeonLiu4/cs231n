#!/usr/bin/env python3
"""
Build the master ShapeNetCore dataset (5000 objects, static nested views).

Nested schedules: 1 ⊂ 2 ⊂ 4 ⊂ 8 ⊂ 16 ⊂ 64 view regions (16 azimuth x 4 elevation).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.camera import (
    MAX_VIEWS,
    NESTED_VIEW_SCHEDULES,
    eval_view_indices,
    jittered_pose_in_bin,
    nested_view_bins,
    pose_for_bin,
    train_view_stem,
    view_index_to_bins,
)
from src.data.master_splits import (
    BROAD_SYNSETS,
    MAIN_CATEGORIES,
    build_master_records,
    save_master_manifest,
    summarize_records,
)
from src.data.render import normalize_mesh, render_rgb_view, sample_surface_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SHAPENET_ROOT = ROOT / "data" / "ShapeNetCore.v2"
DEFAULT_OUTPUT = ROOT / "data" / "master_shapenet"
DEFAULT_JITTER_PER_BIN = 2


def process_sample(
    mesh_path: Path,
    sample_dir: Path,
    split: str,
    sample_id: str,
    seed: int,
    num_gt_points: int,
    image_size: int,
    jitter_per_bin: int,
    overwrite: bool = False,
) -> bool:
    train_dir = sample_dir / "train_views"
    n_eval = len([p for p in sample_dir.glob("view_*.png") if "_pose" not in p.name])
    n_train = len(list(train_dir.glob("az*_el*_j*.png"))) if train_dir.is_dir() else 0
    need_train = split == "train" and jitter_per_bin > 0
    expected_train = MAX_VIEWS * jitter_per_bin if need_train else 0

    if (sample_dir / "gt_points.npy").is_file() and not overwrite:
        if n_eval >= MAX_VIEWS and (not need_train or n_train >= expected_train):
            return False

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = normalize_mesh(mesh)

    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "gt_points.npy", sample_surface_points(mesh, num_gt_points))

    # Fixed eval bank: the 16 bin centers of the largest nested schedule.
    for vi in eval_view_indices(MAX_VIEWS):
        az_bin, el_bin = view_index_to_bins(vi)
        pose = pose_for_bin(az_bin, el_bin)
        rgb = render_rgb_view(mesh, pose, image_size=image_size)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            sample_dir / f"view_{vi:02d}.png"
        )
        np.save(sample_dir / f"view_{vi:02d}_pose.npy", pose.pose_vector())

    # Train pool: jittered renders per bin, only for the bins in the schedule.
    if need_train:
        train_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + hash(sample_id) % (2**31 - 1))
        for az_bin, el_bin in nested_view_bins(MAX_VIEWS):
            for j in range(jitter_per_bin):
                pose = jittered_pose_in_bin(az_bin, el_bin, rng)
                rgb = render_rgb_view(mesh, pose, image_size=image_size)
                stem = train_view_stem(az_bin, el_bin, j)
                Image.fromarray((rgb * 255).astype(np.uint8)).save(
                    train_dir / f"{stem}.png"
                )
                np.save(train_dir / f"{stem}_pose.npy", pose.pose_vector())

    meta = {
        "mesh_path": str(mesh_path),
        "sample_id": sample_id,
        "split": split,
        "num_gt_points": num_gt_points,
        "num_eval_views": MAX_VIEWS,
        "jitter_per_bin": jitter_per_bin if need_train else 0,
        "nested_schedules": {str(k): v for k, v in NESTED_VIEW_SCHEDULES.items()},
        "image_size": image_size,
        "build_seed": seed,
    }
    with (sample_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return True


def _render_entry(payload: tuple) -> tuple[str, str | None]:
    rec_dict, objects_dir, seed, num_gt_points, image_size, jitter_per_bin, overwrite = payload
    sample_dir = objects_dir / rec_dict["sample_id"]
    try:
        changed = process_sample(
            Path(rec_dict["mesh_path"]),
            sample_dir,
            split=rec_dict["split"],
            sample_id=rec_dict["sample_id"],
            seed=seed,
            num_gt_points=num_gt_points,
            image_size=image_size,
            jitter_per_bin=jitter_per_bin,
            overwrite=overwrite,
        )
        return ("ok" if changed else "skipped", None)
    except Exception as exc:
        return (
            "failed",
            json.dumps({"sample_id": rec_dict["sample_id"], "error": str(exc)}),
        )


def run_build(
    shapenet_root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    scale_counts: float = 1.0,
    num_gt_points: int = 2048,
    image_size: int = 224,
    jitter_per_bin: int = DEFAULT_JITTER_PER_BIN,
    max_render: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
    manifest_only: bool = False,
    allow_partial: bool = False,
) -> dict:
    """Build master dataset at output_dir. Returns summary stats."""
    shapenet_root = shapenet_root.resolve()
    output_dir = output_dir.resolve()
    objects_dir = output_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    records = build_master_records(
        shapenet_root,
        seed=seed,
        scale_counts=scale_counts,
        allow_partial=allow_partial,
    )
    summary = summarize_records(records)
    logger.info("Split summary: %s", json.dumps(summary, indent=2))
    logger.info("Nested schedules: %s", NESTED_VIEW_SCHEDULES)

    manifest_path = save_master_manifest(records, output_dir, jitter_per_bin=jitter_per_bin)
    logger.info("Wrote %s", manifest_path)

    if manifest_only:
        return {
            "total": len(records),
            "summary": summary,
            "manifest_path": str(manifest_path),
            "rendered": 0,
            "skipped": 0,
        }

    to_process = records[:max_render] if max_render else records
    fail_log = output_dir / "render_failures.jsonl"
    payloads = [
        (
            rec.to_dict(),
            objects_dir,
            seed,
            num_gt_points,
            image_size,
            jitter_per_bin,
            overwrite,
        )
        for rec in to_process
    ]

    rendered = skipped = 0
    if workers <= 1:
        for status, err in tqdm(
            (_render_entry(p) for p in payloads),
            total=len(payloads),
            desc="Rendering",
        ):
            if err:
                with fail_log.open("a") as f:
                    f.write(err + "\n")
            elif status == "ok":
                rendered += 1
            else:
                skipped += 1
    else:
        with Pool(workers) as pool:
            for status, err in tqdm(
                pool.imap_unordered(_render_entry, payloads),
                total=len(payloads),
                desc="Rendering",
            ):
                if err:
                    with fail_log.open("a") as f:
                        f.write(err + "\n")
                elif status == "ok":
                    rendered += 1
                else:
                    skipped += 1

    logger.info("Rendered %d, skipped %d", rendered, skipped)
    return {
        "total": len(records),
        "summary": summary,
        "manifest_path": str(manifest_path),
        "rendered": rendered,
        "skipped": skipped,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master ShapeNet dataset")
    parser.add_argument("--shapenet-root", type=Path, default=DEFAULT_SHAPENET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-counts", type=float, default=1.0)
    parser.add_argument("--num-gt-points", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--jitter-per-bin",
        type=int,
        default=DEFAULT_JITTER_PER_BIN,
        help="Jittered train renders per (az,el) bin (x64 bins for train split)",
    )
    parser.add_argument("--max-render", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--download-categories", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if args.download_categories:
        print("Seven main categories:")
        print("  python scripts/download_shapenet_hf.py --master-main")
        print("Broad pool:")
        print(f"  python scripts/download_shapenet_hf.py --categories {' '.join(BROAD_SYNSETS[:12])} ...")
        return

    run_build(
        args.shapenet_root,
        args.output_dir,
        seed=args.seed,
        scale_counts=args.scale_counts,
        num_gt_points=args.num_gt_points,
        image_size=args.image_size,
        jitter_per_bin=args.jitter_per_bin,
        max_render=args.max_render,
        workers=args.workers,
        overwrite=args.overwrite,
        manifest_only=args.manifest_only,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
