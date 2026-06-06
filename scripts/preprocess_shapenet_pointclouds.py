#!/usr/bin/env python3
"""Batch preprocess ShapeNetCore meshes into ground-truth point clouds."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

# Allow running as: python scripts/preprocess_shapenet_pointclouds.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapenet.index import PREPROCESS_VERSION, build_manifest, load_manifest, save_manifest
from shapenet.mesh_utils import mesh_to_pointcloud
from shapenet.verify import verify_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MESH_ROOT = Path("data/shapenetcore/ShapeNetCore.v2")
DEFAULT_OUTPUT = Path("data/pointclouds/shapenetcore_v2")
DEFAULT_MANIFEST = Path("data/manifests/shapenetcore_v2.json")
FAILURES_LOG = "preprocess_failures.jsonl"


def process_one(
    entry: dict,
    num_points: int,
    resume: bool,
    overwrite: bool,
    failures_path: Path,
) -> str:
    """Process a single manifest entry. Returns status: ok, skipped, or failed."""
    out_path = Path(entry["pointcloud_path"])
    if resume and not overwrite and out_path.is_file():
        return "skipped"

    try:
        points = mesh_to_pointcloud(entry["mesh_path"], num_points=num_points)
        if points.shape != (num_points, 3):
            raise ValueError(f"Unexpected shape {points.shape}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(out_path, points)
        return "ok"
    except Exception as exc:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "synset_id": entry["synset_id"],
            "model_id": entry["model_id"],
            "mesh_path": entry["mesh_path"],
            "error": str(exc),
        }
        with failures_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ShapeNetCore meshes to point clouds"
    )
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--failures-log", type=Path, default=None)
    args = parser.parse_args()

    mesh_root = args.mesh_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    failures_path = (
        args.failures_log
        if args.failures_log
        else manifest_path.parent / FAILURES_LOG
    )

    if args.overwrite and failures_path.is_file():
        failures_path.unlink()

    if manifest_path.is_file() and args.max_models is None:
        manifest = load_manifest(manifest_path)
        if manifest.get("mesh_root") != str(mesh_root):
            logger.info("Rebuilding manifest (mesh_root changed).")
            manifest = build_manifest(
                mesh_root, output_dir, num_points=args.num_points
            )
    else:
        manifest = build_manifest(
            mesh_root,
            output_dir,
            num_points=args.num_points,
            max_models=args.max_models,
        )

    manifest["preprocess_version"] = PREPROCESS_VERSION
    save_manifest(manifest, manifest_path)

    entries = manifest["entries"]
    if not entries:
        raise SystemExit(f"No models found under {mesh_root}")

    worker = partial(
        process_one,
        num_points=args.num_points,
        resume=args.resume,
        overwrite=args.overwrite,
        failures_path=failures_path.resolve(),
    )

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    if args.workers <= 1:
        for entry in tqdm(entries, desc="Preprocessing"):
            status = worker(entry)
            counts[status] = counts.get(status, 0) + 1
    else:
        with Pool(args.workers) as pool:
            for status in tqdm(
                pool.imap_unordered(worker, entries),
                total=len(entries),
                desc="Preprocessing",
            ):
                counts[status] = counts.get(status, 0) + 1

    manifest["processed_ok"] = counts.get("ok", 0)
    manifest["processed_skipped"] = counts.get("skipped", 0)
    manifest["processed_failed"] = counts.get("failed", 0)
    save_manifest(manifest, manifest_path)

    logger.info(
        "Done: %d ok, %d skipped, %d failed",
        counts.get("ok", 0),
        counts.get("skipped", 0),
        counts.get("failed", 0),
    )
    if counts.get("failed", 0):
        logger.info("Failures logged to %s", failures_path)

    verify_manifest(manifest)


if __name__ == "__main__":
    main()
