#!/usr/bin/env python3
"""
Visualize the master ShapeNet dataset (point clouds + multi-view renders).

Usage:
  python scripts/visualize_dataset.py
  python scripts/visualize_dataset.py --open
  python scripts/visualize_dataset.py --sample-id 02691156_test_model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

from src.data.camera import MAX_VIEWS, eval_view_indices
from src.viz.dataset_plots import DatasetVisualizer

DEFAULT_DATASET = ROOT / "data" / "master_shapenet"
DEFAULT_OUTPUT = ROOT / "figures" / "dataset"

EXPECTED_GT_POINTS = 2048
EXPECTED_VIEWS = MAX_VIEWS


def _print_health_check(viz: DatasetVisualizer, samples: list[str]) -> None:
    import numpy as np

    print("\n--- Dataset health check ---")
    if viz.manifest:
        print(f"Manifest objects: {viz.manifest.get('total_objects')}")
        print(f"Split counts:     {viz.manifest.get('split_counts')}")
    issues = 0
    for sid in samples:
        d = viz.objects_dir / sid
        gt = d / "gt_points.npy"
        meta_path = d / "meta.json"
        num_eval = MAX_VIEWS
        split = None
        if meta_path.is_file():
            import json

            meta = json.loads(meta_path.read_text())
            num_eval = int(meta.get("num_eval_views", MAX_VIEWS))
            split = meta.get("split")
        expected = eval_view_indices(num_eval)
        views_ok = sum(
            1
            for vi in expected
            if (d / f"view_{vi:02d}.png").is_file()
            and (d / f"view_{vi:02d}_pose.npy").is_file()
        )
        line = f"  {sid}:"
        if split:
            line += f" ({split})"
        if gt.is_file():
            pts = np.load(gt)
            ok_shape = pts.shape == (EXPECTED_GT_POINTS, 3)
            line += f" GT {pts.shape}" + ("" if ok_shape else " [BAD SHAPE]")
            if not ok_shape:
                issues += 1
            if not np.isfinite(pts).all():
                line += " [NON-FINITE GT]"
                issues += 1
            r = np.linalg.norm(pts, axis=1).max()
            line += f" max_r={r:.3f}"
        else:
            line += " MISSING gt_points.npy"
            issues += 1
        line += f", eval={views_ok}/{len(expected)}"
        if views_ok != len(expected):
            issues += 1
        train_views = (
            list((d / "train_views").glob("az*_el*_j*.png"))
            if (d / "train_views").is_dir()
            else []
        )
        if train_views:
            line += f", train_pool={len(train_views)}"
        print(line)
    if not samples:
        print("  (no rendered objects)")
        issues += 1
    print("---" + (" OK" if issues == 0 else f" {issues} issue(s)") + " ---\n")


def resolve_dataset_dir(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    if (DEFAULT_DATASET / "master_manifest.json").is_file():
        return DEFAULT_DATASET.resolve()
    if (DEFAULT_DATASET / "objects").is_dir() and any((DEFAULT_DATASET / "objects").iterdir()):
        return DEFAULT_DATASET.resolve()
    raise SystemExit(
        "No dataset found. Expected data/master_shapenet/ with rendered objects.\n"
        "Build first:\n"
        "  python scripts/build_master_dataset.py --allow-partial\n"
        "Then visualize:\n"
        "  python scripts/visualize_dataset.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ShapeNet master dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Root with master_manifest.json and objects/ (default: data/master_shapenet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save PNGs and HTML gallery",
    )
    parser.add_argument("--sample-id", action="append", default=None, help="Only these samples")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-gallery", action="store_true")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open gallery/index.html in a browser after rendering",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    viz = DatasetVisualizer(dataset_dir, args.output_dir)

    print(f"Dataset:  {dataset_dir}")
    print(f"Output:   {args.output_dir.resolve()}")
    samples_on_disk = viz.available_samples()
    print(f"Rendered: {len(samples_on_disk)} object(s) on disk")
    _print_health_check(viz, samples_on_disk)

    paths = viz.run(
        sample_ids=args.sample_id,
        max_samples=args.max_samples,
        make_gallery=not args.no_gallery,
    )

    print("\nSaved:")
    for p in paths:
        print(f"  {p}")

    if args.open:
        viz.open_gallery()
        print(f"\nOpened {args.output_dir / 'gallery' / 'index.html'}")


if __name__ == "__main__":
    main()
