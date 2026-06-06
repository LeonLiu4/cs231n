#!/usr/bin/env python3
"""
Cumulative writeup figure: input RGB view -> GT point cloud, grouped by category.

Scans the rendered objects on disk, groups them by ShapeNet synset, and lays out
a single publication-style figure where each category contributes a row of
example objects (top: input render, bottom: ground-truth point cloud).

Usage:
  python scripts/visualize_categories.py
  python scripts/visualize_categories.py --per-category 6
  python scripts/visualize_categories.py --include-placeholder
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.master_splits import MAIN_CATEGORIES
from src.viz.dataset_plots import CATEGORY_COLORS, DatasetVisualizer

DEFAULT_DATASET = ROOT / "data" / "master_shapenet"
DEFAULT_OUTPUT = ROOT / "figures" / "dataset" / "categories_overview.png"

NAME_MAP = {c["synset_id"]: c["name"] for c in MAIN_CATEGORIES}


def pick_input_view(sample_dir: Path, preferred_index: int) -> np.ndarray | None:
    """Load a representative RGB view (3/4 angle), falling back to any view."""
    candidate = sample_dir / f"view_{preferred_index:02d}.png"
    if candidate.is_file():
        return np.asarray(Image.open(candidate).convert("RGB"))
    others = sorted(p for p in sample_dir.glob("view_*.png") if "_pose" not in p.name)
    if not others:
        return None
    return np.asarray(Image.open(others[0]).convert("RGB"))


def group_samples_by_category(
    viz: DatasetVisualizer, include_placeholder: bool
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sid in viz.available_samples():
        if not include_placeholder and sid.endswith("_test_model"):
            continue
        synset = sid.split("_", 1)[0]
        groups.setdefault(synset, []).append(sid)
    return groups


def build_figure(
    viz: DatasetVisualizer,
    groups: dict[str, list[str]],
    per_category: int,
    view_index: int,
    output_path: Path,
    dpi: int,
) -> Path:
    synsets = list(groups.keys())
    n_rows = len(synsets)
    n_cols = min(per_category, max(len(v) for v in groups.values()))

    fig = plt.figure(figsize=(2.6 * n_cols + 1.2, 5.2 * n_rows))
    outer = gridspec.GridSpec(
        n_rows, 1, figure=fig, hspace=0.35
    )

    for ri, synset in enumerate(synsets):
        samples = groups[synset][:n_cols]
        name = NAME_MAP.get(synset, synset)
        color = CATEGORY_COLORS.get(synset, "#2563eb")

        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_cols, subplot_spec=outer[ri], hspace=0.05, wspace=0.08
        )

        for ci, sid in enumerate(samples):
            sample_dir = viz.objects_dir / sid
            img = pick_input_view(sample_dir, view_index)
            ax_img = fig.add_subplot(inner[0, ci])
            if img is not None:
                ax_img.imshow(img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
            if ci == 0:
                ax_img.set_ylabel(
                    f"{name}\n({synset})",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                    rotation=90,
                    labelpad=12,
                )
            if ri == 0:
                ax_img.set_title("input view" if ci == 0 else "", fontsize=9, loc="left")

            ax_pc = fig.add_subplot(inner[1, ci], projection="3d")
            gt_path = sample_dir / "gt_points.npy"
            if gt_path.is_file():
                pts = np.load(gt_path)
                DatasetVisualizer._scatter_pc(ax_pc, pts, color)
            ax_pc.view_init(elev=22, azim=-58)
            ax_pc.set_axis_off()
            if ri == 0 and ci == 0:
                ax_pc.set_title("GT point cloud", fontsize=9, loc="left")

    fig.suptitle(
        "ShapeNet reconstruction dataset: input RGB render \u2192 ground-truth point cloud",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative per-category dataset figure")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-category", type=int, default=5, help="Max example objects per category")
    parser.add_argument("--view-index", type=int, default=18, help="Preferred eval view index for the input render")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--include-placeholder",
        action="store_true",
        help="Include synthetic *_test_model placeholder objects (e.g. the sphere)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not (dataset_dir / "objects").is_dir():
        raise SystemExit(
            f"No rendered objects under {dataset_dir / 'objects'}.\n"
            "Build first: python scripts/build_master_dataset.py --allow-partial"
        )

    viz = DatasetVisualizer(dataset_dir, args.output.parent)
    groups = group_samples_by_category(viz, include_placeholder=args.include_placeholder)
    if not groups:
        raise SystemExit("No categories with rendered objects found.")

    print("Categories on disk:")
    for synset, sids in groups.items():
        print(f"  {NAME_MAP.get(synset, synset)} ({synset}): {len(sids)} object(s)")

    out = build_figure(
        viz,
        groups,
        per_category=args.per_category,
        view_index=args.view_index,
        output_path=args.output.resolve(),
        dpi=args.dpi,
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
