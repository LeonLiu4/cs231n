#!/usr/bin/env python3
"""
Visualize the master ShapeNet reconstruction dataset.

Works on a local dataset root (``data/master_shapenet`` or ``data/modal_preview``)
or pulls a preview from the Modal volume first.

Examples:
  # Local dataset (default paths)
  python scripts/visualize_master_dataset.py

  # Pull samples from Modal v2 volume, validate, and open gallery
  python scripts/visualize_master_dataset.py --fetch-samples 12 --open

  # Category overview figure only
  python scripts/visualize_master_dataset.py --categories-only --per-category 4

  # Validate without generating figures
  python scripts/visualize_master_dataset.py --validate-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

from scripts.validate_master_dataset import print_report, validate_dataset_dir
from scripts.visualize_categories import build_figure, group_samples_by_category
from src.viz.dataset_plots import DatasetVisualizer

DEFAULT_DATASET = ROOT / "data" / "master_shapenet"
DEFAULT_PREVIEW = ROOT / "data" / "modal_preview"
DEFAULT_OUTPUT = ROOT / "figures" / "dataset"
DEFAULT_VOLUME = "pose-aware-data-v2"
LEGACY_VOLUME = "pose-aware-data"


def resolve_dataset_dir(path: Path | None, *, prefer_preview: bool) -> Path:
    if path is not None:
        return path.resolve()
    if prefer_preview and (DEFAULT_PREVIEW / "objects").is_dir():
        if any((DEFAULT_PREVIEW / "objects").iterdir()):
            return DEFAULT_PREVIEW.resolve()
    if (DEFAULT_DATASET / "master_manifest.json").is_file():
        return DEFAULT_DATASET.resolve()
    if (DEFAULT_DATASET / "objects").is_dir() and any(
        (DEFAULT_DATASET / "objects").iterdir()
    ):
        return DEFAULT_DATASET.resolve()
    raise SystemExit(
        "No dataset found.\n"
        "Build on Modal:\n"
        "  modal run scripts/modal_build_dataset.py --migrate-legacy\n"
        "  modal run scripts/modal_build_dataset.py --skip-download --allow-partial\n"
        "Or fetch a preview:\n"
        "  python scripts/visualize_master_dataset.py --fetch-samples 8"
    )


def fetch_from_modal(
    *,
    volume: str,
    out_dir: Path,
    max_samples: int,
    light: bool,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_modal_samples.py"),
        "--volume",
        volume,
        "--out-dir",
        str(out_dir),
        "--max-samples",
        str(max_samples),
    ]
    if light:
        cmd.append("--light")
    print("Fetching from Modal volume ...")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and visualize the master ShapeNet dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root with master_manifest.json and objects/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for PNG figures and HTML gallery",
    )
    parser.add_argument(
        "--fetch-samples",
        type=int,
        default=None,
        metavar="N",
        help=f"Download N objects from Modal volume into {DEFAULT_PREVIEW.name}/ first",
    )
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help=f"Modal volume name (default: {DEFAULT_VOLUME})",
    )
    parser.add_argument(
        "--legacy-volume",
        action="store_true",
        help=f"Fetch from legacy volume {LEGACY_VOLUME} instead of v2",
    )
    parser.add_argument(
        "--light-fetch",
        action="store_true",
        help="Drop train_views/ after fetch (enough for most figures)",
    )
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--per-category",
        type=int,
        default=4,
        help="Examples per category in categories_overview.png",
    )
    parser.add_argument(
        "--categories-only",
        action="store_true",
        help="Only write categories_overview.png (skip per-sample gallery)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation report and exit",
    )
    parser.add_argument("--no-gallery", action="store_true")
    parser.add_argument(
        "--include-placeholder",
        action="store_true",
        help="Include synthetic *_test_model objects",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open gallery/index.html in a browser after rendering",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if args.fetch_samples:
        volume = LEGACY_VOLUME if args.legacy_volume else args.volume
        preview_dir = DEFAULT_PREVIEW if dataset_dir is None else dataset_dir
        fetch_from_modal(
            volume=volume,
            out_dir=preview_dir,
            max_samples=args.fetch_samples,
            light=args.light_fetch,
        )
        dataset_dir = preview_dir
    else:
        dataset_dir = resolve_dataset_dir(
            dataset_dir, prefer_preview=DEFAULT_PREVIEW.exists()
        )

    report = validate_dataset_dir(dataset_dir)
    print_report(report)

    if args.validate_only:
        if report["incomplete"] > 0:
            raise SystemExit(2)
        return

    viz = DatasetVisualizer(dataset_dir, args.output_dir)
    samples = viz.available_samples()
    if args.sample_id:
        samples = [s for s in args.sample_id if s in samples or True]
        samples = args.sample_id
    elif args.max_samples is not None:
        samples = samples[: args.max_samples]

    if not samples and not args.categories_only:
        raise SystemExit("No rendered objects to visualize.")

    print(f"\nDataset:  {dataset_dir}")
    print(f"Output:   {args.output_dir.resolve()}")
    print(f"Samples:  {len(samples) if samples else 'categories figure only'}")

    saved: list[Path] = []

    groups = group_samples_by_category(viz, include_placeholder=args.include_placeholder)
    if groups:
        cat_path = args.output_dir / "categories_overview.png"
        saved.append(
            build_figure(
                viz,
                groups,
                per_category=args.per_category,
                view_index=18,
                output_path=cat_path.resolve(),
                dpi=150,
            )
        )
        print(f"Categories: {len(groups)} synset(s) with rendered objects")

    if not args.categories_only and samples:
        saved.extend(
            viz.run(
                sample_ids=samples,
                max_samples=None,
                make_gallery=not args.no_gallery,
            )
        )

    print("\nSaved:")
    for p in saved:
        print(f"  {p}")

    if args.open and not args.no_gallery and not args.categories_only:
        viz.open_gallery()


if __name__ == "__main__":
    main()
