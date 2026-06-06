#!/usr/bin/env python3
"""
Validate rendered master dataset objects (local path or Modal volume audit).

Checks per object:
  - gt_points.npy shape (2048, 3), finite values
  - 64 eval views + matching pose files
  - train_views pool for train split (64 * jitter_per_bin)
  - meta.json present

Usage:
  python scripts/validate_master_dataset.py --dataset-dir data/modal_preview
  python scripts/validate_master_dataset.py --dataset-dir data/master_shapenet --max-samples 50
  modal run scripts/modal_build_dataset.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.camera import MAX_VIEWS, eval_view_indices
from src.data.master_splits import MAIN_CATEGORIES

DEFAULT_DATASET = ROOT / "data" / "master_shapenet"
EXPECTED_GT_SHAPE = (2048, 3)
NAME_MAP = {c["synset_id"]: c["name"] for c in MAIN_CATEGORIES}


def validate_sample(
    sample_dir: Path,
    *,
    split: str | None = None,
    jitter_per_bin: int = 2,
) -> list[str]:
    issues: list[str] = []
    if not sample_dir.is_dir():
        return ["missing sample directory"]

    meta_path = sample_dir / "meta.json"
    num_eval_views = MAX_VIEWS
    if meta_path.is_file():
        with meta_path.open() as f:
            meta = json.load(f)
        split = split or meta.get("split")
        jitter_per_bin = meta.get("jitter_per_bin", jitter_per_bin)
        num_eval_views = int(meta.get("num_eval_views", MAX_VIEWS))
    elif split is None:
        split = "train"

    expected_view_indices = eval_view_indices(num_eval_views)

    gt_path = sample_dir / "gt_points.npy"
    if not gt_path.is_file():
        issues.append("missing gt_points.npy")
    else:
        pts = np.load(gt_path)
        if pts.shape != EXPECTED_GT_SHAPE:
            issues.append(f"gt shape {pts.shape}, expected {EXPECTED_GT_SHAPE}")
        if not np.isfinite(pts).all():
            issues.append("non-finite gt points")
        r = float(np.linalg.norm(pts, axis=1).max())
        if r < 0.05 or r > 2.0:
            issues.append(f"suspicious gt radius max_r={r:.3f}")

    eval_pngs = sorted(
        p for p in sample_dir.glob("view_*.png") if "_pose" not in p.name
    )
    if len(eval_pngs) != len(expected_view_indices):
        issues.append(
            f"eval views {len(eval_pngs)}/{len(expected_view_indices)} "
            f"(indices {expected_view_indices[0]:02d}..{expected_view_indices[-1]:02d})"
        )

    missing_eval = [
        vi
        for vi in expected_view_indices
        if not (sample_dir / f"view_{vi:02d}.png").is_file()
        or not (sample_dir / f"view_{vi:02d}_pose.npy").is_file()
    ]
    if missing_eval:
        issues.append(
            f"missing eval view/pose for indices "
            f"{', '.join(f'{vi:02d}' for vi in missing_eval[:6])}"
            + (" ..." if len(missing_eval) > 6 else "")
        )

    if split == "train" and jitter_per_bin > 0:
        train_dir = sample_dir / "train_views"
        expected_train = len(expected_view_indices) * jitter_per_bin
        if not train_dir.is_dir():
            issues.append(f"missing train_views/ (expected {expected_train} images)")
        else:
            train_pngs = list(train_dir.glob("az*_el*_j*.png"))
            if len(train_pngs) < expected_train:
                issues.append(
                    f"train pool {len(train_pngs)}/{expected_train} jitter images"
                )
            missing_train_poses = [
                p
                for p in train_pngs
                if not p.with_name(p.stem + "_pose.npy").is_file()
            ]
            if missing_train_poses:
                issues.append(
                    f"missing {len(missing_train_poses)} train pose file(s)"
                )

    if not meta_path.is_file():
        issues.append("missing meta.json")

    return issues


def _split_for_sample(manifest: dict | None, sample_id: str) -> str | None:
    if manifest is None:
        return None
    for entry in manifest.get("entries", []):
        if entry.get("sample_id") == sample_id:
            return entry.get("split")
    return None


def validate_dataset_dir(
    dataset_dir: Path,
    *,
    max_samples: int | None = None,
    jitter_per_bin: int = 2,
) -> dict:
    dataset_dir = dataset_dir.resolve()
    objects_dir = dataset_dir / "objects"
    manifest_path = dataset_dir / "master_manifest.json"
    manifest = None
    if manifest_path.is_file():
        with manifest_path.open() as f:
            manifest = json.load(f)

    sample_ids = sorted(
        d.name
        for d in objects_dir.iterdir()
        if d.is_dir() and (d / "gt_points.npy").is_file()
    ) if objects_dir.is_dir() else []

    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]

    complete: list[str] = []
    incomplete: dict[str, list[str]] = {}
    by_synset: Counter[str] = Counter()
    complete_by_synset: Counter[str] = Counter()

    for sid in sample_ids:
        split = _split_for_sample(manifest, sid)
        issues = validate_sample(
            objects_dir / sid,
            split=split,
            jitter_per_bin=jitter_per_bin,
        )
        syn = sid.split("_", 1)[0]
        by_synset[syn] += 1
        if issues:
            incomplete[sid] = issues
        else:
            complete.append(sid)
            complete_by_synset[syn] += 1

    manifest_total = manifest.get("total_objects") if manifest else None
    return {
        "dataset_dir": str(dataset_dir),
        "manifest_total": manifest_total,
        "manifest_split_counts": manifest.get("split_counts") if manifest else None,
        "rendered_on_disk": len(sample_ids),
        "complete": len(complete),
        "incomplete": len(incomplete),
        "by_synset": dict(by_synset),
        "complete_by_synset": dict(complete_by_synset),
        "sample_issues": dict(list(incomplete.items())[:30]),
    }


def print_report(report: dict) -> None:
    print(f"\n=== Dataset validation: {report['dataset_dir']} ===")
    if report.get("manifest_total"):
        print(f"Manifest objects:  {report['manifest_total']}")
        print(f"Manifest splits:   {report.get('manifest_split_counts')}")
    print(f"Rendered on disk:  {report['rendered_on_disk']}")
    print(f"Complete:          {report['complete']}")
    print(f"Incomplete:        {report['incomplete']}")

    print("\nBy category (rendered / complete):")
    for syn in sorted(report["by_synset"]):
        name = NAME_MAP.get(syn, syn)
        rendered = report["by_synset"][syn]
        ok = report["complete_by_synset"].get(syn, 0)
        print(f"  {name:10} ({syn}): {ok}/{rendered} complete")

    issues = report.get("sample_issues") or {}
    if issues:
        print(f"\nFirst {len(issues)} incomplete sample(s):")
        for sid, errs in issues.items():
            print(f"  {sid}:")
            for e in errs:
                print(f"    - {e}")
    else:
        print("\nAll checked samples passed.")

    if report["incomplete"] == 0 and report["complete"] > 0:
        print("\nResult: OK — checked samples look good.")
    elif report["incomplete"] > 0:
        print(
            f"\nResult: {report['incomplete']} incomplete object(s) "
            "(often partial writes when volume ran out of inodes)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate master dataset renders")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--jitter-per-bin", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Print JSON report only")
    args = parser.parse_args()

    if not (args.dataset_dir / "objects").is_dir():
        raise SystemExit(f"No objects/ under {args.dataset_dir}")

    report = validate_dataset_dir(
        args.dataset_dir,
        max_samples=args.max_samples,
        jitter_per_bin=args.jitter_per_bin,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if report["complete"] == 0 and report["rendered_on_disk"] > 0:
        raise SystemExit(1)
    if report["incomplete"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
