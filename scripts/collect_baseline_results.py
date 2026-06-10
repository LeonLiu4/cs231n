#!/usr/bin/env python3
"""Fetch baseline eval metrics from Modal volume or local runs/ into one JSON file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL = ROOT / "runs" / "baselines" / "all_eval_results.json"
BASELINE_NAMES = [
    "b1_2d_positional",
    "b2_view_id",
    "b3_camera_pose",
    "b4_single_view",
    "b5_multi_view",
    "main_geometry_aware",
]


def _baseline_dir_names(fast: bool) -> list[str]:
    if fast:
        return [f"{name}_fast" for name in BASELINE_NAMES]
    return list(BASELINE_NAMES)


def collect_local(runs_root: Path, fast: bool = False) -> list[dict]:
    results: list[dict] = []
    for name in _baseline_dir_names(fast):
        eval_path = runs_root / name / "eval_results.json"
        if eval_path.is_file():
            results.append(json.loads(eval_path.read_text()))
    return results


def fetch_from_modal(volume: str, dest: Path, fast: bool) -> None:
    remote_name = "all_eval_results_fast.json" if fast else "all_eval_results.json"
    remote = f"runs/baselines/{remote_name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["modal", "volume", "get", volume, remote, str(dest)],
            check=True,
        )
    except subprocess.CalledProcessError:
        merged: list[dict] = []
        for name in _baseline_dir_names(fast):
            per_run = dest.parent / f"{name}_eval.json"
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    volume,
                    f"runs/baselines/{name}/eval_results.json",
                    str(per_run),
                ],
                check=False,
            )
            if per_run.is_file():
                merged.append(json.loads(per_run.read_text()))
                per_run.unlink(missing_ok=True)
        if not merged:
            raise
        dest.write_text(json.dumps(merged, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline eval results")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs" / "baselines",
        help="Local runs/baselines directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LOCAL,
        help="Where to write merged JSON",
    )
    parser.add_argument(
        "--from-modal",
        action="store_true",
        help="Pull summary JSON from Modal volume first",
    )
    parser.add_argument(
        "--volume",
        default="pose-aware-data-v2",
        help="Modal volume name",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Collect fast-run results (*_fast dirs on volume)",
    )
    args = parser.parse_args()

    if args.from_modal:
        out = args.output
        if args.fast and out == DEFAULT_LOCAL:
            out = ROOT / "runs" / "baselines" / "all_eval_results_fast.json"
        fetch_from_modal(args.volume, out, fast=args.fast)
        print(f"Fetched {out}")
        return

    results = collect_local(args.runs_root, fast=args.fast)
    if not results:
        print(
            "No eval_results.json found under runs/baselines/.\n"
            "Train first, or run with --from-modal after Modal training.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} baseline results to {args.output}")
    for row in results:
        name = row.get("name", "?")
        for split, metrics in row.get("splits", {}).items():
            print(
                f"  {name:22s} {split:20s}  "
                f"chamfer={metrics['chamfer']:.6f}  f={metrics['f_score']:.4f}"
            )


if __name__ == "__main__":
    main()
