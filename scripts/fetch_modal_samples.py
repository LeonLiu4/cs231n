#!/usr/bin/env python3
"""
Download rendered objects from the Modal volume for local inspection.

Examples:
  python scripts/fetch_modal_samples.py --max-samples 3
  python scripts/fetch_modal_samples.py --sample-id 03001627_4e50015368a4f3ea4eb6addc0d23d122
  python scripts/fetch_modal_samples.py --max-samples 5 --visualize --open
  python scripts/fetch_modal_samples.py --max-samples 2 --light   # skip train_views/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME = "pose-aware-data-v2"
LEGACY_VOLUME = "pose-aware-data"
DEFAULT_PREFIX = "master_shapenet"
DEFAULT_OUT = ROOT / "data" / "modal_preview"


def _run_modal(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = ["modal", "volume", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def list_sample_ids(volume: str, prefix: str) -> list[str]:
    remote = f"{prefix}/objects"
    result = _run_modal("ls", volume, remote)
    ids: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # e.g. master_shapenet/objects/03001627_abc123
        name = line.rstrip("/").split("/")[-1]
        if name and name != "objects":
            ids.append(name)
    return sorted(ids)


def fetch_remote_path(
    volume: str,
    remote_path: str,
    local_dest: Path,
    *,
    force: bool = False,
) -> None:
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["get"]
    if force:
        args.append("--force")
    args.extend([volume, remote_path, str(local_dest)])
    _run_modal(*args)


def fetch_sample(
    volume: str,
    prefix: str,
    sample_id: str,
    out_dir: Path,
    *,
    light: bool,
    force: bool,
) -> Path:
    remote = f"{prefix}/objects/{sample_id}"
    objects_dir = out_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    local = objects_dir / sample_id
    if local.exists() and force:
        shutil.rmtree(local)
    # Modal writes into an existing parent dir; do not pre-create `local` as empty.
    fetch_remote_path(volume, remote, objects_dir, force=force)
    if light:
        train_views = local / "train_views"
        if train_views.is_dir():
            shutil.rmtree(train_views)
    return local


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sample objects from Modal volume pose-aware-data"
    )
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Local dataset root (default: {DEFAULT_OUT.relative_to(ROOT)})",
    )
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="How many objects to download when --sample-id is not set (default: 5)",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Skip train_views/ (enough for visualize_dataset.py)",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not download master_manifest.json",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run scripts/visualize_dataset.py after download",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="With --visualize, open the HTML gallery in a browser",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_id:
        sample_ids = args.sample_id
    else:
        all_ids = list_sample_ids(args.volume, args.prefix)
        if not all_ids:
            raise SystemExit(
                f"No rendered objects under {args.prefix}/objects on volume {args.volume!r}."
            )
        n = args.max_samples if args.max_samples is not None else len(all_ids)
        sample_ids = all_ids[:n]
        print(f"Found {len(all_ids)} object(s) on volume; fetching {len(sample_ids)}.")

    if not args.skip_manifest:
        manifest_remote = f"{args.prefix}/master_manifest.json"
        manifest_local = out_dir / "master_manifest.json"
        try:
            fetch_remote_path(
                args.volume, manifest_remote, manifest_local, force=args.force
            )
            print(f"Manifest -> {manifest_local}")
        except subprocess.CalledProcessError:
            print("Warning: master_manifest.json not on volume yet (optional).")

    for sid in sample_ids:
        print(f"Fetching {sid} ...")
        local = fetch_sample(
            args.volume,
            args.prefix,
            sid,
            out_dir,
            light=args.light,
            force=args.force,
        )
        print(f"  -> {local}")

    print(f"\nDone. Local preview dataset: {out_dir}")
    print(
        "Quick look at one render:\n"
        f"  open {out_dir / 'objects' / sample_ids[0] / 'view_00.png'}"
    )

    if args.visualize:
        viz_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "visualize_dataset.py"),
            "--dataset-dir",
            str(out_dir),
        ]
        if args.open:
            viz_cmd.append("--open")
        if args.max_samples is not None and not args.sample_id:
            viz_cmd.extend(["--max-samples", str(len(sample_ids))])
        print("\nRunning visualize_dataset.py ...")
        subprocess.run(viz_cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
