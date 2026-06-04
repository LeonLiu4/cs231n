#!/usr/bin/env python3
"""Download ShapeNetCore.v2 (requires manual registration)."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


SHAPENET_INFO = """
ShapeNetCore.v2 download
========================

1. Register at https://shapenet.org/
2. Accept the terms and download ShapeNetCore.v2
3. Either:
   a) Place the extracted folder at: data/ShapeNetCore.v2
   b) Pass --zip-path /path/to/ShapeNetCore.v2.zip to this script

Expected layout after extraction:
  data/ShapeNetCore.v2/
    03001627/          # chair synset
      <model_id>/
        models/
          model_normalized.obj

Useful synsets for this project:
  03001627  chair
  02691156  airplane
  02958343  car
"""


def extract_shapenet(zip_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_root)
    print(f"Extracted ShapeNet to {output_root}")


def verify_shapenet(root: Path, category: str) -> int:
    category_dir = root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category folder not found: {category_dir}")

    model_dirs = [p for p in category_dir.iterdir() if p.is_dir()]
    valid = 0
    for model_dir in model_dirs:
        obj = model_dir / "models" / "model_normalized.obj"
        if obj.exists():
            valid += 1
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup ShapeNetCore.v2")
    parser.add_argument(
        "--shapenet-root",
        type=Path,
        default=Path("data/ShapeNetCore.v2"),
        help="Target ShapeNet root directory",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional path to ShapeNetCore.v2.zip",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="03001627",
        help="Synset id to verify (default: chair)",
    )
    args = parser.parse_args()

    print(SHAPENET_INFO)

    if args.zip_path is not None:
        if not args.zip_path.exists():
            raise FileNotFoundError(f"Zip not found: {args.zip_path}")
        extract_shapenet(args.zip_path, args.shapenet_root)

    if not args.shapenet_root.exists():
        print(f"ShapeNet root not found at {args.shapenet_root.resolve()}")
        print("Follow the instructions above, then re-run with --zip-path or place files manually.")
        return

    count = verify_shapenet(args.shapenet_root, args.category)
    print(f"Found {count} valid models in category {args.category} under {args.shapenet_root}")


if __name__ == "__main__":
    main()
