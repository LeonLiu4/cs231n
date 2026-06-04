#!/usr/bin/env python3
"""Verify ShapeNetCore.v2 layout (meshes should be downloaded via download_shapenet_hf.py)."""

from __future__ import annotations

import argparse
from pathlib import Path


def verify_shapenet(root: Path, category: str) -> int:
    category_dir = root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category folder not found: {category_dir}")

    count = 0
    for model_dir in category_dir.iterdir():
        if not model_dir.is_dir():
            continue
        obj = model_dir / "models" / "model_normalized.obj"
        if obj.exists():
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ShapeNetCore.v2 installation")
    parser.add_argument(
        "--shapenet-root",
        type=Path,
        default=Path("data/ShapeNetCore.v2"),
    )
    parser.add_argument("--category", type=str, default="03001627")
    args = parser.parse_args()

    if not args.shapenet_root.exists():
        print(f"ShapeNet root not found at {args.shapenet_root.resolve()}")
        print("Download from Hugging Face:")
        print("  hf auth login")
        print("  python scripts/download_shapenet_hf.py --categories 03001627")
        return

    count = verify_shapenet(args.shapenet_root, args.category)
    print(f"Found {count} valid models in category {args.category} under {args.shapenet_root}")


if __name__ == "__main__":
    main()
