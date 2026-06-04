#!/usr/bin/env python3
"""One-command setup helper for the single-view baseline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")])

    print("\nChecking ShapeNet...")
    run([sys.executable, str(root / "scripts/download_shapenet.py")])

    shapenet_root = root / "data" / "ShapeNetCore.v2"
    partannotation_root = root / "data" / "PartAnnotation"
    category_dir = shapenet_root / "03001627"

    if partannotation_root.exists():
        print("\nPartAnnotation found. Preparing processed subset...")
        run(
            [
                sys.executable,
                str(root / "scripts/prepare_partannotation.py"),
                "--max-samples",
                "100",
                "--num-views",
                "1",
            ]
        )
    elif category_dir.exists():
        print("\nShapeNet found. Preparing a small processed subset (50 train objects)...")
        run(
            [
                sys.executable,
                str(root / "scripts/prepare_dataset.py"),
                "--max-samples",
                "50",
                "--num-views",
                "8",
            ]
        )
    else:
        print(
            "\nShapeNet not present yet. Running demo smoke test instead.\n"
            "After downloading ShapeNet, run:\n"
            "  python scripts/prepare_dataset.py\n"
            "  python scripts/train_baseline.py\n"
        )
        run(
            [
                sys.executable,
                str(root / "scripts/train_baseline.py"),
                "--demo",
                "--epochs",
                "1",
            ]
        )


if __name__ == "__main__":
    main()
