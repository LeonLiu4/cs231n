#!/usr/bin/env python3
"""One-command setup helper for ShapeNetCore baselines."""

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

    shapenet_root = root / "data" / "ShapeNetCore.v2" / "03001627"
    if shapenet_root.exists():
        print("\nShapeNetCore found. Preparing 1-view and 2-view processed subsets...")
        run(
            [
                sys.executable,
                str(root / "scripts/prepare_dataset.py"),
                "--num-views",
                "1",
                "--max-samples",
                "200",
                "--processed-dir",
                "data/processed_shapenet_1view",
            ]
        )
        run(
            [
                sys.executable,
                str(root / "scripts/prepare_dataset.py"),
                "--num-views",
                "2",
                "--max-samples",
                "200",
                "--processed-dir",
                "data/processed_shapenet_2view",
            ]
        )
        print("\nSetup complete. Train with:")
        print("  python scripts/train_baseline.py --config configs/baseline_single_view.yaml")
        print("  python scripts/train_baseline.py --config configs/baseline_2view.yaml")
        return

    print(
        "\nShapeNetCore not found at data/ShapeNetCore.v2/.\n"
        "Download chairs from Hugging Face:\n"
        "  hf auth login\n"
        "  python scripts/download_shapenet_hf.py --categories 03001627\n"
        "Then re-run: python scripts/setup_baseline.py\n"
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
