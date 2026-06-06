#!/usr/bin/env python3
"""Verify preprocessed ShapeNetCore point clouds against a manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapenet.index import load_manifest
from shapenet.verify import verify_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/shapenetcore_v2.json"),
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    verify_manifest(manifest)


if __name__ == "__main__":
    main()
