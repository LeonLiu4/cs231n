#!/usr/bin/env python3
"""Download ShapeNetCore.v2 from Hugging Face."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "ShapeNet/ShapeNetCore"
DEFAULT_OUT = Path("data/shapenetcore")
MESH_SUBDIR_NAMES = ("ShapeNetCore.v2", "ShapeNetCoreV2", "ShapeNetCore")


def find_mesh_root(download_dir: Path) -> Path | None:
    """Locate ShapeNetCore.v2 root containing synset subdirectories."""
    for name in MESH_SUBDIR_NAMES:
        candidate = download_dir / name
        if candidate.is_dir() and any(_is_synset_child(c) for c in candidate.iterdir()):
            return candidate
    for candidate in download_dir.rglob("ShapeNetCore.v2"):
        if candidate.is_dir():
            return candidate
    if any(_is_synset_child(c) for c in download_dir.iterdir() if c.is_dir()):
        return download_dir
    return None


def _is_synset_child(path: Path) -> bool:
    return path.is_dir() and len(path.name) == 8 and path.name.isdigit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ShapeNetCore from Hugging Face")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory to store the dataset",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s to %s (~24 GB)...", REPO_ID, out)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(out),
    )

    mesh_root = find_mesh_root(out)
    if mesh_root is None:
        logger.warning(
            "Download finished but could not auto-detect mesh root under %s. "
            "Inspect the folder and pass --mesh-root to preprocess.",
            out,
        )
    else:
        logger.info("Mesh root for preprocessing: %s", mesh_root)
        logger.info(
            "Run: python scripts/preprocess_shapenet_pointclouds.py "
            '--mesh-root "%s" --output-dir data/pointclouds/shapenetcore_v2',
            mesh_root,
        )


if __name__ == "__main__":
    main()
