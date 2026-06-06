"""Verify preprocessed point clouds against a manifest."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def verify_manifest(manifest: dict) -> None:
    """Spot-check preprocessed outputs."""
    entries = manifest["entries"]
    num_points = manifest["num_points"]
    existing = [e for e in entries if Path(e["pointcloud_path"]).is_file()]
    logger.info("Verified %d / %d point clouds on disk", len(existing), len(entries))

    if not existing:
        logger.warning("No point clouds found to verify.")
        return

    for e in existing[: min(5, len(existing))]:
        pts = np.load(e["pointcloud_path"])
        if pts.shape != (num_points, 3):
            raise ValueError(f"Bad shape {pts.shape} for {e['pointcloud_path']}")
        if not np.isfinite(pts).all():
            raise ValueError(f"Non-finite values in {e['pointcloud_path']}")
        radii = np.linalg.norm(pts, axis=1)
        logger.info(
            "Sample %s/%s: max radius %.4f",
            e["synset_id"],
            e["model_id"],
            float(radii.max()),
        )
