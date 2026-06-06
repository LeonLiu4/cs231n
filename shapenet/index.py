"""Discover ShapeNetCore models and build manifests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_REL_PATH = Path("models") / "model_normalized.obj"
EXPECTED_MODEL_COUNT = 51_300
PREPROCESS_VERSION = "1"


def _is_synset_dir(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def discover_models(
    mesh_root: str | Path,
    max_models: int | None = None,
) -> list[dict[str, str]]:
    """Walk mesh_root and return entries with synset_id, model_id, mesh_path."""
    mesh_root = Path(mesh_root)
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"Mesh root not found: {mesh_root}")

    entries: list[dict[str, str]] = []
    for synset_dir in sorted(mesh_root.iterdir()):
        if not synset_dir.is_dir() or not _is_synset_dir(synset_dir.name):
            continue
        synset_id = synset_dir.name
        for model_dir in sorted(synset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            mesh_path = model_dir / MODEL_REL_PATH
            if not mesh_path.is_file():
                continue
            entries.append(
                {
                    "synset_id": synset_id,
                    "model_id": model_dir.name,
                    "mesh_path": str(mesh_path.resolve()),
                }
            )
            if max_models is not None and len(entries) >= max_models:
                return entries
    return entries


def build_manifest(
    mesh_root: str | Path,
    output_dir: str | Path,
    num_points: int = 2048,
    max_models: int | None = None,
) -> dict[str, Any]:
    """Build manifest dict from discovered meshes."""
    mesh_root = Path(mesh_root).resolve()
    output_dir = Path(output_dir).resolve()

    raw_entries = discover_models(mesh_root, max_models=max_models)
    entries = []
    for e in raw_entries:
        rel = Path(e["synset_id"]) / f"{e['model_id']}.npy"
        entries.append(
            {
                "synset_id": e["synset_id"],
                "model_id": e["model_id"],
                "mesh_path": e["mesh_path"],
                "pointcloud_path": str((output_dir / rel).resolve()),
            }
        )

    count = len(entries)
    if max_models is None and count < EXPECTED_MODEL_COUNT * 0.9:
        logger.warning(
            "Found %d models (expected ~%d). Download may be incomplete.",
            count,
            EXPECTED_MODEL_COUNT,
        )

    return {
        "version": PREPROCESS_VERSION,
        "mesh_root": str(mesh_root),
        "output_dir": str(output_dir),
        "num_points": num_points,
        "model_count": count,
        "entries": entries,
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest with %d entries to %s", manifest["model_count"], path)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)
