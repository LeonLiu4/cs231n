"""PyTorch Dataset for preprocessed ShapeNetCore point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from shapenet.index import load_manifest


class ShapeNetPointCloudDataset(Dataset):
    """Load ground-truth point clouds from a preprocessing manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        synset_ids: set[str] | None = None,
    ) -> None:
        manifest = load_manifest(manifest_path)
        entries = manifest["entries"]
        if synset_ids is not None:
            entries = [e for e in entries if e["synset_id"] in synset_ids]
        self.entries = entries
        self.num_points = manifest.get("num_points", 2048)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        points = np.load(entry["pointcloud_path"])
        points = torch.from_numpy(points).float()
        if self.transform is not None:
            points = self.transform(points)
        return {
            "points": points,
            "synset_id": entry["synset_id"],
            "model_id": entry["model_id"],
        }
