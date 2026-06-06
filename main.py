"""Demo: load a batch from preprocessed ShapeNetCore point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from shapenet import ShapeNetPointCloudDataset

DEFAULT_MANIFEST = Path("data/manifests/shapenetcore_v2.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Run preprocessing first (see README.md)."
        )

    dataset = ShapeNetPointCloudDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    batch = next(iter(loader))
    points = batch["points"]
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch points shape: {tuple(points.shape)}")  # (B, 2048, 3)
    print(f"dtype: {points.dtype}")
    print(f"Sample synset_ids: {batch['synset_id'][:2]}")
    print(f"Sample model_ids: {batch['model_id'][:2]}")
    print(f"Max radius (first item): {points[0].norm(dim=-1).max().item():.4f}")


if __name__ == "__main__":
    main()
