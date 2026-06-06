#!/usr/bin/env python3
"""Evaluate a trained baseline on val / test splits (fixed eval view schedule)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_baseline import build_model, evaluate, get_device
from src.data.camera import SUPPORTED_VIEW_COUNTS
from src.data.dataset import ShapeNetReconstructionDataset, collate_batch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_single_view.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test_seen", "test_generalization"],
    )
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    num_views = data_cfg["num_views"]
    if num_views not in SUPPORTED_VIEW_COUNTS and not args.demo:
        print(
            f"Warning: num_views={num_views} not in {SUPPORTED_VIEW_COUNTS}; "
            "nested eval supports those K only."
        )

    device = get_device()

    if args.demo:
        from scripts.train_baseline import build_dataloaders

        _, val_loader = build_dataloaders(cfg, demo=True)
    else:
        ds = ShapeNetReconstructionDataset(
            processed_dir=data_cfg["processed_dir"],
            split=args.split,
            num_views=num_views,
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_val_samples"),
            augment=False,
        )
        val_loader = DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            collate_fn=collate_batch,
        )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate(model, val_loader, device, cfg["eval"]["f_score_threshold"])
    print(
        f"Split={args.split}  views={num_views}  "
        f"chamfer={metrics['chamfer']:.6f}  f_score={metrics['f_score']:.4f}"
    )


if __name__ == "__main__":
    main()
