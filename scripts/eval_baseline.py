#!/usr/bin/env python3
"""Evaluate a trained single-view baseline checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_baseline import build_dataloaders, build_model, evaluate, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-view baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_single_view.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print("Warning: checkpoint has no saved config; using --config YAML.")

    device = get_device()
    _, val_loader = build_dataloaders(cfg, demo=args.demo)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate(model, val_loader, device, cfg["eval"]["f_score_threshold"])
    print(f"Validation metrics: chamfer={metrics['chamfer']:.6f}  f_score={metrics['f_score']:.4f}")


if __name__ == "__main__":
    main()
