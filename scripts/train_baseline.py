#!/usr/bin/env python3
"""Train the single-view RGB reconstruction baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import DemoReconstructionDataset, ShapeNetReconstructionDataset, collate_batch
from src.losses.chamfer import chamfer_distance
from src.metrics.reconstruction import f_score
from src.models.multi_view_baseline import MultiViewBaseline
from src.models.single_view_baseline import SingleViewBaseline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(cfg: dict, demo: bool) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    augment = data_cfg.get("augment", False)

    if demo:
        train_ds = DemoReconstructionDataset(
            num_samples=64,
            num_views=data_cfg["num_views"],
            num_gt_points=data_cfg["num_gt_points"],
            image_size=data_cfg["image_size"],
            seed=cfg["experiment"]["seed"],
        )
        val_ds = DemoReconstructionDataset(
            num_samples=16,
            num_views=data_cfg["num_views"],
            num_gt_points=data_cfg["num_gt_points"],
            image_size=data_cfg["image_size"],
            seed=cfg["experiment"]["seed"] + 1,
        )
    else:
        train_ds = ShapeNetReconstructionDataset(
            processed_dir=data_cfg["processed_dir"],
            split="train",
            num_views=data_cfg["num_views"],
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_train_samples"),
            augment=augment,
        )
        val_ds = ShapeNetReconstructionDataset(
            processed_dir=data_cfg["processed_dir"],
            split="val",
            num_views=data_cfg["num_views"],
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_val_samples"),
            augment=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    num_views = cfg["data"]["num_views"]
    common = dict(
        dinov2_variant=model_cfg["dinov2_variant"],
        num_output_points=model_cfg["num_output_points"],
        hidden_dim=model_cfg["hidden_dim"],
        freeze_backbone=model_cfg["freeze_backbone"],
        feature_mode=model_cfg.get("feature_mode", "cls"),
        head_type=model_cfg.get("head_type", "mlp"),
        unfreeze_last_blocks=model_cfg.get("unfreeze_last_blocks", 0),
    )
    if num_views <= 1:
        return SingleViewBaseline(**common)
    return MultiViewBaseline(
        **common,
        fusion=model_cfg.get("fusion", "mean_pool"),
        num_views=num_views,
        fusion_weights=model_cfg.get("fusion_weights"),
    )


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    default_lr = train_cfg.get("lr", train_cfg.get("head_lr", 1.0e-4))
    head_lr = train_cfg.get("head_lr", default_lr)
    backbone_lr = train_cfg.get("backbone_lr", default_lr * 0.1)

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    return torch.optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    train_cfg = cfg["train"]
    scheduler_type = train_cfg.get("scheduler", "cosine")

    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_cfg.get("plateau_factor", 0.5),
            patience=train_cfg.get("plateau_patience", 5),
            min_lr=train_cfg.get("min_lr", 1.0e-6),
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["num_epochs"],
    )


@torch.no_grad()
def evaluate(model, loader, device, f_threshold: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_f = 0.0
    count = 0

    for batch in loader:
        images = batch["images"].to(device)
        gt_points = batch["gt_points"].to(device)
        pred_points = model(images, batch["poses"].to(device))

        loss, _, _ = chamfer_distance(pred_points, gt_points)
        total_loss += loss.item()
        total_f += f_score(pred_points, gt_points, threshold=f_threshold).item()
        count += 1

    return {
        "chamfer": total_loss / max(count, 1),
        "f_score": total_f / max(count, 1),
    }


def current_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [group["lr"] for group in optimizer.param_groups]


def train(cfg: dict, demo: bool = False, max_epochs: int | None = None) -> None:
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg, demo=demo)
    model_cfg = cfg["model"]

    model = build_model(cfg).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scheduler_type = cfg["train"].get("scheduler", "cosine")
    grad_clip = cfg["train"].get("grad_clip")

    print(
        f"Model: views={cfg['data']['num_views']}, "
        f"fusion={model_cfg.get('fusion', 'n/a')}, "
        f"head={model_cfg.get('head_type', 'mlp')}, "
        f"features={model_cfg.get('feature_mode', 'cls')}, "
        f"unfreeze_blocks={model_cfg.get('unfreeze_last_blocks', 0)}"
    )
    print(f"LRs: head={current_lrs(optimizer)[0]:.2e}", end="")
    if len(current_lrs(optimizer)) > 1:
        print(f", backbone={current_lrs(optimizer)[1]:.2e}", end="")
    print()

    best_val = float("inf")
    patience = cfg["train"]["early_stopping_patience"]
    stale_epochs = 0
    history = []

    num_epochs = max_epochs or cfg["train"]["num_epochs"]
    for epoch in range(1, num_epochs + 1):
        model.train()
        if not model.encoder.has_trainable_backbone:
            model.encoder.eval()
        elif hasattr(model.encoder, "model"):
            model.encoder.model.train()

        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
            images = batch["images"].to(device)
            gt_points = batch["gt_points"].to(device)

            pred_points = model(images, batch["poses"].to(device))
            loss, _, _ = chamfer_distance(pred_points, gt_points)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % cfg["train"]["log_every"] == 0:
                print(f"  step {step + 1}: train chamfer={loss.item():.6f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, cfg["eval"]["f_score_threshold"])

        if scheduler_type == "plateau":
            scheduler.step(val_metrics["chamfer"])
        else:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_chamfer": train_loss,
            "val_chamfer": val_metrics["chamfer"],
            "val_f_score": val_metrics["f_score"],
            "lr_head": current_lrs(optimizer)[0],
            "lr_backbone": current_lrs(optimizer)[1] if len(current_lrs(optimizer)) > 1 else None,
        }
        history.append(record)
        print(
            f"Epoch {epoch}: train_chamfer={train_loss:.6f} "
            f"val_chamfer={val_metrics['chamfer']:.6f} "
            f"val_f={val_metrics['f_score']:.4f} "
            f"lr_head={current_lrs(optimizer)[0]:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_metrics["chamfer"] < best_val:
            best_val = val_metrics["chamfer"]
            stale_epochs = 0
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  saved new best checkpoint (val_chamfer={best_val:.6f})")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        if epoch % cfg["train"]["save_every"] == 0:
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Artifacts saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-view baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_single_view.yaml"))
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, demo=args.demo, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
