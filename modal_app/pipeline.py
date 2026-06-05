"""Modal functions for download, prepare, train, and eval."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .common import (
    HF_SECRET,
    REPO_ROOT,
    VOL_DATA,
    VOLUME_MOUNT,
    app,
    ensure_vol_dirs,
    image,
    resolve_config,
    volume,
)

GPU_TYPE = "A10G"
PREPARE_TIMEOUT = 6 * 60 * 60
TRAIN_TIMEOUT = 12 * 60 * 60


def _run_download(categories: list[str]) -> str:
    ensure_vol_dirs()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/download_shapenet_hf.py"),
        "--output-root",
        str(VOL_DATA / "ShapeNetCore.v2"),
        "--cache-dir",
        str(VOL_DATA / "hf_cache"),
        "--categories",
        *categories,
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    volume.commit()
    return f"Downloaded categories {categories} to {VOL_DATA / 'ShapeNetCore.v2'}"


def _run_prepare(
    num_views: int,
    max_samples: int,
    processed_name: str | None = None,
    render_mode: str = "rgb",
) -> str:
    ensure_vol_dirs()
    if processed_name is None:
        processed_name = "processed_shapenet_1view" if num_views == 1 else "processed_shapenet_2view"

    processed_dir = VOL_DATA / processed_name
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/prepare_dataset.py"),
        "--shapenet-root",
        str(VOL_DATA / "ShapeNetCore.v2"),
        "--processed-dir",
        str(processed_dir),
        "--num-views",
        str(num_views),
        "--max-samples",
        str(max_samples),
        "--render-mode",
        render_mode,
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    volume.commit()
    return f"Prepared {max_samples} samples ({num_views} views) at {processed_dir}"


def _run_train(config: str, demo: bool = False, max_epochs: int | None = None) -> str:
    ensure_vol_dirs()
    config_path = resolve_config(config)

    import yaml

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.train_baseline import train as run_train

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_train(cfg, demo=demo, max_epochs=max_epochs)
    volume.commit()
    return f"Training finished. Checkpoints saved to {cfg['experiment']['output_dir']}"


@app.function(
    image=image,
    volumes=VOLUME_MOUNT,
    secrets=[HF_SECRET],
    timeout=PREPARE_TIMEOUT,
    cpu=4,
)
def download(categories: str = "03001627") -> str:
    """Download ShapeNetCore synsets from Hugging Face into the Modal volume."""
    return _run_download(categories.split())


@app.function(
    image=image,
    volumes=VOLUME_MOUNT,
    timeout=PREPARE_TIMEOUT,
    cpu=8,
)
def prepare(
    num_views: int = 1,
    max_samples: int = 500,
    processed_name: str | None = None,
    render_mode: str = "rgb",
) -> str:
    """Render views + GT point clouds on the Modal volume."""
    return _run_prepare(num_views, max_samples, processed_name, render_mode)


@app.function(
    image=image,
    volumes=VOLUME_MOUNT,
    gpu=GPU_TYPE,
    timeout=TRAIN_TIMEOUT,
)
def train(
    config: str = "single_view",
    demo: bool = False,
    max_epochs: int | None = None,
) -> str:
    """Train a baseline on Modal GPU."""
    return _run_train(config, demo=demo, max_epochs=max_epochs)


@app.function(
    image=image,
    volumes=VOLUME_MOUNT,
    gpu=GPU_TYPE,
    timeout=60 * 60,
)
def evaluate(
    config: str = "single_view",
    checkpoint: str | None = None,
    demo: bool = False,
) -> str:
    """Evaluate a checkpoint on the validation split."""
    ensure_vol_dirs()
    config_path = resolve_config(config)

    import yaml

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.train_baseline import build_dataloaders, build_model, evaluate as run_eval, get_device

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if checkpoint is None:
        checkpoint = str(Path(cfg["experiment"]["output_dir"]) / "best.pt")

    device = get_device()
    _, val_loader = build_dataloaders(cfg, demo=demo)
    model = build_model(cfg).to(device)

    import torch

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("config") is not None:
        cfg = ckpt["config"]
        model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = run_eval(model, val_loader, device, cfg["eval"]["f_score_threshold"])
    return f"chamfer={metrics['chamfer']:.6f}  f_score={metrics['f_score']:.4f}"


@app.function(
    image=image,
    volumes=VOLUME_MOUNT,
    secrets=[HF_SECRET],
    gpu=GPU_TYPE,
    timeout=PREPARE_TIMEOUT + TRAIN_TIMEOUT,
    cpu=8,
)
def setup_and_train(
    max_samples: int = 500,
    config: str = "single_view",
    skip_download: bool = False,
) -> str:
    """Download data, prepare views, and train in one job."""
    if not skip_download:
        _run_download(["03001627"])

    import yaml

    with open(resolve_config(config)) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    num_views = int(data_cfg.get("num_views", 1))
    processed_dir = Path(data_cfg["processed_dir"])
    processed_name = processed_dir.name
    render_mode = data_cfg.get("render_mode", "rgb")
    _run_prepare(num_views, max_samples, processed_name, render_mode)

    return _run_train(config)


@app.local_entrypoint()
def main(
    command: str = "train",
    config: str = "single_view",
    num_views: int = 1,
    max_samples: int = 500,
    max_epochs: int | None = None,
    demo: bool = False,
    categories: str = "03001627",
    checkpoint: str | None = None,
    skip_download: bool = False,
) -> None:
    """
    Run the CS231n pipeline on Modal.

    Examples:
      modal run -m modal_app.pipeline --command download
      modal run -m modal_app.pipeline --command prepare --num-views 1 --max-samples 500
      modal run -m modal_app.pipeline --command train --config single_view
      modal run -m modal_app.pipeline --command eval --config single_view
      modal run -m modal_app.pipeline --command all --max-samples 500
      modal run -m modal_app.pipeline --command all --config 2view_silhouette --max-samples 500
      modal run -m modal_app.pipeline --command demo --max-epochs 1
    """
    if command == "download":
        print(download.remote(categories=categories))
    elif command == "prepare":
        print(prepare.remote(num_views=num_views, max_samples=max_samples))
    elif command == "train":
        print(train.remote(config=config, demo=demo, max_epochs=max_epochs))
    elif command == "eval":
        print(evaluate.remote(config=config, checkpoint=checkpoint, demo=demo))
    elif command == "all":
        print(setup_and_train.remote(max_samples=max_samples, config=config, skip_download=skip_download))
    elif command == "demo":
        print(train.remote(config=config, demo=True, max_epochs=max_epochs or 1))
    else:
        raise SystemExit(
            f"Unknown command: {command}. "
            "Use download | prepare | train | eval | all | demo"
        )
