"""
Train and evaluate reconstruction baselines on Modal GPU with the master dataset volume.

The master dataset and checkpoints live on Modal volume ``pose-aware-data-v2`` at ``/data/``.

Setup:
  pip install -r requirements-modal.txt
  modal secret create huggingface HF_TOKEN=<token>   # optional, dataset already built

Examples:
  # One baseline (full training)
  modal run scripts/modal_train_baseline.py --config configs/baselines/b5_multi_view.yaml

  # All baselines sequentially on one GPU job
  modal run scripts/modal_train_baseline.py --all-baselines

  # Quick smoke test (synthetic demo data, 1 epoch)
  modal run scripts/modal_train_baseline.py --config configs/baselines/b5_multi_view.yaml --demo --epochs 1

  # Evaluate an existing checkpoint only
  modal run scripts/modal_train_baseline.py --config configs/baselines/b5_multi_view.yaml --skip-train

  # Fast run (~15-30 min/baseline, all six in parallel under 2 h wall clock)
  modal run --detach scripts/modal_train_baseline.py --config configs/baselines/b5_multi_view.yaml --fast
  scripts/baselines/run_all_modal_fast.sh
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import modal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

VOLUME_NAME = "pose-aware-data-v2"
VOL_MOUNT = "/data"
MASTER_DATASET = f"{VOL_MOUNT}/master_shapenet"
RUNS_ROOT = f"{VOL_MOUNT}/runs"

BASELINE_CONFIGS = [
    "configs/baselines/b1_2d_positional.yaml",
    "configs/baselines/b2_view_id.yaml",
    "configs/baselines/b3_camera_pose.yaml",
    "configs/baselines/b4_single_view.yaml",
    "configs/baselines/b5_multi_view.yaml",
    "configs/baselines/main_geometry_aware.yaml",
]

app = modal.App("pose-aware-baseline-train")
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
        "trimesh>=4.0.0",
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "TORCH_HOME": f"{VOL_MOUNT}/torch_cache",
        }
    )
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/root/scripts")
    .add_local_dir(REPO_ROOT / "configs", remote_path="/root/configs")
)


# Tuned for <=2 h wall clock when all six baselines run in parallel on A10G.
FAST_PRESET: dict[str, Any] = {
    "epochs": 6,
    "max_train_samples": 256,
    "max_val_samples": 64,
    "max_eval_samples": 80,
    "batch_size": 16,
    "early_stopping_patience": 2,
    "log_every": 4,
}

# Diagnostic: train hard on ONE category, longer, to prove the pipeline can
# produce recognizable shapes (isolates "bug" vs "under-trained / too broad").
SINGLE_CATEGORY_PRESET: dict[str, Any] = {
    "epochs": 60,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_eval_samples": None,
    "batch_size": 16,
    "early_stopping_patience": 15,
    "log_every": 10,
    # Folding head (now LayerNorm-fixed) yields surface structure instead of a blob.
    "head_type": "folding",
    # Fresh decoder needs a higher LR than the 1e-4 default to converge in a short run.
    "lr": 5.0e-4,
}

# Head-to-head: geometry-aware vs blind mean-pool at K=8 (pose info matters more).
COMPARE_PRESET: dict[str, Any] = {
    "epochs": 20,
    "max_train_samples": 500,
    "max_val_samples": 100,
    "max_eval_samples": 210,
    "batch_size": 8,
    "early_stopping_patience": 5,
    "log_every": 10,
    "num_views": 8,
}

COMPARE_CONFIGS = [
    "configs/baselines/b5_multi_view.yaml",
    "configs/baselines/main_geometry_aware.yaml",
]

# Reduced-protocol K-sweep for the Results tables/plots: real numbers, fast.
# (Full protocol would be 60 epochs / 980 objects / K up to 16.)
# Ray-Transformer sweep: the patch-token + Transformer model is larger than the
# old mean-pool CLS baselines, so it needs more data/epochs to leave the
# mean-shape regime. Still a reduced protocol for fast turnaround.
SWEEP_PRESET: dict[str, Any] = {
    "epochs": 60,
    "max_train_samples": 1000,
    "max_val_samples": 100,
    "max_eval_samples": 210,
    "batch_size": 12,
    "early_stopping_patience": 12,
    "log_every": 10,
    "lr": 3.0e-4,
}

# The four embedding variants in the writeup's Results tables.
SWEEP_CONFIGS = [
    "configs/baselines/rt_2d_positional.yaml",
    "configs/baselines/rt_view_id.yaml",
    "configs/baselines/rt_camera_pose.yaml",
    "configs/baselines/rt_geometry_aware.yaml",
]
SWEEP_VIEW_COUNTS = [1, 2, 4, 8]


def _rewrite_config_for_modal(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    processed = cfg["data"]["processed_dir"]
    if processed.startswith("data/"):
        cfg["data"]["processed_dir"] = f"{VOL_MOUNT}/{processed[5:]}"
    elif processed == "data/master_shapenet":
        cfg["data"]["processed_dir"] = MASTER_DATASET

    output_dir = cfg["experiment"]["output_dir"]
    if output_dir.startswith("runs/"):
        cfg["experiment"]["output_dir"] = f"{RUNS_ROOT}/{output_dir[5:]}"
    return cfg


def _apply_fast_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["train"]["num_epochs"] = FAST_PRESET["epochs"]
    cfg["train"]["early_stopping_patience"] = FAST_PRESET["early_stopping_patience"]
    cfg["train"]["batch_size"] = FAST_PRESET["batch_size"]
    cfg["train"]["log_every"] = FAST_PRESET["log_every"]
    cfg["data"]["max_train_samples"] = FAST_PRESET["max_train_samples"]
    cfg["data"]["max_val_samples"] = FAST_PRESET["max_val_samples"]
    cfg["data"]["max_eval_samples"] = FAST_PRESET["max_eval_samples"]
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_fast"
    out = cfg["experiment"]["output_dir"]
    cfg["experiment"]["output_dir"] = out if out.endswith("_fast") else f"{out}_fast"
    return cfg


def _apply_single_category_overrides(cfg: dict[str, Any], category: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    p = SINGLE_CATEGORY_PRESET
    cfg["train"]["num_epochs"] = p["epochs"]
    cfg["train"]["early_stopping_patience"] = p["early_stopping_patience"]
    cfg["train"]["batch_size"] = p["batch_size"]
    cfg["train"]["log_every"] = p["log_every"]
    cfg["data"]["max_train_samples"] = p["max_train_samples"]
    cfg["data"]["max_val_samples"] = p["max_val_samples"]
    cfg["data"]["max_eval_samples"] = p["max_eval_samples"]
    cfg["data"]["category"] = category
    cfg["model"]["head_type"] = p["head_type"]
    cfg["train"]["lr"] = p["lr"]
    tag = f"_cat_{category}"
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}{tag}"
    out = cfg["experiment"]["output_dir"]
    cfg["experiment"]["output_dir"] = out if out.endswith(tag) else f"{out}{tag}"
    return cfg


def _apply_sweep_overrides(cfg: dict[str, Any], num_views: int) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    p = SWEEP_PRESET
    cfg["train"]["num_epochs"] = p["epochs"]
    cfg["train"]["early_stopping_patience"] = p["early_stopping_patience"]
    cfg["train"]["batch_size"] = p["batch_size"]
    cfg["train"]["log_every"] = p["log_every"]
    cfg["train"]["lr"] = p["lr"]
    cfg["data"]["max_train_samples"] = p["max_train_samples"]
    cfg["data"]["max_val_samples"] = p["max_val_samples"]
    cfg["data"]["max_eval_samples"] = p["max_eval_samples"]
    cfg["data"]["num_views"] = num_views
    tag = f"_sweep_k{num_views}"
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}{tag}"
    out = cfg["experiment"]["output_dir"]
    cfg["experiment"]["output_dir"] = out if out.endswith(tag) else f"{out}{tag}"
    return cfg


def _apply_compare_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["train"]["num_epochs"] = COMPARE_PRESET["epochs"]
    cfg["train"]["early_stopping_patience"] = COMPARE_PRESET["early_stopping_patience"]
    cfg["train"]["batch_size"] = COMPARE_PRESET["batch_size"]
    cfg["train"]["log_every"] = COMPARE_PRESET["log_every"]
    cfg["data"]["max_train_samples"] = COMPARE_PRESET["max_train_samples"]
    cfg["data"]["max_val_samples"] = COMPARE_PRESET["max_val_samples"]
    cfg["data"]["max_eval_samples"] = COMPARE_PRESET["max_eval_samples"]
    cfg["data"]["num_views"] = COMPARE_PRESET["num_views"]
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_compare_k8"
    out = cfg["experiment"]["output_dir"]
    suffix = "_compare_k8"
    cfg["experiment"]["output_dir"] = out if out.endswith(suffix) else f"{out}{suffix}"
    return cfg


def _evaluate_checkpoint(cfg: dict[str, Any], demo: bool) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from scripts.train_baseline import build_dataloaders, build_model, evaluate, get_device
    from src.data.dataset import ShapeNetReconstructionDataset, collate_batch

    device = get_device()
    output_dir = Path(cfg["experiment"]["output_dir"])
    ckpt_path = output_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    model = build_model(cfg).to(device)
    ckpt = __import__("torch").load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    results: dict[str, Any] = {
        "name": cfg["experiment"]["name"],
        "output_dir": str(output_dir),
        "num_views": cfg["data"]["num_views"],
        "splits": {},
    }

    if demo:
        _, val_loader = build_dataloaders(cfg, demo=True)
        metrics = evaluate(model, val_loader, device, cfg["eval"]["f_score_threshold"])
        results["splits"]["val"] = metrics
        return results

    data_cfg = cfg["data"]
    eval_cap = data_cfg.get("max_eval_samples", data_cfg.get("max_val_samples"))
    category = data_cfg.get("category")
    # A single-category run only has held-out objects in test_seen.
    splits = ("test_seen",) if category else ("test_seen", "test_generalization")
    for split in splits:
        try:
            ds = ShapeNetReconstructionDataset(
                processed_dir=data_cfg["processed_dir"],
                split=split,
                num_views=data_cfg["num_views"],
                image_size=data_cfg["image_size"],
                max_samples=eval_cap,
                augment=False,
                category=category,
            )
        except ValueError as exc:
            print(f"[eval] skip split={split}: {exc}")
            continue
        loader = DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            collate_fn=collate_batch,
        )
        metrics = evaluate(model, loader, device, cfg["eval"]["f_score_threshold"])
        results["splits"][split] = metrics
    return results


def _train_and_eval_impl(
    config_relpath: str,
    demo: bool = False,
    epochs: int | None = None,
    skip_train: bool = False,
    fast: bool = False,
    compare: bool = False,
    category: str = "",
    sweep: bool = False,
    num_views: int = 0,
) -> dict[str, Any]:
    sys.path.insert(0, "/root")
    data_volume.reload()

    config_path = Path("/root") / config_relpath
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg = _rewrite_config_for_modal(cfg)
    if sweep:
        cfg = _apply_sweep_overrides(cfg, num_views)
    elif category:
        cfg = _apply_single_category_overrides(cfg, category)
    elif compare:
        cfg = _apply_compare_overrides(cfg)
    elif fast:
        cfg = _apply_fast_overrides(cfg)

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = (
        f"sweep_k{num_views}" if sweep
        else (f"category:{category}" if category
        else ("compare_k8" if compare else ("fast" if fast else "full")))
    )
    if sweep:
        print(
            f"Sweep preset: K={cfg['data']['num_views']}, "
            f"epochs={cfg['train']['num_epochs']}, "
            f"train_cap={cfg['data']['max_train_samples']}, "
            f"head={cfg['model'].get('head_type', 'mlp')}"
        )
    print(f"Config: {config_relpath}")
    print(f"Mode:   {mode}")
    print(f"Dataset: {cfg['data']['processed_dir']}")
    print(f"Output:  {output_dir}")
    if category:
        print(
            f"Single-category preset: cat={category}, "
            f"epochs={cfg['train']['num_epochs']}, "
            f"head={cfg['model'].get('head_type')}, "
            f"K={cfg['data']['num_views']}"
        )
    if compare:
        print(
            f"Compare preset: K={cfg['data']['num_views']}, "
            f"epochs={cfg['train']['num_epochs']}, "
            f"train_cap={cfg['data']['max_train_samples']}, "
            f"eval_cap={cfg['data'].get('max_eval_samples')}"
        )
    elif fast:
        print(
            f"Fast preset: epochs={cfg['train']['num_epochs']}, "
            f"train_cap={cfg['data']['max_train_samples']}, "
            f"eval_cap={cfg['data'].get('max_eval_samples')}, "
            f"batch={cfg['train']['batch_size']}"
        )

    if not skip_train:
        from scripts.train_baseline import train

        train(cfg, demo=demo, max_epochs=epochs)
        data_volume.commit()

    eval_results = _evaluate_checkpoint(cfg, demo=demo)
    eval_results["mode"] = mode
    results_path = output_dir / "eval_results.json"
    with results_path.open("w") as f:
        json.dump(eval_results, f, indent=2)
    data_volume.commit()

    print(json.dumps(eval_results, indent=2))
    return eval_results


@app.function(
    image=train_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 60 * 24,
    memory=32768,
)
def train_and_eval_baseline(
    config_relpath: str,
    demo: bool = False,
    epochs: int | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    return _train_and_eval_impl(
        config_relpath,
        demo=demo,
        epochs=epochs,
        skip_train=skip_train,
        fast=False,
        compare=False,
    )


@app.function(
    image=train_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 60 * 2,
    memory=32768,
)
def train_and_eval_baseline_fast(
    config_relpath: str,
    demo: bool = False,
    epochs: int | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    return _train_and_eval_impl(
        config_relpath,
        demo=demo,
        epochs=epochs,
        skip_train=skip_train,
        fast=True,
        compare=False,
    )


@app.function(
    image=train_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 60 * 2,
    memory=32768,
)
def train_and_eval_baseline_compare(
    config_relpath: str,
    demo: bool = False,
    epochs: int | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    return _train_and_eval_impl(
        config_relpath,
        demo=demo,
        epochs=epochs,
        skip_train=skip_train,
        fast=False,
        compare=True,
    )


@app.function(
    image=train_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 60 * 2,
    memory=32768,
)
def train_and_eval_baseline_category(
    config_relpath: str,
    category: str,
    demo: bool = False,
    epochs: int | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    return _train_and_eval_impl(
        config_relpath,
        demo=demo,
        epochs=epochs,
        skip_train=skip_train,
        fast=False,
        compare=False,
        category=category,
    )


@app.function(
    image=train_image,
    gpu="A10G",
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 60 * 3,
    memory=32768,
)
def train_and_eval_baseline_sweep(
    config_relpath: str,
    num_views: int,
    demo: bool = False,
    epochs: int | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    return _train_and_eval_impl(
        config_relpath,
        demo=demo,
        epochs=epochs,
        skip_train=skip_train,
        sweep=True,
        num_views=num_views,
    )


def _select_configs(all_baselines: bool, config: str, only: str) -> list[str]:
    if not all_baselines:
        return [config]
    tokens = [t.strip() for t in only.split() if t.strip()]
    if not tokens:
        return list(BASELINE_CONFIGS)
    return [c for c in BASELINE_CONFIGS if any(t in c for t in tokens)]


@app.local_entrypoint()
def main(
    config: str = "configs/baselines/b5_multi_view.yaml",
    all_baselines: bool = False,
    demo: bool = False,
    epochs: int = 0,
    skip_train: bool = False,
    only: str = "",
    fast: bool = False,
    compare: bool = False,
    category: str = "",
    sweep: bool = False,
    num_views: int = 0,
) -> None:
    epoch_override = epochs if epochs > 0 else None

    if sweep:
        # --sweep alone fans out ALL variants x view counts in parallel (spawn).
        # --sweep --num-views K runs a single (config, K) and blocks (debugging).
        if num_views <= 0:
            combos = [(c, k) for c in SWEEP_CONFIGS for k in SWEEP_VIEW_COUNTS]
            print("\n" + "=" * 70)
            print(f"Launching full ray-transformer sweep: {len(combos)} jobs in parallel")
            print("=" * 70)
            for c, k in combos:
                handle = train_and_eval_baseline_sweep.spawn(
                    c, k, demo=demo, epochs=epoch_override, skip_train=skip_train
                )
                print(f"  spawned {c} [K={k}] -> {handle.object_id}")
            print(
                "\nAll jobs spawned. Run with `modal run --detach` so they finish "
                "server-side, then collect with scripts/collect_sweep_results.py "
                "--from-modal."
            )
            return
        print("\n" + "=" * 70)
        print(f"Sweep run: {config} [K={num_views}]")
        print("=" * 70)
        result = train_and_eval_baseline_sweep.remote(
            config, num_views, demo=demo, epochs=epoch_override, skip_train=skip_train
        )
        print(json.dumps(result, indent=2))
        return

    if compare:
        configs = list(COMPARE_CONFIGS)
    elif category:
        # Diagnostic sweep over the key baselines on one category.
        configs = _select_configs(all_baselines, config, only)
    else:
        configs = _select_configs(all_baselines, config, only)

    all_results: list[dict[str, Any]] = []

    for config_relpath in configs:
        print("\n" + "=" * 70)
        tag = (
            f" [category {category}]" if category
            else (" [compare K=8]" if compare else (" [fast]" if fast else ""))
        )
        print(f"Starting {config_relpath}{tag}")
        print("=" * 70)
        if category:
            result = train_and_eval_baseline_category.remote(
                config_relpath,
                category,
                demo=demo,
                epochs=epoch_override,
                skip_train=skip_train,
            )
        else:
            runner = (
                train_and_eval_baseline_compare if compare
                else (train_and_eval_baseline_fast if fast else train_and_eval_baseline)
            )
            result = runner.remote(
                config_relpath,
                demo=demo,
                epochs=epoch_override,
                skip_train=skip_train,
            )
        all_results.append(result)
        print(json.dumps(result, indent=2))

    if len(all_results) > 1:
        if category:
            summary_name = f"category_{category}_results.json"
        elif compare:
            summary_name = "geometry_comparison_k8.json"
        elif fast:
            summary_name = "all_eval_results_fast.json"
        else:
            summary_name = "all_eval_results.json"
        summary_path = REPO_ROOT / "runs" / "baselines" / summary_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote local summary to {summary_path}")
        print("\nAll requested baselines finished.")
