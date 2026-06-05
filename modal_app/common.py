"""Shared Modal image, volumes, and paths."""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "cs231n-reconstruction"
VOLUME_NAME = "cs231n-data"

VOL_ROOT = Path("/vol")
VOL_DATA = VOL_ROOT / "data"
VOL_RUNS = VOL_ROOT / "runs"
VOL_CACHE = VOL_ROOT / "cache"

# Paths inside the Modal container
REPO_ROOT = Path("/root")
CONFIG_DIR = REPO_ROOT / "configs"

# Paths on the local machine (modal_app/ is one level below repo root)
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(LOCAL_REPO_ROOT / "requirements.txt"))
    .add_local_dir(str(LOCAL_REPO_ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_dir(str(LOCAL_REPO_ROOT / "scripts"), remote_path="/root/scripts", copy=True)
    .add_local_dir(str(LOCAL_REPO_ROOT / "configs"), remote_path="/root/configs", copy=True)
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": str(VOL_CACHE / "hf"),
            "TORCH_HOME": str(VOL_CACHE / "torch"),
        }
    )
)

app = modal.App(APP_NAME)

VOLUME_MOUNT = {str(VOL_ROOT): volume}
HF_SECRET = modal.Secret.from_name("huggingface")

CONFIG_MAP = {
    "single_view": CONFIG_DIR / "modal_single_view.yaml",
    "1view": CONFIG_DIR / "modal_single_view.yaml",
    "single_view_silhouette": CONFIG_DIR / "modal_single_view_silhouette.yaml",
    "silhouette": CONFIG_DIR / "modal_single_view_silhouette.yaml",
    "2view": CONFIG_DIR / "modal_2view.yaml",
    "2view_silhouette": CONFIG_DIR / "modal_2view_silhouette.yaml",
    "2view_sil": CONFIG_DIR / "modal_2view_silhouette.yaml",
}


def ensure_vol_dirs() -> None:
    for path in (VOL_DATA, VOL_RUNS, VOL_CACHE, VOL_CACHE / "hf", VOL_CACHE / "torch"):
        path.mkdir(parents=True, exist_ok=True)


def resolve_config(config: str) -> Path:
    if config in CONFIG_MAP:
        return CONFIG_MAP[config]
    path = Path(config)
    if path.is_file():
        return path
    repo_path = CONFIG_DIR / config
    if repo_path.is_file():
        return repo_path
    repo_path = CONFIG_DIR / f"{config}.yaml"
    if repo_path.is_file():
        return repo_path
    raise FileNotFoundError(
        f"Unknown config '{config}'. Choose one of: {', '.join(CONFIG_MAP)} "
        "or pass a path like modal_single_view.yaml"
    )
