"""
Build the master ShapeNet dataset on a Modal volume (parallel cloud build).

Volume layout:
  /data/ShapeNetCore.v2/   raw meshes
  /data/master_shapenet/   built dataset

Parallelism:
  - Downloads: one container per synset (up to download_parallelism)
  - Renders: one container per batch of objects (up to render_parallelism)

Setup:
  pip install -r requirements-modal.txt
  modal secret create huggingface HF_TOKEN=<your_hf_token>

Run:
  modal run scripts/modal_build_dataset.py
  modal run scripts/modal_build_dataset.py --skip-download
  modal run scripts/modal_build_dataset.py --max-render 40 --batch-size 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

VOLUME_NAME = "pose-aware-data-v2"
LEGACY_VOLUME_NAME = "pose-aware-data"
VOL_MOUNT = "/data"
SHAPENET_ROOT = f"{VOL_MOUNT}/ShapeNetCore.v2"
MASTER_DATASET_ROOT = f"{VOL_MOUNT}/master_shapenet"
HF_CACHE = f"{VOL_MOUNT}/hf_cache"

DEFAULT_BATCH_SIZE = 8
DEFAULT_RENDER_PARALLELISM = 120
DEFAULT_DOWNLOAD_PARALLELISM = 20

app = modal.App("pose-aware-master-dataset")
# v1 volumes cap at 500k inodes; full dataset needs ~1M+ files (train jitter pool).
data_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True, version=2
)
legacy_volume = modal.Volume.from_name(LEGACY_VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "trimesh>=4.0.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.20.0",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/root/scripts")
)


def _all_synsets() -> list[str]:
    from scripts.download_shapenet_hf import MASTER_MAIN_SYNSETS
    from src.data.master_splits import BROAD_SYNSETS

    return [*MASTER_MAIN_SYNSETS, *BROAD_SYNSETS]


@app.function(
    image=image,
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 10,
    cpu=1,
    memory=2048,
)
def synset_counts_on_volume() -> dict[str, int]:
    sys.path.insert(0, "/root")
    from scripts.download_shapenet_hf import verify_category

    data_volume.reload()
    shapenet_root = Path(SHAPENET_ROOT)
    return {
        synset: verify_category(shapenet_root, synset)
        for synset in _all_synsets()
    }


@app.function(
    image=image,
    volumes={VOL_MOUNT: data_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 3,
    cpu=2,
    memory=4096,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    max_containers=DEFAULT_DOWNLOAD_PARALLELISM,
)
def download_synset(synset_id: str) -> dict[str, Any]:
    import os

    sys.path.insert(0, "/root")
    from scripts.download_shapenet_hf import download_categories, verify_category

    data_volume.reload()
    shapenet_root = Path(SHAPENET_ROOT)
    cache_dir = Path(HF_CACHE)
    existing = verify_category(shapenet_root, synset_id)
    if existing > 0:
        return {"synset_id": synset_id, "count": existing, "skipped": True}

    token = os.environ.get("HF_TOKEN")
    counts = download_categories(
        [synset_id], shapenet_root, cache_dir, token=token
    )
    data_volume.commit()
    return {
        "synset_id": synset_id,
        "count": counts[synset_id],
        "skipped": False,
    }


@app.function(
    image=image,
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 30,
    cpu=2,
    memory=4096,
)
def prepare_manifest(
    allow_partial: bool = False,
    jitter_per_bin: int = 2,
    seed: int = 42,
    scale_counts: float = 1.0,
) -> dict[str, Any]:
    sys.path.insert(0, "/root")
    from src.data.master_splits import build_master_records, save_master_manifest, summarize_records

    data_volume.reload()
    records = build_master_records(
        Path(SHAPENET_ROOT),
        seed=seed,
        scale_counts=scale_counts,
        allow_partial=allow_partial,
    )
    summary = summarize_records(records)
    manifest_path = save_master_manifest(
        records, Path(MASTER_DATASET_ROOT), jitter_per_bin=jitter_per_bin
    )
    data_volume.commit()
    return {
        "summary": summary,
        "manifest_path": str(manifest_path),
        "records": [r.to_dict() for r in records],
    }


@app.function(
    image=image,
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 45,
    cpu=2,
    memory=4096,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=3.0),
    max_containers=DEFAULT_RENDER_PARALLELISM,
)
def render_batch(
    records: list[dict[str, str]],
    seed: int = 42,
    num_gt_points: int = 2048,
    image_size: int = 224,
    jitter_per_bin: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    sys.path.insert(0, "/root")
    from scripts.build_master_dataset import process_sample

    data_volume.reload()
    objects_dir = Path(MASTER_DATASET_ROOT) / "objects"
    rendered = skipped = failed = 0
    errors: list[dict[str, str]] = []

    for rec in records:
        sample_dir = objects_dir / rec["sample_id"]
        try:
            changed = process_sample(
                Path(rec["mesh_path"]),
                sample_dir,
                split=rec["split"],
                sample_id=rec["sample_id"],
                seed=seed,
                num_gt_points=num_gt_points,
                image_size=image_size,
                jitter_per_bin=jitter_per_bin,
                overwrite=overwrite,
            )
            if changed:
                rendered += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            errors.append({"sample_id": rec["sample_id"], "error": str(exc)})

    data_volume.commit()
    return {
        "rendered": rendered,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _map_in_waves(
    fn,
    items: list,
    *,
    wave_size: int,
    map_kwargs: dict | None = None,
) -> list:
    """Run Modal .map() in waves; Modal 1.x map() has no parallelism kwarg."""
    if not items:
        return []
    map_kwargs = map_kwargs or {}
    results: list = []
    for wave in _chunk(items, max(1, wave_size)):
        results.extend(list(fn.map(wave, **map_kwargs)))
    return results


@app.function(
    image=image,
    volumes={VOL_MOUNT: data_volume},
    timeout=60 * 30,
    cpu=2,
    memory=4096,
)
def audit_dataset(max_check: int | None = None) -> dict[str, Any]:
    """Scan rendered objects on the active volume and report completeness."""
    sys.path.insert(0, "/root")
    from scripts.validate_master_dataset import validate_dataset_dir

    data_volume.reload()
    report = validate_dataset_dir(
        Path(MASTER_DATASET_ROOT),
        max_samples=max_check,
    )
    data_volume.commit()
    return report


@app.function(
    image=image,
    volumes={VOL_MOUNT: data_volume, "/legacy": legacy_volume},
    timeout=60 * 60 * 12,
    cpu=4,
    memory=8192,
)
def migrate_from_legacy(subpath: str, *, force: bool = False) -> dict[str, Any]:
    """Copy one top-level tree from the v1 volume into the v2 volume."""
    import shutil

    data_volume.reload()
    legacy_volume.reload()
    src = Path("/legacy") / subpath
    dst = Path(VOL_MOUNT) / subpath
    if not src.exists():
        return {"subpath": subpath, "skipped": True, "reason": "missing on legacy"}
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            return {"subpath": subpath, "skipped": True, "reason": "already on v2"}
    shutil.copytree(src, dst, dirs_exist_ok=False)
    data_volume.commit()
    return {"subpath": subpath, "copied": True, "bytes": _tree_size(dst)}


def _tree_size(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _aggregate_batch_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = skipped = failed = 0
    errors: list[dict[str, str]] = []
    for result in results:
        rendered += result.get("rendered", 0)
        skipped += result.get("skipped", 0)
        failed += result.get("failed", 0)
        errors.extend(result.get("errors", []))
    return {
        "rendered": rendered,
        "skipped": skipped,
        "failed": failed,
        "num_errors": len(errors),
        "errors": errors[:20],
    }


@app.function(
    image=image,
    volumes={"/legacy": legacy_volume},
    timeout=60 * 30,
    cpu=2,
    memory=4096,
)
def audit_legacy_dataset(max_check: int | None = None) -> dict[str, Any]:
    sys.path.insert(0, "/root")
    from scripts.validate_master_dataset import validate_dataset_dir

    legacy_volume.reload()
    return validate_dataset_dir(
        Path("/legacy/master_shapenet"),
        max_samples=max_check,
    )


@app.local_entrypoint()
def main(
    skip_download: bool = False,
    allow_partial: bool = True,
    jitter_per_bin: int = 2,
    overwrite: bool = False,
    max_render: int | None = None,
    manifest_only: bool = False,
    audit_only: bool = False,
    audit_legacy: bool = False,
    migrate_legacy: bool = False,
    migrate_force: bool = False,
    audit_max: int | None = None,
    seed: int = 42,
    scale_counts: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    render_parallelism: int = DEFAULT_RENDER_PARALLELISM,
    download_parallelism: int = DEFAULT_DOWNLOAD_PARALLELISM,
) -> None:
    if migrate_legacy:
        trees = ["ShapeNetCore.v2", "master_shapenet"]
        print(f"Migrating {trees} from {LEGACY_VOLUME_NAME} -> {VOLUME_NAME} ...")
        results = list(
            migrate_from_legacy.map(trees, kwargs={"force": migrate_force})
        )
        print(json.dumps(results, indent=2))
        return

    if audit_only:
        fn = audit_legacy_dataset if audit_legacy else audit_dataset
        report = fn.remote(max_check=audit_max)
        from scripts.validate_master_dataset import print_report

        vol = LEGACY_VOLUME_NAME if audit_legacy else VOLUME_NAME
        print(f"Volume: {vol}")
        print_report(report)
        return

    if not skip_download:
        counts = synset_counts_on_volume.remote()
        to_fetch = [synset for synset, n in counts.items() if n == 0]
        if to_fetch:
            print(f"Downloading {len(to_fetch)} synsets in parallel (limit={download_parallelism})...")
            dl_results = _map_in_waves(
                download_synset,
                to_fetch,
                wave_size=download_parallelism,
                map_kwargs={"return_exceptions": True},
            )
            ok = [r for r in dl_results if isinstance(r, dict)]
            bad = [r for r in dl_results if not isinstance(r, dict)]
            print(f"Download finished: {len(ok)} ok, {len(bad)} errors")
            if bad:
                print("First download error:", bad[0])
        else:
            print("All synsets already on volume; skipping download.")

    manifest = prepare_manifest.remote(
        allow_partial=allow_partial,
        jitter_per_bin=jitter_per_bin,
        seed=seed,
        scale_counts=scale_counts,
    )
    print("Split summary:", json.dumps(manifest["summary"], indent=2))

    if manifest_only:
        print(f"Manifest only: {manifest['manifest_path']}")
        return

    records: list[dict[str, str]] = manifest["records"]
    if max_render is not None:
        records = records[:max_render]

    batches = _chunk(records, max(1, batch_size))
    print(
        f"Rendering {len(records)} objects in {len(batches)} batches "
        f"(batch_size={batch_size}, parallelism={render_parallelism})..."
    )

    batch_results = _map_in_waves(
        render_batch,
        batches,
        wave_size=render_parallelism,
        map_kwargs={
            "kwargs": {
                "seed": seed,
                "jitter_per_bin": jitter_per_bin,
                "overwrite": overwrite,
            },
            "return_exceptions": True,
        },
    )

    ok_results = [r for r in batch_results if isinstance(r, dict)]
    bad_results = [r for r in batch_results if not isinstance(r, dict)]
    stats = _aggregate_batch_results(ok_results)
    stats["batch_errors"] = len(bad_results)
    if bad_results:
        stats["first_batch_error"] = str(bad_results[0])

    print("Build finished:")
    print(json.dumps(stats, indent=2))
    print(f"\nDataset on Modal volume '{VOLUME_NAME}':")
    print(f"  meshes:  {SHAPENET_ROOT}")
    print(f"  dataset: {MASTER_DATASET_ROOT}")
