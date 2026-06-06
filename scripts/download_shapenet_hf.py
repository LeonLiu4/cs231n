#!/usr/bin/env python3
"""Download ShapeNetCore from Hugging Face (ShapeNet/ShapeNetCore)."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ID = "ShapeNet/ShapeNetCore"

# Master dataset: seven main categories
MASTER_MAIN_SYNSETS = [
    "03001627",  # chair
    "04379243",  # table
    "04256520",  # sofa
    "02958343",  # car
    "02691156",  # airplane
    "03642806",  # lamp
    "02933112",  # cabinet
]
DEFAULT_CATEGORIES = MASTER_MAIN_SYNSETS[:1]  # chair only for quick tests

SYNSETS = {
    "02691156": "airplane",
    "02773838": "bag",
    "02933112": "cabinet",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03261776": "earphone",
    "03467517": "guitar",
    "03636649": "knife",
    "03642806": "lamp",
    "03790512": "motorbike",
    "03797390": "mug",
    "03948459": "pistol",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04379243": "table",
    "04530566": "watercraft",
}


def resolve_token(explicit_token: str | None) -> str | None:
    import os

    from huggingface_hub.utils import get_token

    if explicit_token:
        return explicit_token.strip()
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()
    return get_token()


def check_auth(token: str | None) -> str:
    resolved = resolve_token(token)
    if resolved is None:
        raise SystemExit(
            "Not logged in to Hugging Face.\n\n"
            "You tried `hf login` — that command does not exist. Use:\n\n"
            "  hf auth login\n\n"
            "Or skip interactive login and pass a token:\n\n"
            "  export HF_TOKEN=hf_...\n"
            "  python3 scripts/download_shapenet_hf.py --categories 03001627\n\n"
            "Create a token at https://huggingface.co/settings/tokens\n"
            "Accept dataset access at https://huggingface.co/datasets/ShapeNet/ShapeNetCore\n\n"
            "Verify login with: hf auth whoami"
        )
    return resolved


def download_synset_zip(category: str, cache_dir: Path, token: str) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    zip_name = f"{category}.zip"
    print(f"Downloading {zip_name} from {REPO_ID} ...")
    try:
        return Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_name,
                repo_type="dataset",
                local_dir=cache_dir,
                token=token,
            )
        )
    except HfHubHTTPError as exc:
        if "403" in str(exc) and "gated" in str(exc).lower():
            raise SystemExit(
                "403 Forbidden: your HF token cannot access gated datasets.\n\n"
                "Fix (pick one):\n"
                "  A) Create a classic token with Read access:\n"
                "     https://huggingface.co/settings/tokens -> New token -> Read\n"
                "  B) If using a fine-grained token, enable:\n"
                "     'Access public gated repositories' in token settings\n\n"
                "Also confirm you accepted access at:\n"
                "  https://huggingface.co/datasets/ShapeNet/ShapeNetCore\n\n"
                "Then re-login:\n"
                "  hf auth login\n"
                "  python3 scripts/download_shapenet_hf.py --categories 03001627"
            ) from exc
        raise


def extract_synset(zip_path: Path, output_root: Path, category: str) -> Path:
    category_dir = output_root / category
    category_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        # Some zips contain 03001627/<id>/... ; others may nest one level deeper.
        zf.extractall(category_dir)

    # Flatten if zip extracted as category/category/<model_id>/...
    nested = category_dir / category
    while nested.is_dir():
        for child in nested.iterdir():
            dest = category_dir / child.name
            if dest.exists():
                continue
            shutil.move(str(child), str(dest))
        nested.rmdir()
        nested = category_dir / category

    return category_dir


def verify_category(root: Path, category: str) -> int:
    category_dir = root / category
    if not category_dir.exists():
        return 0
    count = 0
    for model_dir in category_dir.iterdir():
        if not model_dir.is_dir():
            continue
        obj = model_dir / "models" / "model_normalized.obj"
        if obj.exists():
            count += 1
    return count


def download_categories(
    categories: list[str],
    output_root: Path,
    cache_dir: Path,
    token: str | None = None,
) -> dict[str, int]:
    """Download and extract synset zips. Returns synset_id -> mesh count."""
    resolved_token = check_auth(token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for category in categories:
        zip_path = download_synset_zip(category, cache_dir, resolved_token)
        print(f"Extracting {zip_path.name} -> {output_root / category}")
        extract_synset(zip_path, output_root, category)
        count = verify_category(output_root, category)
        counts[category] = count
        name = SYNSETS.get(category, "unknown")
        print(f"Ready: {count} meshes in {output_root / category} ({name})")
        if zip_path.is_file():
            zip_path.unlink()
            print(f"Removed zip to save disk: {zip_path.name}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ShapeNetCore from Hugging Face")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ShapeNetCore.v2"),
        help="Extract meshes to this directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf_cache"),
        help="Where to store downloaded zip files",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help=f"Synset ids to download (default: chair {DEFAULT_CATEGORIES[0]})",
    )
    parser.add_argument(
        "--master-main",
        action="store_true",
        help="Download all seven master main categories (chair, table, sofa, car, airplane, lamp, cabinet)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print common synset ids and exit",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (or set HF_TOKEN env var). Avoid putting tokens in shell history.",
    )
    args = parser.parse_args()

    if args.list_categories:
        print("Master main categories:")
        for synset in MASTER_MAIN_SYNSETS:
            print(f"  {synset}  {SYNSETS.get(synset, '?')}")
        print("\nOther common synsets:")
        for synset, name in sorted(SYNSETS.items()):
            if synset not in MASTER_MAIN_SYNSETS:
                print(f"  {synset}  {name}")
        return

    if args.master_main:
        args.categories = MASTER_MAIN_SYNSETS

    download_categories(
        args.categories,
        args.output_root,
        args.cache_dir,
        token=args.token,
    )

    print(f"\nShapeNetCore root: {args.output_root.resolve()}")
    print("Next: python scripts/prepare_dataset.py --num-views 2 --max-samples 500")


if __name__ == "__main__":
    main()
