#!/usr/bin/env python3
"""Print the Hugging Face token used for ShapeNetCore dataset access."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_shapenet_hf import check_auth


def main() -> None:
    token = check_auth(None)
    print(token)


if __name__ == "__main__":
    main()
