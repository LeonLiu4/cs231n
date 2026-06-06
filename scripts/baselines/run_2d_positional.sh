#!/usr/bin/env bash
# Baseline 1: 2D Positional Embedding (ViT patch positions only, no pose).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/_run_baseline.sh" configs/baselines/b1_2d_positional.yaml "$@"
