#!/usr/bin/env bash
# Proposed Model: Geometry-Aware Positional Embedding (Fourier camera geometry).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/_run_baseline.sh" configs/baselines/main_geometry_aware.yaml "$@"
