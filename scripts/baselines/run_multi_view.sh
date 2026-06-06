#!/usr/bin/env bash
# Baseline 5: Multi-View Model (mean-pool fusion, no geometry-aware embeddings).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/_run_baseline.sh" configs/baselines/b5_multi_view.yaml "$@"
