#!/usr/bin/env bash
# Baseline 4: Single-View Model (one input image, maximum ambiguity).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/_run_baseline.sh" configs/baselines/b4_single_view.yaml "$@"
