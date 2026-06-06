#!/usr/bin/env bash
# Baseline 2: View-ID Embedding (learned per-view-index embedding).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/_run_baseline.sh" configs/baselines/b2_view_id.yaml "$@"
