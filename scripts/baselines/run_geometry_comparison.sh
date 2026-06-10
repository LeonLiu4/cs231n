#!/usr/bin/env bash
# Train geometry-aware vs blind multi-view at K=8 (parallel, ~1 h each).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for config in \
  configs/baselines/b5_multi_view.yaml \
  configs/baselines/main_geometry_aware.yaml; do
  echo "[launch compare K=8] $config"
  modal run --detach scripts/modal_train_baseline.py --config "$config" --compare &
done
wait || true
echo "Compare jobs submitted. After they finish:"
echo "  modal run scripts/modal_visualize_baselines.py --fetch-local"
