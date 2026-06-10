#!/usr/bin/env bash
# Launch all six baselines in parallel with the fast preset (~15-30 min each).
# Wall clock stays under 2 h even if you close the laptop.
#
# Fast preset: 8 epochs, 400 train / 120 eval samples, batch 12, early stop 3.
# Checkpoints: /data/runs/baselines/<name>_fast/ on pose-aware-data-v2
#
# Usage:
#   scripts/baselines/run_all_modal_fast.sh
#   ONLY="b5 geometry" scripts/baselines/run_all_modal_fast.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIGS=(
  "configs/baselines/b1_2d_positional.yaml"
  "configs/baselines/b2_view_id.yaml"
  "configs/baselines/b3_camera_pose.yaml"
  "configs/baselines/b4_single_view.yaml"
  "configs/baselines/b5_multi_view.yaml"
  "configs/baselines/main_geometry_aware.yaml"
)

for config in "${CONFIGS[@]}"; do
  if [[ -n "${ONLY:-}" ]]; then
    match=0
    for token in $ONLY; do
      [[ "$config" == *"$token"* ]] && match=1
    done
    [[ "$match" == "1" ]] || { echo "[skip] $config"; continue; }
  fi
  echo "[launch] modal run --detach --fast $config"
  modal run --detach scripts/modal_train_baseline.py --config "$config" --fast &
done
wait || true

echo
echo "Six fast jobs submitted (parallel). Monitor: https://modal.com/apps"
echo "When done:"
echo "  python scripts/collect_baseline_results.py --from-modal --fast"
