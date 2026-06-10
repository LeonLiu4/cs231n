#!/usr/bin/env bash
# Launch each baseline as its own detached Modal GPU job (safe to close laptop).
#
# Usage:
#   scripts/baselines/run_all_modal.sh
#   EPOCHS=5 scripts/baselines/run_all_modal.sh
#   ONLY="b5 geometry" scripts/baselines/run_all_modal.sh
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

EXTRA=()
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(--epochs "$EPOCHS")
[[ "${DEMO:-0}" == "1" ]] && EXTRA+=(--demo)
[[ "${SKIP_TRAIN:-0}" == "1" ]] && EXTRA+=(--skip-train)

for config in "${CONFIGS[@]}"; do
  if [[ -n "${ONLY:-}" ]]; then
    match=0
    for token in $ONLY; do
      [[ "$config" == *"$token"* ]] && match=1
    done
    [[ "$match" == "1" ]] || { echo "[skip] $config"; continue; }
  fi
  echo "[launch] modal run --detach $config"
  modal run --detach scripts/modal_train_baseline.py --config "$config" "${EXTRA[@]}"
done

echo
echo "All jobs submitted. Monitor at https://modal.com/apps"
echo "After they finish: python scripts/collect_baseline_results.py --from-modal"
