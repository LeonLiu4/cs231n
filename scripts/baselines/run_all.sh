#!/usr/bin/env bash
# Run every baseline sequentially (one at a time), in table order.
#
# Examples:
#   scripts/baselines/run_all.sh                 # full training of all baselines
#   DEMO=1 EPOCHS=1 scripts/baselines/run_all.sh # quick end-to-end smoke test
#   ONLY="b2_view_id main_geometry_aware" scripts/baselines/run_all.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINES=(
  "run_2d_positional.sh"
  "run_view_id.sh"
  "run_camera_pose.sh"
  "run_single_view.sh"
  "run_multi_view.sh"
  "run_geometry_aware.sh"
)

for script in "${BASELINES[@]}"; do
  # Optional filter: ONLY substring match against the script name.
  if [[ -n "${ONLY:-}" ]]; then
    match=0
    for token in $ONLY; do
      [[ "$script" == *"$token"* ]] && match=1
    done
    [[ "$match" == "1" ]] || { echo "[skip] $script"; continue; }
  fi
  echo
  echo "##############################################################"
  echo "# Running $script"
  echo "##############################################################"
  "$DIR/$script"
done

echo
echo "All requested baselines complete."
