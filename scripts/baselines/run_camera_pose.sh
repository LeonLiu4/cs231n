#!/usr/bin/env bash
# Baseline 3: Camera Pose Embedding (MLP over the raw 7-D pose vector).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/_run_baseline.sh" configs/baselines/b3_camera_pose.yaml "$@"
