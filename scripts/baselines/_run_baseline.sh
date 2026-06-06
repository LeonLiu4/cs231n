#!/usr/bin/env bash
# Shared helper: train one baseline, then evaluate best.pt on the test splits.
#
# Usage: _run_baseline.sh <config_path> [extra train args...]
# Env:
#   EPOCHS        override epoch count (passed to train as --epochs)
#   DEMO=1        use synthetic demo data (smoke test)
#   SKIP_TRAIN=1  skip training, only evaluate an existing checkpoint
#   SKIP_EVAL=1   skip evaluation after training
set -euo pipefail

CONFIG="${1:?Usage: _run_baseline.sh <config_path>}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Pull experiment.output_dir from the YAML config.
OUTPUT_DIR="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['experiment']['output_dir'])" "$CONFIG")"

TRAIN_ARGS=()
[[ -n "${EPOCHS:-}" ]] && TRAIN_ARGS+=(--epochs "$EPOCHS")
[[ "${DEMO:-0}" == "1" ]] && TRAIN_ARGS+=(--demo)

echo "=============================================================="
echo "Baseline config : $CONFIG"
echo "Output dir      : $OUTPUT_DIR"
echo "=============================================================="

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "[train] starting..."
  python3 scripts/train_baseline.py --config "$CONFIG" "${TRAIN_ARGS[@]}" "$@"
else
  echo "[train] skipped (SKIP_TRAIN=1)"
fi

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[eval] skipped (SKIP_EVAL=1)"
  exit 0
fi

CKPT="$OUTPUT_DIR/best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[eval] no checkpoint at $CKPT; skipping eval."
  exit 0
fi

EVAL_ARGS=()
[[ "${DEMO:-0}" == "1" ]] && EVAL_ARGS+=(--demo)

if [[ "${DEMO:-0}" == "1" ]]; then
  echo "[eval] demo mode -> evaluating on val only"
  python3 scripts/eval_baseline.py --config "$CONFIG" --checkpoint "$CKPT" --split val "${EVAL_ARGS[@]}"
else
  for split in test_seen test_generalization; do
    echo "[eval] split=$split"
    python3 scripts/eval_baseline.py --config "$CONFIG" --checkpoint "$CKPT" --split "$split"
  done
fi

echo "[done] $CONFIG"
