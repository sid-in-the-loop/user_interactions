#!/usr/bin/env bash
# Step 2 — Run FKL probe signal measurement for each checkpoint sequentially.
# Inference only (no optimizer): runs on 1 GPU with batched forward passes.
#
# Usage:
#   PROBE_SET=results/probe_set.json \
#   CHECKPOINTS="/path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3" \
#   ./jobs/eval/run_probe_signal_all_ckpts.sh
#
# Or with a single run dir and checkpoint subdirs:
#   RUN_DIR=/data/.../baseline_v2 \
#   CKPT_NAMES="step-500 step-1000 final" \
#   ./jobs/eval/run_probe_signal_all_ckpts.sh
#
# Optional: BATCH_SIZE=8 (default), MAX_LENGTH=2048
# Output: for each checkpoint, writes OUTPUT_DIR/<ckpt_name>_signal.jsonl
# (ckpt_name = basename of checkpoint path).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

PROBE_SET="${PROBE_SET:-$REPO_ROOT/results/probe_set.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Either CHECKPOINTS (space-separated full paths) or RUN_DIR + CKPT_NAMES
if [[ -n "${CHECKPOINTS:-}" ]]; then
  # Full paths given directly
  CKPT_LIST="$CHECKPOINTS"
elif [[ -n "${RUN_DIR:-}" && -n "${CKPT_NAMES:-}" ]]; then
  # e.g. RUN_DIR=/path/to/run CKPT_NAMES="step-500 step-1000 final"
  CKPT_LIST=""
  for name in $CKPT_NAMES; do
    CKPT_LIST="$CKPT_LIST $RUN_DIR/$name"
  done
else
  echo "ERROR: Set either CHECKPOINTS (space-separated paths) or RUN_DIR + CKPT_NAMES"
  echo "  Example: CHECKPOINTS=\"/path/to/baseline_v1_s500 /path/to/baseline_v1\" $0"
  echo "  Or:     RUN_DIR=/path/to/run CKPT_NAMES=\"step-500 final\" $0"
  exit 1
fi

if [[ ! -f "$PROBE_SET" ]]; then
  echo "ERROR: Probe set not found: $PROBE_SET"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

echo "════════════════════════════════════════"
echo "Step 2 — FKL probe signal (1 GPU, sequential checkpoints)"
echo "  PROBE_SET=$PROBE_SET"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MAX_LENGTH=$MAX_LENGTH  BATCH_SIZE=$BATCH_SIZE"
echo "════════════════════════════════════════"

for CKPT in $CKPT_LIST; do
  CKPT_NAME="$(basename "$CKPT")"
  OUT_FILE="$OUTPUT_DIR/${CKPT_NAME}_signal.jsonl"
  echo "[$(date '+%F %T')] Checkpoint: $CKPT -> $OUT_FILE"
  python scripts/fkl/measure_fkl_signal.py \
    --probe_set "$PROBE_SET" \
    --checkpoint "$CKPT" \
    --output "$OUT_FILE" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE"
  echo "[$(date '+%F %T')] Done: $OUT_FILE"
  echo "----------------------------------------"
done

echo "[$(date '+%F %T')] All checkpoints done."
