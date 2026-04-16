#!/bin/bash
# Evaluate all checkpoints for a method on AlpacaEval with Qwen3-14B judge.
#
# Usage:
#   sbatch --job-name=eval_jsd jobs/deltaai/eval_method.sh jsd_p30 [STEP_INTERVAL]
#
# Args:
#   $1  METHOD_DIR_NAME  e.g. jsd_p30, fkl_p30, sft_p30
#   $2  STEP_INTERVAL    eval every N steps (default: 10 = all checkpoints)
#
# Requires: 14B judge server running (reads URL from tmp/qwen3_14b_server_url.txt)

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
# job-name set via sbatch --job-name=eval_<method>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

METHOD="${1:?Usage: sbatch eval_method.sh METHOD_DIR_NAME [STEP_INTERVAL]}"
STEP_INTERVAL="${2:-10}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"

CKPT_BASE="/projects/bgtw/ssredharan/checkpoints/${METHOD}"
EVAL_BASE="/projects/bgtw/ssredharan/eval_results/${METHOD}"
JUDGE_URL_FILE="tmp/qwen3_14b_server_url.txt"

# Wait for judge server
echo "Waiting for 14B judge server..."
while [ ! -f "$JUDGE_URL_FILE" ]; do sleep 10; done
JUDGE_URL=$(cat "$JUDGE_URL_FILE")
until curl -sf "${JUDGE_URL}/health" > /dev/null 2>&1; do
    echo "  Judge not ready..."
    sleep 10
done
echo "Judge ready: ${JUDGE_URL}"

# Get list of checkpoints
STEPS=($(ls -d "${CKPT_BASE}"/step-* 2>/dev/null | sed 's/.*step-//' | sort -n))
echo "Found ${#STEPS[@]} checkpoints for ${METHOD}"

# Eval each checkpoint
for STEP in "${STEPS[@]}"; do
    # Skip if not on interval
    if (( STEP % STEP_INTERVAL != 0 )); then
        continue
    fi

    CKPT_DIR="${CKPT_BASE}/step-${STEP}"
    OUT_DIR="${EVAL_BASE}/step-${STEP}"

    # Skip if already evaluated
    if [ -f "${OUT_DIR}/scores.json" ]; then
        echo "SKIP step-${STEP} (already evaluated)"
        continue
    fi

    echo ""
    echo "════════════════════════════════════════"
    echo "  Evaluating ${METHOD} step-${STEP}"
    echo "════════════════════════════════════════"

    python scripts/eval/eval_checkpoint.py \
        --checkpoint "$CKPT_DIR" \
        --judge-url "$JUDGE_URL" \
        --output-dir "$OUT_DIR" \
        --gen-batch-size 4 \
        --judge-workers 8

done

# Also eval final if exists
if [ -d "${CKPT_BASE}/final" ] && [ ! -f "${EVAL_BASE}/final/scores.json" ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "  Evaluating ${METHOD} final"
    echo "════════════════════════════════════════"

    python scripts/eval/eval_checkpoint.py \
        --checkpoint "${CKPT_BASE}/final" \
        --judge-url "$JUDGE_URL" \
        --output-dir "${EVAL_BASE}/final" \
        --gen-batch-size 4 \
        --judge-workers 8
fi

echo ""
echo "Done! Results in ${EVAL_BASE}/"

# Print summary
echo ""
echo "Method: ${METHOD}"
echo "Step | WR (inc ties) | WR (exc ties) | Avg Len"
echo "-----|---------------|---------------|--------"
for d in "${EVAL_BASE}"/step-* "${EVAL_BASE}"/final; do
    [ -f "$d/scores.json" ] || continue
    python3 -c "
import json
s = json.load(open('$d/scores.json'))
step = '$d'.split('/')[-1]
print(f\"{step:>5} | {s['win_rate']:>12.1f}% | {s['win_rate_no_ties']:>12.1f}% | {s['avg_length']:>6.0f}\")
"
done
