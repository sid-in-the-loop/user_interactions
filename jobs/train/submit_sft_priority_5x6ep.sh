#!/bin/bash
# Submit 5 SFT jobs: 4 wildfeedback (think best/full × lr 5e-6, 2e-6) + 1 wildchat (think full 5e-6).
# 6 epochs each; epoch-1..epoch-6 + final; minimal step-* checkpoints (large SAVE_STEPS).
# Run from repo root: bash jobs/train/submit_sft_priority_5x6ep.sh
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
WC="$REPO/datasets/wildchat"
WF="$REPO/datasets/wildfeedback"
SCRIPT="jobs/train/sbatch_sft_one.sh"
PORT=29600
EPOCHS=6
# Few step saves (only when global_step hits multiples; keep high to save disk)
SAVE_STEPS=999999

for f in "$WF/ystar_thinking_best.jsonl" "$WF/ystar_thinking_full.jsonl"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f — generate y* for wildfeedback first."
    exit 1
  fi
done

submit() {
  local job_name=$1
  local input=$2
  local run_name=$3
  local lr=$4
  export INPUT="$input" RUN_NAME="$run_name" LR="$lr" MASTER_PORT=$PORT EPOCHS=$EPOCHS SAVE_STEPS=$SAVE_STEPS
  export MODEL="${MODEL:-Qwen/Qwen3-4B}"
  sbatch --job-name="$job_name" "$SCRIPT"
  echo "  -> $job_name  RUN_NAME=$run_name  LR=$lr"
  ((PORT++)) || true
}

echo "Submitting 5 SFT jobs (6 epochs each, epoch checkpoints + final)..."
submit "sft_wf_best_5e6_6ep"  "$WF/ystar_thinking_best.jsonl" "sft_wf_thinking_best_bs8_ga32_lr5e6_6ep"  5e-6
submit "sft_wf_best_2e6_6ep"  "$WF/ystar_thinking_best.jsonl" "sft_wf_thinking_best_bs8_ga32_lr2e6_6ep"  2e-6
submit "sft_wf_full_5e6_6ep"  "$WF/ystar_thinking_full.jsonl"  "sft_wf_thinking_full_bs8_ga32_lr5e6_6ep"  5e-6
submit "sft_wf_full_2e6_6ep"  "$WF/ystar_thinking_full.jsonl"  "sft_wf_thinking_full_bs8_ga32_lr2e6_6ep"  2e-6
submit "sft_wc_full_5e6_6ep"  "$WC/ystar_thinking_full.jsonl"  "sft_wc_thinking_full_bs8_ga32_lr5e6_6ep"  5e-6
echo "Done. See docs/wc_wf_priority.md for paths."
