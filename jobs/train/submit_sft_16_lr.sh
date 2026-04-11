#!/bin/bash
# Submit 16 SFT jobs: 4 wildchat + 4 wildfeedback configs, each with LR 5e-6 and 2e-6.
# Run from repo root: bash jobs/train/submit_sft_16_lr.sh
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
WC="$REPO/datasets/wildchat"
WF="$REPO/datasets/wildfeedback"
SCRIPT="jobs/train/sbatch_sft_one.sh"
PORT=29521

submit() {
  local job_name=$1
  local input=$2
  local run_name=$3
  local lr=$4
  export INPUT="$input" RUN_NAME="$run_name" LR="$lr" MASTER_PORT=$PORT
  sbatch --job-name="$job_name" "$SCRIPT"
  ((PORT++)) || true
}

# Wildchat: 4 configs × 2 LRs
echo "Submitting 8 wildchat SFT jobs..."
submit "sft_wc_think_full_lr5e6"   "$WC/ystar_thinking_full.jsonl"     "sft_wc_thinking_full_bs8_ga32_lr5e6"   5e-6
submit "sft_wc_think_full_lr2e6"   "$WC/ystar_thinking_full.jsonl"     "sft_wc_thinking_full_bs8_ga32_lr2e6"   2e-6
submit "sft_wc_think_best_lr5e6"   "$WC/ystar_thinking_best.jsonl"     "sft_wc_thinking_best_bs8_ga32_lr5e6"   5e-6
submit "sft_wc_think_best_lr2e6"   "$WC/ystar_thinking_best.jsonl"     "sft_wc_thinking_best_bs8_ga32_lr2e6"   2e-6
submit "sft_wc_nothink_full_lr5e6" "$WC/ystar_nonthinking_full.jsonl"  "sft_wc_nonthinking_full_bs8_ga32_lr5e6" 5e-6
submit "sft_wc_nothink_full_lr2e6" "$WC/ystar_nonthinking_full.jsonl"  "sft_wc_nonthinking_full_bs8_ga32_lr2e6" 2e-6
submit "sft_wc_nothink_best_lr5e6" "$WC/ystar_nonthinking_best.jsonl"   "sft_wc_nonthinking_best_bs8_ga32_lr5e6"  5e-6
submit "sft_wc_nothink_best_lr2e6" "$WC/ystar_nonthinking_best.jsonl"   "sft_wc_nonthinking_best_bs8_ga32_lr2e6"  2e-6

# Wildfeedback: 4 configs × 2 LRs
echo "Submitting 8 wildfeedback SFT jobs..."
submit "sft_wf_think_full_lr5e6"   "$WF/ystar_thinking_full.jsonl"     "sft_wf_thinking_full_bs8_ga32_lr5e6"   5e-6
submit "sft_wf_think_full_lr2e6"   "$WF/ystar_thinking_full.jsonl"     "sft_wf_thinking_full_bs8_ga32_lr2e6"   2e-6
submit "sft_wf_think_best_lr5e6"   "$WF/ystar_thinking_best.jsonl"     "sft_wf_thinking_best_bs8_ga32_lr5e6"   5e-6
submit "sft_wf_think_best_lr2e6"   "$WF/ystar_thinking_best.jsonl"     "sft_wf_thinking_best_bs8_ga32_lr2e6"   2e-6
submit "sft_wf_nothink_full_lr5e6" "$WF/ystar_nonthinking_full.jsonl"  "sft_wf_nonthinking_full_bs8_ga32_lr5e6" 5e-6
submit "sft_wf_nothink_full_lr2e6" "$WF/ystar_nonthinking_full.jsonl"  "sft_wf_nonthinking_full_bs8_ga32_lr2e6" 2e-6
submit "sft_wf_nothink_best_lr5e6" "$WF/ystar_nonthinking_best.jsonl"   "sft_wf_nonthinking_best_bs8_ga32_lr5e6"  5e-6
submit "sft_wf_nothink_best_lr2e6" "$WF/ystar_nonthinking_best.jsonl"   "sft_wf_nonthinking_best_bs8_ga32_lr2e6"  2e-6

echo "Submitted 16 SFT jobs (LR 5e-6 and 2e-6 for each of 8 y* configs)."
