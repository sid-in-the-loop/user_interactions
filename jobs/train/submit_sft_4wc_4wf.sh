#!/bin/bash
# Submit 4 wildchat SFT + 4 wildfeedback SFT jobs (LR 8e-7 each).
# Run from repo root: bash jobs/train/submit_sft_4wc_4wf.sh
# To submit only wildfeedback: bash jobs/train/submit_sft_4wc_4wf.sh wf
# To submit only wildchat:     bash jobs/train/submit_sft_4wc_4wf.sh wc
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
WC="$REPO/datasets/wildchat"
WF="$REPO/datasets/wildfeedback"
SCRIPT="jobs/train/sbatch_sft_one.sh"
PORT=29521

submit() {
  local job_name=$1 input=$2 run_name=$3
  export INPUT="$input" RUN_NAME="$run_name" LR=8e-7 MASTER_PORT=$PORT
  sbatch --job-name="$job_name" "$SCRIPT"
  ((PORT++)) || true
}

if [[ "${1:-all}" != "wf" ]]; then
  echo "Submitting 4 wildchat SFT jobs..."
  submit sft_wc_think_full   "$WC/ystar_thinking_full.jsonl"      sft_wc_thinking_full_bs8_ga32
  submit sft_wc_think_best   "$WC/ystar_thinking_best.jsonl"      sft_wc_thinking_best_bs8_ga32
  submit sft_wc_nothink_full "$WC/ystar_nonthinking_full.jsonl"   sft_wc_nonthinking_full_bs8_ga32
  submit sft_wc_nothink_best "$WC/ystar_nonthinking_best.jsonl"   sft_wc_nonthinking_best_bs8_ga32
fi
if [[ "${1:-all}" != "wc" ]]; then
  echo "Submitting 4 wildfeedback SFT jobs..."
  submit sft_wf_think_full   "$WF/ystar_thinking_full.jsonl"      sft_wf_thinking_full_bs8_ga32
  submit sft_wf_think_best   "$WF/ystar_thinking_best.jsonl"      sft_wf_thinking_best_bs8_ga32
  submit sft_wf_nothink_full "$WF/ystar_nonthinking_full.jsonl"   sft_wf_nonthinking_full_bs8_ga32
  submit sft_wf_nothink_best "$WF/ystar_nonthinking_best.jsonl"   sft_wf_nonthinking_best_bs8_ga32
fi
echo "Done."
