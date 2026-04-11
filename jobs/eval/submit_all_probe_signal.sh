#!/bin/bash
# Submit 6 separate 1-GPU jobs for FKL probe signal (adv experiment).
# Usage: ./jobs/eval/submit_all_probe_signal.sh

SUBMIT_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_probe_signal.sh"
BASE_PATH="/data/group_data/cx_group/ssmurali/offpolicy/fkl"

echo "Submitting 6 probe-signal jobs (1 GPU each)..."

submit_ckpt() {
    local name=$1
    local path=$2
    sbatch "$SUBMIT_SCRIPT" "$name" "$path"
}

submit_ckpt baseline_sft_fp32    "$BASE_PATH/baseline_sft_fp32/final"
submit_ckpt baseline_v1_s500     "$BASE_PATH/baseline_sft_fp32/baseline_v1/step-500"
submit_ckpt baseline_v1         "$BASE_PATH/baseline_sft_fp32/baseline_v1/final"
submit_ckpt baseline_v2_s500     "$BASE_PATH/baseline_sft_fp32/baseline_v2/step-500"
submit_ckpt baseline_v2_s1000    "$BASE_PATH/baseline_sft_fp32/baseline_v2/step-1000"
submit_ckpt baseline_v2         "$BASE_PATH/baseline_sft_fp32/baseline_v2/final"

echo "All 6 probe-signal jobs submitted."
