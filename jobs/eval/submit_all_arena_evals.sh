#!/bin/bash
# jobs/submit_all_arena_evals.sh
# Submit all 6 models for Arena-Hard-v2.0 using the correct paths.

SUBMIT_SCRIPT="jobs/eval_arena_hard.sh"
BASE_PATH="/data/group_data/cx_group/ssmurali/offpolicy/fkl"

echo "Submitting all 6 Arena-Hard evaluations (1 GPU per job)..."

submit_model() {
    local name=$1
    local path=$2
    sbatch "$SUBMIT_SCRIPT" "$name" "$path"
}

# 1. Base Model
submit_model base Qwen/Qwen3-4B

# 2. Local Models (Hardcoded exact names from filesystem)
submit_model baseline_sft_fp32 "$BASE_PATH/baseline_sft_fp32/final"
submit_model fkl_relaxed_1e-4_fp32 "$BASE_PATH/fkl_relaxed_1e-4_fp32/final"
submit_model fkl_mid_5e-4_fp32 "$BASE_PATH/fkl_mid_5e-4_fp32/final"
submit_model fkl_aggressive_5e_6_fp32 "$BASE_PATH/fkl_aggressive_5e-6_fp32/final"
submit_model fkl_very_agg_1e-5_fp32 "$BASE_PATH/fkl_very_agg_1e-5_fp32/final"

echo "All 6 Arena-Hard jobs submitted."
