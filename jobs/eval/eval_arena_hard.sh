#!/bin/bash
# jobs/eval_arena_hard.sh
# Usage: sbatch jobs/eval_arena_hard.sh <model_name> <model_path>
#
# Examples:
#   sbatch jobs/eval_arena_hard.sh base Qwen/Qwen3-4B
#   sbatch jobs/eval_arena_hard.sh fkl_mid /data/group_data/cx_group/ssmurali/offpolicy/fkl/fkl_mid_5e-4_fp32/final

#SBATCH --job-name=arena_eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=logs/arena_eval_%j_%x.out
#SBATCH --error=logs/arena_eval_%j_%x.err

# Exit on any error
set -e

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# ── NCCL & Stability Flags ───────────────────────────────────
# These prevent the "stuck after NCCL init" hang on Slurm nodes
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# ─────────────────────────────────────────────────────────────

# Arguments
MODEL_NAME=$1
MODEL_PATH=$2

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: sbatch jobs/eval_arena_hard.sh <model_name> <model_path>"
    exit 1
fi

echo "════════════════════════════════════════"
echo "ARENA HARD EVALUATION: $MODEL_NAME"
echo "PATH: $MODEL_PATH"
echo "GPUs: 1"
echo "DIRECTORY CHECK:"
if [ -d "$MODEL_PATH" ]; then
    echo "  [OK] Directory exists."
    ls -l "$MODEL_PATH/config.json" || echo "  [FAIL] Cannot see config.json"
else
    echo "  [ERROR] Path not found on this node!"
fi
echo "════════════════════════════════════════"

cd arena-hard-auto

# 1. Generate Model Answers
python gen_answer_vllm.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH"

# 2. Run Judgment (requires OPENAI_API_KEY)
# Note: You can also run judgment separately once answers are ready.
# If you want to skip judgment and just generate answers, comment below:
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Starting Judgments..."
    # Update config for this specific run
    cat > config/arena-hard-v2.0.yaml <<EOF
judge_model: gpt-4o-mini-2024-07-18
bench_name: arena-hard-v2.0
reference:
  - gpt-4.1
temperature: 0.0
max_tokens: 16000
regex_patterns:
  - \[\[([AB<>=]+)\]\]
  - \[([AB<>=]+)\]
prompt_template: "<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
model_list:
  - $MODEL_NAME
EOF
    python gen_judgment.py
else
    echo "OPENAI_API_KEY not set. Skipping judgment step."
fi

echo "Done."
