#!/bin/bash
# Evaluate 4 evenly-spaced checkpoints under a given run dir using Gemini 2.5 Flash.
# Usage: sbatch --export=ALL jobs/eval/eval_ckpts_gemini.sh <ckpt_dir>
#
# Example:
#   GOOGLE_API_KEY=... sbatch --export=ALL \
#       jobs/eval/eval_ckpts_gemini.sh checkpoints/qwen3_8b_sft_p30_wc

#SBATCH --job-name=eval_gemini
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_eval_gemini_%x.out
#SBATCH --error=logs/%j_eval_gemini_%x.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
cd /home/ssmurali/user_interactions

CKPT_DIR="${1:?Usage: sbatch eval_ckpts_gemini.sh <ckpt_dir>}"

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY not set" && exit 1
fi

python scripts/eval/eval_ckpts_alpaca.py "$CKPT_DIR" \
    --judge        gemini \
    --results_root eval_results/alpaca_gemini \
    --max_ckpts    4 \
    --concurrency  200 \
    --max_judge_tokens 16
