#!/bin/bash
# Evaluate all checkpoints under a given run dir using GPT-4o-mini.
# Usage: sbatch --export=ALL jobs/eval/eval_ckpts_gpt4omini.sh <ckpt_dir>
#
# Example:
#   OPENAI_API_KEY=... sbatch --export=ALL \
#       jobs/eval/eval_ckpts_gpt4omini.sh checkpoints/qwen3_8b_sft_p30_wc

#SBATCH --job-name=eval_gpt4omini
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_eval_gpt4omini_%x.out
#SBATCH --error=logs/%j_eval_gpt4omini_%x.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
cd /home/ssmurali/user_interactions

CKPT_DIR="${1:?Usage: sbatch eval_ckpts_gpt4omini.sh <ckpt_dir>}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set" && exit 1
fi

python scripts/eval/eval_ckpts_alpaca.py "$CKPT_DIR" \
    --judge        gpt4omini \
    --results_root eval_results/alpaca \
    --concurrency  200 \
    --max_judge_tokens 16
