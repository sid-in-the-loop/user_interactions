#!/bin/bash
#SBATCH --job-name=eval_gemini_4ckpts
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_eval_gemini_4ckpts.out
#SBATCH --error=logs/%j_eval_gemini_4ckpts.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
cd /home/ssmurali/user_interactions

# GOOGLE_API_KEY must be set in environment before submitting:
#   GOOGLE_API_KEY=... sbatch jobs/eval/eval_gemini_4ckpts.sh
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY not set" && exit 1
fi

python scripts/eval/eval_ckpts_alpaca.py checkpoints/ \
    --judge        gemini \
    --results_root eval_results/alpaca_gemini \
    --max_ckpts    6 \
    --concurrency  200 \
    --max_judge_tokens 16
