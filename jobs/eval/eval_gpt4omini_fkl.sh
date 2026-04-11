#!/bin/bash
#SBATCH --job-name=eval_gpt4omini_fkl
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_eval_gpt4omini_fkl.out
#SBATCH --error=logs/%j_eval_gpt4omini_fkl.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
cd /home/ssmurali/user_interactions

# OPENAI_API_KEY must be set before submitting:
#   OPENAI_API_KEY=... sbatch jobs/eval/eval_gpt4omini_fkl.sh
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set" && exit 1
fi

# Run each FKL dir separately — skips already-done steps automatically
for run in fkl_full_wfbest fkl_full_wffull fkl_p30_wfbest fkl_p30_wffull; do
    echo "=== Evaluating checkpoints/$run ==="
    python scripts/eval/eval_ckpts_alpaca.py checkpoints/qwen3_8b_$run/ \
        --judge        gpt4omini \
        --results_root eval_results/alpaca \
        --concurrency  200 \
        --max_judge_tokens 16
done
