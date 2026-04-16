#!/bin/bash
#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=logprob_diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=logs/logprob_diag_%j.out
#SBATCH --error=logs/logprob_diag_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"

python scripts/eval/logprob_diagnostics.py \
    --dataset datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
    --adapter step-15 \
    --model Qwen/Qwen3-8B \
    --subsample 1000 \
    --output-dir diagnostics
