#!/bin/bash
#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=token_heatmaps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=01:00:00
#SBATCH --output=logs/token_heatmaps_%j.out
#SBATCH --error=logs/token_heatmaps_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"

python scripts/eval/token_heatmaps.py \
    --dataset datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
    --model Qwen/Qwen3-8B \
    --y_star_field y_star_prefix30 \
    --n_samples 10 \
    --output-dir diagnostics/token_heatmaps
