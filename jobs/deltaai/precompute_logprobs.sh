#!/bin/bash
#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=precompute_lp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=01:00:00
#SBATCH --output=logs/precompute_lp_%j.out
#SBATCH --error=logs/precompute_lp_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"

python scripts/fkl/precompute_logprobs.py \
    --input datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
    --y_star_field y_star_prefix30 \
    --model Qwen/Qwen3-8B \
    --output datasets/wildchat/init_logprobs_qwen3_8b.pt \
    --batch_size 4
