#!/bin/bash
# Merge many 2-GPU FSDP sharded dirs (rank_0.pt, rank_1.pt) -> HF folders for vLLM.
# Usage: sbatch jobs/train/sbatch_merge_fsdp_2gpu.sh
#
#SBATCH --job-name=merge_fsdp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/merge_fsdp_%j.out
#SBATCH --error=logs/merge_fsdp_%j.err

set -euo pipefail
# Under Slurm, script may run from spool dir; use submit dir so logs/ is in the repo.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TASKS="${MERGE_TASKS_FILE:-eval_results/merge_fsdp_tasks.txt}"
BASE_MODEL="${MERGE_BASE_MODEL:-Qwen/Qwen3-4B}"

echo "Tasks: $TASKS | base: $BASE_MODEL | 2 GPUs"

torchrun --nproc_per_node=2 scripts/fkl/merge_fsdp_sharded_to_hf.py \
  --tasks_file "$TASKS" \
  --base_model "$BASE_MODEL" \
  --fp32

echo "Merge job finished."
