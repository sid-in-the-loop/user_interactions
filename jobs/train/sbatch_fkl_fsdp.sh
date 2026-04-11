#!/bin/bash
#SBATCH --job-name=fkl_fsdp_2gpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Exit on error
set -e

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Force offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Config
Y_STAR="datasets/wildchat/y_star.jsonl"
BASE_OUTPUT_DIR="/data/group_data/cx_group/ssmurali/offpolicy/fkl"
MODEL="Qwen/Qwen3-4B"

# Hyperparams from environment variables (with defaults)
RUN_NAME="${RUN_NAME:-fkl_v1}"
LR="${LR:-2e-6}"
MASK_TAU="${MASK_TAU:-0.001}"
OPTIMIZER="${OPTIMIZER:-adamw_8bit}"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo "PHASE 2 — SFT Training with FSDP (2 GPUs, Full Shard)"
echo "Run: $RUN_NAME | LR: $LR | Tau: $MASK_TAU | Opt: $OPTIMIZER"
echo "════════════════════════════════════════"

            torchrun --nproc_per_node=2 train_fkl_fsdp.py \
                --input      "$Y_STAR"     \
                --output_dir "$OUTPUT_DIR" \
                --model      "$MODEL"      \
                --batch_size  4            \
                --grad_accum  32           \
                --epochs      2            \
                --lr          "$LR"        \
                --optimizer   "$OPTIMIZER" \
                --max_length  2048         \
                --mask_tau    "$MASK_TAU"  \
                --wandb_project "fkl-distill-fsdp" \
                --run_name    "$RUN_NAME"

echo "Done."
