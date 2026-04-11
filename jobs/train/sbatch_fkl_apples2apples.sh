#!/bin/bash
#SBATCH --job-name=fkl_apples2apples
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
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

# Hyperparams aligned with SDPO apples-to-apples config
# Effective Batch Size = 4 (BS) * 4 (GA) * 4 (GPUs) = 64
RUN_NAME="${RUN_NAME:-fkl_apples2apples_fp32_v1}"
LR="2e-6"
BS="2"
GA="8"
MASK_TAU="${MASK_TAU:-0.001}"
OPTIMIZER="adamw"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"

# Intermediate checkpoints: 1060 total steps / 5 = 212
SAVE_STEPS=212

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo "PHASE 2 — FKL SFT Training (Apples-to-Apples with SDPO, FP32)"
echo "Run: $RUN_NAME | LR: $LR | Tau: $MASK_TAU | Eff BS: 64"
echo "════════════════════════════════════════"

# Running from workspace root
export PYTHONPATH=$PYTHONPATH:.

torchrun --nproc_per_node=4 scripts/fkl/train_fkl_fsdp.py \
    --input      "$Y_STAR"     \
    --output_dir "$OUTPUT_DIR" \
    --model      "$MODEL"      \
    --batch_size "$BS"         \
    --grad_accum "$GA"         \
    --epochs      2            \
    --lr          "$LR"        \
    --optimizer   "$OPTIMIZER" \
    --max_length  2048         \
    --mask_tau    "$MASK_TAU"  \
    --save_steps  "$SAVE_STEPS" \
    --fp32                     \
    --wandb_project "fkl-distill-fsdp" \
    --run_name    "$RUN_NAME"

echo "Done."
