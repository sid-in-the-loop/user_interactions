#!/bin/bash
#SBATCH --job-name=fkl_v2_baseline
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

# Config from user request
INPUT="/home/ssmurali/user_interactions/datasets/wildchat/y_star_processed.jsonl"
BASE_OUTPUT_DIR="/data/group_data/cx_group/ssmurali/offpolicy/fkl/baseline_sft_fp32"
RUN_NAME="baseline_v2"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"
MODEL="Qwen/Qwen3-4B"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo "PHASE 2 — Baseline SFT (v2) | FP32 | Mask Tau: 0"
echo "Input: $INPUT"
echo "Config: BS=1 | GA=16 | GPUs=4 | Eff BS=64"
echo "════════════════════════════════════════"

# Running from workspace root
export PYTHONPATH=$PYTHONPATH:.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use a different port to avoid conflicts
MASTER_PORT=29504

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT scripts/fkl/train_fkl_fsdp.py \
    --input      "$INPUT"      \
    --output_dir "$OUTPUT_DIR"  \
    --model      "$MODEL"       \
    --batch_size  1             \
    --grad_accum  16            \
    --epochs      2             \
    --lr          0.000002      \
    --optimizer   "adamw"       \
    --max_length  2048          \
    --mask_tau    0             \
    --save_steps  500           \
    --fp32                      \
    --wandb_project "fkl-distill-fsdp" \
    --run_name    "$RUN_NAME"

echo "Done. Model saved to $OUTPUT_DIR/final"
