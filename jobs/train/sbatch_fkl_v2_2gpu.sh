#!/bin/bash
#SBATCH --job-name=fkl_v2_2gpu
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

# Config
INPUT="datasets/wildchat/processed_tuples.jsonl"
Y_STAR="datasets/wildchat/y_star_processed_v2.jsonl"
MODEL="Qwen/Qwen3-4B"
RUN_NAME="fkl_v2"
BASE_OUTPUT_DIR="/data/group_data/cx_group/ssmurali/offpolicy/fkl"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo "PHASE 1 — Generating y* with vLLM (2 GPUs, tensor parallel)"
echo "Input: $INPUT"
echo "Output: $Y_STAR"
echo "════════════════════════════════════════"

# Running from workspace root
export PYTHONPATH=$PYTHONPATH:.

python scripts/fkl/generate_ystar.py \
    --input      "$INPUT"   \
    --output     "$Y_STAR"  \
    --model      "$MODEL"   \
    --max_tokens  2048       \
    --temperature 0.7       \
    --gpu_util    0.90      \
    --tp_size     2

echo ""
echo "════════════════════════════════════════"
echo "PHASE 2 — FKL SFT Training with FSDP (2 GPUs, FP32)"
echo "Run: $RUN_NAME | Input: $Y_STAR | Opt: adamw"
echo "════════════════════════════════════════"

# Force offline mode for training phase
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

torchrun --nproc_per_node=2 scripts/fkl/train_fkl_fsdp.py \
    --input      "$Y_STAR"     \
    --output_dir "$OUTPUT_DIR" \
    --model      "$MODEL"      \
    --batch_size  4            \
    --grad_accum  32           \
    --epochs      2            \
    --lr          2e-6         \
    --optimizer   "adamw"      \
    --max_length  2048         \
    --mask_tau    0.001        \
    --fp32                     \
    --wandb_project "fkl-distill-fsdp" \
    --run_name    "$RUN_NAME"

echo "Done. Model saved to $OUTPUT_DIR/final"
