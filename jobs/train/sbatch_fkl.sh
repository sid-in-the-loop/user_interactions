#!/bin/bash
#SBATCH --job-name=fkl_distill
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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

# Force offline mode to prevent connection errors on compute nodes
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_OFFLINE=1

# ─────────────────────────────────────────────────────────────────────────────
# FKL Distillation — Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

INPUT="datasets/wildchat/filtered_tuples.jsonl"
Y_STAR="datasets/wildchat/y_star.jsonl"
OUTPUT_DIR="./checkpoints"
MODEL="Qwen/Qwen3-4B"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo "PHASE 1 — Generating y* with vLLM (4 GPUs, tensor parallel)"
echo "════════════════════════════════════════"
# vLLM handles multi-GPU internally with tensor_parallel_size=4
python generate_ystar.py \
    --input      "$INPUT"   \
    --output     "$Y_STAR"  \
    --model      "$MODEL"   \
    --max_tokens  2048      \
    --temperature 0.7       \
    --gpu_util    0.92

echo ""
echo "════════════════════════════════════════"
echo "PHASE 2 — SFT Training with DDP (4 GPUs, data parallel)"
echo "════════════════════════════════════════"
# DDP requires torchrun
torchrun --nproc_per_node=4 train_fkl.py \
    --input      "$Y_STAR"     \
    --output_dir "$OUTPUT_DIR" \
    --model      "$MODEL"      \
    --batch_size  16           \
    --grad_accum  4            \
    --epochs      2            \
    --lr          2e-6         \
    --max_length  2048         \
    --mask_tau    0.001        \
    --save_steps  500

echo ""
echo "Done. Final model saved to $OUTPUT_DIR/final"
