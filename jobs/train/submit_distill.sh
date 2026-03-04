#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FKL Distillation — Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   bash run.sh
#
# Phase 1: python  generate_y_star.py  (vLLM, all 4 GPUs via tensor_parallel)
# Phase 2: torchrun train_fkl.py       (DDP,  all 4 GPUs via data parallel)
#
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

INPUT="datasets/wildchat/filtered_tuples.jsonl"
Y_STAR="datasets/wildchat/y_star.jsonl"
OUTPUT_DIR="/data/group_data/cx_group/ssmurali/offpolicy/fkl"
MODEL="Qwen/Qwen3-4B"

echo "════════════════════════════════════════"
echo "PHASE 1 — Generating y* with vLLM (4 GPUs, tensor parallel)"
echo "════════════════════════════════════════"
python generate_ystar.py \
    --input      "$INPUT"   \
    --output     "$Y_STAR"  \
    --model      "$MODEL"   \
    --max_tokens  2048       \
    --temperature 0.7       \
    --gpu_util    0.92

echo ""
echo "════════════════════════════════════════"
echo "PHASE 2 — SFT Training with DDP (4 GPUs, data parallel)"
echo "════════════════════════════════════════"
torchrun --nproc_per_node=4 train_fkl.py \
    --input      "$Y_STAR"     \
    --output_dir "$OUTPUT_DIR" \
    --model      "$MODEL"      \
    --batch_size  16           \
    --grad_accum  1            \
    --epochs      2            \
    --lr          2e-6         \
    --max_length  2048         \
    --mask_tau    0.001        \
    --save_steps  500

echo ""
echo "Done. Model saved to $OUTPUT_DIR/final"