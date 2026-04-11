#!/bin/bash
#SBATCH --job-name=fkl_train_only
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Exit on any error
set -e

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Force offline mode for model/tokenizer loading
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Config
Y_STAR="datasets/wildchat/y_star.jsonl"
OUTPUT_DIR="./checkpoints"
MODEL="Qwen/Qwen3-4B"

echo "════════════════════════════════════════"
echo "RESUMING: Phase 2 Training Only"
echo "════════════════════════════════════════"

torchrun --nproc_per_node=4 train_fkl.py \
    --input      "$Y_STAR"     \
    --output_dir "$OUTPUT_DIR" \
    --model      "$MODEL"      \
    --batch_size  8            \
    --grad_accum  8            \
    --epochs      2            \
    --lr          2e-6         \
    --max_length  2048         \
    --mask_tau    0.001        \
    --save_steps  500

echo "Done."
