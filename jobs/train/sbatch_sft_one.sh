#!/bin/bash
# Parameterized SFT job. Set env vars before sbatch:
#   INPUT, RUN_NAME, LR, MASTER_PORT
#   Optional: NGPUS (default 2), BATCH_SIZE (default 8), GRAD_ACCUM (default 32),
#             SAVE_STEPS, SAVE_EPOCH_EVERY (1=every epoch), SHARDED_EPOCH_PREFIX (e.g. ext-)
# Example (4B, 2 GPU):
#   export INPUT=... RUN_NAME=sft_wc_think_full_lr5e6 LR=5e-6
#   sbatch jobs/train/sbatch_sft_one.sh
# Example (8B, 4 GPU):
#   export INPUT=... RUN_NAME=sft_wf_8b_think LR=5e-6 MODEL=Qwen/Qwen3-8B NGPUS=4 BATCH_SIZE=4 GRAD_ACCUM=16
#   sbatch --gres=gpu:4 jobs/train/sbatch_sft_one.sh
# Extend from a checkpoint: set MODEL to path to previous run's final dir, EPOCHS to extra epochs, RUN_NAME to a new name (e.g. ..._ep4).
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j_%x.out
#SBATCH --error=logs/sft_%j_%x.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ssmurali/user_interactions}"
cd "$REPO_ROOT" && export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT" && export PYTHONUNBUFFERED=1

INPUT="${INPUT:?Set INPUT to ystar JSONL path}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
LR="${LR:-8e-7}"
MASTER_PORT="${MASTER_PORT:-29500}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"

BASE_OUTPUT_DIR="/data/group_data/cx_group/ssmurali/offpolicy/fkl"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
EPOCHS="${EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_EPOCH_EVERY="${SAVE_EPOCH_EVERY:-1}"
SHARDED_EPOCH_PREFIX="${SHARDED_EPOCH_PREFIX:-}"
NGPUS="${NGPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
MASK_TAU="${MASK_TAU:-0}"
mkdir -p logs "$OUTPUT_DIR"

# Longer NCCL heartbeat so one slow checkpoint save (FSDP all-gather + I/O) doesn't abort the job
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

echo "SFT | Input: $INPUT | Output: $OUTPUT_DIR | Model: $MODEL | Epochs: $EPOCHS | LR: $LR | GPUs: $NGPUS | BS: $BATCH_SIZE | GA: $GRAD_ACCUM | scheduler: $LR_SCHEDULER"
torchrun --nproc_per_node=$NGPUS --master_port=$MASTER_PORT scripts/fkl/train_fkl_fsdp.py \
  --input "$INPUT" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --lr_scheduler "$LR_SCHEDULER" \
  --warmup_ratio "$WARMUP_RATIO" \
  --optimizer "adamw" \
  --max_length 2048 \
  --mask_tau "$MASK_TAU" \
  --save_steps "$SAVE_STEPS" \
  --save_epoch_every "$SAVE_EPOCH_EVERY" \
  --sharded_epoch_prefix "$SHARDED_EPOCH_PREFIX" \
  --wandb_project "sft-ystar" \
  --run_name "$RUN_NAME"
echo "Done. Model saved to $OUTPUT_DIR/final"
