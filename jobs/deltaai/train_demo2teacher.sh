#!/usr/bin/env bash
# Generic submitter for SFT / FKL / PC-SDPO training in the
# demonstrator-to-teacher experiment. One LoRA + single GPU.
#
# Usage:
#   sbatch jobs/deltaai/train_demo2teacher.sh \
#       <objective> <run_id> <model> <dataset_path> <output_dir>
#
#   objective    — sft | fkl | pc_sdpo
#   run_id       — short tag, e.g. WI-1, WC-7
#   model        — Qwen/Qwen3-4B  or  Qwen/Qwen2.5-Math-7B
#   dataset_path — path to a teacher_wins_* / teacher_loses_* jsonl
#   output_dir   — where to save checkpoints (will be created)
#
# Example:
#   sbatch jobs/deltaai/train_demo2teacher.sh \
#       fkl WC-3 Qwen/Qwen3-4B \
#       experiments-for-apr26-may1/wildchat_prefix_decision/data/wildchat/teacher_wins_cond_xyo.jsonl \
#       /work/nvme/bgtw/ssredharan/checkpoints/WC-3-fkl
#
# Optional env overrides:
#   EPOCHS         (default 5)
#   BATCH_SIZE     (default 2)
#   GRAD_ACCUM     (default 32)
#   LR             (default 2e-6)
#   MAX_LENGTH     (default 2048)
#   LORA_R         (default 16)
#   WANDB_PROJECT  (default demonstrator-to-teacher)

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=demo2t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=12:00:00
#SBATCH --output=logs/demo2t_%j.out
#SBATCH --error=logs/demo2t_%j.err

set -euo pipefail

OBJ="${1:?usage: sbatch ${0##*/} <obj> <run_id> <model> <dataset_path> <output_dir>}"
RUN_ID="${2:?}"
MODEL="${3:?}"
DATASET="${4:?}"
OUTDIR="${5:?}"

case "$OBJ" in
  sft|fkl|sdpo|pc_sdpo) ;;
  *) echo "unknown objective: $OBJ"; exit 1 ;;
esac

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LR="${LR:-2e-6}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-demonstrator-to-teacher}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs "$OUTDIR"

export HF_HOME="${HF_HOME:-/projects/bgtw/ssredharan/models}"

echo "════════════════════════════════════════"
echo "  demo→teacher training"
echo "  objective : $OBJ"
echo "  run_id    : $RUN_ID"
echo "  model     : $MODEL"
echo "  dataset   : $DATASET"
echo "  output    : $OUTDIR"
echo "  epochs=$EPOCHS  batch=$BATCH_SIZE  grad_accum=$GRAD_ACCUM  lr=$LR"
echo "  max_len=$MAX_LENGTH  lora_r=$LORA_R  lora_alpha=$LORA_ALPHA"
echo "  wandb     : $WANDB_PROJECT"
echo "  job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "════════════════════════════════════════"

case "$OBJ" in
  sdpo|pc_sdpo)
    SCRIPT="experiments-for-apr26-may1/training/train_modeseeking.py"
    EXTRA=(--objective "$OBJ")
    ;;
  sft|fkl)
    SCRIPT="experiments-for-apr26-may1/training/train_modecovering.py"
    EXTRA=(--objective "$OBJ")
    ;;
esac

python "$SCRIPT" \
  --dataset_path "$DATASET" \
  --run_id "$RUN_ID" \
  --output_dir "$OUTDIR" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --max_length "$MAX_LENGTH" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --wandb_project "$WANDB_PROJECT" \
  "${EXTRA[@]}"
