#!/bin/bash
# Single mode-covering / mode-seeking training run. One LoRA, one GPU.
#
# Usage:
#   sbatch jobs/babel/train_demo2teacher.sh \
#       <objective> <run_id> <model> <dataset_path> <output_dir>
#
#   objective    — sft | fkl | sdpo | pc_sdpo
#   run_id       — short tag (e.g. WI-1, WC-7)
#   model        — Qwen/Qwen2.5-Math-7B  or  Qwen/Qwen3-4B
#   dataset_path — teacher_wins_* / teacher_loses_* jsonl
#   output_dir   — where to save LoRA checkpoints
#
# Optional env overrides:
#   EPOCHS BATCH_SIZE GRAD_ACCUM LR MAX_LENGTH LORA_R LORA_ALPHA WANDB_PROJECT
#
#SBATCH --job-name=demo2t
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/demo2t_%j.out
#SBATCH --error=logs/demo2t_%j.err

set -euo pipefail

OBJ="${1:?usage: sbatch ${0##*/} <obj> <run_id> <model> <dataset_path> <output_dir>}"
RUN_ID="${2:?}"
MODEL="${3:?}"
DATASET="${4:?}"
OUTDIR="${5:?}"

case "$OBJ" in
  sft|fkl)        SCRIPT="experiments-for-apr26-may1/training/train_modecovering.py" ;;
  sdpo|pc_sdpo)   SCRIPT="experiments-for-apr26-may1/training/train_modeseeking.py" ;;
  dpo)            SCRIPT="experiments-for-apr26-may1/training/train_dpo.py" ;;
  *) echo "ERROR: unknown objective '$OBJ'" >&2; exit 1 ;;
esac

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ssmurali/user_interactions}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

mkdir -p logs "$OUTDIR"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LR="${LR:-2e-6}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
NUM_INTERMEDIATE_CKPTS="${NUM_INTERMEDIATE_CKPTS:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-demonstrator-to-teacher}"

echo "════════════════════════════════════════"
echo "  demo→teacher training (Babel)"
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

python "$SCRIPT" \
  --dataset_path "$DATASET" \
  --run_id "$RUN_ID" \
  --output_dir "$OUTDIR" \
  --model "$MODEL" \
  --objective "$OBJ" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --max_length "$MAX_LENGTH" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --num_intermediate_ckpts "$NUM_INTERMEDIATE_CKPTS" \
  --wandb_project "$WANDB_PROJECT"

echo "Done: $RUN_ID ($OBJ) → $OUTDIR"
