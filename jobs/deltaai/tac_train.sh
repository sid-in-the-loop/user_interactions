#!/bin/bash
# Train all 4 methods (SFT, JSD, DPO, RKL) sequentially on one (dataset, winrate)
# mixture. One GPU per sbatch. ~18h total walltime.
#
# Usage:
#   sbatch jobs/deltaai/tac_train.sh DATASET WINRATE
# DATASET in {wildchat, webinstruct, polaris}
# WINRATE in {100, 70, 50}

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=24:00:00
#SBATCH --output=logs/tac_train_%x_%j.out
#SBATCH --error=logs/tac_train_%x_%j.err

set -euo pipefail

DATASET="${1:?Usage: sbatch tac_train.sh DATASET WINRATE}"
WINRATE="${2:?Usage: sbatch tac_train.sh DATASET WINRATE}"

case "$DATASET" in wildchat|webinstruct|polaris) ;; *) echo "bad dataset: $DATASET" >&2; exit 1 ;; esac
case "$WINRATE" in 100|70|50|20) ;; *) echo "bad winrate: $WINRATE" >&2; exit 1 ;; esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ──────────────────────────────────────────────────────────────────────────────
TRAIN_DIR="experiments/tac_winrates/data/training_inputs"
IN_METHODS="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_for_methods.jsonl"
IN_SDPO="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_for_sdpo.jsonl"

if [ ! -f "$IN_METHODS" ] || [ ! -f "$IN_SDPO" ]; then
  echo "Training input files missing; running build_training_inputs.py first." >&2
  python experiments/tac_winrates/build_training_inputs.py
fi

if [ ! -f "$IN_METHODS" ] || [ ! -f "$IN_SDPO" ]; then
  echo "ERROR: still missing $IN_METHODS or $IN_SDPO" >&2
  exit 1
fi

CKPT_ROOT="/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/${DATASET}_w${WINRATE}"
mkdir -p "$CKPT_ROOT"

BASE_MODEL="Qwen/Qwen3-4B"

LORA_ARGS=(
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --lora_target all-linear
)

TAG="${DATASET}_w${WINRATE}"
COMMON_METHODS_ARGS=(
  --input         "$IN_METHODS"
  --y_star_field  y_star_prefix30
  --model         "$BASE_MODEL"
  --epochs        2
  --batch_size    2
  --grad_accum    32
  --lr            2e-6
  --max_length    2048
  --num_ckpts     4
  --save_every    0                 # triggers num_ckpts-based scheduling
  "${LORA_ARGS[@]}"
  --wandb_project tac-winrates
)

echo "=============== tac_train: $TAG ==============="
echo "methods input  : $IN_METHODS"
echo "sdpo   input   : $IN_SDPO"
echo "ckpt root      : $CKPT_ROOT"
echo "==============================================="

# ── 1/4 SFT ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] SFT ==="
python scripts/fkl/train_methods.py \
    --objective  sft \
    --output_dir "$CKPT_ROOT/sft" \
    --run_name   "${TAG}_sft" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 2/4 JSD ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] JSD ==="
python scripts/fkl/train_methods.py \
    --objective  jsd \
    --beta       0.5 \
    --pure_kl \
    --output_dir "$CKPT_ROOT/jsd" \
    --run_name   "${TAG}_jsd" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 3/4 DPO ─────────────────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] DPO ==="
python scripts/fkl/train_methods.py \
    --objective  dpo \
    --dpo_beta   0.1 \
    --output_dir "$CKPT_ROOT/dpo" \
    --run_name   "${TAG}_dpo" \
    "${COMMON_METHODS_ARGS[@]}"

# ── 4/4 RKL (offline SDPO) with LoRA ────────────────────────────────────────
echo "=== [$(date +%H:%M)] RKL (offline SDPO, LoRA) ==="
python experiments/tac_winrates/train_sdpo_lora.py \
    --base_model    "$BASE_MODEL" \
    --train_jsonl   "$IN_SDPO" \
    --output_dir    "$CKPT_ROOT/rkl" \
    --num_epochs    2 \
    --batch_size    4 \
    --grad_accum    8 \
    --learning_rate 2e-6 \
    --num_ckpts     4 \
    "${LORA_ARGS[@]}" \
    --wandb_project tac-winrates \
    --run_name      "${TAG}_rkl"

echo "=== [$(date +%H:%M)] DONE: $TAG ==="
du -sh "$CKPT_ROOT"/*/ 2>/dev/null || true
