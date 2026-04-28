#!/bin/bash
# Resume tac_train: runs only methods whose $CKPT/$method/final is missing.
# Same pipeline as tac_train.sh (SFT, FKL, JSD@1e-4, DPO, RKL at 2e-6).
#
# Usage:
#   sbatch jobs/deltaai/tac_train_resume.sh DATASET WINRATE MATCH_N

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_train_resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=logs/tac_train_%x_%j.out
#SBATCH --error=logs/tac_train_%x_%j.err

set -euo pipefail

DATASET="${1:?Usage: sbatch tac_train_resume.sh DATASET WINRATE MATCH_N}"
WINRATE="${2:?Usage: sbatch tac_train_resume.sh DATASET WINRATE MATCH_N}"
MATCH_N="${3:?Usage: sbatch tac_train_resume.sh DATASET WINRATE MATCH_N}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

TRAIN_DIR="experiments/tac_winrates/data/training_inputs"
IN_METHODS="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_matched${MATCH_N}_for_methods.jsonl"
IN_SDPO="$TRAIN_DIR/mix_${DATASET}_teacher_xo_w${WINRATE}_matched${MATCH_N}_for_sdpo.jsonl"
[ -f "$IN_METHODS" ] || { echo "ERROR: missing $IN_METHODS (run tac_train.sh first to materialize matched-N)" >&2; exit 1; }
[ -f "$IN_SDPO"    ] || { echo "ERROR: missing $IN_SDPO" >&2; exit 1; }

CKPT_ROOT="/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/${DATASET}_w${WINRATE}_n${MATCH_N}"
BASE_MODEL="Qwen/Qwen3-4B"
TAG="${DATASET}_w${WINRATE}_n${MATCH_N}"

LORA_ARGS=( --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_target all-linear )
COMMON=(
  --input "$IN_METHODS" --y_star_field y_star_prefix30 --model "$BASE_MODEL"
  --epochs 2 --batch_size 2 --grad_accum 32 --max_length 2048
  --num_ckpts 4 --save_every 0 "${LORA_ARGS[@]}"
  --wandb_project tac-winrates
)

have_final() { [ -d "$CKPT_ROOT/$1/final" ]; }
need() { have_final "$1" && { echo "[skip] $1 (final exists)"; return 1; } || return 0; }

echo "=============== tac_train_resume: $TAG ==============="
echo "ckpt root: $CKPT_ROOT"
for m in sft fkl jsd dpo rkl; do
  if have_final "$m"; then echo "  $m: final ✓ (skip)"; else echo "  $m: MISSING (will run)"; fi
done
echo "======================================================"

if need sft; then
  echo "=== [$(date +%H:%M)] SFT (lr=2e-6) ==="
  rm -rf "$CKPT_ROOT/sft"
  python scripts/fkl/train_methods.py --objective sft --lr 2e-6 \
      --output_dir "$CKPT_ROOT/sft" --run_name "${TAG}_sft" "${COMMON[@]}"
fi

if need fkl; then
  echo "=== [$(date +%H:%M)] FKL (lr=2e-6) ==="
  rm -rf "$CKPT_ROOT/fkl"
  python scripts/fkl/train_methods.py --objective fkl --pure_kl --lr 2e-6 \
      --output_dir "$CKPT_ROOT/fkl" --run_name "${TAG}_fkl" "${COMMON[@]}"
fi

if need jsd; then
  echo "=== [$(date +%H:%M)] JSD (lr=1e-4) ==="
  rm -rf "$CKPT_ROOT/jsd"
  python scripts/fkl/train_methods.py --objective jsd --beta 0.5 --pure_kl --lr 1e-4 \
      --output_dir "$CKPT_ROOT/jsd" --run_name "${TAG}_jsd_lr1e4" "${COMMON[@]}"
fi

if need dpo; then
  echo "=== [$(date +%H:%M)] DPO (lr=2e-6) ==="
  rm -rf "$CKPT_ROOT/dpo"
  python scripts/fkl/train_methods.py --objective dpo --dpo_beta 0.1 --lr 2e-6 \
      --output_dir "$CKPT_ROOT/dpo" --run_name "${TAG}_dpo" "${COMMON[@]}"
fi

if need rkl; then
  echo "=== [$(date +%H:%M)] RKL (offline SDPO, LoRA, lr=2e-6) ==="
  rm -rf "$CKPT_ROOT/rkl"
  python experiments/tac_winrates/train_sdpo_lora.py \
      --base_model "$BASE_MODEL" --train_jsonl "$IN_SDPO" \
      --output_dir "$CKPT_ROOT/rkl" \
      --num_epochs 2 --batch_size 4 --grad_accum 8 --learning_rate 2e-6 \
      --num_ckpts 4 "${LORA_ARGS[@]}" \
      --wandb_project tac-winrates --run_name "${TAG}_rkl"
fi

echo "=== [$(date +%H:%M)] DONE: $TAG ==="
du -sh "$CKPT_ROOT"/*/ 2>/dev/null || true
