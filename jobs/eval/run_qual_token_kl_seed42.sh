#!/usr/bin/env bash
#SBATCH --job-name=qual_token_kl_seed42
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qual_token_kl_seed42_%j_%x.out
#SBATCH --error=logs/qual_token_kl_seed42_%j_%x.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

PROBE_SET="$REPO_ROOT/results/qual_probe_set_seed42.json"

TOKEN_OUT_DIR="$REPO_ROOT/results/qual_token_kl_seed42/token_kl_txt"
SIGNALS_OUT_DIR="$REPO_ROOT/results/qual_token_kl_seed42/signals_jsonl"

mkdir -p "$TOKEN_OUT_DIR" "$SIGNALS_OUT_DIR" logs
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"

CHECKPOINTS=(
  "Qwen/Qwen3-4B"
  "/data/group_data/cx_group/ssmurali/offpolicy/fkl/sft_wc_thinking_full_bs8_ga32_lr5e6_6ep/epoch-4_hf"
  "/data/group_data/cx_group/ssmurali/offpolicy/fkl/sft_wc_thinking_best_bs8_ga32_lr5e6/extended_final_hf"
  "/data/group_data/cx_group/ssmurali/offpolicy/fkl/sft_wf_thinking_full_bs8_ga32_lr2e6_6ep/epoch-6_hf"
  "/data/group_data/cx_group/ssmurali/offpolicy/fkl/sft_wf_thinking_best_bs8_ga32_lr5e6_6ep/ext-epoch-12_hf"
)

echo "════════════════════════════════════════"
echo "Qualitative token KL logging"
echo "  PROBE_SET: $PROBE_SET"
echo "  TOKEN_OUT_DIR: $TOKEN_OUT_DIR"
echo "  SIGNALS_OUT_DIR: $SIGNALS_OUT_DIR"
echo "  MAX_LENGTH: $MAX_LENGTH"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "════════════════════════════════════════"

for CKPT in "${CHECKPOINTS[@]}"; do
  CKPT_NAME="$(basename "$CKPT")"
  OUT_FILE="$SIGNALS_OUT_DIR/${CKPT_NAME}_signal.jsonl"
  echo "[$(date '+%F %T')] Running checkpoint: $CKPT"
  echo "  -> $OUT_FILE"
  python scripts/fkl/measure_fkl_signal.py \
    --probe_set "$PROBE_SET" \
    --checkpoint "$CKPT" \
    --output "$OUT_FILE" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --token_kl_txt_out_dir "$TOKEN_OUT_DIR"
  echo "----------------------------------------"
done

echo "[$(date '+%F %T')] Done."

