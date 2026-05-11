#!/bin/bash
# Use the existing eval_ckpts_alpaca.py pipeline against the WC v2 ckpts.
# Proper Qwen3 hyperparams: temp=1.0, enable_thinking=False, LoRA hot-swap, LC winrate via gpt-4o-mini.
#
# Submit: sbatch jobs/babel/eval_alpaca_proper.sh
#
#SBATCH --job-name=demo2t-alpaca-v2
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/alpaca_v2_%j.out
#SBATCH --error=logs/alpaca_v2_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY required for gpt4o-mini judge" >&2; exit 1
fi

REPO=/home/ssmurali/user_interactions
cd $REPO

CKPT_ROOT=/data/group_data/cx_group/ssmurali/demo2teacher_v2
RESULTS_ROOT=$REPO/eval_results/alpaca_demo2teacher_v2

# Walks all 24 runs × 6 ckpts under CKPT_ROOT, hot-swaps LoRA per ckpt.
python3 -u scripts/eval/eval_ckpts_alpaca.py \
  "$CKPT_ROOT" \
  --base_model Qwen/Qwen3-4B \
  --judge gpt4omini \
  --max_gen_tokens 2048 \
  --concurrency 200 \
  --gpu_util 0.92 \
  --results_root "$RESULTS_ROOT"

echo "Done. results in $RESULTS_ROOT"
