#!/bin/bash
# Compute KL(π_ckpt ∥ π_base) for all 6 ckpts of one WC run.
# Usage: sbatch jobs/babel/eval_kl_one_run.sh <run_id> <run_ckpt_dir>
#
#SBATCH --job-name=demo2t-kl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/kl_%j.out
#SBATCH --error=logs/kl_%j.err

set -euo pipefail

RUN_ID="${1:?usage: sbatch ${0##*/} <run_id> <run_ckpt_dir>}"
RUN_CKPT_DIR="${2:?}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/home/ssmurali/user_interactions
OUT=$REPO/CFT-Eric-Zhu/eval_results/demo2teacher/$RUN_ID/kl_vs_base.json
mkdir -p logs "$(dirname $OUT)"

cd $REPO
python3 -u jobs/babel/eval_kl_vs_base.py \
  --run_ckpt_dir "$RUN_CKPT_DIR" \
  --base_model Qwen/Qwen3-4B \
  --out_path "$OUT" \
  --n_prompts 80 --max_new 256

echo "Done KL eval for $RUN_ID"
cat "$OUT"
