#!/usr/bin/env bash
# WildChat judging pipeline (Prompt 3).
# Loads Qwen3-4B once, runs both judges (student offline + gpt4o-mini API),
# writes data/judgments.jsonl + data/judgments_summary.txt.
#
# Usage:
#   sbatch jobs/deltaai/wildchat_judge.sh smoke   # 50 rows
#   sbatch jobs/deltaai/wildchat_judge.sh full

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=wc_judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --output=logs/wildchat_judge_%j.out
#SBATCH --error=logs/wildchat_judge_%j.err

set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
  smoke) LIMIT_FLAG=(--limit 50) ;;
  full)  LIMIT_FLAG=() ;;
  *) echo "usage: sbatch ${0##*/} {smoke|full}"; exit 1 ;;
esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs

export HF_HOME="${HF_HOME:-/projects/bgtw/ssredharan/models}"

echo "════════════════════════════════════════"
echo "  wildchat_judge  mode=$MODE"
echo "  job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "════════════════════════════════════════"

python experiments-for-apr26-may1/wildchat_prefix_decision/02_judge.py \
  "${LIMIT_FLAG[@]}"

echo "── now build datasets (CPU-only post-processing) ──"
python experiments-for-apr26-may1/wildchat_prefix_decision/03_build_datasets.py
