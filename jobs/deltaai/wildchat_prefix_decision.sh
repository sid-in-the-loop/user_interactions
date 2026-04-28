#!/usr/bin/env bash
# Single-job offline vLLM generation for the WildChat prefix-decision experiment.
# Loads Qwen3-4B once and runs all 33,920 × 4 prompts through one llm.generate()
# call. No HTTP server, no client/server roundtrips.
#
# Usage:
#   sbatch jobs/deltaai/wildchat_prefix_decision.sh smoke   # 50 rows
#   sbatch jobs/deltaai/wildchat_prefix_decision.sh full

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=wildchat_prefix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --output=logs/wildchat_prefix_%j.out
#SBATCH --error=logs/wildchat_prefix_%j.err

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
echo "  wildchat_prefix_decision  mode=$MODE"
echo "  job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "  HF_HOME=$HF_HOME"
echo "════════════════════════════════════════"

python experiments-for-apr26-may1/wildchat_prefix_decision/01_generate.py "${LIMIT_FLAG[@]}"
