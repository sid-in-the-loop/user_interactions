#!/bin/bash
# Judge all Phase-1 generations under eval_results/tac/ with GPT-4o-mini
# (plus programmatic judging for aime). Resumable: skips any (ckpt, bench)
# pair whose scores.json already exists.
#
# Usage:
#   sbatch jobs/deltaai/tac_eval_judge.sh
#
# Prereq: OPENAI_API_KEY must be in the submitting shell's env.

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_eval_judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --output=logs/tac_eval_%x_%j.out
#SBATCH --error=logs/tac_eval_%x_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set in the submitting shell." >&2
  exit 1
fi

RESULTS_ROOT="$REPO/eval_results/tac"
[ -d "$RESULTS_ROOT" ] || { echo "ERROR: $RESULTS_ROOT missing (run gen first)" >&2; exit 1; }

echo "=== judging $RESULTS_ROOT (gpt-4o-mini + programmatic) ==="
python scripts/eval/judge_all.py \
    --results_root "$RESULTS_ROOT" \
    --benchmarks   alpaca_eval arena_hard writingbench aime

echo "=== aggregating ==="
python scripts/eval/aggregate_scores.py \
    --results_root "$RESULTS_ROOT" \
    --output       "$RESULTS_ROOT/aggregate.csv" \
    --plots_dir    "$REPO/plots/tac_benchmark_curves"

echo "=== done ==="
ls -la "$RESULTS_ROOT"
