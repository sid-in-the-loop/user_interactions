#!/bin/bash
# Build 12 TAC training mixtures (3 datasets × 4 winrate targets).
# Depends on:
#   - tac_scaleup_judge_wc_wi finished (writes eval_wildchat.csv, eval_webinstruct.csv)
#   - tac_scaleup_polaris finished (writes polaris_generations.jsonl)
# Math-verify runs inside the builder for POLARIS.

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_mixtures
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=logs/tac_mixtures_%j.out
#SBATCH --error=logs/tac_mixtures_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python experiments/tac_winrates/build_mixtures.py

echo "=== mixture files ==="
ls -la experiments/tac_winrates/data/mixtures/
