#!/bin/bash
# Generate y_base + y_star_{0,30,70,100} for one TAC dataset.
#
# Usage:
#   sbatch jobs/deltaai/tac_gen.sh DATASET
# DATASET in {wildchat, webinstruct, polaris}

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/tac_gen_%x_%j.out
#SBATCH --error=logs/tac_gen_%x_%j.err

set -euo pipefail

DATASET="${1:?Usage: sbatch tac_gen.sh DATASET  (wildchat|webinstruct|polaris)}"

case "$DATASET" in
  wildchat|webinstruct|polaris) ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATA_DIR="experiments/tac_winrates/data"
GEN_DIR="experiments/tac_winrates/results/generations"
mkdir -p "$GEN_DIR"

# Unified input file: wildchat_unified.jsonl, webinstruct_unified.jsonl, polaris_unified.jsonl
INPUT="$DATA_DIR/${DATASET}_unified.jsonl"
OUTPUT="$GEN_DIR/${DATASET}_generations.jsonl"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input not found: $INPUT" >&2
  exit 1
fi

echo "=== tac_gen: $DATASET ==="
echo "input:  $INPUT"
echo "output: $OUTPUT"

python experiments/tac_winrates/generate/generate.py \
    --input  "$INPUT" \
    --output "$OUTPUT"

echo "=== done: $DATASET ==="
wc -l "$OUTPUT"
