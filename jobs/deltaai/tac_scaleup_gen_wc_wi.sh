#!/bin/bash
# Full-scale gen for wildchat (34k) + webinstruct (50k) in a single 1-GPU job.
# Preps both datasets from source, then runs generate.py --only_xo
# (y_base + y_star_0) for each, reusing the same loaded Qwen3-4B engine
# across datasets (separate LLM instantiations per dataset but same process).

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_scaleup_wcwi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=logs/tac_scaleup_wcwi_%j.out
#SBATCH --error=logs/tac_scaleup_wcwi_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATA_DIR="experiments/tac_winrates/data"
GEN_DIR="experiments/tac_winrates/results/generations"
mkdir -p "$GEN_DIR"

echo "=== prep: wildchat (full 34k) ==="
python experiments/tac_winrates/prep/prep_wildchat.py \
    --output "$DATA_DIR/wildchat_unified.jsonl" --n 0
wc -l "$DATA_DIR/wildchat_unified.jsonl"

echo "=== prep: webinstruct (full 50k from parquet) ==="
python experiments/tac_winrates/prep/prep_webinstruct.py \
    --output "$DATA_DIR/webinstruct_unified.jsonl" --n 0
wc -l "$DATA_DIR/webinstruct_unified.jsonl"

echo "=== gen: wildchat y_base + y_star_0 ==="
python experiments/tac_winrates/generate/generate.py \
    --input  "$DATA_DIR/wildchat_unified.jsonl" \
    --output "$GEN_DIR/wildchat_generations.jsonl" \
    --only_xo

echo "=== gen: webinstruct y_base + y_star_0 ==="
python experiments/tac_winrates/generate/generate.py \
    --input  "$DATA_DIR/webinstruct_unified.jsonl" \
    --output "$GEN_DIR/webinstruct_generations.jsonl" \
    --only_xo

echo "=== done ==="
wc -l "$GEN_DIR/wildchat_generations.jsonl" "$GEN_DIR/webinstruct_generations.jsonl"
