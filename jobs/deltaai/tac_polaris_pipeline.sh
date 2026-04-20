#!/bin/bash
# POLARIS construction pipeline for TAC winrates (experiments/tac_winrates).
# Runs all 3 stages back-to-back on a single GH200:
#   1. Pass-rate filter (Qwen3-4B, T=0.7, N=8)  — hard-for-4B subset
#   2. Solvability filter (Qwen3-14B non-think, greedy)
#   3. Critique generation (Qwen3-14B non-think, greedy)
#
# Stages 2+3 use Qwen3-14B (swapped from Polaris-4B per spec) because smoke
# testing showed Polaris's <think> blocks blow past 4k tokens on every problem.
# Each stage loads its model fresh; cheaper than juggling two engines in RAM.
#
# Usage:
#   sbatch jobs/deltaai/tac_polaris_pipeline.sh [LIMIT]
# LIMIT=0 (default) processes all 10k pool items (stage 1 is the long pole).

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_polaris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=18:00:00
#SBATCH --output=logs/tac_polaris_%j.out
#SBATCH --error=logs/tac_polaris_%j.err

set -euo pipefail

LIMIT="${1:-0}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATA_DIR="experiments/tac_winrates/data"

echo "=== Stage 1: pass-rate filter (Qwen3-4B) ==="
python experiments/tac_winrates/polaris_pipeline/01_pass_rate_filter.py \
    --input "$DATA_DIR/polaris_pool.jsonl" \
    --output "$DATA_DIR/polaris_stage1_passrate.jsonl" \
    --limit "$LIMIT"

echo "=== Stage 2: Polaris-4B solvability filter ==="
python experiments/tac_winrates/polaris_pipeline/02_polaris_solve_filter.py \
    --input "$DATA_DIR/polaris_stage1_passrate.jsonl" \
    --output "$DATA_DIR/polaris_stage2_solvable.jsonl"

echo "=== Stage 3: critique generation (Polaris-4B) ==="
python experiments/tac_winrates/polaris_pipeline/03_generate_critiques.py \
    --input "$DATA_DIR/polaris_stage2_solvable.jsonl" \
    --output "$DATA_DIR/polaris_unified.jsonl" \
    --sanity_output "$DATA_DIR/polaris_sanity_samples.txt" \
    --target_n 1000

echo "=== done ==="
wc -l "$DATA_DIR"/polaris_{pool,stage1_passrate,stage2_solvable,unified}.jsonl
