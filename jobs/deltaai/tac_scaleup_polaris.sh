#!/bin/bash
# Full POLARIS pipeline: pool (53k) -> stage 1 pass-rate (Qwen3-4B T=0.7 N=8)
# -> stage 2 solvability (Qwen3-14B) -> stage 3 critiques (Qwen3-14B)
# -> gen y_base + y_star_0 (Qwen3-4B).
#
# Stage 1 is the long pole (~25h at full 53k). Total fits in 48h walltime.

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_scaleup_pol
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=48:00:00
#SBATCH --output=logs/tac_scaleup_pol_%j.out
#SBATCH --error=logs/tac_scaleup_pol_%j.err

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

echo "=== prep: polaris pool (full 53k) ==="
python experiments/tac_winrates/prep/prep_polaris_pool.py \
    --output "$DATA_DIR/polaris_pool.jsonl" --n 0
wc -l "$DATA_DIR/polaris_pool.jsonl"

echo "=== stage 1: pass-rate (Qwen3-4B) ==="
python experiments/tac_winrates/polaris_pipeline/01_pass_rate_filter.py \
    --input  "$DATA_DIR/polaris_pool.jsonl" \
    --output "$DATA_DIR/polaris_stage1_passrate.jsonl"

echo "=== stage 2: Qwen3-14B solvability ==="
python experiments/tac_winrates/polaris_pipeline/02_polaris_solve_filter.py \
    --input  "$DATA_DIR/polaris_stage1_passrate.jsonl" \
    --output "$DATA_DIR/polaris_stage2_solvable.jsonl"

echo "=== stage 3: Qwen3-14B critiques ==="
python experiments/tac_winrates/polaris_pipeline/03_generate_critiques.py \
    --input         "$DATA_DIR/polaris_stage2_solvable.jsonl" \
    --output        "$DATA_DIR/polaris_unified.jsonl" \
    --sanity_output "$DATA_DIR/polaris_sanity_samples.txt" \
    --target_n      1000000

echo "=== gen: polaris y_base + y_star_0 ==="
python experiments/tac_winrates/generate/generate.py \
    --input  "$DATA_DIR/polaris_unified.jsonl" \
    --output "$GEN_DIR/polaris_generations.jsonl" \
    --only_xo

echo "=== done ==="
wc -l "$DATA_DIR"/polaris_{pool,stage1_passrate,stage2_solvable,unified}.jsonl \
      "$GEN_DIR/polaris_generations.jsonl"
