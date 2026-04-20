#!/bin/bash
# Run judge / verifier over all 3 TAC-winrate datasets and aggregate.
# Needs OPENAI_API_KEY in env for wildchat + webinstruct (judge).
#
# Note: DeltaAI partitions require --gpus-per-node=1; the GPU is idle here
# (all work is OpenAI API + math_verify on CPU).

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --output=logs/tac_eval_%j.out
#SBATCH --error=logs/tac_eval_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATA_DIR="experiments/tac_winrates/data"
GEN_DIR="experiments/tac_winrates/results/generations"
RES_DIR="experiments/tac_winrates/results"
mkdir -p "$RES_DIR"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set. Add to ~/.bashrc before submitting." >&2
  exit 1
fi

run_one() {
  local ds="$1"
  echo "=== eval: $ds ==="
  python experiments/tac_winrates/eval/eval.py \
      --unified      "$DATA_DIR/${ds}_unified.jsonl" \
      --generations  "$GEN_DIR/${ds}_generations.jsonl" \
      --output_csv   "$RES_DIR/eval_${ds}.csv"
}

run_one wildchat
run_one webinstruct
run_one polaris

echo "=== aggregating ==="
python experiments/tac_winrates/eval/aggregate.py \
    --raw_csvs \
        "$RES_DIR/eval_wildchat.csv" \
        "$RES_DIR/eval_webinstruct.csv" \
        "$RES_DIR/eval_polaris.csv" \
    --summary_csv       "$RES_DIR/phase1_summary.csv" \
    --raw_combined_csv  "$RES_DIR/phase1_raw.csv" \
    --plot_dir          "$RES_DIR"

echo "=== done ==="
ls -la "$RES_DIR"
