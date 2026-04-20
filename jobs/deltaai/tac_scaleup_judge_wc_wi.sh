#!/bin/bash
# Judge wildchat + webinstruct at prefix=0 only (y_star_vs_y_base).
# Runs sequentially in one job; no GPU needed but DeltaAI requires 1.

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_scaleup_judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --output=logs/tac_scaleup_judge_%j.out
#SBATCH --error=logs/tac_scaleup_judge_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2; exit 1
fi

DATA_DIR="experiments/tac_winrates/data"
GEN_DIR="experiments/tac_winrates/results/generations"
RES_DIR="experiments/tac_winrates/results"

for ds in wildchat webinstruct; do
  echo "=== judge: $ds ==="
  python experiments/tac_winrates/eval/eval.py \
      --unified      "$DATA_DIR/${ds}_unified.jsonl" \
      --generations  "$GEN_DIR/${ds}_generations.jsonl" \
      --output_csv   "$RES_DIR/eval_${ds}.csv" \
      --max_workers  32
done

echo "=== done ==="
wc -l "$RES_DIR"/eval_{wildchat,webinstruct}.csv
