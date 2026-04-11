#!/bin/bash
# Step 2 — FKL probe signal for one checkpoint (1 GPU).
# Usage: sbatch jobs/eval/eval_probe_signal.sh <output_name> <checkpoint_path>
#   e.g. sbatch jobs/eval/eval_probe_signal.sh baseline_sft_fp32 /data/.../baseline_sft_fp32/final
# Output: results/<output_name>_signal.jsonl

#SBATCH --job-name=probe_signal
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/probe_signal_%j_%x.out
#SBATCH --error=logs/probe_signal_%j_%x.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

NAME=$1
CKPT=$2
if [ -z "$NAME" ] || [ -z "$CKPT" ]; then
    echo "Usage: sbatch jobs/eval/eval_probe_signal.sh <output_name> <checkpoint_path>"
    exit 1
fi

# Under sbatch, use submit dir so paths resolve correctly (BASH_SOURCE can point into /var/spool)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

PROBE_SET="${PROBE_SET:-$REPO_ROOT/results/probe_set.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results}"
mkdir -p "$OUTPUT_DIR" logs

OUT_FILE="$OUTPUT_DIR/${NAME}_signal.jsonl"
echo "Probe signal: $NAME <- $CKPT -> $OUT_FILE"
python scripts/fkl/measure_fkl_signal.py \
    --probe_set "$PROBE_SET" \
    --checkpoint "$CKPT" \
    --output "$OUT_FILE" \
    --max_length "${MAX_LENGTH:-2048}" \
    --batch_size "${BATCH_SIZE:-8}"
echo "Done: $OUT_FILE"
