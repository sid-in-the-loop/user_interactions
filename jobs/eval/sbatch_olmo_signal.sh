#!/bin/bash
# Measure how much feedback (y, o) shifts OLMo's token distribution along y_base.
# Runs two forward passes per sample (no gradients), then plots results.
#
# Submit:
#   sbatch jobs/eval/sbatch_olmo_signal.sh

#SBATCH --job-name=olmo_signal
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo_signal_%j.out
#SBATCH --error=logs/olmo_signal_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

YBASE="datasets/wildfeedback/olmo_3_7b_500/ybase_olmo.jsonl"
WF_DIR="datasets/wildfeedback"
OUT_DIR="data/olmo_signal"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"
mkdir -p logs "$OUT_DIR"

echo "Job ID  : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"
echo "ybase   : $YBASE  ($(wc -l < $YBASE) samples)"
echo "output  : $OUT_DIR"
echo "──────────────────────────────────────────"

# Step 1 — measure KL signal
echo ""
echo "=== Step 1: measure signal ==="
python scripts/eval/measure_olmo_signal.py \
    --ybase      "$YBASE" \
    --wf_dir     "$WF_DIR" \
    --model      "$MODEL" \
    --output_dir "$OUT_DIR" \
    --batch_size 8 \
    --max_length 4096

# Step 2 — plot
echo ""
echo "=== Step 2: plot ==="
python scripts/eval/plot_olmo_signal.py \
    --input      "$OUT_DIR/olmo_signal.jsonl" \
    --output_dir "$OUT_DIR"

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Outputs:"
echo "  $OUT_DIR/olmo_signal.jsonl"
echo "  $OUT_DIR/olmo_signal_summary.txt"
echo "  $OUT_DIR/signal_histogram.png"
echo "  $OUT_DIR/signal_position.png"
echo "  $OUT_DIR/signal_scatter.png"
