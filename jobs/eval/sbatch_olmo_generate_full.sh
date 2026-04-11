#!/bin/bash
# Full y_base + y* generation for OLMo-3-7B-Instruct-SFT on all WildFeedback tuples.
# Outputs:
#   datasets/wildfeedback/olmo_3_7b/ybase_olmo.jsonl
#   datasets/wildfeedback/olmo_3_7b/ystar_olmo.jsonl
#
# Submit from repo root:
#   sbatch jobs/eval/sbatch_olmo_generate_full.sh

#SBATCH --job-name=olmo_gen_full
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo_gen_full_%j.out
#SBATCH --error=logs/olmo_gen_full_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

INPUT="datasets/wildfeedback/tuples.jsonl"
OUTPUT_DIR="datasets/wildfeedback/olmo_3_7b"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"
mkdir -p logs "$OUTPUT_DIR"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Model      : $MODEL"
echo "Input      : $INPUT  ($(wc -l < $INPUT) samples)"
echo "Output dir : $OUTPUT_DIR"
echo "Started    : $(date)"
echo "──────────────────────────────────────────"

# Pass 1: y_base (OLMo sees only conversation history)
echo ""
echo "=== Pass 1: y_base ==="
python scripts/eval/generate_olmo.py \
    --target      ybase \
    --input       "$INPUT" \
    --output_dir  "$OUTPUT_DIR" \
    --model       "$MODEL" \
    --tp_size     1 \
    --max_num_seqs 512 \
    --gpu_util    0.92 \
    --max_tokens  1024 \
    --max_model_len 4096

# Pass 2: y* (OLMo sees history + GPT-4 response + user feedback)
echo ""
echo "=== Pass 2: y* ==="
python scripts/eval/generate_olmo.py \
    --target      ystar \
    --input       "$INPUT" \
    --output_dir  "$OUTPUT_DIR" \
    --model       "$MODEL" \
    --tp_size     1 \
    --max_num_seqs 512 \
    --gpu_util    0.92 \
    --max_tokens  1024 \
    --max_model_len 4096

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Outputs:"
echo "  $OUTPUT_DIR/ybase_olmo.jsonl  ($(wc -l < $OUTPUT_DIR/ybase_olmo.jsonl) lines)"
echo "  $OUTPUT_DIR/ystar_olmo.jsonl  ($(wc -l < $OUTPUT_DIR/ystar_olmo.jsonl) lines)"
