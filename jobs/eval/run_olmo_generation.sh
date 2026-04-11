#!/bin/bash
# Generate y_base and y* for OLMo-3-7B-Instruct on WildFeedback full dataset.
#
# Runs both passes (y_base first, then y*) in a single vLLM instance.
# Outputs:
#   datasets/wildfeedback/olmo_3_7b/ybase_olmo.jsonl
#   datasets/wildfeedback/olmo_3_7b/ystar_olmo.jsonl
#
# Tunable via env vars before calling sbatch/bash:
#   MODEL        — HF model ID (default: allenai/OLMo-2-7B-Instruct)
#   GPUS         — number of GPUs (default: 1)
#   TARGET       — ybase | ystar | both (default: both)
#   MAX_TOKENS   — max generation length (default: 1024)
#   INPUT        — input JSONL (default: datasets/wildfeedback/tuples.jsonl)
#   IDS_FILE     — optional JSON ID list (default: none — use full dataset)
#
# Example (full dataset, 4 GPUs):
#   export GPUS=4 MODEL=allenai/OLMo-2-7B-Instruct
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=128G \
#          --job-name=olmo_gen --time=12:00:00 \
#          jobs/eval/run_olmo_generation.sh

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/olmo_gen_%j_%x.out
#SBATCH --error=logs/olmo_gen_%j_%x.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct-SFT}"
GPUS="${GPUS:-1}"
TARGET="${TARGET:-both}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
INPUT="${INPUT:-datasets/wildfeedback/tuples.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/wildfeedback/olmo_3_7b}"
IDS_FILE="${IDS_FILE:-}"

# ── vLLM tuning based on GPU count ────────────────────────────────────────────
if [[ "$GPUS" -ge 4 ]]; then
  TP="${TP:-$GPUS}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
  GPU_UTIL="${GPU_UTIL:-0.92}"
else
  TP="${TP:-1}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
  GPU_UTIL="${GPU_UTIL:-0.95}"
fi

mkdir -p logs "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo "  OLMo generation"
echo "  model     : $MODEL"
echo "  target    : $TARGET"
echo "  input     : $INPUT"
echo "  output    : $OUTPUT_DIR"
echo "  tp_size   : $TP  |  max_num_seqs: $MAX_NUM_SEQS"
echo "════════════════════════════════════════"

IDS_ARG=""
[[ -n "$IDS_FILE" ]] && IDS_ARG="--ids-file $IDS_FILE"

python "$REPO/scripts/eval/generate_olmo.py" \
  --input         "$INPUT" \
  --output_dir    "$OUTPUT_DIR" \
  --target        "$TARGET" \
  --model         "$MODEL" \
  --max_tokens    "$MAX_TOKENS" \
  --temperature   "$TEMPERATURE" \
  --tp_size       "$TP" \
  --max_num_seqs  "$MAX_NUM_SEQS" \
  --gpu_util      "$GPU_UTIL" \
  $IDS_ARG

echo "Done. Outputs in $OUTPUT_DIR"
