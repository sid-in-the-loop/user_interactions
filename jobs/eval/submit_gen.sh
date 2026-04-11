#!/bin/bash
# Unified y* generation submission — wraps generate_ystar_prefix.py
#
# Usage:
#   sbatch jobs/eval/submit_gen.sh <model> [input] [output] [extra_args...]
#
# Arguments:
#   model   HF model name (required), e.g. Qwen/Qwen3-8B
#   input   input JSONL  (optional, default: datasets/wildfeedback/filtered_BEST.jsonl)
#   output  output JSONL (optional, default: derived from model basename)
#
# extra_args are forwarded verbatim to generate_ystar_prefix.py
#
# Examples:
#   sbatch jobs/eval/submit_gen.sh Qwen/Qwen3-8B
#   sbatch jobs/eval/submit_gen.sh Qwen/Qwen3-4B datasets/wildfeedback/filtered_BEST.jsonl
#   sbatch jobs/eval/submit_gen.sh Qwen/Qwen3-8B datasets/wildfeedback/filtered_BEST.jsonl \
#       datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl --prefix_frac 0.5 --max_tokens 2048
#   sbatch --gres=gpu:L40S:1 jobs/eval/submit_gen.sh Qwen/Qwen3-8B
#
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

MODEL="${1:?Usage: submit_gen.sh <model> [input] [output] [extra_args...]}"
INPUT="${2:-datasets/wildfeedback/filtered_BEST.jsonl}"

# 3rd arg: output path if it looks like a file (*.jsonl), else treat as extra_args
if [[ "${3:-}" == *.jsonl ]]; then
    OUTPUT="$3"
    shift 3 2>/dev/null || shift $#
else
    MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
    OUTPUT="datasets/wildfeedback/ystar_prefix_${MODEL_TAG}.jsonl"
    shift 2 2>/dev/null || shift $#
fi
EXTRA_ARGS="${*:-}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DATASETS_TRUST_REMOTE_CODE=1
mkdir -p logs "$(dirname "$OUTPUT")"
cd /home/ssmurali/user_interactions
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

echo "════════════════════════════════════════"
echo " Model  : $MODEL"
echo " Input  : $INPUT"
echo " Output : $OUTPUT"
echo " Extra  : ${EXTRA_ARGS:-none}"
echo "════════════════════════════════════════"

python scripts/fkl/generate_ystar_prefix.py \
    --input         "$INPUT"   \
    --output        "$OUTPUT"  \
    --model         "$MODEL"   \
    --prefix_frac   0.30       \
    --max_tokens    1024       \
    --temperature   1.0        \
    --tp_size       1          \
    --gpu_util      0.92       \
    --max_num_seqs  256        \
    --max_model_len 8192       \
    $EXTRA_ARGS

echo "Done → $OUTPUT"
