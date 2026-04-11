#!/bin/bash
# Generate y* for ONE condition using a running vLLM server.
# CPU-only job — no GPU needed (the server has the GPU).
#
# Args:
#   $1  CONDITION      prefix30 | noprefix | full
#   $2  INPUT          e.g. datasets/wildfeedback/filter_raw.jsonl
#   $3  OUTPUT         e.g. datasets/wildfeedback/ystar_prefix30_qwen3_8b.jsonl
#   $4  MODEL_NAME     e.g. Qwen/Qwen3-8B  (passed to API, must match what server serves)
#   $5  SERVER_URL_FILE  e.g. tmp/qwen3_8b_server_url.txt
#
# Example:
#   sbatch jobs/train/gen_condition.sh \
#       prefix30 \
#       datasets/wildfeedback/filter_raw.jsonl \
#       datasets/wildfeedback/ystar_prefix30_qwen3_8b.jsonl \
#       Qwen/Qwen3-8B \
#       tmp/qwen3_8b_server_url.txt

#SBATCH --job-name=gen_condition
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j_gen_%x.out
#SBATCH --error=logs/%j_gen_%x.err

set -euo pipefail

CONDITION="${1:?Usage: sbatch gen_condition.sh CONDITION INPUT OUTPUT MODEL_NAME SERVER_URL_FILE}"
INPUT="${2:?}"
OUTPUT="${3:?}"
MODEL_NAME="${4:?}"
SERVER_URL_FILE="${5:?}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

mkdir -p logs "$(dirname "$OUTPUT")"

echo "════════════════════════════════════════"
echo "  CONDITION : $CONDITION"
echo "  INPUT     : $INPUT"
echo "  OUTPUT    : $OUTPUT"
echo "  MODEL     : $MODEL_NAME"
echo "  SERVER    : $SERVER_URL_FILE"
echo "════════════════════════════════════════"

python scripts/fkl/generate_ystar_client.py \
    --input            "$INPUT"           \
    --output           "$OUTPUT"          \
    --condition        "$CONDITION"       \
    --model_name       "$MODEL_NAME"      \
    --server_url_file  "$SERVER_URL_FILE" \
    --max_tokens       1024               \
    --temperature      1.0                \
    --max_concurrent   256                \
    --server_wait_timeout 1200

echo "Done → $OUTPUT"
