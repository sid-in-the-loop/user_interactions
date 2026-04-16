#!/bin/bash
# Generate y_base responses using a running vLLM server.
# Waits for the server URL file to appear before starting.
#
# Usage:
#   sbatch jobs/deltaai/gen_ybase.sh <SERVER_TAG> <INPUT> <OUTPUT_MERGED>
#
# Example:
#   sbatch jobs/deltaai/gen_ybase.sh qwen3_8b \
#     datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl \
#     datasets/wildfeedback/ystar_ybase_qwen3_8b.jsonl

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=gen_ybase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_ybase_%j.out
#SBATCH --error=logs/gen_ybase_%j.err

set -euo pipefail

SERVER_TAG="${1:?Usage: sbatch gen_ybase.sh SERVER_TAG INPUT OUTPUT_MERGED}"
INPUT="${2:?}"
OUTPUT_MERGED="${3:?}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

URL_FILE="tmp/${SERVER_TAG}_server_url.txt"

echo "Waiting for server URL file: ${URL_FILE}"
while [ ! -f "$URL_FILE" ]; do
    sleep 10
done
SERVER_URL=$(cat "$URL_FILE")
echo "Server URL: ${SERVER_URL}"

# Wait for server to be healthy
until curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
    echo "  Server not healthy yet, retrying..."
    sleep 10
done
echo "Server is healthy."

# Generate y_base and merge into y* file
python scripts/eval/gen_ybase_and_merge.py \
    --input "$INPUT" \
    --output "$OUTPUT_MERGED" \
    --server-url "$SERVER_URL" \
    --max-workers 32

echo "Done. Output: ${OUTPUT_MERGED}"
