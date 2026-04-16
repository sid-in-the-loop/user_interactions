#!/bin/bash
# Run pairwise winrate: y_base vs y* (prefix30, noprefix, full)
# using local Qwen3-14B vLLM server as judge.
#
# Usage:
#   sbatch jobs/deltaai/winrate_ybase.sh <JUDGE_TAG> <INPUT> <OUTPUT_DIR>
#
# Example:
#   sbatch jobs/deltaai/winrate_ybase.sh qwen3_14b \
#     datasets/wildfeedback/ystar_ybase_qwen3_8b.jsonl \
#     data/winrate_results/ybase_vs_ystar_qwen3_8b

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=winrate_ybase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --output=logs/winrate_ybase_%j.out
#SBATCH --error=logs/winrate_ybase_%j.err

set -euo pipefail

JUDGE_TAG="${1:?Usage: sbatch winrate_ybase.sh JUDGE_TAG INPUT OUTPUT_DIR}"
INPUT="${2:?}"
OUTPUT_DIR="${3:?}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

URL_FILE="tmp/${JUDGE_TAG}_server_url.txt"

echo "Waiting for judge server URL file: ${URL_FILE}"
while [ ! -f "$URL_FILE" ]; do
    sleep 10
done
SERVER_URL=$(cat "$URL_FILE")
echo "Judge server URL: ${SERVER_URL}"

until curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
    echo "  Judge not healthy yet, retrying..."
    sleep 10
done
echo "Judge is healthy."

MODEL=$(curl -s "${SERVER_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Judge model: ${MODEL}"

# Run pairwise winrate using local judge
python scripts/eval/winrate_pairwise_ybase.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --server-url "$SERVER_URL" \
    --model "$MODEL" \
    --max-workers 8 \
    --subsample 500

echo "Done. Results in: ${OUTPUT_DIR}"
