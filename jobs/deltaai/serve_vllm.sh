#!/bin/bash
# vLLM server on DeltaAI (1 GPU batch job).
#
# Usage:
#   sbatch jobs/deltaai/serve_vllm.sh <MODEL_ID> <TAG> <PORT> [MAX_MODEL_LEN]
#
# Examples:
#   sbatch jobs/deltaai/serve_vllm.sh Qwen/Qwen3-8B qwen3_8b 8001
#   sbatch jobs/deltaai/serve_vllm.sh Qwen/Qwen3-14B qwen3_14b 8002 4096
#
# Writes URL to: tmp/<TAG>_server_url.txt (e.g. http://gh045:8001)

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=vllm_serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --output=logs/serve_%j.out
#SBATCH --error=logs/serve_%j.err

set -euo pipefail

MODEL_ID="${1:?Usage: sbatch serve_vllm.sh MODEL_ID TAG PORT [MAX_MODEL_LEN]}"
TAG="${2:?}"
PORT="${3:?}"
MAX_MODEL_LEN="${4:-8192}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs tmp

URL_FILE="tmp/${TAG}_server_url.txt"
rm -f "$URL_FILE"

echo "════════════════════════════════════════"
echo "  vLLM SERVER: ${MODEL_ID}"
echo "  Port       : ${PORT}"
echo "  Max len    : ${MAX_MODEL_LEN}"
echo "  Node       : $(hostname)"
echo "════════════════════════════════════════"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    &
SERVER_PID=$!

echo "Waiting for server to be ready..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
    kill -0 $SERVER_PID 2>/dev/null || { echo "Server process died!"; exit 1; }
done

echo "http://$(hostname):${PORT}" > "$URL_FILE"
echo "Server ready → $(cat "$URL_FILE")"

wait $SERVER_PID
