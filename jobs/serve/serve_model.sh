#!/bin/bash
# Generic vLLM OpenAI-compatible server job.
#
# Args:
#   $1  MODEL_ID   e.g. Qwen/Qwen3-8B
#   $2  MODEL_TAG  e.g. qwen3_8b   (used for the URL file: tmp/<tag>_server_url.txt)
#   $3  PORT       e.g. 8100
#
# URL file written to: $REPO/tmp/<tag>_server_url.txt
# Content: http://<hostname>:<port>
#
# Example submits:
#   sbatch jobs/serve/serve_model.sh Qwen/Qwen3-8B            qwen3_8b 8100
#   sbatch jobs/serve/serve_model.sh allenai/OLMo-3-7B-Instruct-SFT olmo_7b  8200

#SBATCH --job-name=vllm_server
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_serve_%x.out
#SBATCH --error=logs/%j_serve_%x.err

set -euo pipefail

MODEL_ID="${1:?Usage: sbatch serve_model.sh MODEL_ID MODEL_TAG PORT}"
MODEL_TAG="${2:?}"
PORT="${3:-8000}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs tmp

URL_FILE="tmp/${MODEL_TAG}_server_url.txt"
rm -f "$URL_FILE"

echo "════════════════════════════════════════"
echo "  vLLM SERVER: ${MODEL_ID}"
echo "  Port       : ${PORT}"
echo "  URL file   : ${URL_FILE}"
echo "════════════════════════════════════════"

# Start server in background
vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --dtype bfloat16 \
    --trust-remote-code \
    &
SERVER_PID=$!

# Poll until healthy
echo "Waiting for server to be ready..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
    # Exit if server process died
    kill -0 $SERVER_PID 2>/dev/null || { echo "Server process died!"; exit 1; }
done

# Write URL for clients
echo "http://$(hostname):${PORT}" > "$URL_FILE"
echo "Server ready → $(cat $URL_FILE)"

# Stay alive until cancelled or server dies
wait $SERVER_PID
