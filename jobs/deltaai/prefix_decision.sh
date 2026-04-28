#!/usr/bin/env bash
# Single sbatch job for the WebInstruct prefix-decision experiment.
#
# 1. starts a Qwen/Qwen2.5-Math-7B vLLM server on the allocated GH200 GPU
# 2. waits for /health to pass
# 3. runs the 4-step pipeline (gen → student-judge → openai-judge → summarize)
# 4. shuts vLLM down on exit
#
# Usage:
#   sbatch jobs/deltaai/prefix_decision.sh smoke    # 30 rows, ~5 min
#   sbatch jobs/deltaai/prefix_decision.sh full     # full 50k, ~3-6 hours
#
# Required env (export before sbatch, sbatch does NOT inherit shell env unless
# you set `export=ALL` — but `--export=ALL` is the default on slurm here):
#   OPENAI_API_KEY  for the gpt4o-mini judge step

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=prefix_decision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=12:00:00
#SBATCH --output=logs/prefix_decision_%j.out
#SBATCH --error=logs/prefix_decision_%j.err

set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: sbatch ${0##*/} {smoke|full}"; exit 1 ;;
esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs

PORT="${PORT:-8001}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"  # Qwen2.5-Math-7B native ctx is 4096
JOB_ID="${SLURM_JOB_ID:-local$$}"
VLLM_LOG="logs/vllm_${JOB_ID}.log"

echo "════════════════════════════════════════"
echo "  prefix_decision  mode=$MODE  job=$JOB_ID"
echo "  model    : $MODEL"
echo "  port     : $PORT"
echo "  max-len  : $MAX_MODEL_LEN"
echo "  vllm log : $VLLM_LOG"
echo "  node     : $(hostname)"
echo "════════════════════════════════════════"

# 1. Launch vLLM in background
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 1024 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "shutting down vLLM (pid=$VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# 2. Wait for ready (give it up to ~5 minutes for cold model load)
echo "waiting for vLLM /health on localhost:${PORT} ..."
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM died during startup; tail of $VLLM_LOG:"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done
if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "vLLM not ready after 5 min; tail of $VLLM_LOG:"
    tail -50 "$VLLM_LOG"
    exit 1
fi
echo "vLLM ready."

# 3. Run pipeline
export VLLM_URL="http://localhost:${PORT}/v1"
bash experiments-for-apr26-may1/prefix_decision/run.sh "$MODE"

echo "pipeline complete; trap will shut down vLLM."
