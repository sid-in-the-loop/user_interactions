#!/bin/bash
# Run AlpacaEval judging using local Qwen3-8B as judge (via vLLM).
# Targets checkpoint-0 and checkpoint-300-38144samples — the two with GPT-4T ground truth.
#
# Usage:
#   bash scripts/eval/run_alpaca_judge_qwen3_8b.sh
#
# Requires: conda env "opf" with vllm + alpaca_eval installed, 2 GPUs.

set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

ANNOTATOR="weighted_alpaca_eval_qwen3_8b"
ANNOTATOR_DIR="$(pwd)/alpaca_eval_data/annotators"
MODEL="Qwen/Qwen3-8B"
PORT=8000
TP=2

CHECKPOINTS=(
  "checkpoint-0"
  "checkpoint-300-38144samples"
)

# ── Start vLLM server ────────────────────────────────────────────────────────
echo "Starting vLLM server for $MODEL on port $PORT (TP=$TP)..."
VLLM_LOG=/tmp/vllm_qwen3_judge.log
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --disable-log-stats \
    &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Shutting down vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died. Check $VLLM_LOG"
        exit 1
    fi
    sleep 5
done

# ── Run judge on each checkpoint ─────────────────────────────────────────────
for CKPT in "${CHECKPOINTS[@]}"; do
    OUTDIR="dist=sample-sdpo-logpscale-filtered/${CKPT}"
    OUTJSON="${OUTDIR}/model_outputs.json"
    LB="${OUTDIR}/${ANNOTATOR}/leaderboard.csv"

    if [ ! -f "$OUTJSON" ]; then
        echo "[SKIP] $CKPT: missing model_outputs.json"
        continue
    fi

    if [ -f "$LB" ]; then
        echo "[DONE] $CKPT: already judged, skipping"
        continue
    fi

    echo ""
    echo "=== Judging $CKPT ==="
    OPENAI_API_KEY="EMPTY" \
    OPENAI_API_BASE="http://localhost:${PORT}/v1" \
    alpaca_eval \
        --model_outputs "$OUTJSON" \
        --annotators_config "$ANNOTATOR" \
        --annotators_config_kwargs "{\"annotation_type\": \"pairwise\"}" \
        --output_path "$OUTDIR" \
        --caching_path "None"

    echo "[OK] $CKPT — results in $LB"
done

echo ""
echo "=== Done ==="
echo "Compare LC win rates:"
for CKPT in "${CHECKPOINTS[@]}"; do
    LB="dist=sample-sdpo-logpscale-filtered/${CKPT}/${ANNOTATOR}/leaderboard.csv"
    if [ -f "$LB" ]; then
        LC=$(tail -1 "$LB" | awk -F',' '{print $10}')
        echo "  $CKPT: LC=$LC"
    fi
done
echo ""
echo "GPT-4 Turbo ground truth:"
echo "  checkpoint-0:                 LC=49.5%"
echo "  checkpoint-300-38144samples:  LC=59.9%"
