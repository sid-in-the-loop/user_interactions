#!/usr/bin/env bash
# WebInstruct prefix-decision pipeline.
#
# Prereq: vLLM server for Qwen/Qwen2.5-Math-7B is up at $VLLM_URL.
# Start it on a compute node with:
#   sbatch jobs/deltaai/serve_vllm.sh Qwen/Qwen2.5-Math-7B qmath7b 8001
# then read tmp/qmath7b_server_url.txt and export VLLM_URL=<that>/v1
#
# Usage:
#   bash experiments-for-apr26-may1/prefix_decision/run.sh smoke    # 30 rows
#   bash experiments-for-apr26-may1/prefix_decision/run.sh full     # all 50k

set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke) LIMIT=30 ;;
  full)  LIMIT=0 ;;
  *)     echo "usage: run.sh {smoke|full}"; exit 1 ;;
esac

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

INPUT="experiments/tac_winrates/data/webinstruct_unified.jsonl"
OUTDIR="experiments-for-apr26-may1/prefix_decision/data"
mkdir -p "$OUTDIR"

GENS="$OUTDIR/01_generations.jsonl"
SCSV="$OUTDIR/02_verdicts_student.csv"
OCSV="$OUTDIR/03_verdicts_gpt4o_mini.csv"
FINAL="$OUTDIR/webinstruct_prefix_decision.jsonl"
SUMMARY="$OUTDIR/summary.txt"

VLLM_URL="${VLLM_URL:-http://localhost:8001/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B}"
GEN_WORKERS="${GEN_WORKERS:-256}"
JUDGE_WORKERS="${JUDGE_WORKERS:-1024}"
OPENAI_WORKERS="${OPENAI_WORKERS:-64}"

echo "── mode=$MODE  limit=$LIMIT ──"
echo "vLLM: $VLLM_URL  model: $MODEL"
echo "input: $INPUT"
echo "outdir: $OUTDIR"

LIMIT_ARG=()
if [[ "$LIMIT" -gt 0 ]]; then LIMIT_ARG=(--limit "$LIMIT"); fi

echo "── step 1/4: generate y_star_full / y_star_prefix / y_base ──"
python experiments-for-apr26-may1/prefix_decision/01_generate.py \
  --input "$INPUT" --output "$GENS" \
  --vllm_url "$VLLM_URL" --model "$MODEL" \
  --workers "$GEN_WORKERS" "${LIMIT_ARG[@]}"

echo "── step 2/4: student judge (logprob comparison) ──"
python experiments-for-apr26-may1/prefix_decision/02_judge_local.py \
  --gens "$GENS" --output "$SCSV" \
  --vllm_url "$VLLM_URL" --model "$MODEL" \
  --workers "$JUDGE_WORKERS"

echo "── step 3/4: gpt4o_mini judge (free-form) ──"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: OPENAI_API_KEY not set; skipping step 3."
else
  python experiments-for-apr26-may1/prefix_decision/03_judge_openai.py \
    --gens "$GENS" --output "$OCSV" \
    --workers "$OPENAI_WORKERS"
fi

echo "── step 4/4: summarize ──"
python experiments-for-apr26-may1/prefix_decision/04_summarize.py \
  --gens "$GENS" \
  --student_csv "$SCSV" \
  --openai_csv "$OCSV" \
  --output "$FINAL" \
  --summary "$SUMMARY"
