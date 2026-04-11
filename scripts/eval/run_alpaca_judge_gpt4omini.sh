#!/bin/bash
# Run AlpacaEval judging for all models using weighted_alpaca_eval_gpt-4o-mini-2024-07-18
# Usage: OPENAI_API_KEY="sk-..." bash scripts/eval/run_alpaca_judge_gpt4omini.sh

set -euo pipefail

cd /home/ssmurali/user_interactions

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set. Run with:"
  echo "  OPENAI_API_KEY=\"sk-...\" bash scripts/eval/run_alpaca_judge_gpt4omini.sh"
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export OPENAI_MAX_CONCURRENCY="${OPENAI_MAX_CONCURRENCY:-3}"

ANNOTATOR="weighted_alpaca_eval_gpt-4o-mini-2024-07-18"
MANIFEST="eval_results/benchmark_manifest.csv"

echo "Using annotator: $ANNOTATOR"
echo "Concurrency: $OPENAI_MAX_CONCURRENCY"
echo ""

TOTAL=0
DONE=0
SKIP=0
FAIL=0

tail -n +2 "$MANIFEST" | cut -d, -f1 | while read -r MID; do
  TOTAL=$((TOTAL + 1))
  OUTDIR="alpaca_eval_data/results/${MID}"
  OUTJSON="${OUTDIR}/model_outputs.json"
  LB="${OUTDIR}/${ANNOTATOR}/leaderboard.csv"

  if [ ! -f "$OUTJSON" ]; then
    echo "[SKIP] $MID: missing model_outputs.json"
    SKIP=$((SKIP + 1))
    continue
  fi

  if [ -f "$LB" ]; then
    echo "[DONE] $MID: already has leaderboard.csv"
    DONE=$((DONE + 1))
    continue
  fi

  echo "[RUN] $MID"
  if alpaca_eval \
    --model_outputs "$OUTJSON" \
    --annotators_config "$ANNOTATOR" \
    --output_path "$OUTDIR"; then
    echo "[OK] $MID"
    DONE=$((DONE + 1))
  else
    echo "[FAIL] $MID"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=== Summary ==="
echo "Total models in manifest: checked above"
echo "If you see [FAIL], check the error output and re-run."
