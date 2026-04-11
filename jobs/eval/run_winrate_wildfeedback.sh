#!/bin/bash
# Run winrate eval (y* vs y) on wildfeedback y* files — 500 samples, seed 42.
# Usage: from repo root, with OPENAI_API_KEY set:
#   bash jobs/eval/run_winrate_wildfeedback.sh
# Or run the python commands below manually for the 3 that exist.
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
WF="$REPO/datasets/wildfeedback"
OUT="$REPO/data/winrate_results"
mkdir -p "$OUT"

run_one() {
  local name=$1
  local input=$2
  [[ -f "$input" ]] || { echo "Skip (missing): $input"; return 0; }
  echo "=== $name ==="
  python scripts/eval/winrate_eval.py \
    --input "$input" \
    --output-dir "$OUT/wf_$name" \
    --subsample 500 \
    --seed 42 \
    --max-concurrent 500
  echo ""
}

run_one "nonthinking_best"   "$WF/ystar_nonthinking_best.jsonl"
run_one "nonthinking_full"   "$WF/ystar_nonthinking_full.jsonl"
run_one "thinking_best"      "$WF/ystar_thinking_best.jsonl"
run_one "thinking_full"      "$WF/ystar_thinking_full.jsonl"

echo "Done. Summaries in $OUT/wf_*/winrate_summary.txt"
