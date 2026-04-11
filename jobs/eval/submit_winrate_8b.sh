#!/bin/bash
# Winrate eval (y vs y*) for all 4 Qwen3-8B WF y* files.
# Usage: bash jobs/eval/submit_winrate_8b.sh
# Requires: OPENAI_API_KEY set in env.
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
YSTAR="$REPO/datasets/wildfeedback/qwen3_8b"
OUT="$REPO/data/winrate_results/qwen3_8b"
mkdir -p "$OUT"

for f in thinking_full thinking_best nonthinking_full nonthinking_best; do
  INPUT="$YSTAR/ystar_${f}.jsonl"
  [[ -f "$INPUT" ]] || { echo "SKIP (missing): $INPUT"; continue; }
  echo "=== winrate: $f ==="
  python scripts/eval/winrate_eval.py \
    --input "$INPUT" \
    --output-dir "$OUT/wf_${f}" \
    --subsample 500 --seed 42 --max-concurrent 500
done
echo "Done. Summaries in $OUT/wf_*/winrate_summary.txt"
