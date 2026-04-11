#!/bin/bash
# Submit Arena-Hard + AlpacaEval + MMLU-Pro for every row in eval_results/benchmark_manifest.csv.
#
# Usage (from repo root):
#   bash jobs/eval/submit_benchmark_manifest.sh              # submit all
#   bash jobs/eval/submit_benchmark_manifest.sh --dry-run
#   DRY_RUN=1 bash jobs/eval/submit_benchmark_manifest.sh
#
# Requires per row: model_id, model_path. Uses model_id as the job name tag everywhere
# (arena model_answer/<model_id>.jsonl, alpaca results/<model_id>/, mmlu_pro/<model_id>/).
#
# Skips local paths without config.json (FSDP sharded epoch dirs are not vLLM-loadable).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
MANIFEST="${MANIFEST:-eval_results/benchmark_manifest.csv}"
DRY="${DRY_RUN:-}"
[[ "${1:-}" == "--dry-run" ]] && DRY=1

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST"
  exit 1
fi

submit_three() {
  local mid="$1" path="$2"
  if [[ "$path" != /* && "$path" != *:* ]]; then
    path="$REPO/$path"
  fi
  if [[ -d "$path" ]] && [[ ! -f "$path/config.json" ]]; then
    echo "SKIP $mid — no config.json at $path (merge FSDP shard to HF or use final/)"
    return 0
  fi
  echo "QUEUE $mid | $path"
  if [[ -n "$DRY" ]]; then
    echo "  sbatch jobs/eval/eval_arena_hard.sh \"$mid\" \"$path\""
    echo "  sbatch jobs/eval/eval_alpaca_eval.sh \"$mid\" \"$path\""
    echo "  sbatch jobs/eval/eval_mmlu_pro.sh \"$mid\" \"$path\""
    return 0
  fi
  sbatch jobs/eval/eval_arena_hard.sh "$mid" "$path"
  sbatch jobs/eval/eval_alpaca_eval.sh "$mid" "$path"
  sbatch jobs/eval/eval_mmlu_pro.sh "$mid" "$path"
}

tail -n +2 "$MANIFEST" | while IFS= read -r line; do
  mid="${line%%,*}"
  rest="${line#*,}"
  path="${rest%%,*}"
  mid="${mid//$'\r'/}"
  path="${path//$'\r'/}"
  [[ -z "$mid" || "$mid" == model_id ]] && continue
  submit_three "$mid" "$path"
done

echo "Done. Set OPENAI_API_KEY on the login node or in job env for Arena+Alpaca judging."
echo "After jobs finish: python scripts/eval/fill_benchmark_manifest.py"
