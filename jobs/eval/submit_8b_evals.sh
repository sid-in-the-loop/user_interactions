#!/bin/bash
# Submit Arena-Hard + AlpacaEval for 8B final checkpoints + base.
# Usage:
#   bash jobs/eval/submit_8b_evals.sh
#   bash jobs/eval/submit_8b_evals.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

MANIFEST="eval_results/benchmark_manifest.csv"

# 6 models: base_qwen3_8b + 5 finals
MODELS=(
  "base_qwen3_8b,Qwen/Qwen3-8B"
  "8b_think_best_final,/data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/sft_wf_thinking_best_lr5e6/final"
  "8b_think_best_ext_final,/data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/sft_wf_thinking_best_lr5e6_ext/final"
  "8b_think_full_final,/data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/sft_wf_thinking_full_lr5e6/final"
  "8b_nothink_best_final,/data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/sft_wf_nonthinking_best_lr5e6/final"
  "8b_nothink_full_final,/data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/sft_wf_nonthinking_full_lr5e6/final"
)

for entry in "${MODELS[@]}"; do
  mid="${entry%%,*}"
  path="${entry#*,}"
  echo "QUEUE $mid | $path"
  if [[ -n "$DRY" ]]; then
    echo "  sbatch jobs/eval/eval_arena_hard.sh $mid $path"
    echo "  sbatch jobs/eval/eval_alpaca_eval.sh $mid $path"
    continue
  fi
  sbatch --job-name="arena_${mid}" jobs/eval/eval_arena_hard.sh "$mid" "$path"
  sbatch --job-name="alpaca_${mid}" jobs/eval/eval_alpaca_eval.sh "$mid" "$path"
done

echo ""
echo "Submitted 12 jobs (6 models × 2 evals)."
echo "After jobs finish: python scripts/eval/fill_benchmark_manifest.py --write"
