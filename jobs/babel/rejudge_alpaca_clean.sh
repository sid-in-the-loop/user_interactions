#!/bin/bash
# Re-judge existing alpaca outputs against gpt-4-turbo with <think> blocks stripped.
# Outputs themselves stay untouched on disk; the judge does the strip in-memory.
#
#SBATCH --job-name=demo2t-rejudge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rejudge_%j.out
#SBATCH --error=logs/rejudge_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then echo "ERROR: OPENAI_API_KEY"; exit 1; fi

REPO=/home/ssmurali/user_interactions
RES=$REPO/CFT-Eric-Zhu/eval_results
cd $REPO

mapfile -t INPUTS < <(find "$RES/demo2teacher" -path '*/alpaca/outputs.jsonl' | grep "/WC-" | sort)
echo "Re-judging ${#INPUTS[@]} alpaca outputs (stripping <think> blocks first)..."

python3 -u $REPO/CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/judge_alpaca_async.py \
  --inputs "${INPUTS[@]}" \
  --output_dir "$RES/demo2teacher/_alpaca_judge_clean" \
  --reference_config alpaca_eval_gpt4_baseline \
  --concurrency 200 \
  --overwrite

echo
echo "── done ──"
ls "$RES/demo2teacher/_alpaca_judge_clean/" | head
echo
echo "── new winrates (sample) ──"
python3 -c "
import json
j = json.load(open('$RES/demo2teacher/_alpaca_judge_clean/summary.json'))
items = sorted(j.items(), key=lambda x: x[0])[:20]
for k, v in items: print(f'  {k:<30s}  winrate={v.get(\"winrate\",0):.3f}')
"
