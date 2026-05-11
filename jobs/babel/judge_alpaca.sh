#!/bin/bash
# Judge all alpaca generations (ckpts + baseline) against gpt-4-1106-preview reference
# (AlpacaEval 2 setup). Async, no GPU.
#
# Submit: sbatch jobs/babel/judge_alpaca.sh
#
#SBATCH --job-name=demo2t-judge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set in env at submit time" >&2
  exit 1
fi

REPO=/home/ssmurali/user_interactions
RESULTS=$REPO/CFT-Eric-Zhu/eval_results

cd $REPO

# Glob all alpaca outputs (WC ckpts under demo2teacher/ + baseline)
INPUTS=()
while IFS= read -r f; do
  [[ -f "$f" ]] && INPUTS+=("$f")
done < <(find "$RESULTS/demo2teacher" -path '*/alpaca/outputs.jsonl' 2>/dev/null | grep "/WC-")
[[ -f "$RESULTS/baseline_qwen3_4b/alpaca/outputs.jsonl" ]] && INPUTS+=("$RESULTS/baseline_qwen3_4b/alpaca/outputs.jsonl")

echo "Judging ${#INPUTS[@]} alpaca outputs against gpt-4-1106-preview reference"
echo "Output dir: $RESULTS/demo2teacher/_alpaca_judge_v2"

python3 CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/judge_alpaca_async.py \
  --inputs "${INPUTS[@]}" \
  --output_dir "$RESULTS/demo2teacher/_alpaca_judge_v2" \
  --reference_config alpaca_eval_gpt4_baseline \
  --concurrency 200

echo "── done ──"
ls "$RESULTS/demo2teacher/_alpaca_judge_v2/" | head
