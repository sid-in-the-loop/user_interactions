#!/bin/bash
# LLM-as-judge for math outputs (gpt-4o-mini). Runs over all WC math eval files.
#
#SBATCH --job-name=demo2t-mjudge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mjudge_%j.out
#SBATCH --error=logs/mjudge_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY required" >&2; exit 1
fi

REPO=/home/ssmurali/user_interactions
RES=$REPO/CFT-Eric-Zhu/eval_results

# All math output JSONLs — old (math/) AND seeded (math_s0/, math_s1/, math_s2/)
mapfile -t INPUTS < <(find "$RES/demo2teacher" -path '*/math*/*/test_*.jsonl' | grep "/WC-" | sort)

echo "Judging ${#INPUTS[@]} math output files (each ~30-500 rows)"

cd $REPO
python3 -u CFT-Eric-Zhu/evaluation_script/evaluate_math/llm_judge_math.py \
  --inputs "${INPUTS[@]}" \
  --output_dir "$RES/demo2teacher/_math_llmjudge" \
  --concurrency 200

echo "── done ──"
ls "$RES/demo2teacher/_math_llmjudge/" | head
