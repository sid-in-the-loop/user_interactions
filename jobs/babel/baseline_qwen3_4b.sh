#!/bin/bash
# Baseline Qwen3-4B (no LoRA) on amc23+aime24 + alpaca gen.
# Reference numbers for forgetting plot + alpaca winrate.
#
# Submit: sbatch jobs/babel/baseline_qwen3_4b.sh
#
#SBATCH --job-name=demo2t-baseline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/home/ssmurali/user_interactions
OUT=$REPO/CFT-Eric-Zhu/eval_results/baseline_qwen3_4b
MODEL=Qwen/Qwen3-4B
mkdir -p logs "$OUT/math" "$OUT/alpaca"

echo "════════════════════════════════════════"
echo "  Qwen3-4B baseline"
echo "  job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "  out: $OUT"
echo "════════════════════════════════════════"

echo
echo "── 1/2  math: amc23,aime24 ──"
pushd "$REPO/CFT-Eric-Zhu/evaluation_script/evaluate_math" > /dev/null
python3 -u math_eval.py \
  --model_name_or_path "$MODEL" \
  --data_name "amc23,aime24" --data_dir ./data \
  --output_dir "$OUT/math" \
  --summary_path "$OUT/math/summary.txt" \
  --split test --prompt_type qwen25-math-cot --apply_chat_template \
  --num_test_sample -1 --max_tokens_per_call 2048 \
  --seed 0 --temperature 0 --n_sampling 1 --top_p 1 --start 0 --end -1 \
  --use_vllm --save_outputs
popd > /dev/null

echo
echo "── 2/2  alpaca gen ──"
python3 -u "$REPO/CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/alpaca_eval.py" \
  --model_name_or_path "$MODEL" \
  --output_path "$OUT/alpaca/outputs.jsonl" \
  --generator_name qwen3-4b-base \
  --max_tokens 2048 --temperature 0.7 --top_p 0.9 --apply_chat_template

echo
echo "── done ──"
cat "$OUT/math/summary.txt" 2>&1
wc -l "$OUT/alpaca/outputs.jsonl" 2>&1
