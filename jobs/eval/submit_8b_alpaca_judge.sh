#!/bin/bash
# Submit alpaca_eval judging (annotation only) for 8B models.
# Outputs already exist — this just runs the GPT-4-turbo judge.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

MODELS=(
  base_qwen3_8b
  8b_think_best_final
  8b_think_best_ext_final
  8b_think_full_final
  8b_nothink_best_final
  8b_nothink_full_final
)

for MODEL in "${MODELS[@]}"; do
  sbatch --job-name="alpaca_${MODEL}" \
    --partition=general --nodes=1 --ntasks-per-node=1 \
    --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=1:00:00 \
    --output="logs/alpaca_eval_%j_${MODEL}.out" \
    --error="logs/alpaca_eval_%j_${MODEL}.err" \
    --wrap="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export OPENAI_API_KEY=${OPENAI_API_KEY}
cd $(pwd)
alpaca_eval \
  --model_outputs alpaca_eval_data/results/${MODEL}/model_outputs.json \
  --annotators_config weighted_alpaca_eval_gpt4_turbo \
  --output_path alpaca_eval_data/results/${MODEL}
echo DONE ${MODEL}
"
done

echo "Submitted ${#MODELS[@]} judging jobs."
echo "Track with: squeue -u \$USER"
echo "Results in: alpaca_eval_data/results/<model>/leaderboard.csv"
