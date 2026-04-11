#!/bin/bash
# Submit y_base generation for WildFeedback — 4B and 8B, nonthinking mode.
# y_base = Qwen(x) with no feedback, used for apples-to-apples winrate analysis.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

WF=datasets/wildfeedback

# ── 8B ──────────────────────────────────────────────────────────────────────
sbatch --job-name="ybase_8b_full" \
  --partition=general --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --gres=gpu:1 --mem=64G --time=4:00:00 \
  --output=logs/gen_ybase_%j_8b_full.out \
  --error=logs/gen_ybase_%j_8b_full.err \
  --wrap="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/fkl/generate_ybase.py \
  --input $WF/tuples.jsonl \
  --output_dir $WF/qwen3_8b \
  --output_suffix '' \
  --ids-file eval_results/winrate_sample_500_seed42.json \
  --mode A \
  --model Qwen/Qwen3-8B \
  --max_tokens 1024 --tp_size 1 --gpu_util 0.95 --max_num_seqs 512
"

sbatch --job-name="ybase_8b_best" \
  --partition=general --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --gres=gpu:1 --mem=64G --time=2:00:00 \
  --output=logs/gen_ybase_%j_8b_best.out \
  --error=logs/gen_ybase_%j_8b_best.err \
  --wrap="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/fkl/generate_ybase.py \
  --input $WF/filtered_BEST.jsonl \
  --output_dir $WF/qwen3_8b \
  --output_suffix '_best' \
  --ids-file eval_results/winrate_sample_500_seed42.json \
  --mode A \
  --model Qwen/Qwen3-8B \
  --max_tokens 1024 --tp_size 1 --gpu_util 0.95 --max_num_seqs 512
"

# ── 4B ──────────────────────────────────────────────────────────────────────
sbatch --job-name="ybase_4b_full" \
  --partition=general --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --gres=gpu:1 --mem=48G --time=3:00:00 \
  --output=logs/gen_ybase_%j_4b_full.out \
  --error=logs/gen_ybase_%j_4b_full.err \
  --wrap="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/fkl/generate_ybase.py \
  --input $WF/tuples.jsonl \
  --output_dir $WF/qwen3_4b \
  --output_suffix '' \
  --ids-file eval_results/winrate_sample_500_seed42.json \
  --mode A \
  --model Qwen/Qwen3-4B \
  --max_tokens 1024 --tp_size 1 --gpu_util 0.95 --max_num_seqs 512
"

sbatch --job-name="ybase_4b_best" \
  --partition=general --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --gres=gpu:1 --mem=48G --time=1:30:00 \
  --output=logs/gen_ybase_%j_4b_best.out \
  --error=logs/gen_ybase_%j_4b_best.err \
  --wrap="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/fkl/generate_ybase.py \
  --input $WF/filtered_BEST.jsonl \
  --output_dir $WF/qwen3_4b \
  --output_suffix '_best' \
  --ids-file eval_results/winrate_sample_500_seed42.json \
  --mode A \
  --model Qwen/Qwen3-4B \
  --max_tokens 1024 --tp_size 1 --gpu_util 0.95 --max_num_seqs 512
"

echo "Submitted 4 y_base generation jobs (8B full/best, 4B full/best)."
echo "Outputs: datasets/wildfeedback/qwen3_{8b,4b}/ybase_nonthinking{,_best}.jsonl"
