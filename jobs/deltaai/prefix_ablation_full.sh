#!/bin/bash
# Run full prefix ablation pipeline:
#   1. Generate y_base + y* at 0-100% prefix fractions
#   2. Run winrate eval with GPT-4o-mini
#
# Waits for the 8B server to be ready before starting.
#
# Usage:
#   sbatch jobs/deltaai/prefix_ablation_full.sh

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=prefix_ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=122g
#SBATCH --time=04:00:00
#SBATCH --output=logs/prefix_ablation_%j.out
#SBATCH --error=logs/prefix_ablation_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"
mkdir -p logs

# Wait for 8B server
URL_FILE="tmp/qwen3_8b_server_url.txt"
echo "Waiting for 8B server URL file..."
while [ ! -f "$URL_FILE" ]; do
    sleep 10
done
SERVER_URL=$(cat "$URL_FILE")
until curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
    echo "  Server not healthy yet..."
    sleep 10
done
echo "8B server ready: ${SERVER_URL}"

# Step 1: Generate y_base + y* at all prefix fractions
echo ""
echo "════════════════════════════════════════"
echo "  Step 1: Prefix ablation generation"
echo "════════════════════════════════════════"
export SERVER_URL INPUT OUTPUT_DIR SUBSAMPLE
INPUT=datasets/wildfeedback/filtered_BEST.jsonl
OUTPUT_DIR=datasets/wildfeedback/prefix_ablation_orig
SUBSAMPLE=1000
bash scripts/eval/run_prefix_ablation.sh \
    "$SERVER_URL" \
    "$INPUT" \
    "$OUTPUT_DIR" \
    "$SUBSAMPLE"

# Step 2: Winrate eval with GPT-4o-mini
echo ""
echo "════════════════════════════════════════"
echo "  Step 2: Winrate eval (GPT-4o-mini)"
echo "════════════════════════════════════════"
python scripts/eval/winrate_eval.py \
    --input datasets/wildfeedback/prefix_ablation_orig/prefix_ablation_merged.jsonl \
    --judge gpt4omini \
    --output-dir data/winrate_results/prefix_ablation_orig_gpt4omini \
    --comparisons \
        p0:y_star_0:ybase:y_base \
        p10:y_star_10:ybase:y_base \
        p20:y_star_20:ybase:y_base \
        p30:y_star_30:ybase:y_base \
        p40:y_star_40:ybase:y_base \
        p50:y_star_50:ybase:y_base \
        p60:y_star_60:ybase:y_base \
        p70:y_star_70:ybase:y_base \
        p80:y_star_80:ybase:y_base \
        p90:y_star_90:ybase:y_base \
        p100:y_star_100:ybase:y_base \
    --subsample 500

echo ""
echo "Done! Results in data/winrate_results/prefix_ablation_orig_gpt4omini/"
