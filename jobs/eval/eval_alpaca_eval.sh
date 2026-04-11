#!/bin/bash
# jobs/eval_alpaca_eval.sh
# Usage: sbatch jobs/eval_alpaca_eval.sh <model_name> <model_path>

#SBATCH --job-name=alpaca_eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/alpaca_eval_%j_%x.out
#SBATCH --error=logs/alpaca_eval_%j_%x.err

# Exit on any error
set -e

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Stability Flags for NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DATASETS_TRUST_REMOTE_CODE=1
export JOBLIB_TEMP_FOLDER=/tmp/$USER_joblib
mkdir -p $JOBLIB_TEMP_FOLDER

# Arguments
MODEL_NAME=$1
MODEL_PATH=$2

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: sbatch jobs/eval_alpaca_eval.sh <model_name> <model_path>"
    exit 1
fi

RESULTS_DIR="alpaca_eval_data/results/$MODEL_NAME"
mkdir -p "$RESULTS_DIR"

echo "════════════════════════════════════════"
echo "ALPACA EVAL 2.0: $MODEL_NAME"
echo "PATH: $MODEL_PATH"
echo "════════════════════════════════════════"

# 1. Generate Answers
python scripts/eval/alpaca_eval_gen.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --input_file arena-hard-auto/arena-hard-auto/alpaca_eval_data/alpaca_eval_prompts.jsonl \
    --output_file "$RESULTS_DIR/model_outputs.json" \
    --tensor_parallel_size 1

# 2. Run Evaluation (requires OPENAI_API_KEY)
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Starting AlpacaEval Annotations..."
    alpaca_eval --model_outputs "$RESULTS_DIR/model_outputs.json" \
                --annotators_config "gpt-4o-mini" \
                --output_path "$RESULTS_DIR"
else
    echo "OPENAI_API_KEY not set. Skipping annotation step."
fi

echo "Done."
