#!/bin/bash
# jobs/eval/run_ystar_prefix_pipeline.sh
#
# Full y* prefix ablation pipeline:
#   Step 1 — Generate y*_prefix30 and y*_noprefix via Qwen3-4B / vLLM
#   Step 2 — Win rate evaluation with GPT-4o-mini judge
#   Step 3 — Plot results and save curated dataset
#
# Usage:
#   sbatch jobs/eval/run_ystar_prefix_pipeline.sh
#   # or override paths:
#   sbatch jobs/eval/run_ystar_prefix_pipeline.sh \
#       datasets/wildfeedback/filtered_BEST.jsonl \
#       datasets/wildfeedback/ystar_prefix_best.jsonl \
#       data/winrate_results/prefix_ablation

#SBATCH --job-name=ystar_prefix_pipeline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j_ystar_prefix.out
#SBATCH --error=logs/%j_ystar_prefix.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# vLLM / NCCL stability flags
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DATASETS_TRUST_REMOTE_CODE=1

# Work from repo root
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

# ─────────────────────────────────────────────────────────────────────────────
# Configurable paths (override via positional args)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_JSONL="${1:-datasets/wildfeedback/filtered_BEST.jsonl}"
TUPLES_OUT="${2:-datasets/wildfeedback/ystar_prefix_best.jsonl}"
RESULTS_DIR="${3:-data/winrate_results/prefix_ablation}"

MODEL="Qwen/Qwen3-4B"
TP_SIZE=1
MAX_TOKENS=1024
TEMPERATURE=1.0
GPU_UTIL=0.92
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=256

mkdir -p logs "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════"
echo "  y* PREFIX ABLATION PIPELINE"
echo "  Input  : $INPUT_JSONL"
echo "  Tuples : $TUPLES_OUT"
echo "  Results: $RESULTS_DIR"
echo "════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — y* Generation (GPU)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Generating y*_prefix30 and y*_noprefix ──"
echo "   Model : $MODEL  (tp=$TP_SIZE)"
echo "   Input : $INPUT_JSONL"
echo "   Output: $TUPLES_OUT"
echo ""

python scripts/fkl/generate_ystar_prefix.py \
    --input         "$INPUT_JSONL"  \
    --output        "$TUPLES_OUT"   \
    --model         "$MODEL"        \
    --prefix_frac   0.30            \
    --max_tokens    "$MAX_TOKENS"   \
    --temperature   "$TEMPERATURE"  \
    --tp_size       "$TP_SIZE"      \
    --gpu_util      "$GPU_UTIL"     \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_seqs  "$MAX_NUM_SEQS"

echo "Step 1 complete. Tuples saved to $TUPLES_OUT"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Win Rate Evaluation (OpenAI API — no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: Win rate evaluation with GPT-4o-mini ──"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before submitting this job."
    echo "       e.g.  export OPENAI_API_KEY=sk-..."
    exit 1
fi

python scripts/eval/winrate_prefix_eval.py \
    --input        "$TUPLES_OUT"  \
    --output_dir   "$RESULTS_DIR" \
    --max_concurrent 500

echo "Step 2 complete. Results in $RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Plotting and curated dataset export
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Plotting results ──"

python scripts/eval/plot_prefix_winrate.py \
    --results    "$RESULTS_DIR/winrate_prefix_results.jsonl" \
    --tuples     "$TUPLES_OUT"                                \
    --output_dir "$RESULTS_DIR"

echo "Step 3 complete."
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline finished."
echo "  Plots         : $RESULTS_DIR/prefix_winrate_*.png"
echo "  Curated data  : $RESULTS_DIR/ystar_prefix30_wins.jsonl"
echo "  Summary       : $RESULTS_DIR/winrate_prefix_summary.txt"
echo "════════════════════════════════════════════════════════"
