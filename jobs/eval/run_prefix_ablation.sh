#!/bin/bash
# Generic prefix ablation job: generation + winrate + plots for one model.
#
# Positional args:
#   $1  MODEL_ID   — HuggingFace model ID   (required)
#   $2  MODEL_TAG  — short name for paths   (required, e.g. qwen3_4b)
#
# Example:
#   sbatch jobs/eval/run_prefix_ablation.sh Qwen/Qwen3-4B      qwen3_4b
#   sbatch jobs/eval/run_prefix_ablation.sh Qwen/Qwen3-8B      qwen3_8b
#   sbatch jobs/eval/run_prefix_ablation.sh allenai/OLMo-3-7B-Instruct-SFT olmo_7b
#
# Override SLURM resources at submit time:
#   sbatch --gres=gpu:L40S:1 --time=08:00:00 jobs/eval/run_prefix_ablation.sh ...

#SBATCH --job-name=prefix_ablation
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_prefix_ablation_%x.out
#SBATCH --error=logs/%j_prefix_ablation_%x.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID="${1:-}"
MODEL_TAG="${2:-}"
if [[ -z "$MODEL_ID" || -z "$MODEL_TAG" ]]; then
    echo "Usage: sbatch $0 <model_id> <model_tag>"
    echo "  e.g. sbatch $0 Qwen/Qwen3-4B qwen3_4b"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DATASETS_TRUST_REMOTE_CODE=1

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
INPUT="datasets/wildfeedback/filtered_BEST.jsonl"
TUPLES_OUT="datasets/wildfeedback/ystar_prefix_${MODEL_TAG}.jsonl"
RESULTS_DIR="data/winrate_results/prefix_ablation_${MODEL_TAG}"

mkdir -p logs "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════"
echo "  PREFIX ABLATION — ${MODEL_TAG}"
echo "  Model  : ${MODEL_ID}"
echo "  Output : ${TUPLES_OUT}"
echo "  Results: ${RESULTS_DIR}"
echo "════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generation (GPU)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Generating y* (prefix30 / noprefix / full) ──"

python scripts/fkl/generate_ystar_prefix.py \
    --input         "$INPUT"       \
    --output        "$TUPLES_OUT"  \
    --model         "$MODEL_ID"    \
    --prefix_frac   0.30           \
    --max_tokens    1024           \
    --temperature   1.0            \
    --tp_size       1              \
    --gpu_util      0.92           \
    --max_model_len 8192           \
    --max_num_seqs  256

echo "Step 1 complete → ${TUPLES_OUT}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Win rate evaluation (OpenAI API)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: Win rate evaluation ──"

python scripts/eval/winrate_prefix_eval.py \
    --input        "$TUPLES_OUT"  \
    --output_dir   "$RESULTS_DIR" \
    --max_concurrent 500

echo "Step 2 complete → ${RESULTS_DIR}/winrate_prefix_results.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Plots + curated dataset
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Plotting ──"

python scripts/eval/plot_prefix_winrate.py \
    --results    "${RESULTS_DIR}/winrate_prefix_results.jsonl" \
    --tuples     "$TUPLES_OUT"                                 \
    --output_dir "$RESULTS_DIR"                                \
    --model_name "$MODEL_TAG"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done — ${MODEL_TAG}"
echo "  Plots  : ${RESULTS_DIR}/prefix_winrate_*.png"
echo "  Data   : ${RESULTS_DIR}/ystar_prefix30_wins.jsonl"
echo "  Summary: ${RESULTS_DIR}/winrate_prefix_summary.txt"
echo "════════════════════════════════════════════════════════"
