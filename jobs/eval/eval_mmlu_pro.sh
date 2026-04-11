#!/bin/bash
# MMLU-Pro (CoT, vLLM) — same spirit as Arena-Hard / Alpaca eval jobs.
# Usage: sbatch jobs/eval/eval_mmlu_pro.sh <run_name> <model_path>
#
# Optional env:
#   MMLU_PRO_SUBJECTS   default "all"; or e.g. "Math,Physics" (substring match per subject)
#   MMLU_PRO_SMOKE      if set to 1, only 20 questions per subject (quick test)
#   MMLU_PRO_TP         tensor parallel size (default 1)
#
# Example:
#   sbatch jobs/eval/eval_mmlu_pro.sh wf_best_5e6 /data/.../sft_wf_.../final

#SBATCH --job-name=mmlu_pro
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mmlu_pro_%j_%x.out
#SBATCH --error=logs/mmlu_pro_%j_%x.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_NAME="${1:?usage: sbatch jobs/eval/eval_mmlu_pro.sh <run_name> <model_path>}"
MODEL_PATH="${2:?model path required}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"

OUT="eval_results/mmlu_pro/${RUN_NAME}"
mkdir -p logs "$OUT"

SUBJECTS="${MMLU_PRO_SUBJECTS:-all}"
EXTRA=()
if [[ "${MMLU_PRO_SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--max_questions_per_subject 20)
fi
TP="${MMLU_PRO_TP:-1}"

echo "MMLU-Pro | run=$RUN_NAME | model=$MODEL_PATH | out=$OUT | subjects=$SUBJECTS | tp=$TP"

python scripts/eval/mmlu_pro_eval.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT" \
  --subjects "$SUBJECTS" \
  --tensor_parallel_size "$TP" \
  "${EXTRA[@]}"

echo "Done. See $OUT/summary.json"
