#!/bin/bash
# SFT on all 4 Qwen3-8B WF y* datasets (4 GPU, lr=5e-6, mask_tau=0).
# Usage: bash jobs/train/submit_sft_8b.sh
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
YSTAR="$REPO/datasets/wildfeedback/qwen3_8b"
SCRIPT="jobs/train/sbatch_sft_one.sh"
PORT=29600

for tag in thinking_full thinking_best nonthinking_full nonthinking_best; do
  INPUT="$YSTAR/ystar_${tag}.jsonl"
  [[ -f "$INPUT" ]] || { echo "SKIP (missing): $INPUT"; continue; }
  export INPUT
  export RUN_NAME="qwen3_8b/sft_wf_${tag}_lr5e6"
  export LR=5e-6 MODEL=Qwen/Qwen3-8B NGPUS=4 BATCH_SIZE=4 GRAD_ACCUM=16
  export MASTER_PORT=$PORT MASK_TAU=0 SAVE_STEPS=999999
  sbatch --gres=gpu:4 --job-name="sft_8b_wf_${tag}" "$SCRIPT"
  echo "  submitted: sft_8b_wf_${tag} (port $PORT)"
  ((PORT++)) || true
done
echo "Done. Checkpoints → /data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/"
