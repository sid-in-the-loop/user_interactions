#!/bin/bash
# T-4: Contrastive FKL on y*_C tokens.
#   p_T   = frozen_OLMo(· | x_o, y*_{<n})   teacher sees x + "Note: o"
#   p_ref = frozen_OLMo(· | x,   y*_{<n})   same frozen model, x-only context
#   loss  = Σ_v (p_T − p_ref) · log(p_T / p_S)
#
# One frozen model, two sequential forward passes per step (different inputs).
# Memory same as T2/T5: ~14GB frozen + ~3.5GB student shard on 48GB L40S.
#
# Submit:
#   sbatch jobs/train/sbatch_olmo_T4.sh

#SBATCH --job-name=olmo_T4
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/olmo_T4_%j.out
#SBATCH --error=logs/olmo_T4_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="allenai/OLMo-3-7B-Instruct-SFT"
YSTAR="/data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3/ystar_olmo_xo_C.jsonl"
OUTPUT_DIR="/data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3/T4"
mkdir -p logs "$OUTPUT_DIR"

echo "Job ID : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"
echo "Mode   : T4 (contrastive FKL, two frozen teachers)"
echo "Data   : $YSTAR  ($(wc -l < "$YSTAR") samples)"
echo "Output : $OUTPUT_DIR"
echo "──────────────────────────────────────────"

# Effective batch: 32 per GPU × 2 GPUs × 1 grad_accum = 64
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/fkl/train_olmo_fkl.py \
        --input          "$YSTAR" \
        --output_dir     "$OUTPUT_DIR" \
        --model          "$MODEL" \
        --mode           T4 \
        --batch_size     8 \
        --grad_accum     4 \
        --epochs         2 \
        --lr             5e-6 \
        --warmup_ratio   0.05 \
        --max_prompt_len 2048 \
        --max_compl_len  2048 \
        --eval_steps     40 \
        --run_name       olmo_fkl_T4 \
        --wandb_project  olmo-fkl

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Final checkpoint: $OUTPUT_DIR/final"
