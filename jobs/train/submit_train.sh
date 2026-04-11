#!/bin/bash
# Unified LoRA training submission — replaces all sbatch_{obj}_{prefix}_{data}.sh
#
# Usage:
#   sbatch jobs/train/submit_train.sh <objective> <prefix> <dataset> [model] [extra_args...]
#
# Arguments:
#   objective   sft | fkl | jsd
#   prefix      p30 | full          (which y_star field to train on)
#   dataset     wfbest | wffull     (which WildFeedback split)
#   model       (optional) HF model name, default: Qwen/Qwen3-8B
#
# Examples:
#   sbatch jobs/train/submit_train.sh fkl full wfbest
#   sbatch jobs/train/submit_train.sh jsd p30 wffull Qwen/Qwen3-4B
#   sbatch jobs/train/submit_train.sh sft full wfbest Qwen/Qwen3-8B --epochs 5 --lr 1e-5
#
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
OBJ="${1:?Usage: submit_train.sh <sft|fkl|jsd> <p30|full> <wfbest|wffull> [model]}"
PREFIX="${2:?Usage: submit_train.sh <sft|fkl|jsd> <p30|full> <wfbest|wffull> [model]}"
DATA="${3:?Usage: submit_train.sh <sft|fkl|jsd> <p30|full> <wfbest|wffull> [model]}"
MODEL="${4:-Qwen/Qwen3-8B}"
shift 4 2>/dev/null || shift $#
EXTRA_ARGS="$@"

# Validate
case "$OBJ" in sft|fkl|jsd) ;; *) echo "Invalid objective: $OBJ (sft|fkl|jsd)"; exit 1 ;; esac
case "$PREFIX" in p30|full) ;; *) echo "Invalid prefix: $PREFIX (p30|full)"; exit 1 ;; esac
case "$DATA" in wfbest|wffull) ;; *) echo "Invalid dataset: $DATA (wfbest|wffull)"; exit 1 ;; esac

# ── Derived variables ─────────────────────────────────────────────────────────
RUN_NAME="$(basename $MODEL | tr '/' '_')_${OBJ}_${PREFIX}_${DATA}"

# y_star field: p30 → y_star_prefix30, full → y_star_full
if [[ "$PREFIX" == "p30" ]]; then
    YSTAR_FIELD="y_star_prefix30"
else
    YSTAR_FIELD="y_star_full"
fi

# Input file: wfbest = BEST-tier, wffull = full WF split
if [[ "$DATA" == "wfbest" ]]; then
    INPUT="datasets/wildfeedback/ystar_prefix_qwen3_8b.jsonl"
else
    INPUT="datasets/wildfeedback/ystar_full_qwen3_8b_full_wf.jsonl"
fi

OUTPUT_DIR="checkpoints/${RUN_NAME}"

# --pure_kl for fkl/jsd (pure objective, no SFT mixing)
PURE_KL=""
[[ "$OBJ" != "sft" ]] && PURE_KL="--pure_kl"

# ── SLURM job name / logs (set dynamically if not already set by #SBATCH) ─────
export SLURM_JOB_NAME="${RUN_NAME}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
cd /home/ssmurali/user_interactions

echo "════════════════════════════════════════"
echo " Objective  : $OBJ"
echo " Prefix     : $PREFIX  →  $YSTAR_FIELD"
echo " Dataset    : $DATA    →  $INPUT"
echo " Model      : $MODEL"
echo " Output dir : $OUTPUT_DIR"
echo " Pure KL    : ${PURE_KL:-no}"
echo " Extra args : ${EXTRA_ARGS:-none}"
echo "════════════════════════════════════════"

python scripts/fkl/train_lora.py \
    --input          "$INPUT"        \
    --y_star_field   "$YSTAR_FIELD"  \
    --objective      "$OBJ"          \
    $PURE_KL                         \
    --model          "$MODEL"        \
    --output_dir     "$OUTPUT_DIR"   \
    --lora_r 64 --lora_alpha 128     \
    --batch_size 2 --grad_accum 64   \
    --epochs 3 --lr 2e-6             \
    --max_length 2048                \
    --teacher_max_length 4096        \
    --num_ckpts 8                    \
    --wandb_project prefix-ablation-lora \
    --run_name "$RUN_NAME"           \
    $EXTRA_ARGS
