#!/bin/bash
# Generic training job for any method in train_methods.py.
#
# Usage:
#   sbatch jobs/deltaai/train_method.sh <OBJECTIVE> [EXTRA_ARGS...]
#
# Examples:
#   sbatch jobs/deltaai/train_method.sh jsd --pure_kl
#   sbatch jobs/deltaai/train_method.sh dpo
#   sbatch jobs/deltaai/train_method.sh jsd_is1 --pure_kl --logprobs_file datasets/wildchat/init_logprobs_qwen3_8b.pt
#   sbatch jobs/deltaai/train_method.sh zg_jsd --pure_kl --logprobs_file datasets/wildchat/init_logprobs_qwen3_8b.pt
#   sbatch jobs/deltaai/train_method.sh rkl --rollout_max_tokens 1024
#   sbatch jobs/deltaai/train_method.sh distillm2 --rollout_max_tokens 1024

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
# job-name set via sbatch --job-name=<objective> at submission
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

OBJECTIVE="${1:?Usage: sbatch train_method.sh <OBJECTIVE> [EXTRA_ARGS...]}"
shift
EXTRA_ARGS="$@"

# Override job name so logs are named like: sft_2132500.out, rkl_2132501.out
scontrol update JobId=$SLURM_JOB_ID JobName="${OBJECTIVE}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"
mkdir -p logs checkpoints

INPUT="datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl"
OUTPUT_DIR="/projects/bgtw/ssredharan/checkpoints/${OBJECTIVE}_p30"
RUN_NAME="qwen3_8b_${OBJECTIVE}_p30"

echo "════════════════════════════════════════"
echo "  Method   : ${OBJECTIVE}"
echo "  Input    : ${INPUT}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Extra    : ${EXTRA_ARGS}"
echo "════════════════════════════════════════"

# Memory-heavy methods need batch_size=1
case "$OBJECTIVE" in
    dpo|rkl|rlad|distillm2|zg_jsd)
        BS=1; GA=64 ;;
    *)
        BS=2; GA=32 ;;
esac

python scripts/fkl/train_methods.py \
    --input "$INPUT" \
    --y_star_field y_star_prefix30 \
    --objective "$OBJECTIVE" \
    --model Qwen/Qwen3-8B \
    --output_dir "$OUTPUT_DIR" \
    --save_every 10 \
    --batch_size $BS \
    --grad_accum $GA \
    --epochs 2 \
    --lr 2e-6 \
    --run_name "$RUN_NAME" \
    --wandb_project distillation-methods \
    $EXTRA_ARGS

echo "Done. Checkpoints in ${OUTPUT_DIR}/"
