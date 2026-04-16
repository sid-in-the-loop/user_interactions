#!/bin/bash
# Generate benchmark outputs for all checkpoints of one method.
#
# Usage:
#   sbatch --job-name=gen_jsd jobs/deltaai/gen_benchmarks.sh jsd_p30
#   sbatch --job-name=gen_sft jobs/deltaai/gen_benchmarks.sh sft_p30

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
# job-name set via sbatch --job-name=gen_<method>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

METHOD="${1:?Usage: sbatch gen_benchmarks.sh METHOD_DIR_NAME}"

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

cd "${SLURM_SUBMIT_DIR:-/u/ssredharan/user_interactions}"
mkdir -p logs

CKPT_DIR="/projects/bgtw/ssredharan/checkpoints/${METHOD}"
OUTPUT_DIR="eval_results/${METHOD}"

echo "════════════════════════════════════════"
echo "  Generation: ${METHOD}"
echo "  Checkpoints: ${CKPT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "════════════════════════════════════════"

python scripts/eval/generate_all.py \
    --method_dir "$CKPT_DIR" \
    --output_root "$OUTPUT_DIR" \
    --base_model Qwen/Qwen3-8B \
    --benchmarks all \
    --max_num_seqs 512

echo "Done."
