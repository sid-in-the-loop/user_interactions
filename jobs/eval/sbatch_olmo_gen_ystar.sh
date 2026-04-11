#!/bin/bash
# Generate y* for all WildFeedback tuples (OLMo-3-7B-Instruct-SFT).
# Run in parallel with sbatch_olmo_gen_ybase.sh.
#
# sbatch jobs/eval/sbatch_olmo_gen_ystar.sh

#SBATCH --job-name=olmo_gen_ystar
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo_gen_ystar_%j.out
#SBATCH --error=logs/olmo_gen_ystar_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

INPUT="datasets/wildfeedback/tuples.jsonl"
OUTPUT_DIR="datasets/wildfeedback/olmo_3_7b"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"
mkdir -p logs "$OUTPUT_DIR"

echo "Job ID  : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"

python scripts/eval/generate_olmo.py \
    --target       ystar \
    --input        "$INPUT" \
    --output_dir   "$OUTPUT_DIR" \
    --model        "$MODEL" \
    --tp_size      1 \
    --max_num_seqs 512 \
    --gpu_util     0.92 \
    --max_tokens   1024 \
    --max_model_len 4096

echo "Done: $(date)  →  $OUTPUT_DIR/ystar_olmo.jsonl  ($(wc -l < $OUTPUT_DIR/ystar_olmo.jsonl) lines)"
