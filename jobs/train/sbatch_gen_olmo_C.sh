#!/bin/bash
# Generate y*_C (Prompt C: "Note: o", x+o only) for the full WildFeedback dataset.
# Output: $BASE_OUT/ystar_olmo_xo_C.jsonl
#
# Submit:
#   sbatch jobs/train/sbatch_gen_olmo_C.sh

#SBATCH --job-name=gen_olmo_C
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_olmo_C_%j.out
#SBATCH --error=logs/gen_olmo_C_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

MODEL="allenai/OLMo-3-7B-Instruct-SFT"
BASE_OUT="/data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3"
mkdir -p logs "$BASE_OUT"

echo "Job ID : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"
echo "Input  : datasets/wildfeedback/tuples.jsonl  ($(wc -l < datasets/wildfeedback/tuples.jsonl) samples)"
echo "Output : $BASE_OUT/ystar_olmo_xo_C.jsonl"
echo "──────────────────────────────────────────"

python scripts/eval/generate_olmo.py \
    --input          datasets/wildfeedback/tuples.jsonl \
    --output_dir     "$BASE_OUT" \
    --target         ystar \
    --teacher-prompt xo_C \
    --model          "$MODEL"

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Output: $BASE_OUT/ystar_olmo_xo_C.jsonl  ($(wc -l < "$BASE_OUT/ystar_olmo_xo_C.jsonl") samples)"
