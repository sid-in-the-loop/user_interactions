#!/bin/bash
# Parameterized y* generation job. Set env vars before sbatch:
#   INPUT        — path to filtered JSONL (required)
#   OUTPUT_DIR   — where to write ystar_*.jsonl (required)
#   MODE         — A (nonthinking), B (thinking), or both (required)
#   SUFFIX       — output suffix, e.g. _best → ystar_thinking_best.jsonl (default: "")
#   GPUS         — number of GPUs (default 1)
#   TIME         — SBATCH time limit (default 12:00:00)
#   MODEL        — base model (default Qwen/Qwen3-4B)
#   MAX_TOKENS   — (default 1024)
#   TEMPERATURE  — (default 1.0)
#
# Example:
#   export INPUT=datasets/wildfeedback/filtered_BEST.jsonl OUTPUT_DIR=datasets/wildfeedback MODE=B SUFFIX=_best
#   sbatch --job-name=ystar_wf_think_best jobs/train/sbatch_gen_ystar.sh
#
# 4-GPU example (larger datasets):
#   export GPUS=4 TIME=12:00:00 INPUT=... OUTPUT_DIR=... MODE=B
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=128G jobs/train/sbatch_gen_ystar.sh

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_ystar_%j_%x.out
#SBATCH --error=logs/gen_ystar_%j_%x.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
REPO="${SLURM_SUBMIT_DIR:-/home/ssmurali/user_interactions}"
cd "$REPO" && export PYTHONPATH="${PYTHONPATH:-}:$REPO" && export PYTHONUNBUFFERED=1

INPUT="${INPUT:?Set INPUT to filtered JSONL path}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
MODE="${MODE:?Set MODE to A, B, or both}"
SUFFIX="${SUFFIX:-}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
GPUS="${GPUS:-1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# Auto-tune vLLM params based on GPU count
if [[ "$GPUS" -ge 4 ]]; then
  TP="${TP:-$GPUS}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
  GPU_UTIL="${GPU_UTIL:-0.92}"
  BATCH="${BATCH:+--batch_size $BATCH}"
else
  TP="${TP:-1}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
  GPU_UTIL="${GPU_UTIL:-0.95}"
  BATCH=""
fi

mkdir -p logs "$OUTPUT_DIR"

echo "y* gen | input=$INPUT | out=$OUTPUT_DIR | mode=$MODE | suffix=$SUFFIX | tp=$TP"
python "$REPO/scripts/fkl/generate_ystar_fkl.py" \
  --input "$INPUT" \
  --output_dir "$OUTPUT_DIR" \
  --output_suffix "$SUFFIX" \
  --mode "$MODE" \
  --model "$MODEL" \
  --max_tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
  --tp_size "$TP" --max_num_seqs "$MAX_NUM_SEQS" --gpu_util "$GPU_UTIL" \
  $BATCH
echo "Done."
