#!/bin/bash
# Full Qwen3-8B pipeline on WildFeedback: y* gen → filter best → winrate → SFT (4 configs).
#
# Usage (from repo root):
#   bash jobs/train/submit_qwen3_8b_wf.sh           # submit all
#   bash jobs/train/submit_qwen3_8b_wf.sh --dry-run # preview commands
#
# Prerequisites: OPENAI_API_KEY set (for winrate step).
# GPU allocation: y* gen = 1 GPU each, SFT = 4 GPUs each.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=echo

MODEL="Qwen/Qwen3-8B"
WF="$REPO/datasets/wildfeedback"
YSTAR_DIR="$REPO/datasets/wildfeedback/qwen3_8b"
WINRATE_DIR="$REPO/data/winrate_results/qwen3_8b"
REF_BEST="$WF/filtered_BEST.jsonl"
SFT_SCRIPT="jobs/train/sbatch_sft_one.sh"
YSTAR_SCRIPT="jobs/train/sbatch_gen_ystar.sh"

mkdir -p "$YSTAR_DIR" "$WINRATE_DIR" logs

# ── Step 1: y* generation (2 jobs, 1 GPU each, run in parallel) ──────────
echo "=== Step 1: y* generation (thinking + nonthinking on full WF) ==="

export INPUT="$WF/tuples.jsonl" OUTPUT_DIR="$YSTAR_DIR" SUFFIX="_full" MODEL="$MODEL"

export MODE=B
JOB_THINK=$($DRY sbatch --parsable --job-name=ystar_8b_wf_think_full "$YSTAR_SCRIPT")
echo "  thinking full: job $JOB_THINK"

export MODE=A
JOB_NOTHINK=$($DRY sbatch --parsable --job-name=ystar_8b_wf_nothink_full "$YSTAR_SCRIPT")
echo "  nonthinking full: job $JOB_NOTHINK"

# ── Step 2: Filter best + winrate (runs after y* gen finishes) ───────────
# This is a lightweight job (1 GPU for 15 min) that:
#   a) filters full → best using conversation IDs
#   b) runs winrate eval on all 4 files
echo "=== Step 2: filter best + winrate (depends on step 1) ==="

STEP2_DEPS=""
if [[ -z "$DRY" ]]; then
  STEP2_DEPS="--dependency=afterok:${JOB_THINK}:${JOB_NOTHINK}"
fi

STEP2_JOB=$($DRY sbatch --parsable $STEP2_DEPS \
  --job-name=filter_winrate_8b \
  --partition=general --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=02:00:00 \
  --output=logs/filter_winrate_8b_%j.out \
  --error=logs/filter_winrate_8b_%j.err \
  --wrap "#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate opf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $REPO && export PYTHONPATH=\${PYTHONPATH:-}:$REPO

echo '── Filtering full → best ──'
python scripts/fkl/filter_ystar_by_ids.py \
  --ystar $YSTAR_DIR/ystar_thinking_full.jsonl \
  --ref   $REF_BEST \
  --out   $YSTAR_DIR/ystar_thinking_best.jsonl

python scripts/fkl/filter_ystar_by_ids.py \
  --ystar $YSTAR_DIR/ystar_nonthinking_full.jsonl \
  --ref   $REF_BEST \
  --out   $YSTAR_DIR/ystar_nonthinking_best.jsonl

echo '── Winrate eval (y vs y*) ──'
for f in thinking_full thinking_best nonthinking_full nonthinking_best; do
  INPUT_FILE=$YSTAR_DIR/ystar_\${f}.jsonl
  if [[ -f \"\$INPUT_FILE\" ]]; then
    echo \"=== winrate: \$f ===\"
    python scripts/eval/winrate_eval.py \
      --input \"\$INPUT_FILE\" \
      --output-dir $WINRATE_DIR/wf_\${f} \
      --subsample 500 --seed 42 --max-concurrent 500
  else
    echo \"SKIP (missing): \$INPUT_FILE\"
  fi
done
echo 'Done. Summaries in $WINRATE_DIR/wf_*/winrate_summary.txt'
")
echo "  filter+winrate: job $STEP2_JOB"

# ── Step 3: SFT (4 jobs, 4 GPUs each, depend on step 2) ─────────────────
echo "=== Step 3: SFT on all 4 y* datasets (4 GPU, lr=5e-6, mask_tau=0) ==="

SFT_DEPS=""
if [[ -z "$DRY" ]]; then
  SFT_DEPS="--dependency=afterok:${STEP2_JOB}"
fi

# 8B on 4×L40 48GB: BS=4, GA=16, effective BS = 4*16*4 = 256
PORT=29600
submit_sft() {
  local tag=$1 input=$2
  export INPUT="$input"
  export RUN_NAME="qwen3_8b/sft_wf_${tag}_lr5e6"
  export LR=5e-6 MODEL="$MODEL" NGPUS=4 BATCH_SIZE=4 GRAD_ACCUM=16
  export MASTER_PORT=$PORT MASK_TAU=0 SAVE_STEPS=999999
  local jid=$($DRY sbatch --parsable $SFT_DEPS \
    --gres=gpu:4 --job-name="sft_8b_wf_${tag}" "$SFT_SCRIPT")
  echo "  sft_wf_${tag}: job $jid (port $PORT)"
  ((PORT++)) || true
}

submit_sft thinking_full      "$YSTAR_DIR/ystar_thinking_full.jsonl"
submit_sft thinking_best      "$YSTAR_DIR/ystar_thinking_best.jsonl"
submit_sft nonthinking_full   "$YSTAR_DIR/ystar_nonthinking_full.jsonl"
submit_sft nonthinking_best   "$YSTAR_DIR/ystar_nonthinking_best.jsonl"

echo ""
echo "=== Summary ==="
echo "  y* output:      $YSTAR_DIR/"
echo "  winrate output:  $WINRATE_DIR/"
echo "  SFT output:      /data/group_data/cx_group/ssmurali/offpolicy/fkl/qwen3_8b/"
echo "  Total jobs:      2 (y* gen) + 1 (filter+winrate) + 4 (SFT) = 7"
echo "  Job chain:       y* gen → filter+winrate → SFT (all via --dependency)"
