#!/usr/bin/env bash
# Generate ~/jobs.txt — 24 mode-covering training jobs (12 WI + 12 WC).
#
# Each line:  <objective> <run_id> <model> <dataset_path> <output_dir>
# Run from anywhere:  bash jobs/babel/gen_jobs_txt.sh
#
# Override with env vars:
#   WI_DATASET_DIR  WC_DATASET_DIR  CKPT_ROOT  OUT
#   OBJECTIVES (default "sft fkl"; set to "sft fkl sdpo pc_sdpo" for all 48)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

WI_DATASET_DIR="${WI_DATASET_DIR:-$REPO/experiments-for-apr26-may1/build_datasets/data/webinstruct}"
WC_DATASET_DIR="${WC_DATASET_DIR:-$REPO/experiments-for-apr26-may1/wildchat_prefix_decision/data/wildchat}"
CKPT_ROOT="${CKPT_ROOT:-/data/group_data/cx_group/ssmurali/demo2teacher}"
OUT="${OUT:-$HOME/jobs.txt}"

WI_MODEL="${WI_MODEL:-Qwen/Qwen2.5-Math-7B}"
WC_MODEL="${WC_MODEL:-Qwen/Qwen3-4B}"

DATASETS=(
  teacher_wins_cond_xo
  teacher_loses_cond_xo
  teacher_wins_cond_xyo
  teacher_loses_cond_xyo
  teacher_wins_cond_xyo_ystart
  teacher_loses_cond_xyo_ystart
)
read -r -a OBJ_ARR <<< "${OBJECTIVES:-sft fkl}"

: > "$OUT"

emit_family() {
  local family="$1" model="$2" dsdir="$3"
  local idx=0 ds obj run_id outdir
  for ds in "${DATASETS[@]}"; do
    for obj in "${OBJ_ARR[@]}"; do
      idx=$((idx + 1))
      run_id="${family}-${idx}"
      outdir="${CKPT_ROOT}/${run_id}_${obj}_${ds}"
      printf '%s %s %s %s %s\n' "$obj" "$run_id" "$model" "$dsdir/$ds.jsonl" "$outdir" >> "$OUT"
    done
  done
}

emit_family WI "$WI_MODEL" "$WI_DATASET_DIR"
emit_family WC "$WC_MODEL" "$WC_DATASET_DIR"

NLINES=$(wc -l < "$OUT")
echo "Wrote $NLINES lines to $OUT"
echo
echo "Preview:"
nl -ba "$OUT" | head -6
echo "  ..."
nl -ba "$OUT" | tail -3
echo
echo "Submit with:  sbatch --array=1-${NLINES}%8 jobs/babel/train_demo2teacher.sh"
