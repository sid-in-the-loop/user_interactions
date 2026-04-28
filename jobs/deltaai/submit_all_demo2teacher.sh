#!/usr/bin/env bash
# Fire-and-forget launcher: submits all 48 demonstrator-to-teacher training
# runs (24 WebInstruct × 24 WildChat) as separate single-GPU sbatch jobs.
#
# SLURM scheduler will run them in parallel up to your concurrent-GPU cap.
#
# Usage:
#   bash jobs/deltaai/submit_all_demo2teacher.sh         # submit all 48
#   bash jobs/deltaai/submit_all_demo2teacher.sh dry     # print sbatch lines, don't submit
#   bash jobs/deltaai/submit_all_demo2teacher.sh wi      # only WebInstruct (24)
#   bash jobs/deltaai/submit_all_demo2teacher.sh wc      # only WildChat (24)
#   bash jobs/deltaai/submit_all_demo2teacher.sh sft     # only sft objective (12)
#   bash jobs/deltaai/submit_all_demo2teacher.sh fkl     # only fkl
#   bash jobs/deltaai/submit_all_demo2teacher.sh sdpo    # only sdpo
#   bash jobs/deltaai/submit_all_demo2teacher.sh pc_sdpo # only pc_sdpo
#   bash jobs/deltaai/submit_all_demo2teacher.sh wi sft  # filter both
#
# Resulting jobs are named demo2t in squeue. Watch with `squeue -u $USER`.
#
# Adjust paths at the top if your dataset / checkpoint roots differ.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# ── PATHS — edit these to your actual data + ckpt locations ─────────────────
WI_DATASET_DIR="experiments-for-apr26-may1/build_datasets/data/webinstruct"
WC_DATASET_DIR="experiments-for-apr26-may1/wildchat_prefix_decision/data/wildchat"
CKPT_ROOT="/work/nvme/bgtw/ssredharan/checkpoints/demo2teacher"

WI_MODEL="Qwen/Qwen2.5-Math-7B"
WC_MODEL="Qwen/Qwen3-4B"

# WebInstruct dataset filenames (Prompt 1's build_datasets.py output).
# build_datasets currently writes teacher_wins_student / teacher_loses_student
# only — when you re-run it with the cond-aware schema, file names will be
# teacher_wins_cond_xo etc. Update WI_DATASETS if the names differ.
WI_DATASETS=(
  "teacher_wins_cond_xo"
  "teacher_loses_cond_xo"
  "teacher_wins_cond_xyo"
  "teacher_loses_cond_xyo"
  "teacher_wins_cond_xyo_ystart"
  "teacher_loses_cond_xyo_ystart"
)
WC_DATASETS=(
  "teacher_wins_cond_xo"
  "teacher_loses_cond_xo"
  "teacher_wins_cond_xyo"
  "teacher_loses_cond_xyo"
  "teacher_wins_cond_xyo_ystart"
  "teacher_loses_cond_xyo_ystart"
)

OBJECTIVES=("sft" "fkl" "sdpo" "pc_sdpo")

# Filters: anything in $@ that matches "wi"|"wc"|"sft"|"fkl"|"sdpo"|"pc_sdpo"|"dry"
DRY=0
DATASET_FILTER=""
OBJ_FILTER=""
for arg in "$@"; do
  case "$arg" in
    dry|--dry|-n) DRY=1 ;;
    wi|WI)  DATASET_FILTER="wi" ;;
    wc|WC)  DATASET_FILTER="wc" ;;
    sft|fkl|sdpo|pc_sdpo) OBJ_FILTER="$arg" ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

mkdir -p logs "$CKPT_ROOT"

# ── Run-id counters ──────────────────────────────────────────────────────────
WI_IDX=0
WC_IDX=0

submit() {
  local family="$1"   # WI or WC
  local model="$2"
  local dataset_path="$3"
  local objective="$4"
  local dataset_dir_short="$5"  # for output dir naming

  local idx run_id
  if [[ "$family" == "WI" ]]; then
    WI_IDX=$((WI_IDX + 1)); idx=$WI_IDX
  else
    WC_IDX=$((WC_IDX + 1)); idx=$WC_IDX
  fi
  run_id="${family}-${idx}"
  local outdir="${CKPT_ROOT}/${run_id}_${objective}_${dataset_dir_short}"

  echo "[$run_id] obj=$objective  ds=$dataset_dir_short  → $outdir"

  if [[ "$DRY" -eq 1 ]]; then
    echo "  DRY: sbatch jobs/deltaai/train_demo2teacher.sh $objective $run_id $model $dataset_path $outdir"
    return 0
  fi
  sbatch jobs/deltaai/train_demo2teacher.sh \
    "$objective" "$run_id" "$model" "$dataset_path" "$outdir"
}

# ── WebInstruct (24) ─────────────────────────────────────────────────────────
if [[ -z "$DATASET_FILTER" || "$DATASET_FILTER" == "wi" ]]; then
  for ds in "${WI_DATASETS[@]}"; do
    for obj in "${OBJECTIVES[@]}"; do
      if [[ -n "$OBJ_FILTER" && "$OBJ_FILTER" != "$obj" ]]; then continue; fi
      submit "WI" "$WI_MODEL" "${WI_DATASET_DIR}/${ds}.jsonl" "$obj" "$ds"
    done
  done
fi

# ── WildChat (24) ────────────────────────────────────────────────────────────
if [[ -z "$DATASET_FILTER" || "$DATASET_FILTER" == "wc" ]]; then
  for ds in "${WC_DATASETS[@]}"; do
    for obj in "${OBJECTIVES[@]}"; do
      if [[ -n "$OBJ_FILTER" && "$OBJ_FILTER" != "$obj" ]]; then continue; fi
      submit "WC" "$WC_MODEL" "${WC_DATASET_DIR}/${ds}.jsonl" "$obj" "$ds"
    done
  done
fi

echo
echo "Submitted. Watch with: squeue -u \$USER"
