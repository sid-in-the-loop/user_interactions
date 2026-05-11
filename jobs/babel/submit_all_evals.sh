#!/usr/bin/env bash
# Submit one eval sbatch per training run. Auto-detects WI vs WC for the
# base model + chat template. Skips runs whose ckpt dir doesn't exist yet.
#
# Usage:
#   bash jobs/babel/submit_all_evals.sh             # all runs found
#   bash jobs/babel/submit_all_evals.sh dry         # preview
#   bash jobs/babel/submit_all_evals.sh wi          # only WI-*
#   bash jobs/babel/submit_all_evals.sh wc          # only WC-*

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/data/group_data/cx_group/ssmurali/demo2teacher}"
EVAL_SBATCH="$REPO/jobs/babel/eval_one_run.sh"

WI_MODEL="Qwen/Qwen2.5-Math-7B"
WC_MODEL="Qwen/Qwen3-4B"

DRY=0
FAMILY_FILTER=""
for arg in "$@"; do
  case "$arg" in
    dry|--dry|-n) DRY=1 ;;
    wi|WI) FAMILY_FILTER="wi" ;;
    wc|WC) FAMILY_FILTER="wc" ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "ERROR: $CKPT_ROOT not found (run from a node with /data mounted)"
  exit 1
fi

mkdir -p logs

n=0
for d in "$CKPT_ROOT"/*/; do
  d="${d%/}"
  base="$(basename "$d")"

  # Match WI-N_obj_dataset or WC-N_obj_dataset → extract run_id (WI-N / WC-N)
  if   [[ "$base" =~ ^(WI-[0-9]+)_ ]]; then RUN_ID="${BASH_REMATCH[1]}"; FAMILY="wi"; MODEL="$WI_MODEL"; CHAT="0"
  elif [[ "$base" =~ ^(WC-[0-9]+)_ ]]; then RUN_ID="${BASH_REMATCH[1]}"; FAMILY="wc"; MODEL="$WC_MODEL"; CHAT="1"
  else continue
  fi

  [[ -n "$FAMILY_FILTER" && "$FAMILY_FILTER" != "$FAMILY" ]] && continue

  # Need at least one ckpt to be present
  if ! ls -d "$d"/step-* > /dev/null 2>&1 && [[ ! -d "$d/final" ]]; then
    echo "[skip] $RUN_ID ($base) — no step-*/ or final/ yet"
    continue
  fi

  n=$((n + 1))
  echo "[$n] $RUN_ID  family=$FAMILY  $base"
  if [[ $DRY -eq 1 ]]; then
    echo "    DRY: sbatch $EVAL_SBATCH $RUN_ID $d $MODEL $CHAT"
  else
    sbatch "$EVAL_SBATCH" "$RUN_ID" "$d" "$MODEL" "$CHAT"
  fi
done

echo
echo "Submitted $n eval jobs. Watch with:  squeue -u \$USER -n demo2t-eval"
