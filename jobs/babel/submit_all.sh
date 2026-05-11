#!/usr/bin/env bash
# Submit each line of ~/jobs.txt as its own sbatch (one 24h, 1-GPU job per line).
#
# Usage:
#   bash jobs/babel/submit_all.sh                # submit every line
#   bash jobs/babel/submit_all.sh dry            # print sbatch commands, don't submit
#   JOBS_FILE=/path/to/other.txt bash jobs/babel/submit_all.sh
#
# Each line of jobs.txt: <objective> <run_id> <model> <dataset_path> <output_dir>

set -euo pipefail

JOBS_FILE="${JOBS_FILE:-$HOME/jobs.txt}"
SBATCH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/train_demo2teacher.sh"

[[ -f "$JOBS_FILE" ]]     || { echo "ERROR: $JOBS_FILE not found (run gen_jobs_txt.sh first)"; exit 1; }
[[ -f "$SBATCH_SCRIPT" ]] || { echo "ERROR: $SBATCH_SCRIPT not found"; exit 1; }

DRY=0
[[ "${1:-}" =~ ^(dry|--dry|-n)$ ]] && DRY=1

mkdir -p logs

n=0
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue
  n=$((n + 1))
  read -r OBJ RUN_ID MODEL DATASET OUTDIR <<< "$line"
  echo "[$n] $RUN_ID  $OBJ  $(basename "$DATASET")"
  if [[ $DRY -eq 1 ]]; then
    echo "    DRY: sbatch $SBATCH_SCRIPT $OBJ $RUN_ID $MODEL $DATASET $OUTDIR"
  else
    sbatch "$SBATCH_SCRIPT" "$OBJ" "$RUN_ID" "$MODEL" "$DATASET" "$OUTDIR"
  fi
done < "$JOBS_FILE"

echo
echo "Submitted $n jobs. Watch with:  squeue -u \$USER -n demo2t"
