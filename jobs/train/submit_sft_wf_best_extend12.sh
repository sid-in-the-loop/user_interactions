#!/bin/bash
# Extend WF thinking-best (both LRs) by 12 epochs from each run's final/.
# Writes into the SAME output dirs as the 6ep runs:
#   ext-epoch-6, ext-epoch-12  (sharded, every 6 epochs)
#   final/  (overwritten with merged HF weights at end)
# Prereq: .../sft_wf_thinking_best_bs8_ga32_lr{5e6,2e6}_6ep/final exist.
# Run from repo root: bash jobs/train/submit_sft_wf_best_extend12.sh
set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
WF_JSONL="$REPO/datasets/wildfeedback/ystar_thinking_best.jsonl"
FKL="/data/group_data/cx_group/ssmurali/offpolicy/fkl"
SCRIPT="jobs/train/sbatch_sft_one.sh"
PORT=29700

for name in lr5e6 lr2e6; do
  case $name in
    lr5e6) lr=5e-6; run="sft_wf_thinking_best_bs8_ga32_lr5e6_6ep" ;;
    lr2e6) lr=2e-6; run="sft_wf_thinking_best_bs8_ga32_lr2e6_6ep" ;;
  esac
  final="$FKL/$run/final"
  if [[ ! -d "$final" ]] || [[ ! -f "$final/config.json" ]]; then
    echo "ERROR: missing checkpoint: $final (need 6ep run finished)"
    exit 1
  fi
done
[[ -f "$WF_JSONL" ]] || { echo "ERROR: missing $WF_JSONL"; exit 1; }

submit_one() {
  local lr=$1 run=$2 port=$3
  export INPUT="$WF_JSONL"
  export RUN_NAME="$run"
  export LR="$lr"
  export MODEL="$FKL/$run/final"
  export EPOCHS=12
  export SAVE_STEPS=999999
  export SAVE_EPOCH_EVERY=6
  export SHARDED_EPOCH_PREFIX="ext-"
  export MASTER_PORT="$port"
  sbatch --job-name="wf_best_ext12_${run##*_}" "$SCRIPT"
  echo "Submitted extend12: $run (MODEL=$FKL/$run/final)"
}

submit_one 5e-6 sft_wf_thinking_best_bs8_ga32_lr5e6_6ep $PORT
((PORT++)) || true
submit_one 2e-6 sft_wf_thinking_best_bs8_ga32_lr2e6_6ep $PORT
echo "Done. Sharded dirs: ext-epoch-6, ext-epoch-12 under each run; final/ updated."
