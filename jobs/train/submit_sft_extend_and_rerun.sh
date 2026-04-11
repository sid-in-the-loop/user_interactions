#!/bin/bash
# Submit: (1) 3 extend jobs from existing finals, (2) 5 re-run jobs from base.
# Run from repo root: bash jobs/train/submit_sft_extend_and_rerun.sh

set -e
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
BASE="/data/group_data/cx_group/ssmurali/offpolicy/fkl"
DATA="$REPO/datasets/wildchat"

# -------- 1. Extend (3 jobs): MODEL=path to final, EPOCHS=2 or 6, new RUN_NAME --------
echo "=== Submitting EXTEND jobs (load from final, more epochs) ==="

export INPUT="$DATA/ystar_thinking_full.jsonl"
export RUN_NAME="sft_wc_think_full_lr2e6_ext2"
export MODEL="$BASE/sft_wc_thinking_full_bs8_ga32_lr2e6/final"
export EPOCHS=2
export LR=2e-6
export MASTER_PORT=29540
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

export INPUT="$DATA/ystar_thinking_best.jsonl"
export RUN_NAME="sft_wc_think_best_lr5e6_ext6"
export MODEL="$BASE/sft_wc_thinking_best_bs8_ga32_lr5e6/final"
export EPOCHS=6
export LR=5e-6
export MASTER_PORT=29541
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

export INPUT="$DATA/ystar_thinking_best.jsonl"
export RUN_NAME="sft_wc_think_best_lr2e6_ext6"
export MODEL="$BASE/sft_wc_thinking_best_bs8_ga32_lr2e6/final"
export EPOCHS=6
export LR=2e-6
export MASTER_PORT=29542
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

# -------- 2. Re-run from base (5 jobs): 4 epochs, default MODEL --------
echo "=== Submitting RE-RUN jobs (from base, 4 epochs) ==="

unset MODEL
export EPOCHS=4

export INPUT="$DATA/ystar_thinking_full.jsonl"
export RUN_NAME="sft_wc_think_full_lr5e6"
export LR=5e-6
export MASTER_PORT=29543
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

export INPUT="$DATA/ystar_nonthinking_full.jsonl"
export RUN_NAME="sft_wc_nothink_full_lr5e6"
export LR=5e-6
export MASTER_PORT=29544
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

export INPUT="$DATA/ystar_nonthinking_full.jsonl"
export RUN_NAME="sft_wc_nothink_full_lr2e6"
export LR=2e-6
export MASTER_PORT=29545
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

export INPUT="$DATA/ystar_nonthinking_best.jsonl"
export RUN_NAME="sft_wc_nothink_best_lr5e6"
export LR=5e-6
export MASTER_PORT=29546
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

export INPUT="$DATA/ystar_nonthinking_best.jsonl"
export RUN_NAME="sft_wc_nothink_best_lr2e6"
export LR=2e-6
export MASTER_PORT=29547
sbatch --job-name="$RUN_NAME" jobs/train/sbatch_sft_one.sh

echo "Done. 3 extend + 5 re-run = 8 jobs submitted."
