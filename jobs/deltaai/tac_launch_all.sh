#!/bin/bash
# One-shot launcher for the full TAC-winrates pipeline. Uses SLURM dependencies
# so nothing runs out of order. Each GPU job is 1-GPU; eval is 1-GPU (idle).
#
# Usage:
#   bash jobs/deltaai/tac_launch_all.sh
#
# Flow:
#   - wildchat gen + webinstruct gen: fire now (unified data already on disk)
#   - polaris pipeline (3 stages): fires now; polaris gen waits for it
#   - eval: waits for all 3 gen jobs

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

echo "launching from $REPO"
echo

# Sanity: make sure the two already-prepped unified files exist.
for f in experiments/tac_winrates/data/wildchat_unified.jsonl \
         experiments/tac_winrates/data/webinstruct_unified.jsonl \
         experiments/tac_winrates/data/polaris_pool.jsonl; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required file missing: $f" >&2
    exit 1
  fi
done

GEN_WC=$(sbatch --parsable jobs/deltaai/tac_gen.sh wildchat)
echo "submitted gen wildchat       : $GEN_WC"

GEN_WI=$(sbatch --parsable jobs/deltaai/tac_gen.sh webinstruct)
echo "submitted gen webinstruct    : $GEN_WI"

POL=$(sbatch --parsable jobs/deltaai/tac_polaris_pipeline.sh)
echo "submitted polaris pipeline   : $POL"

GEN_POL=$(sbatch --parsable --dependency=afterok:"$POL" jobs/deltaai/tac_gen.sh polaris)
echo "submitted gen polaris        : $GEN_POL  (after $POL)"

EVAL=$(sbatch --parsable \
  --dependency=afterok:"$GEN_WC":"$GEN_WI":"$GEN_POL" \
  jobs/deltaai/tac_eval.sh)
echo "submitted eval + aggregate   : $EVAL  (after $GEN_WC,$GEN_WI,$GEN_POL)"

echo
echo "track progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/tac_{gen,polaris,eval}_*.{out,err}"
echo
echo "final outputs will land in experiments/tac_winrates/results/:"
echo "  phase1_raw.csv"
echo "  phase1_summary.csv"
echo "  plot_A_ystar_vs_ybase.{png,pdf}"
echo "  plot_B_ystar_vs_y.{png,pdf}"
