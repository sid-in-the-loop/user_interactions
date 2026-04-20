#!/bin/bash
# Master launcher for the TAC scale-up (Option B):
#   wildchat  34k   -> gen + judge
#   webinstruct 50k -> gen + judge
#   polaris   53k   -> 3-stage pipeline + gen (48h long pole)
#   mixtures: 3 datasets × {100,70,50,20} winrate = 12 training files
#
# Usage:
#   bash jobs/deltaai/tac_scaleup_launch.sh
#
# Requires OPENAI_API_KEY in env before submission (inherited by judge job).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set. Add to ~/.bashrc first." >&2
  exit 1
fi

echo "launching TAC scale-up from $REPO"
echo "API key prefix: ${OPENAI_API_KEY:0:10}..."
echo

# Chain 1: wildchat + webinstruct gen -> judge
GEN_WCWI=$(sbatch --parsable jobs/deltaai/tac_scaleup_gen_wc_wi.sh)
echo "submitted gen wildchat+webinstruct : $GEN_WCWI"

JUDGE_WCWI=$(sbatch --parsable --dependency=afterok:"$GEN_WCWI" \
             jobs/deltaai/tac_scaleup_judge_wc_wi.sh)
echo "submitted judge wildchat+webinstruct: $JUDGE_WCWI  (after $GEN_WCWI)"

# Chain 2: polaris full pipeline (independent — can start now)
POL=$(sbatch --parsable jobs/deltaai/tac_scaleup_polaris.sh)
echo "submitted polaris pipeline + gen   : $POL"

# Chain 3: mixtures (after both chains' final jobs)
MIX=$(sbatch --parsable \
       --dependency=afterok:"$JUDGE_WCWI":"$POL" \
       jobs/deltaai/tac_scaleup_mixtures.sh)
echo "submitted mixture builder          : $MIX  (after $JUDGE_WCWI,$POL)"

echo
echo "track: squeue -u \$USER"
echo "logs:  tail -f logs/tac_{scaleup,mixtures}_*.out"
echo
echo "end artifacts in experiments/tac_winrates/data/mixtures/:"
echo "  mix_wildchat_teacher_xo_w{100,70,50,20}.jsonl"
echo "  mix_webinstruct_teacher_xo_w{100,70,50,20}.jsonl"
echo "  mix_polaris_teacher_xo_w{100,70,50,20}.jsonl"
echo "  _mixtures_summary.csv"
