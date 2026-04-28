#!/bin/bash
# Launch JSD-only retraining for wildchat (N=8550) and webinstruct (N=11998)
# across winrates {100, 50, 20}, with LR=1e-4 and matched-N subsampling.
# Overwrites existing $CKPT/jsd directories.
#
# Usage:
#   bash jobs/deltaai/tac_train_jsd_launch.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

WINRATES=(100 50 20)
declare -A MATCH_N=( [wildchat]=8550 [webinstruct]=11998 )

echo "submitting 6 JSD re-runs (wildchat@8550, webinstruct@11998) x {100,50,20}..."
JOBIDS=()
for ds in wildchat webinstruct; do
  n="${MATCH_N[$ds]}"
  for w in "${WINRATES[@]}"; do
    jid=$(sbatch --parsable \
            --job-name="tac_jsd_${ds}_w${w}_n${n}" \
            jobs/deltaai/tac_train_jsd.sh "$ds" "$w" "$n")
    JOBIDS+=("$jid")
    printf "  %-12s w%-3s  n=%-5s  ->  jobid %s\n" "$ds" "$w" "$n" "$jid"
  done
done

echo
echo "all 6 JSD jobs submitted:"
printf '  %s\n' "${JOBIDS[@]}"
echo "track: squeue -u \$USER -o '%.12i %.20j %.8T %.10M %R'"
