#!/bin/bash
# Launch 6 matched-N TAC training jobs:
#   datasets : wildchat (N=8550), webinstruct (N=11998)
#   regimes  : w100, w50, w20
# Each sbatch runs SFT -> FKL -> JSD -> DPO -> RKL on one GPU.
#
# Usage:
#   bash jobs/deltaai/tac_train_launch.sh [MIXTURE_JOBID]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

DEP_ARG=""
if [ $# -ge 1 ] && [ -n "$1" ]; then
  DEP_ARG="--dependency=afterok:$1"
  echo "all training jobs will wait on jobid $1"
fi

WINRATES=(100 50 20)
declare -A MATCH_N=( [wildchat]=8550 [webinstruct]=11998 )

echo "submitting 6 training jobs (wildchat@8550, webinstruct@11998) x {100,50,20}..."
echo

JOBIDS=()
for ds in wildchat webinstruct; do
  n="${MATCH_N[$ds]}"
  for w in "${WINRATES[@]}"; do
    if [ -n "$DEP_ARG" ]; then
      jid=$(sbatch --parsable $DEP_ARG \
              --job-name="tac_${ds}_w${w}_n${n}" \
              jobs/deltaai/tac_train.sh "$ds" "$w" "$n")
    else
      jid=$(sbatch --parsable \
              --job-name="tac_${ds}_w${w}_n${n}" \
              jobs/deltaai/tac_train.sh "$ds" "$w" "$n")
    fi
    JOBIDS+=("$jid")
    printf "  %-12s w%-3s  n=%-5s  ->  jobid %s\n" "$ds" "$w" "$n" "$jid"
  done
done

echo
echo "all 6 training jobs submitted:"
printf '  %s\n' "${JOBIDS[@]}"
echo
echo "track: squeue -u \$USER -o '%.12i %.20j %.8T %.10M %R'"
echo "logs:  tail -f logs/tac_train_tac_*_*.out"
echo
echo "checkpoints will land under: /work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/{ds}_w{w}_n{N}/"
