#!/bin/bash
# Launch the 9 TAC training jobs.
#   datasets : wildchat, webinstruct, polaris
#   regimes  : w100, w70, w50  (w20 dropped for this week — already covered by prior-week work)
#
# Each job is ONE sbatch that runs SFT -> JSD -> DPO -> RKL sequentially on one
# GPU (~18h walltime). With 2-3 concurrent GPUs the 9 jobs finish in ~52h wall.
#
# Usage:
#   bash jobs/deltaai/tac_train_launch.sh [MIXTURE_JOBID]
#
# If MIXTURE_JOBID is given, each tac_train job is made dependent on it with
# --dependency=afterok:<JOBID>, so they won't start until mixtures are built.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

DEP_ARG=""
if [ $# -ge 1 ] && [ -n "$1" ]; then
  DEP_ARG="--dependency=afterok:$1"
  echo "all training jobs will wait on jobid $1"
fi

DATASETS=(wildchat webinstruct polaris)
WINRATES=(100 70 50)

echo "submitting 9 training jobs (3 datasets x 3 regimes)..."
echo

JOBIDS=()
for ds in "${DATASETS[@]}"; do
  for w in "${WINRATES[@]}"; do
    if [ -n "$DEP_ARG" ]; then
      jid=$(sbatch --parsable $DEP_ARG \
              --job-name="tac_${ds}_w${w}" \
              jobs/deltaai/tac_train.sh "$ds" "$w")
    else
      jid=$(sbatch --parsable \
              --job-name="tac_${ds}_w${w}" \
              jobs/deltaai/tac_train.sh "$ds" "$w")
    fi
    JOBIDS+=("$jid")
    printf "  %-12s w%-3s  ->  jobid %s\n" "$ds" "$w" "$jid"
  done
done

echo
echo "all 9 training jobs submitted:"
printf '  %s\n' "${JOBIDS[@]}"
echo
echo "track: squeue -u \$USER -o '%.12i %.20j %.8T %.10M %R'"
echo "logs:  tail -f logs/tac_train_tac_*_*.out"
echo
echo "checkpoints will land under: /work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/"
