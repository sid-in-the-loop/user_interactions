#!/bin/bash
# Launch one generation sbatch per matched-N run_dir (6 total). Each sbatch
# iterates 4-5 methods × their ckpts × 4 benchmarks. generate_all.py is
# resumable — re-run is safe.
#
# Usage:
#   bash jobs/deltaai/tac_eval_gen_launch.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

CKPT_ROOT="/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates"

JOBIDS=()
for run_path in "$CKPT_ROOT"/*_n*/; do
  run=$(basename "$run_path")
  nadapters=$(find "$run_path" -maxdepth 3 -name adapter_config.json 2>/dev/null | wc -l)
  if [ "$nadapters" -eq 0 ]; then
    printf "  %-34s  skip (no adapters yet)\n" "$run"
    continue
  fi
  jid=$(sbatch --parsable --job-name="taceval_${run}" \
          jobs/deltaai/tac_eval_gen.sh "$run")
  JOBIDS+=("$jid")
  printf "  %-34s  n_adapters=%d  ->  %s\n" "$run" "$nadapters" "$jid"
done

echo
echo "submitted ${#JOBIDS[@]} generation jobs"
echo "track: squeue -u \$USER -o '%.12i %.30j %.8T %.10M %R'"
