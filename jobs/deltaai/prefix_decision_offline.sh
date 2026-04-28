#!/usr/bin/env bash
# Offline vLLM pipeline for WebInstruct prefix-decision.
# No HTTP server. One Python process loads Qwen2.5-Math-7B and runs:
#   stage A → generation (3 variants × 50k rows)
#   stage B → student judge (logprob comparison)
#   stage C → gpt4o-mini judge (OpenAI API)
#   stage D → summarize
#
# Usage:
#   sbatch jobs/deltaai/prefix_decision_offline.sh smoke   # 50 rows
#   sbatch jobs/deltaai/prefix_decision_offline.sh full

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=prefix_dec_off
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --output=logs/prefix_decision_offline_%j.out
#SBATCH --error=logs/prefix_decision_offline_%j.err

set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
  smoke) LIMIT_FLAG=(--limit 50) ;;
  full)  LIMIT_FLAG=() ;;
  *) echo "usage: sbatch ${0##*/} {smoke|full}"; exit 1 ;;
esac

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs

export HF_HOME="${HF_HOME:-/projects/bgtw/ssredharan/models}"

OUTDIR="experiments-for-apr26-may1/prefix_decision/data"
GENS="$OUTDIR/01_generations.jsonl"
SCSV="$OUTDIR/02_verdicts_student.csv"
OCSV="$OUTDIR/03_verdicts_gpt4o_mini.csv"
FINAL="$OUTDIR/webinstruct_prefix_decision.jsonl"
SUMMARY="$OUTDIR/summary.txt"
mkdir -p "$OUTDIR"

echo "════════════════════════════════════════"
echo "  prefix_decision OFFLINE  mode=$MODE"
echo "  job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo "  HF_HOME=$HF_HOME"
echo "════════════════════════════════════════"

echo "── stages A+B: offline gen + student judge ──"
python experiments-for-apr26-may1/prefix_decision/01_pipeline_offline.py \
  --input  experiments/tac_winrates/data/webinstruct_unified.jsonl \
  --gens   "$GENS" --student_csv "$SCSV" \
  --skip_existing_gens \
  "${LIMIT_FLAG[@]}"

echo "── stage C: gpt4o-mini judge (OpenAI API) ──"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: OPENAI_API_KEY not set; skipping stage C."
else
  python experiments-for-apr26-may1/prefix_decision/03_judge_openai.py \
    --gens "$GENS" --output "$OCSV" --workers 64
fi

echo "── stage D: summarize ──"
python experiments-for-apr26-may1/prefix_decision/04_summarize.py \
  --gens "$GENS" --student_csv "$SCSV" --openai_csv "$OCSV" \
  --output "$FINAL" --summary "$SUMMARY"

echo "── stage E: build 6 training datasets (CPU, instant) ──"
python experiments-for-apr26-may1/build_datasets/01_build.py \
  --gens         "$GENS" \
  --student_csv  "$SCSV" \
  --openai_csv   "$OCSV"

echo "done. Training datasets:"
ls experiments-for-apr26-may1/build_datasets/data/webinstruct/teacher_*.jsonl 2>/dev/null
