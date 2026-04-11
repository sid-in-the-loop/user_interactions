#!/bin/bash
# Win-rate eval for OLMo-3-7B-Instruct on WildFeedback (500 samples, seed 42).
# Runs all three comparisons then plots the bar chart.
#
# Prerequisites:
#   - OPENAI_API_KEY must be set
#   - Generation must have already run (see run_olmo_generation.sh)
#
# Usage (interactive):
#   export OPENAI_API_KEY=...
#   bash jobs/eval/run_olmo_winrate.sh
#
# Env vars:
#   WF_DIR      — base dir for OLMo outputs (default: datasets/wildfeedback/olmo_3_7b)
#   OUT_DIR     — results output dir      (default: data/winrate_results/olmo_wf)
#   SUBSAMPLE   — samples to evaluate     (default: 500)
#   SEED        — random seed             (default: 42)
#   PLOT_OUT    — chart PNG path          (default: $OUT_DIR/winrate_olmo.png)

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

# ── Params ────────────────────────────────────────────────────────────────────
WF_DIR="${WF_DIR:-datasets/wildfeedback/olmo_3_7b}"
OUT_DIR="${OUT_DIR:-data/winrate_results/olmo_wf}"
SUBSAMPLE="${SUBSAMPLE:-500}"
SEED="${SEED:-42}"
YBASE_FILE="${WF_DIR}/ybase_olmo.jsonl"
YSTAR_FILE="${WF_DIR}/ystar_olmo.jsonl"
RESULTS_FILE="${OUT_DIR}/winrate_olmo_results.jsonl"
PLOT_OUT="${PLOT_OUT:-${OUT_DIR}/winrate_olmo.png}"

# ── Preflight checks ──────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY not set" >&2
  exit 1
fi
for f in "$YBASE_FILE" "$YSTAR_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "error: missing generated file: $f" >&2
    echo "       Run run_olmo_generation.sh first." >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════"
echo "  OLMo win-rate eval"
echo "  y_base : $YBASE_FILE"
echo "  y*     : $YSTAR_FILE"
echo "  output : $OUT_DIR"
echo "  n      : $SUBSAMPLE  (seed $SEED)"
echo "════════════════════════════════════════"

# ── Step 1: run win-rate eval ─────────────────────────────────────────────────
python "$REPO/scripts/eval/winrate_olmo.py" \
  --ybase-file   "$YBASE_FILE" \
  --ystar-file   "$YSTAR_FILE" \
  --output-dir   "$OUT_DIR" \
  --subsample    "$SUBSAMPLE" \
  --seed         "$SEED" \
  --max-concurrent 500

echo ""

# ── Step 2: plot ──────────────────────────────────────────────────────────────
python "$REPO/scripts/eval/plot_olmo_winrate.py" \
  --results "$RESULTS_FILE" \
  --output  "$PLOT_OUT"

echo ""
echo "Results table : ${OUT_DIR}/winrate_olmo_summary.txt"
echo "Raw judge data: ${OUT_DIR}/winrate_olmo_results.jsonl"
echo "Bar chart     : ${PLOT_OUT}"
