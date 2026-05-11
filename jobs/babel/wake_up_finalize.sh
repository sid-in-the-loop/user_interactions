#!/bin/bash
# Run when you wake up. Does everything to produce final plots + per-model readout.
# Steps:
#   1. Check if eval jobs all finished (warns if not)
#   2. Judge alpaca outputs against gpt-4-turbo (async, ~3 min)
#   3. Pull adv mean/var from wandb for fig3/4
#   4. Run the styled plotter (all real data)
#   5. Generate master.csv + per-model summary readout
#
# Usage:  bash jobs/babel/wake_up_finalize.sh
# Requires: OPENAI_API_KEY exported

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not exported. Set it and re-run." >&2; exit 1
fi

REPO=/home/ssmurali/user_interactions
RES=$REPO/CFT-Eric-Zhu/eval_results

cd $REPO

# ─── 1. Sanity check: are all evals done? ───────────────────────────────────
echo "── 1. Eval status ──"
n_math_s0=$(find $RES/demo2teacher -path '*/math_s0/summary.txt' 2>/dev/null | wc -l)
n_math_s1=$(find $RES/demo2teacher -path '*/math_s1/summary.txt' 2>/dev/null | wc -l)
n_math_s2=$(find $RES/demo2teacher -path '*/math_s2/summary.txt' 2>/dev/null | wc -l)
n_alpaca=$(find $RES/demo2teacher -path '*/alpaca/outputs.jsonl' 2>/dev/null | grep "/WC-" | wc -l)
n_kl=$(find $RES/demo2teacher -name kl_vs_base.json 2>/dev/null | wc -l)
echo "  math seed 0: $n_math_s0 / 144"
echo "  math seed 1: $n_math_s1 / 144"
echo "  math seed 2: $n_math_s2 / 144"
echo "  alpaca outs: $n_alpaca / 144"
echo "  KL files   : $n_kl / 24"

if (( n_alpaca < 100 )); then
  echo "⚠ less than 100 alpaca outputs — evals likely incomplete. Run anyway? (Ctrl-C to stop, 10s)"
  sleep 10
fi

# ─── 2. Judge alpaca async (~3 min) ─────────────────────────────────────────
echo
echo "── 2. Judge alpaca (vs gpt-4-1106-preview) ──"
mapfile -t INPUTS < <(find "$RES/demo2teacher" -path '*/alpaca/outputs.jsonl' | grep "/WC-" | sort)
echo "  judging ${#INPUTS[@]} alpaca outputs ..."
python3 -u $REPO/CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/judge_alpaca_async.py \
  --inputs "${INPUTS[@]}" \
  --output_dir "$RES/demo2teacher/_alpaca_judge_final" \
  --reference_config alpaca_eval_gpt4_baseline \
  --concurrency 200 2>&1 | tail -10

# ─── 3. Pull adv from wandb ──────────────────────────────────────────────────
echo
echo "── 3. Pull advantage data from wandb ──"
python3 - << 'PY' 2>&1 | tail -10
import sys, json, os
for p in list(sys.path):
    if "user_interactions" in p: sys.path.remove(p)
import wandb
api = wandb.Api()
runs = list(api.runs("demonstrator-to-teacher-v2", per_page=200))
print(f"  found {len(runs)} runs")
out = {}
for r in runs:
    tags = set(r.tags)
    obj  = next((o for o in ["sft","fkl","sdpo","pc_sdpo"] if o in tags), None)
    cond = next((c for c in ["cond_xo","cond_xyo_ystart","cond_xyo"] if c in tags), None)
    direction = "wins" if "wins" in tags else ("loses" if "loses" in tags else None)
    if not (obj and cond and direction): continue
    try:
        h = r.history(samples=2000, pandas=True)
        if h.empty: continue
        # extract adv if present
        rec = {"obj": obj, "cond": cond, "direction": direction, "name": r.name}
        for k in ["advantage/mean", "advantage/var", "advantage/std", "kl/T_S", "kl/S_T", "loss", "step"]:
            if k in h.columns:
                rec[k] = h[k].dropna().tolist()
        out[f"{obj}__{direction}_{cond}"] = rec
    except Exception as e:
        print(f"  ! {r.name}: {e}")
os.makedirs("/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/_wandb_pulled", exist_ok=True)
with open("/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/_wandb_pulled/training_data.json", "w") as f:
    json.dump(out, f)
print(f"  pulled {len(out)} runs → _wandb_pulled/training_data.json")
PY

# ─── 4. Run plotter ──────────────────────────────────────────────────────────
echo
echo "── 4. Generate plots ──"
python3 -u $REPO/jobs/babel/make_template_figs.py 2>&1 | tail -15

# ─── 5. Master CSV + per-model summary ───────────────────────────────────────
echo
echo "── 5. Aggregate master.csv + summary readout ──"
python3 -u $REPO/jobs/babel/aggregate_meeting_readout.py 2>&1 | tail -50

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  DONE."
echo
echo "  Plots:      $RES/_plots/"
echo "  Master CSV: $RES/_plots/master.csv"
echo "  Readout:    $RES/_plots/per_model_readout.txt"
echo "═══════════════════════════════════════════════════════════════════"
