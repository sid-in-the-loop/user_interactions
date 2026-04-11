#!/bin/bash
# Test (x, o)-only teacher context for OLMo y* — paper Table 1 template.
# Reuses the 500 sample IDs from the original winrate job.
# Comparisons:
#   1. y*_xo vs y_base  — does removing y fix the 3:1 loss we saw before?
#   2. y*_xo vs y*_old  — is the new template better than (x,y,o)?
#
# Submit:
#   sbatch jobs/eval/sbatch_olmo_xo_ablation.sh

#SBATCH --job-name=olmo_xo_ablation
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo_xo_%j.out
#SBATCH --error=logs/olmo_xo_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

IDS_FILE="datasets/wildfeedback/olmo_3_7b_500/winrate_500_ids.json"
YBASE="datasets/wildfeedback/olmo_3_7b_500/ybase_olmo.jsonl"
YSTAR_OLD="datasets/wildfeedback/olmo_3_7b_500/ystar_olmo.jsonl"
GEN_OUT_DIR="datasets/wildfeedback/olmo_3_7b_500"
WR_XO_VS_YBASE="data/winrate_results/olmo_xo_ablation/xo_vs_ybase"
WR_XO_VS_OLD="data/winrate_results/olmo_xo_ablation/xo_vs_old"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"
mkdir -p logs "$WR_XO_VS_YBASE" "$WR_XO_VS_OLD"

echo "Job ID  : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"
echo "IDs file: $IDS_FILE  ($(python -c "import json; d=json.load(open('$IDS_FILE')); print(len(d))") samples)"
echo "──────────────────────────────────────────"

# Step 1 — generate y*_xo
echo ""
echo "=== Step 1: generate y*_xo (x, o only — paper Table 1 template) ==="
python scripts/eval/generate_olmo.py \
    --input          datasets/wildfeedback/tuples.jsonl \
    --output_dir     "$GEN_OUT_DIR" \
    --target         ystar \
    --teacher-prompt xo \
    --model          "$MODEL" \
    --ids-file       "$IDS_FILE"

YSTAR_XO="$GEN_OUT_DIR/ystar_olmo_xo.jsonl"
echo "Generated: $YSTAR_XO  ($(wc -l < $YSTAR_XO) samples)"

# Step 2 — winrate: y*_xo vs y_base
echo ""
echo "=== Step 2: y*_xo vs y_base ==="
python scripts/eval/winrate_olmo.py \
    --ybase-file "$YBASE" \
    --ystar-file "$YSTAR_XO" \
    --output-dir "$WR_XO_VS_YBASE" \
    --subsample  0

# Step 3 — winrate: y*_xo vs y*_old (pair mode)
echo ""
echo "=== Step 3: y*_xo vs y*_old ==="
python scripts/eval/winrate_eval.py \
    --file-a     "$YSTAR_XO" \
    --file-b     "$YSTAR_OLD" \
    --field-a    y_star \
    --field-b    y_star \
    --label-a    "y*_xo" \
    --label-b    "y*_old" \
    --output-dir "$WR_XO_VS_OLD" \
    --subsample  9999

# Step 4 — print summary
echo ""
echo "=== Step 4: summary ==="
python - "$WR_XO_VS_YBASE/winrate_olmo_results.jsonl" \
         "$WR_XO_VS_OLD/winrate_results.jsonl" \
         "$YSTAR_XO" "$YSTAR_OLD" <<'PYEOF'
import json, math, sys

def wilson_ci(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, center - margin, center + margin

def avg_len_words(path, field):
    total, count = 0, 0
    for line in open(path):
        r = json.loads(line)
        text = r.get(field, "") or ""
        total += len(text.split())
        count += 1
    return total / count if count else 0

xo_vs_ybase_file, xo_vs_old_file, xo_file, old_file = sys.argv[1:]

# y*_xo vs y_base (winrate_olmo_results.jsonl)
recs = [json.loads(l) for l in open(xo_vs_ybase_file) if l.strip()]
cmp  = [r for r in recs if r.get("comparison") == "y* vs y_base"]
w1 = sum(1 for r in cmp if r["outcome"] == "win")
l1 = sum(1 for r in cmp if r["outcome"] == "loss")
t1 = sum(1 for r in cmp if r["outcome"] == "tie")
wr1, lo1, hi1 = wilson_ci(w1, l1)

# y*_xo vs y*_old (winrate_results.jsonl — pair mode, winner=1 means A=xo wins)
recs2 = [json.loads(l) for l in open(xo_vs_old_file) if l.strip()]
comp_name = list(recs2[0]["comparisons"].keys())[0] if recs2 else "y*_xo vs y*_old"
w2 = sum(1 for r in recs2 if r["comparisons"][comp_name]["winner"] == "1")
l2 = sum(1 for r in recs2 if r["comparisons"][comp_name]["winner"] == "2")
t2 = sum(1 for r in recs2 if r["comparisons"][comp_name]["winner"] == "tie")
wr2, lo2, hi2 = wilson_ci(w2, l2)

# Avg lengths
xo_len  = avg_len_words(xo_file,  "y_star")
old_len = avg_len_words(old_file, "y_star")

print("\n" + "="*62)
print("  OLMo (x,o)-only teacher ablation")
print("="*62)
print(f"\n{'Comparison':<30} {'Win Rate':>10}  {'95% CI':>18}  Wins  Losses  Ties")
print("─"*62)
print(f"{'y*_xo vs y_base':<30} {wr1*100:>9.1f}%  [{lo1*100:.1f}, {hi1*100:.1f}]  {w1:>4}  {l1:>6}  {t1:>4}")
print(f"{'y*_xo vs y*_old':<30} {wr2*100:>9.1f}%  [{lo2*100:.1f}, {hi2*100:.1f}]  {w2:>4}  {l2:>6}  {t2:>4}")
print("─"*62)
print(f"\nAvg response length (words):")
print(f"  y*_xo  : {xo_len:.1f}")
print(f"  y*_old : {old_len:.1f}")
print("="*62 + "\n")
PYEOF

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Outputs:"
echo "  $YSTAR_XO"
echo "  $WR_XO_VS_YBASE/winrate_olmo_results.jsonl"
echo "  $WR_XO_VS_OLD/winrate_results.jsonl"
