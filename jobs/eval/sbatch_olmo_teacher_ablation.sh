#!/bin/bash
# OLMo teacher-prompt ablation — test 4 (x,o)-family variants on 500 samples.
#
# Prompts:
#   A — paper xo template (already generated; reused)
#   B — explicit revision instruction
#   C — minimal "Note: o"
#   D — Socratic (reason before answering)
#
# Winrate comparisons (GPT-4o-mini, position-bias removed):
#   1. y*_A vs y_base      (reference; re-run for consistency)
#   2. y*_B vs y_base
#   3. y*_C vs y_base
#   4. y*_D vs y_base
#   5. y*_B vs y*_A
#   6. y*_C vs y*_A
#   7. y*_D vs y*_A
#
# Submit:
#   sbatch jobs/eval/sbatch_olmo_teacher_ablation.sh

#SBATCH --job-name=olmo_teacher_abl
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/olmo_teacher_abl_%j.out
#SBATCH --error=logs/olmo_teacher_abl_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

IDS_FILE="datasets/wildfeedback/olmo_3_7b_500/winrate_500_ids.json"
GEN_DIR="datasets/wildfeedback/olmo_3_7b_500"
WR_DIR="data/winrate_results/olmo_teacher_ablation"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"

YBASE="$GEN_DIR/ybase_olmo.jsonl"
YSTAR_A="$GEN_DIR/ystar_olmo_xo.jsonl"     # already exists
YSTAR_B="$GEN_DIR/ystar_olmo_xo_B.jsonl"
YSTAR_C="$GEN_DIR/ystar_olmo_xo_C.jsonl"
YSTAR_D="$GEN_DIR/ystar_olmo_xo_D.jsonl"

mkdir -p logs "$WR_DIR"

echo "Job ID  : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"
echo "IDs     : $IDS_FILE  ($(python -c "import json; print(len(json.load(open('$IDS_FILE'))))" ) samples)"
echo "──────────────────────────────────────────"

# ── Generation ─────────────────────────────────────────────────────────────────
# Prompt A is already generated. Generate B, C, D sequentially (one vLLM per pass).

for VARIANT in xo_B xo_C xo_D; do
    echo ""
    echo "=== Generating y* prompt=${VARIANT} ==="
    python scripts/eval/generate_olmo.py \
        --input          datasets/wildfeedback/tuples.jsonl \
        --output_dir     "$GEN_DIR" \
        --target         ystar \
        --teacher-prompt "$VARIANT" \
        --model          "$MODEL" \
        --ids-file       "$IDS_FILE"
done

echo ""
echo "Generation done.  $(date)"
echo "  A : $YSTAR_A  ($(wc -l < $YSTAR_A) samples)"
echo "  B : $YSTAR_B  ($(wc -l < $YSTAR_B) samples)"
echo "  C : $YSTAR_C  ($(wc -l < $YSTAR_C) samples)"
echo "  D : $YSTAR_D  ($(wc -l < $YSTAR_D) samples)"

# ── Winrate comparisons ────────────────────────────────────────────────────────
# All 7 use winrate_eval.py pair mode → winrate_results.jsonl per subdirectory.
# winner="1" → A wins (first file), "2" → B wins (second file).

run_winrate() {
    local FILE_A="$1" FILE_B="$2" FIELD_A="$3" FIELD_B="$4"
    local LABEL_A="$5" LABEL_B="$6" OUT_SUBDIR="$7"
    echo ""
    echo "=== Winrate: ${LABEL_A} vs ${LABEL_B} ==="
    mkdir -p "$WR_DIR/$OUT_SUBDIR"
    python scripts/eval/winrate_eval.py \
        --file-a   "$FILE_A" \
        --file-b   "$FILE_B" \
        --field-a  "$FIELD_A" \
        --field-b  "$FIELD_B" \
        --label-a  "$LABEL_A" \
        --label-b  "$LABEL_B" \
        --output-dir "$WR_DIR/$OUT_SUBDIR" \
        --subsample 9999
}

# vs y_base (field = y_base in the ybase file)
run_winrate "$YSTAR_A" "$YBASE"  y_star y_base  "y*_A"  "y_base"  "ystar_A_vs_ybase"
run_winrate "$YSTAR_B" "$YBASE"  y_star y_base  "y*_B"  "y_base"  "ystar_B_vs_ybase"
run_winrate "$YSTAR_C" "$YBASE"  y_star y_base  "y*_C"  "y_base"  "ystar_C_vs_ybase"
run_winrate "$YSTAR_D" "$YBASE"  y_star y_base  "y*_D"  "y_base"  "ystar_D_vs_ybase"

# vs y*_A
run_winrate "$YSTAR_B" "$YSTAR_A" y_star y_star "y*_B" "y*_A" "ystar_B_vs_A"
run_winrate "$YSTAR_C" "$YSTAR_A" y_star y_star "y*_C" "y*_A" "ystar_C_vs_A"
run_winrate "$YSTAR_D" "$YSTAR_A" y_star y_star "y*_D" "y*_A" "ystar_D_vs_A"

echo ""
echo "Winrate comparisons done.  $(date)"

# ── Plot ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Plotting ==="
python scripts/eval/plot_teacher_ablation.py \
    --results-dir "$WR_DIR" \
    --output      "$WR_DIR/teacher_ablation.png"

# ── Summary table ──────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
python - "$WR_DIR" "$YSTAR_A" "$YSTAR_B" "$YSTAR_C" "$YSTAR_D" <<'PYEOF'
import json, math, sys
from pathlib import Path

def wilson_ci(w, l):
    n = w + l
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = w / n; z = 1.96
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, center - margin, center + margin

def load_pair(path, comp):
    w = l = t = 0
    for line in open(path):
        r = json.loads(line)
        v = r.get("comparisons", {}).get(comp, {}).get("winner", "")
        if v == "1": w += 1
        elif v == "2": l += 1
        elif v == "tie": t += 1
    return w, l, t

def avg_words(path, field):
    total = count = 0
    for line in open(path):
        r = json.loads(line)
        total += len((r.get(field) or "").split())
        count += 1
    return total / count if count else 0

wr_dir = Path(sys.argv[1])
files = {p: f for p, f in zip("ABCD", sys.argv[2:])}

print()
print("=" * 72)
print("  OLMo teacher-prompt ablation — WildFeedback 500 samples")
print("=" * 72)
print(f"\n{'Prompt':<8} {'vs y_base WR':>14}  {'95% CI':>18}  {'W':>5}  {'L':>5}  {'T':>5}")
print("─" * 62)

best_p, best_wr = None, -1.0
for p in "ABCD":
    subdir = f"ystar_{p}_vs_ybase"
    comp   = f"y*_{p} vs y_base"
    f = wr_dir / subdir / "winrate_results.jsonl"
    w, l, t = load_pair(f, comp)
    wr, lo, hi = wilson_ci(w, l)
    marker = " ◀ BEST" if p == "A" else ""  # placeholder
    print(f"  {p}      {wr*100:>12.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  {w:>5}  {l:>5}  {t:>5}")
    if not math.isnan(wr) and wr > best_wr:
        best_wr = wr; best_p = p

print()
print(f"\n{'Prompt':<8} {'vs y*_A WR':>14}  {'95% CI':>18}  {'W':>5}  {'L':>5}  {'T':>5}")
print("─" * 62)
for p in "BCD":
    subdir = f"ystar_{p}_vs_A"
    comp   = f"y*_{p} vs y*_A"
    f = wr_dir / subdir / "winrate_results.jsonl"
    w, l, t = load_pair(f, comp)
    wr, lo, hi = wilson_ci(w, l)
    print(f"  {p}      {wr*100:>12.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  {w:>5}  {l:>5}  {t:>5}")

print()
print("─" * 62)
print(f"\nAvg response length (words):")
for p, fpath in files.items():
    field = "y_star"
    print(f"  Prompt {p}: {avg_words(fpath, field):.1f}")

print()
print(f"  ★  WINNER (highest y* vs y_base): Prompt {best_p}  ({best_wr*100:.1f}%)")
print("=" * 72)
PYEOF

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Chart: $WR_DIR/teacher_ablation.png"
