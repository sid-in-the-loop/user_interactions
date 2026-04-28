#!/bin/bash
# Full per-run eval pipeline: generate → judge (gpt-4o-mini) → aggregate → print.
# One sbatch per run_dir. OPENAI_API_KEY must be in the submitting shell.
#
# Usage:
#   sbatch jobs/deltaai/tac_eval_gen.sh RUN_DIR

#SBATCH --account=bgtw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=tac_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/tac_eval_%x_%j.out
#SBATCH --error=logs/tac_eval_%x_%j.err

set -euo pipefail

RUN_DIR="${1:?Usage: sbatch tac_eval_gen.sh RUN_DIR}"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: OPENAI_API_KEY not propagated (submit with key set in shell)" >&2; exit 1; }

module load python/miniforge3_pytorch/2.10.0
conda activate skywork

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export HF_HOME=/work/hdd/bgtw/ssredharan/models
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RUN_BASE="/work/hdd/bgtw/ssredharan/checkpoints/tac_winrates/${RUN_DIR}"
OUT_BASE="${REPO}/eval_results/tac/${RUN_DIR}"
[ -d "$RUN_BASE" ] || { echo "ERROR: $RUN_BASE missing" >&2; exit 1; }

echo "============ tac_eval: $RUN_DIR ============"
echo "ckpts   : $RUN_BASE"
echo "results : $OUT_BASE"
for method in sft fkl jsd dpo rkl; do
  md="$RUN_BASE/$method"
  [ -d "$md" ] || { printf "  %-3s  --\n" "$method"; continue; }
  n=$(find "$md" -maxdepth 2 -name adapter_config.json 2>/dev/null | wc -l)
  printf "  %-3s  %d adapter(s)\n" "$method" "$n"
done
echo "============================================"

# ── 1/3 GENERATION (GPU, vLLM) ───────────────────────────────────────────────
for method in sft fkl jsd dpo rkl; do
  md="$RUN_BASE/$method"
  [ -d "$md" ] || continue
  n=$(find "$md" -maxdepth 2 -name adapter_config.json 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || continue

  OUT_ROOT="${OUT_BASE}/${method}"
  mkdir -p "$OUT_ROOT"
  echo "=== [$(date +%H:%M)] GEN: $RUN_DIR / $method ($n ckpts) ==="
  python scripts/eval/generate_all.py \
      --method_dir    "$md" \
      --output_root   "$OUT_ROOT" \
      --base_model    "Qwen/Qwen3-4B" \
      --benchmarks    alpaca_eval arena_hard writingbench aime \
      --max_num_seqs  512 \
      --max_model_len 16384
done

# ── 2/3 JUDGING (gpt-4o-mini + programmatic aime) ────────────────────────────
echo "=== [$(date +%H:%M)] JUDGE: $RUN_DIR ==="
python scripts/eval/judge_all.py \
    --results_root "$OUT_BASE" \
    --benchmarks   alpaca_eval arena_hard writingbench aime

# ── 3/3 AGGREGATE + PRINT ────────────────────────────────────────────────────
echo "=== [$(date +%H:%M)] AGGREGATE: $RUN_DIR ==="
python scripts/eval/aggregate_scores.py \
    --results_root "$OUT_BASE" \
    --output       "$OUT_BASE/aggregate.csv" \
    --plots_dir    "${REPO}/plots/tac_eval/${RUN_DIR}"

echo
echo "################ SUMMARY: $RUN_DIR ################"
python - <<PY
import csv, collections, pathlib
p = pathlib.Path("${OUT_BASE}/aggregate.csv")
if not p.exists():
    print("(no aggregate.csv produced)"); raise SystemExit
rows = list(csv.DictReader(open(p)))
# Pick "final" if present per (method, benchmark); else best step.
by = collections.defaultdict(dict)  # method -> benchmark -> metric
for r in rows:
    m, b, s, v = r["method"], r["benchmark"], r["step_name"], float(r["metric"])
    if b not in by[m] or s == "final":
        by[m][b] = (s, v)
benches = ["alpaca_eval", "arena_hard", "writingbench", "aime"]
methods = sorted(by.keys())
print()
print("| method |", " | ".join(benches), "|")
print("|" + "---|" * (len(benches)+1))
for m in methods:
    cells = []
    for b in benches:
        if b in by[m]:
            s, v = by[m][b]
            cells.append(f"{v:.1f} ({s})")
        else:
            cells.append("—")
    print(f"| {m} | " + " | ".join(cells) + " |")
PY

echo
echo "=== [$(date +%H:%M)] DONE: $RUN_DIR ==="
