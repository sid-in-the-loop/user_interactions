#!/bin/bash
# Prompt ablation for OLMo y* teacher: test prompts A and B on all 477 samples.
# Reuses y_base from the existing 500-sample run.
#
# Steps:
#   0. Extract all IDs from the existing ybase file
#   1. Generate ystar_A on all 477 samples (prompt A)
#   2. Generate ystar_B on all 477 samples (prompt B)
#   3. Winrate: prompt A y* vs y_base
#   4. Winrate: prompt B y* vs y_base
#   5. Winrate: prompt A y* vs prompt B y* (pair mode)
#
# Requires OPENAI_API_KEY set before submitting:
#   export OPENAI_API_KEY=sk-...
#   sbatch jobs/eval/sbatch_olmo_prompt_ablation.sh

#SBATCH --job-name=olmo_prompt_ablation
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo_prompt_ablation_%j.out
#SBATCH --error=logs/olmo_prompt_ablation_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "error: OPENAI_API_KEY not set" >&2; exit 1
fi

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT="datasets/wildfeedback/tuples.jsonl"
EXISTING_YBASE="datasets/wildfeedback/olmo_3_7b_500/ybase_olmo.jsonl"
GEN_DIR="datasets/wildfeedback/olmo_3_7b_ablation"
OUT_DIR="data/winrate_results/olmo_prompt_ablation"
IDS_FILE="$GEN_DIR/ablation_50_ids.json"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"
mkdir -p logs "$GEN_DIR" "$OUT_DIR"

echo "Job ID  : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Started: $(date)"
echo "──────────────────────────────────────────"

# ── Step 0: extract all IDs from the existing ybase file ──────────────────────
echo ""
echo "=== Step 0: extract all ybase IDs ==="
python - <<'EOF'
import json, os
data = [json.loads(l) for l in open("datasets/wildfeedback/olmo_3_7b_500/ybase_olmo.jsonl")]
ids = [{"conversation_id": r["conversation_id"], "turn_index": r["turn_index"]} for r in data]
os.makedirs("datasets/wildfeedback/olmo_3_7b_ablation", exist_ok=True)
with open("datasets/wildfeedback/olmo_3_7b_ablation/ablation_50_ids.json", "w") as f:
    json.dump(ids, f)
print(f"Extracted {len(ids)} IDs")
EOF

# Copy y_base slice for these 50 IDs (winrate_olmo.py joins on id)
# (nothing needed — winrate_olmo.py will join against the full ybase file)

VLLM_ARGS="--model $MODEL --tp_size 1 --max_num_seqs 512 --gpu_util 0.92 --max_tokens 1024 --max_model_len 4096 --ids-file $IDS_FILE --target ystar --input $INPUT"

# ── Step 1: generate y* with prompt A ─────────────────────────────────────────
echo ""
echo "=== Step 1: y* prompt A ==="
python scripts/eval/generate_olmo.py \
    $VLLM_ARGS \
    --teacher-prompt A \
    --output_dir "$GEN_DIR"

# ── Step 2: generate y* with prompt B ─────────────────────────────────────────
echo ""
echo "=== Step 2: y* prompt B ==="
python scripts/eval/generate_olmo.py \
    $VLLM_ARGS \
    --teacher-prompt B \
    --output_dir "$GEN_DIR"

# ── Step 3: winrate — prompt A y* vs y_base ───────────────────────────────────
echo ""
echo "=== Step 3: winrate — prompt A vs y_base ==="
python scripts/eval/winrate_olmo.py \
    --ybase-file  "$EXISTING_YBASE" \
    --ystar-file  "$GEN_DIR/ystar_olmo_A.jsonl" \
    --output-dir  "$OUT_DIR/prompt_A_vs_ybase" \
    --subsample   0 \
    --seed        42 \
    --max-concurrent 200

# ── Step 4: winrate — prompt B y* vs y_base ───────────────────────────────────
echo ""
echo "=== Step 4: winrate — prompt B vs y_base ==="
python scripts/eval/winrate_olmo.py \
    --ybase-file  "$EXISTING_YBASE" \
    --ystar-file  "$GEN_DIR/ystar_olmo_B.jsonl" \
    --output-dir  "$OUT_DIR/prompt_B_vs_ybase" \
    --subsample   0 \
    --seed        42 \
    --max-concurrent 200

# ── Step 5: winrate — prompt A vs prompt B (pair mode) ────────────────────────
echo ""
echo "=== Step 5: winrate — prompt A vs prompt B ==="
python scripts/eval/winrate_eval.py \
    --file-a      "$GEN_DIR/ystar_olmo_A.jsonl" \
    --file-b      "$GEN_DIR/ystar_olmo_B.jsonl" \
    --field-a     y_star \
    --field-b     y_star \
    --label-a     "prompt_A" \
    --label-b     "prompt_B" \
    --output-dir  "$OUT_DIR/A_vs_B" \
    --subsample   9999 \
    --seed        42 \
    --max-concurrent 200

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  ABLATION SUMMARY"
echo "══════════════════════════════════════════"
echo ""
echo "--- Prompt A y* vs y_base ---"
cat "$OUT_DIR/prompt_A_vs_ybase/winrate_olmo_summary.txt"
echo ""
echo "--- Prompt B y* vs y_base ---"
cat "$OUT_DIR/prompt_B_vs_ybase/winrate_olmo_summary.txt"
echo ""
echo "--- Prompt A vs Prompt B ---"
cat "$OUT_DIR/A_vs_B/winrate_summary.txt"
echo ""
echo "Done: $(date)"
