#!/bin/bash
# Generate y_base + y* for 500 WildFeedback samples on the fly, then run winrate.
# Does NOT require the full generation job to have run first.
#
# Requires OPENAI_API_KEY to be set in your environment before submitting:
#   export OPENAI_API_KEY=sk-...
#   sbatch jobs/eval/sbatch_olmo_winrate_500.sh
#
# Outputs:
#   datasets/wildfeedback/olmo_3_7b_500/ybase_olmo.jsonl
#   datasets/wildfeedback/olmo_3_7b_500/ystar_olmo.jsonl
#   data/winrate_results/olmo_wf_500/winrate_olmo_results.jsonl
#   data/winrate_results/olmo_wf_500/winrate_olmo_summary.txt
#   data/winrate_results/olmo_wf_500/winrate_olmo.png

#SBATCH --job-name=olmo_winrate_500
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo_winrate_500_%j.out
#SBATCH --error=logs/olmo_winrate_500_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO"
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "error: OPENAI_API_KEY not set — set it before sbatch" >&2
    exit 1
fi

INPUT="datasets/wildfeedback/tuples.jsonl"
GEN_DIR="datasets/wildfeedback/olmo_3_7b_500"
OUT_DIR="data/winrate_results/olmo_wf_500"
IDS_FILE="$GEN_DIR/winrate_500_ids.json"
MODEL="allenai/OLMo-3-7B-Instruct-SFT"
mkdir -p logs "$GEN_DIR" "$OUT_DIR"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Model      : $MODEL"
echo "Started    : $(date)"
echo "──────────────────────────────────────────"

# Step 0: sample 500 IDs (seed 42, reproducible)
echo ""
echo "=== Step 0: sample 500 IDs ==="
python - <<'EOF'
import json, random
random.seed(42)
data = [json.loads(l) for l in open("datasets/wildfeedback/tuples.jsonl")]
sample = random.sample(data, 500)
ids = [{"conversation_id": r["conversation_id"], "turn_index": r["turn_index"]} for r in sample]
import os; os.makedirs("datasets/wildfeedback/olmo_3_7b_500", exist_ok=True)
with open("datasets/wildfeedback/olmo_3_7b_500/winrate_500_ids.json", "w") as f:
    json.dump(ids, f)
print(f"Sampled {len(ids)} IDs → datasets/wildfeedback/olmo_3_7b_500/winrate_500_ids.json")
EOF

# Step 1: generate y_base for 500 samples
echo ""
echo "=== Step 1: y_base (500 samples) ==="
python scripts/eval/generate_olmo.py \
    --target       ybase \
    --input        "$INPUT" \
    --output_dir   "$GEN_DIR" \
    --ids-file     "$IDS_FILE" \
    --model        "$MODEL" \
    --tp_size      1 \
    --max_num_seqs 512 \
    --gpu_util     0.92 \
    --max_tokens   1024 \
    --max_model_len 4096

# Step 2: generate y* for 500 samples
echo ""
echo "=== Step 2: y* (500 samples) ==="
python scripts/eval/generate_olmo.py \
    --target       ystar \
    --input        "$INPUT" \
    --output_dir   "$GEN_DIR" \
    --ids-file     "$IDS_FILE" \
    --model        "$MODEL" \
    --tp_size      1 \
    --max_num_seqs 512 \
    --gpu_util     0.92 \
    --max_tokens   1024 \
    --max_model_len 4096

# Step 3: winrate eval (all 3 comparisons, GPT-4o-mini judge)
echo ""
echo "=== Step 3: winrate eval ==="
python scripts/eval/winrate_olmo.py \
    --ybase-file   "$GEN_DIR/ybase_olmo.jsonl" \
    --ystar-file   "$GEN_DIR/ystar_olmo.jsonl" \
    --output-dir   "$OUT_DIR" \
    --subsample    0 \
    --seed         42 \
    --max-concurrent 500

# Step 4: plot
echo ""
echo "=== Step 4: plot ==="
python scripts/eval/plot_olmo_winrate.py \
    --results "$OUT_DIR/winrate_olmo_results.jsonl" \
    --output  "$OUT_DIR/winrate_olmo.png"

echo ""
echo "──────────────────────────────────────────"
echo "Done: $(date)"
echo "Results : $OUT_DIR/winrate_olmo_summary.txt"
echo "Chart   : $OUT_DIR/winrate_olmo.png"
