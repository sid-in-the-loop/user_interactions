#!/bin/bash
# Smoke test: minerva (20 Qs) + alpaca-gen (5 prompts) on one final/ ckpt.
# Prints minerva accuracy and first 2 alpaca outputs.
# Runs in foreground — expect ~3–5 min.

set -euo pipefail

RUN_DIR="${1:-/data/group_data/cx_group/ssmurali/demo2teacher/WI-1_sft_teacher_wins_cond_xo}"
MODEL="${2:-Qwen/Qwen2.5-Math-7B}"
CKPT="$RUN_DIR/final"
OUT="/tmp/smoke_eval_$$"

[[ -d "$CKPT" ]] || { echo "ERROR: $CKPT not found"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/home/ssmurali/user_interactions
mkdir -p "$OUT"

echo "════════════ SMOKE TEST ════════════"
echo "  run dir : $RUN_DIR"
echo "  ckpt    : $CKPT"
echo "  model   : $MODEL"
echo "  out     : $OUT"
echo "═════════════════════════════════════"

echo
echo "── 1/2  minerva_math (20 Qs) ──"
pushd "$REPO/CFT-Eric-Zhu/evaluation_script/evaluate_math" > /dev/null
python3 -u math_eval.py \
  --model_name_or_path "$MODEL" \
  --lora_path "$CKPT" \
  --data_name minerva_math \
  --data_dir ./data \
  --output_dir "$OUT/math" \
  --summary_path "$OUT/math/summary.txt" \
  --split test \
  --prompt_type qwen25-math-cot \
  --num_test_sample 20 \
  --seed 0 --temperature 0 --n_sampling 1 --top_p 1 --start 0 --end -1 \
  --use_vllm --save_outputs 2>&1 | tail -20
popd > /dev/null

echo
echo "── 2/2  alpaca_eval gen (5 prompts) ──"
python3 -u "$REPO/CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/alpaca_eval.py" \
  --model_name_or_path "$MODEL" \
  --lora_path "$CKPT" \
  --output_path "$OUT/alpaca/outputs.jsonl" \
  --generator_name "smoke" \
  --n_prompts 5 \
  --max_tokens 1024 --temperature 0.7 --top_p 0.9 2>&1 | tail -10

echo
echo "── 3/3  alpaca judge (5 prompts via gpt-4o-mini async) ──"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  python3 -u "$REPO/CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/judge_alpaca_async.py" \
    --inputs "$OUT/alpaca/outputs.jsonl" \
    --output_dir "$OUT/alpaca_judge" \
    --concurrency 10 2>&1 | tail -20
else
  echo "[skip] OPENAI_API_KEY not set — judge step skipped"
fi

echo
echo "════════════ RESULTS ════════════"
echo
echo "=== minerva_math summary ==="
cat "$OUT/math/summary.txt" 2>/dev/null || echo "(no summary file)"
echo
echo "=== alpaca outputs (first 2 of 5) ==="
python3 -c "
import json
with open('$OUT/alpaca/outputs.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 2: break
        d = json.loads(line)
        print(f'\n[{i}] PROMPT: {d[\"instruction\"][:140]}')
        print(f'    OUTPUT: {d[\"output\"][:240]}')
"
echo
echo "=== alpaca judge summary ==="
cat "$OUT/alpaca_judge/summary.json" 2>/dev/null || echo "(no judge summary — run with OPENAI_API_KEY)"
echo
echo "✓ smoke complete. cleanup: rm -rf $OUT"
