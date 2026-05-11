#!/bin/bash
# 3-seed minerva eval at temp=0.7 on a chosen ckpt of one run.
# Shares one base+LoRA load across the 3 seeds (only sampling differs).
#
# Usage:
#   sbatch jobs/babel/eval_seeded.sh <run_id> <run_ckpt_dir> <model> <ckpt_name>
#   ckpt_name = step-182 | step-91 | final | etc.
#
#SBATCH --job-name=demo2t-seeded
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/seeded_%j.out
#SBATCH --error=logs/seeded_%j.err

set -euo pipefail

RUN_ID="${1:?usage: sbatch ${0##*/} <run_id> <run_ckpt_dir> <model> <ckpt_name>}"
RUN_CKPT_DIR="${2:?}"
MODEL="${3:?}"
CKPT_NAME="${4:-step-182}"

CKPT="$RUN_CKPT_DIR/$CKPT_NAME"
[[ -d "$CKPT" ]] || { echo "ERROR: $CKPT not found"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/home/ssmurali/user_interactions
EVAL_REPO="$REPO/CFT-Eric-Zhu/evaluation_script"
RESULTS="$REPO/CFT-Eric-Zhu/eval_results/demo2teacher/$RUN_ID/$CKPT_NAME/math_seeded"
mkdir -p logs "$RESULTS"

echo "════════════════════════════════════════"
echo "  seeded minerva eval (temp=0.7, 3 seeds)"
echo "  run_id : $RUN_ID"
echo "  ckpt   : $CKPT_NAME"
echo "  model  : $MODEL"
echo "  out    : $RESULTS"
echo "════════════════════════════════════════"

cd "$EVAL_REPO/evaluate_math"
for SEED in 0 1 2; do
  OUT_SEED="$RESULTS/seed${SEED}"
  if [[ -f "$OUT_SEED/done" ]]; then
    echo "[skip] seed=$SEED already done"
    continue
  fi
  mkdir -p "$OUT_SEED"
  echo
  echo "── seed=$SEED ──"
  python3 -u math_eval.py \
    --model_name_or_path "$MODEL" \
    --lora_path "$CKPT" \
    --data_name minerva_math \
    --data_dir ./data \
    --output_dir "$OUT_SEED" \
    --summary_path "$OUT_SEED/summary.txt" \
    --split test --prompt_type qwen25-math-cot \
    --num_test_sample -1 \
    --max_tokens_per_call 2048 \
    --seed "$SEED" --temperature 0.7 --n_sampling 1 --top_p 0.95 \
    --start 0 --end -1 \
    --use_vllm --save_outputs
  touch "$OUT_SEED/done"
done

echo
echo "── seeded scores ──"
for s in 0 1 2; do
  acc=$(grep -oE "Final Accuracy:\s*[0-9.]+" "$RESULTS/seed$s/summary.txt" 2>/dev/null | grep -oE "[0-9.]+")
  echo "  seed=$s  acc=${acc:-?}"
done

# write a single aggregate JSON
python3 -c "
import json, os, re
results = {}
for s in [0,1,2]:
    p = '$RESULTS/seed%d/summary.txt' % s
    if os.path.exists(p):
        m = re.search(r'Final Accuracy:\s*([0-9.]+)', open(p).read())
        if m: results[f'seed{s}'] = float(m.group(1))
vals = list(results.values())
out = {'run_id': '$RUN_ID', 'ckpt': '$CKPT_NAME', 'seeds': results}
if vals:
    out['mean'] = sum(vals)/len(vals)
    out['min']  = min(vals)
    out['max']  = max(vals)
    out['range'] = max(vals) - min(vals)
with open('$RESULTS/aggregate.json','w') as f:
    json.dump(out, f, indent=2)
print('aggregate:', json.dumps(out, indent=2))
"
echo "Done: $RUN_ID @ $CKPT_NAME"
