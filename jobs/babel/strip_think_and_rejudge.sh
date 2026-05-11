#!/bin/bash
# Post-hoc fix: strip <think>...</think> blocks from existing alpaca outputs,
# then re-judge against gpt-4-turbo. No re-gen needed.
#
#SBATCH --job-name=demo2t-strip-judge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/strip_judge_%j.out
#SBATCH --error=logs/strip_judge_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY required" >&2; exit 1
fi

REPO=/home/ssmurali/user_interactions
RES=$REPO/CFT-Eric-Zhu/eval_results
RAW_DIR=$RES/demo2teacher
STRIPPED_DIR=$RES/demo2teacher_stripped

cd $REPO

# Phase 1 — strip think blocks
echo "── stripping <think>...</think> from alpaca outputs ──"
python3 - << 'PY'
import json, os, re, glob, shutil
SRC = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/demo2teacher"
DST = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/demo2teacher_stripped"

def strip_think(text):
    if not isinstance(text, str): return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = text.split("<think>")[0].strip()
    return text

n_files = 0; n_with_think = 0
for path in glob.glob(f"{SRC}/WC-*/*/alpaca/outputs.jsonl"):
    rel = os.path.relpath(path, SRC)
    out_path = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = [json.loads(l) for l in open(path)]
    has_think = any("<think>" in r.get("output","") for r in rows)
    if has_think: n_with_think += 1
    new_rows = []
    for r in rows:
        r = dict(r)
        r["output"] = strip_think(r.get("output",""))
        new_rows.append(r)
    with open(out_path, "w") as f:
        for r in new_rows: f.write(json.dumps(r) + "\n")
    n_files += 1
print(f"  stripped {n_files} files ({n_with_think} had <think>)")
PY

# Phase 2 — judge stripped outputs
echo
echo "── judging stripped alpaca outputs vs gpt-4-1106-preview ──"
mapfile -t INPUTS < <(find "$STRIPPED_DIR" -path '*/alpaca/outputs.jsonl' | sort)
echo "  ${#INPUTS[@]} files to judge"

python3 -u $REPO/CFT-Eric-Zhu/evaluation_script/evaluate_alpaca/judge_alpaca_async.py \
  --inputs "${INPUTS[@]}" \
  --output_dir "$RES/demo2teacher/_alpaca_judge_stripped" \
  --reference_config alpaca_eval_gpt4_baseline \
  --concurrency 200

echo
echo "── done. Results in $RES/demo2teacher/_alpaca_judge_stripped/ ──"
ls "$RES/demo2teacher/_alpaca_judge_stripped/" | head
