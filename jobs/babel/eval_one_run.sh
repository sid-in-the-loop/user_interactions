#!/bin/bash
# Evaluate all 5 LoRA ckpts of a single training run on one GPU.
# Chains: minerva_math (272 Qs)  +  alpaca_eval-gen (805 prompts) per ckpt.
# Alpaca judge phase runs separately (CPU + API).
#
# Usage:
#   sbatch jobs/babel/eval_one_run.sh <run_id> <run_ckpt_dir> <model> [chat_template]
# Args:
#   run_id        — e.g. WI-1
#   run_ckpt_dir  — dir containing step-*/ and final/ subdirs from training
#   model         — base HF id, e.g. Qwen/Qwen2.5-Math-7B
#   chat_template — "1" to apply chat template (set for Qwen3-4B); default "0"
#
#SBATCH --job-name=demo2t-eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

RUN_ID="${1:?usage: sbatch ${0##*/} <run_id> <run_ckpt_dir> <model> [chat_template]}"
RUN_CKPT_DIR="${2:?}"
MODEL="${3:?}"
CHAT_TPL="${4:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ssmurali/user_interactions}"
EVAL_REPO="$REPO_ROOT/CFT-Eric-Zhu/evaluation_script"
RESULTS_ROOT="$REPO_ROOT/CFT-Eric-Zhu/eval_results/demo2teacher/$RUN_ID"

mkdir -p logs "$RESULTS_ROOT"

# Find ckpts to eval. Default = "all" (auto-discover step-*/ + final/ in numeric order).
# Override with EVAL_CKPTS=step-N1,step-N2,final to subset.
case "${EVAL_CKPTS:-all}" in
  all|default|"")
    mapfile -t CKPTS < <(
      ls -d "$RUN_CKPT_DIR"/step-* 2>/dev/null \
        | awk -F'step-' '{print $2"\t"$0}' | sort -n -k1 | cut -f2
    )
    [[ -d "$RUN_CKPT_DIR/final" ]] && CKPTS+=("$RUN_CKPT_DIR/final")
    ;;
  *)
    CKPTS=()
    IFS=',' read -ra _picks <<< "$EVAL_CKPTS"
    for p in "${_picks[@]}"; do
      [[ -d "$RUN_CKPT_DIR/$p" ]] && CKPTS+=("$RUN_CKPT_DIR/$p")
    done
    ;;
esac

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "ERROR: no ckpts under $RUN_CKPT_DIR (looked for step-*/ and final/)"
  exit 1
fi

echo "════════════════════════════════════════"
echo "  eval run: $RUN_ID"
echo "  base    : $MODEL"
echo "  ckpt dir: $RUN_CKPT_DIR"
echo "  ckpts   : ${#CKPTS[@]}"
for c in "${CKPTS[@]}"; do echo "    - $(basename "$c")"; done
echo "  results : $RESULTS_ROOT"
echo "  chat tpl: $CHAT_TPL"
echo "════════════════════════════════════════"

CHAT_FLAG=()
[[ "$CHAT_TPL" == "1" ]] && CHAT_FLAG=(--apply_chat_template)

# Seed + temperature for math (defaults: greedy temp=0; for 3-seed runs set MATH_TEMP=0.7)
SEED="${SEED:-0}"
MATH_TEMP="${MATH_TEMP:-0}"
ALPACA_TEMP="${ALPACA_TEMP:-1.0}"
ALPACA_TOP_P="${ALPACA_TOP_P:-1.0}"
SEED_TAG="${SEED_TAG:-}"  # appended to per-ckpt out-dir names if set

# ---- per ckpt: minerva, then alpaca-gen ----
for CKPT in "${CKPTS[@]}"; do
  CKPT_NAME="$(basename "$CKPT")"
  OUT="$RESULTS_ROOT/$CKPT_NAME${SEED_TAG}"
  mkdir -p "$OUT"

  echo
  echo "── $RUN_ID :: $CKPT_NAME ──"

  # ---- 1. Math benchmark (default math-500; override with MATH_DATA env var) ----
  MATH_DATA="${MATH_DATA:-math-500}"
  if [[ -f "$OUT/math_done" ]]; then
    echo "[skip] math ($MATH_DATA) (already done)"
  else
    echo "[run] math: $MATH_DATA …"
    pushd "$EVAL_REPO/evaluate_math" > /dev/null
    python3 -u math_eval.py \
        --model_name_or_path "$MODEL" \
        --lora_path "$CKPT" \
        --data_name "$MATH_DATA" \
        --data_dir "./data" \
        --output_dir "$OUT/math" \
        --summary_path "$OUT/math/summary.txt" \
        --split test \
        --prompt_type "qwen25-math-cot" \
        --num_test_sample -1 \
        --seed "$SEED" --temperature "$MATH_TEMP" --n_sampling 1 --top_p 1 \
        --start 0 --end -1 \
        --use_vllm --save_outputs \
        "${CHAT_FLAG[@]}"
    popd > /dev/null
    touch "$OUT/math_done"
  fi

  # ---- 2. Alpaca-eval generation ----
  # Set ALPACA_FINAL_ONLY=1 to skip alpaca on intermediate (step-*) ckpts.
  if [[ "${ALPACA_FINAL_ONLY:-0}" == "1" && "$CKPT_NAME" != "final" ]]; then
    echo "[skip] alpaca (ALPACA_FINAL_ONLY=1, ckpt is $CKPT_NAME)"
    continue
  fi
  if [[ -f "$OUT/alpaca_done" ]]; then
    echo "[skip] alpaca (already done)"
  else
    echo "[run] alpaca_eval (gen only) …"
    python3 -u "$EVAL_REPO/evaluate_alpaca/alpaca_eval.py" \
        --model_name_or_path "$MODEL" \
        --lora_path "$CKPT" \
        --output_path "$OUT/alpaca/outputs.jsonl" \
        --generator_name "${RUN_ID}_${CKPT_NAME}${SEED_TAG}" \
        --seed "$SEED" \
        --max_tokens 2048 --temperature "$ALPACA_TEMP" --top_p "$ALPACA_TOP_P" \
        "${CHAT_FLAG[@]}"
    touch "$OUT/alpaca_done"
  fi
done

echo
echo "Done with $RUN_ID. Results in $RESULTS_ROOT"
