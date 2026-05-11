#!/bin/bash
# All-in-one eval for one WC run:
#   Phase A — math (amc23, aime24) at temp=0.7, 3 seeds (0,1,2), all ckpts
#   Phase B — alpaca-gen at temp=1.0 enable_thinking=False, single seed, all ckpts
#   Phase C — KL(π_ckpt ∥ π_base) on a 80-prompt sample, all ckpts
#
# Usage: sbatch eval_all_one_run.sh <run_id> <run_ckpt_dir> <model> <chat_tpl 0|1>
# Env overrides: SEEDS=0,1,2  EVAL_CKPTS=all  MATH_DATA="amc23,aime24"
#                MATH_TEMP=0.7  ALPACA_TEMP=1.0
#
#SBATCH --job-name=demo2t-full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

set -euo pipefail

RUN_ID="${1:?usage: sbatch ${0##*/} <run_id> <run_ckpt_dir> <model> <chat_tpl>}"
RUN_CKPT_DIR="${2:?}"
MODEL="${3:?}"
CHAT_TPL="${4:-1}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/home/ssmurali/user_interactions
EVAL_REPO="$REPO/CFT-Eric-Zhu/evaluation_script"
RESULTS_ROOT="$REPO/CFT-Eric-Zhu/eval_results/demo2teacher/$RUN_ID"
mkdir -p logs "$RESULTS_ROOT"

SEEDS_CSV="${SEEDS:-0,1,2}"
MATH_DATA="${MATH_DATA:-amc23,aime24}"
MATH_TEMP="${MATH_TEMP:-0.7}"
ALPACA_TEMP="${ALPACA_TEMP:-1.0}"
ALPACA_TOP_P="${ALPACA_TOP_P:-1.0}"

CHAT_FLAG=()
[[ "$CHAT_TPL" == "1" ]] && CHAT_FLAG=(--apply_chat_template)

# Discover ckpts
case "${EVAL_CKPTS:-all}" in
  all|"")
    mapfile -t CKPTS < <(
      ls -d "$RUN_CKPT_DIR"/step-* 2>/dev/null \
        | awk -F'step-' '{print $2"\t"$0}' | sort -n -k1 | cut -f2
    )
    [[ -d "$RUN_CKPT_DIR/final" ]] && CKPTS+=("$RUN_CKPT_DIR/final")
    ;;
  *) IFS=',' read -ra _picks <<< "$EVAL_CKPTS"
     CKPTS=()
     for p in "${_picks[@]}"; do
       [[ -d "$RUN_CKPT_DIR/$p" ]] && CKPTS+=("$RUN_CKPT_DIR/$p")
     done
     ;;
esac

echo "════════════════════════════════════════"
echo "  FULL eval: $RUN_ID"
echo "  base : $MODEL  | chat_tpl=$CHAT_TPL"
echo "  ckpts: ${#CKPTS[@]}"; for c in "${CKPTS[@]}"; do echo "    - $(basename "$c")"; done
echo "  math : $MATH_DATA   temp=$MATH_TEMP   seeds=$SEEDS_CSV"
echo "  alpa : temp=$ALPACA_TEMP top_p=$ALPACA_TOP_P (1 seed)"
echo "════════════════════════════════════════"

# ─── Phase A — math, 3 seeds per ckpt ────────────────────────────────────────
IFS=',' read -ra SEEDS_ARR <<< "$SEEDS_CSV"
echo
echo "── Phase A: math (seeds=${SEEDS_CSV}) ──"
for CKPT in "${CKPTS[@]}"; do
  CKPT_NAME="$(basename "$CKPT")"
  for SEED in "${SEEDS_ARR[@]}"; do
    OUT="$RESULTS_ROOT/$CKPT_NAME/math_s${SEED}"
    if [[ -f "$OUT/done" ]]; then
      echo "[skip] math $CKPT_NAME s$SEED"
      continue
    fi
    mkdir -p "$OUT"
    pushd "$EVAL_REPO/evaluate_math" > /dev/null
    python3 -u math_eval.py \
        --model_name_or_path "$MODEL" \
        --lora_path "$CKPT" \
        --data_name "$MATH_DATA" --data_dir "./data" \
        --output_dir "$OUT" \
        --summary_path "$OUT/summary.txt" \
        --split test --prompt_type "qwen25-math-cot" \
        --num_test_sample -1 --max_tokens_per_call 2048 \
        --seed "$SEED" --temperature "$MATH_TEMP" --n_sampling 1 --top_p 1 \
        --start 0 --end -1 \
        --use_vllm --save_outputs \
        "${CHAT_FLAG[@]}"
    popd > /dev/null
    touch "$OUT/done"
  done
done

# ─── Phase B — alpaca, 1 seed ────────────────────────────────────────────────
echo
echo "── Phase B: alpaca (1 seed) ──"
for CKPT in "${CKPTS[@]}"; do
  CKPT_NAME="$(basename "$CKPT")"
  OUT="$RESULTS_ROOT/$CKPT_NAME/alpaca"
  if [[ -f "$OUT/outputs.jsonl" ]] && [[ $(wc -l < "$OUT/outputs.jsonl") -ge 800 ]]; then
    echo "[skip] alpaca $CKPT_NAME (already done)"
    continue
  fi
  mkdir -p "$OUT"
  python3 -u "$EVAL_REPO/evaluate_alpaca/alpaca_eval.py" \
      --model_name_or_path "$MODEL" \
      --lora_path "$CKPT" \
      --output_path "$OUT/outputs.jsonl" \
      --generator_name "${RUN_ID}_${CKPT_NAME}" \
      --seed 0 \
      --max_tokens 2048 --temperature "$ALPACA_TEMP" --top_p "$ALPACA_TOP_P" \
      "${CHAT_FLAG[@]}"
done

# ─── Phase C — KL(π_ckpt ∥ π_base) ───────────────────────────────────────────
echo
echo "── Phase C: KL vs base ──"
KL_OUT="$RESULTS_ROOT/kl_vs_base.json"
if [[ -f "$KL_OUT" ]]; then
  echo "[skip] KL (already done)"
else
  python3 -u "$REPO/jobs/babel/eval_kl_vs_base.py" \
      --run_ckpt_dir "$RUN_CKPT_DIR" \
      --base_model "$MODEL" \
      --out_path "$KL_OUT" \
      --n_prompts 80 --max_new 256
fi

echo
echo "Done: $RUN_ID — results in $RESULTS_ROOT"
