#!/usr/bin/env bash
# Run Arena-Hard v2.0 pairwise judging with OpenAI GPT-4.1 as the judge model.
# - Models: every model_id in eval_results/benchmark_manifest.csv that has
#   arena-hard-auto/data/arena-hard-v2.0/model_answer/<id>.jsonl
# - Skips missing answer files (e.g. wc_full_2e6_extended, wf_full_2e6_ep12/18).
# - High concurrency via ThreadPoolExecutor (see ARENA_PARALLEL, default 100).
#
# Note: Pairwise *baselines* are still category-specific (gemini-2.5-flash for
# hard_prompt / creative_writing per arena-hard-auto/utils/judge_utils.py).
# GPT-4.1 here is only the *judge* that scores A vs B.
#
# Usage (repo root):
#   export OPENAI_API_KEY=sk-...
#   bash scripts/eval/run_arena_judge_gpt41.sh
# Optional:
#   ARENA_PARALLEL=100 bash scripts/eval/run_arena_judge_gpt41.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARENA_DIR="arena-hard-auto"
JUDGE_MODEL="gpt-4.1"
PARALLEL="${ARENA_PARALLEL:-100}"
MANIFEST="eval_results/benchmark_manifest.csv"
ANSWER_DIR="$ARENA_DIR/data/arena-hard-v2.0/model_answer"
JUDGMENT_DIR="$ARENA_DIR/data/arena-hard-v2.0/model_judgment/$JUDGE_MODEL"
# Arena-Hard v2.0 has 750 prompts; one jsonl line per prompt when complete.
EXPECTED_LINES="${ARENA_EXPECTED_JUDGMENTS:-750}"

mkdir -p "$JUDGMENT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  exit 1
fi

echo "========================================"
echo "Arena-Hard judging | judge: $JUDGE_MODEL | parallel: $PARALLEL"
echo "Complete when each model has ~$EXPECTED_LINES judgment lines"
echo "========================================"

mapfile -t MODEL_IDS < <(tail -n +2 "$MANIFEST" | cut -d, -f1 | tr -d '\r')

PENDING=()
DONE=0
MISSING=0

for mid in "${MODEL_IDS[@]}"; do
  [[ -z "$mid" ]] && continue
  answer_file="$ANSWER_DIR/${mid}.jsonl"
  judgment_file="$JUDGMENT_DIR/${mid}.jsonl"

  if [[ ! -f "$answer_file" ]]; then
    echo "[SKIP] $mid — no model_answer file"
    MISSING=$((MISSING + 1))
    continue
  fi

  if [[ -f "$judgment_file" ]]; then
    lines=$(wc -l < "$judgment_file" | tr -d ' ')
    if [[ "$lines" -ge "$EXPECTED_LINES" ]]; then
      echo "[DONE] $mid ($lines judgments)"
      DONE=$((DONE + 1))
      continue
    fi
    echo "[PARTIAL] $mid ($lines/$EXPECTED_LINES) — will resume"
  fi

  PENDING+=("$mid")
done

echo ""
echo "Summary: ${#PENDING[@]} to judge/resume, $DONE complete, $MISSING skipped (no answers)"
echo ""

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "Nothing to run."
  exit 0
fi

MODEL_LIST=""
for mid in "${PENDING[@]}"; do
  MODEL_LIST+="  - $mid"$'\n'
done

TMP_CONFIG="$(mktemp)"
TMP_API_CONFIG="$(mktemp)"
cleanup() { rm -f "$TMP_CONFIG" "$TMP_API_CONFIG"; }
trap cleanup EXIT

cat > "$TMP_CONFIG" <<EOF
judge_model: $JUDGE_MODEL
temperature: 0.0
max_tokens: 16000
bench_name: arena-hard-v2.0
reference: null
regex_patterns:
  - \\[\\[([AB<>=]+)\\]\\]
  - \\[([AB<>=]+)\\]
prompt_template: "<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
model_list:
$MODEL_LIST
EOF

cat > "$TMP_API_CONFIG" <<EOF
$JUDGE_MODEL:
    model: gpt-4.1
    endpoints: null
    api_type: openai
    parallel: $PARALLEL
    max_tokens: 32000
    temperature: 0.0
EOF

echo "Starting gen_judgment.py with $PARALLEL workers..."
echo "Models: ${PENDING[*]}"
echo ""

cd "$ARENA_DIR"
python gen_judgment.py \
  --setting-file "$TMP_CONFIG" \
  --endpoint-file "$TMP_API_CONFIG"

echo ""
echo "========================================"
echo "Judgments under: $REPO_ROOT/$JUDGMENT_DIR/"
echo "Leaderboard: cd $ARENA_DIR && python show_result.py ... (see Arena-Hard README)"
echo "========================================"
