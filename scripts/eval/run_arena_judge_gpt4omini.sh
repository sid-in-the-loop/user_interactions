#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARENA_DIR="arena-hard-auto"
CONFIG_FILE="config/arena-hard-v2.0.yaml"
API_CONFIG="config/api_config.yaml"

JUDGE_MODEL="gpt-4o-mini-2024-07-18"
PARALLEL="${ARENA_PARALLEL:-128}"

# Models to judge (from benchmark_manifest.csv)
MODELS=(
  base_qwen3_4b
  wc_best_5e6_final
  wc_best_5e6_extended
  wc_best_2e6_final
  wc_best_2e6_extended
  wc_full_2e6_final
  wc_full_5e6_ep1
  wc_full_5e6_ep2
  wc_full_5e6_ep3
  wc_full_5e6_ep4
  wf_best_2e6_ep6
  wf_best_2e6_ep12
  wf_best_2e6_ep18
  wf_best_5e6_ep6
  wf_best_5e6_ep12
  wf_best_5e6_ep18
  wf_full_2e6_ep6
  wf_full_2e6_ep12
  wf_full_2e6_ep18
)

ANSWER_DIR="$ARENA_DIR/data/arena-hard-v2.0/model_answer"
JUDGMENT_DIR="$ARENA_DIR/data/arena-hard-v2.0/model_judgment/$JUDGE_MODEL"

mkdir -p "$JUDGMENT_DIR"

echo "========================================"
echo "Arena-Hard Judging with $JUDGE_MODEL"
echo "Parallel: $PARALLEL"
echo "========================================"

# Check which models need judging
PENDING=()
DONE=0
MISSING=0

for mid in "${MODELS[@]}"; do
  answer_file="$ANSWER_DIR/${mid}.jsonl"
  judgment_file="$JUDGMENT_DIR/${mid}.jsonl"
  
  if [[ ! -f "$answer_file" ]]; then
    echo "[SKIP] $mid - no answer file"
    ((MISSING++))
    continue
  fi
  
  if [[ -f "$judgment_file" ]]; then
    lines=$(wc -l < "$judgment_file")
    if [[ "$lines" -ge 500 ]]; then
      echo "[DONE] $mid ($lines judgments)"
      ((DONE++))
      continue
    else
      echo "[PARTIAL] $mid ($lines/500+ judgments) - will resume"
    fi
  fi
  
  PENDING+=("$mid")
done

echo ""
echo "Summary: ${#PENDING[@]} pending, $DONE done, $MISSING missing answers"
echo ""

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "All models already judged!"
  exit 0
fi

# Build model list for config
MODEL_LIST=""
for mid in "${PENDING[@]}"; do
  MODEL_LIST+="  - $mid"$'\n'
done

# Create temporary config with our models and high parallelism
TMP_CONFIG=$(mktemp)
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

# Update api_config to use our parallel setting
TMP_API_CONFIG=$(mktemp)
cat > "$TMP_API_CONFIG" <<EOF
$JUDGE_MODEL:
    model: gpt-4o-mini
    endpoints: null
    api_type: openai
    parallel: $PARALLEL
    max_tokens: 8196
    temperature: 0.0
EOF

echo "Starting judgment with $PARALLEL parallel requests..."
echo "Models to judge: ${PENDING[*]}"
echo ""

cd "$ARENA_DIR"

python gen_judgment.py \
  --setting-file "$TMP_CONFIG" \
  --endpoint-file "$TMP_API_CONFIG"

rm -f "$TMP_CONFIG" "$TMP_API_CONFIG"

echo ""
echo "========================================"
echo "Judging complete!"
echo "Results in: $JUDGMENT_DIR/"
echo "========================================"
