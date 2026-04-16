#!/bin/bash
# Download and set up all benchmark datasets.
# Run once from the repo root with internet access.

set -euo pipefail

cd "$(dirname "$0")/../.."
DATA="data/benchmark_data"
mkdir -p "$DATA"/{arena_hard,mt_bench,math500,reasoning_gym,writingbench,wildfeedback_held}

echo "=== Arena-Hard v2.0 ==="
if [ ! -f "$DATA/arena_hard/question.jsonl" ]; then
    git clone --depth 1 https://github.com/lmarena/arena-hard-auto.git /tmp/arena-hard-auto 2>/dev/null || true
    cp /tmp/arena-hard-auto/data/arena-hard-v2.0/question.jsonl "$DATA/arena_hard/question.jsonl" 2>/dev/null || \
    echo "  WARNING: Could not find question.jsonl. Download manually from https://github.com/lmarena/arena-hard-auto"
    rm -rf /tmp/arena-hard-auto
fi
echo "  $(wc -l < "$DATA/arena_hard/question.jsonl" 2>/dev/null || echo 0) prompts"

echo ""
echo "=== MT-Bench ==="
if [ ! -f "$DATA/mt_bench/question.jsonl" ]; then
    curl -sL "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl" \
        -o "$DATA/mt_bench/question.jsonl" 2>/dev/null || \
    echo "  WARNING: Could not download. Get it from https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/mt_bench"
fi
echo "  $(wc -l < "$DATA/mt_bench/question.jsonl" 2>/dev/null || echo 0) questions"

echo ""
echo "=== MATH500 ==="
if [ ! -f "$DATA/math500/problems.json" ]; then
    python3 -c "
from datasets import load_dataset
import json, random
random.seed(42)
ds = load_dataset('hendrycks/competition_math', split='test', trust_remote_code=True)
# Take 500 random problems
indices = random.sample(range(len(ds)), min(500, len(ds)))
problems = [{'problem': ds[i]['problem'], 'answer': ds[i]['solution'], 'level': ds[i].get('level',''), 'type': ds[i].get('type','')} for i in indices]
with open('$DATA/math500/problems.json', 'w') as f:
    json.dump(problems, f, indent=2)
print(f'  Saved {len(problems)} problems')
" 2>/dev/null || echo "  WARNING: Need 'datasets' package. pip install datasets"
fi
echo "  $(python3 -c "import json; print(len(json.load(open('$DATA/math500/problems.json'))))" 2>/dev/null || echo 0) problems"

echo ""
echo "=== Reasoning Gym ==="
if [ ! -f "$DATA/reasoning_gym/problems.json" ]; then
    python3 -c "
import reasoning_gym, json
tasks = ['knights_and_knaves', 'binary_matrix', 'shortest_path']
problems = []
for task in tasks:
    env = reasoning_gym.make(task, seed=42, size=200)
    for i, entry in enumerate(env):
        q = entry['question'] if isinstance(entry, dict) else str(entry)
        problems.append({'task': task, 'problem_index': i, 'problem': q, 'seed': 42, 'entry': entry if isinstance(entry, dict) else {'question': str(entry)}})
with open('$DATA/reasoning_gym/problems.json', 'w') as f:
    json.dump(problems, f, indent=2)
print(f'  Saved {len(problems)} problems')
" 2>/dev/null || echo "  WARNING: Need reasoning-gym. pip install reasoning-gym"
fi
echo "  $(python3 -c "import json; print(len(json.load(open('$DATA/reasoning_gym/problems.json'))))" 2>/dev/null || echo 0) problems"

echo ""
echo "=== WritingBench ==="
if [ ! -f "$DATA/writingbench/tasks.json" ]; then
    git clone --depth 1 https://github.com/X-PLUG/WritingBench.git /tmp/WritingBench 2>/dev/null || true
    # Look for task data
    find /tmp/WritingBench -name "*.json" -o -name "*.jsonl" 2>/dev/null | head -10
    echo "  Check /tmp/WritingBench for task files and copy to $DATA/writingbench/tasks.json"
    rm -rf /tmp/WritingBench
fi

echo ""
echo "=== WildFeedback held-out ==="
if [ ! -f "$DATA/wildfeedback_held/held_out_ids.json" ]; then
    python3 -c "
import json, random
random.seed(42)
# Training set IDs
train_ids = set()
with open('datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl') as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            train_ids.add((row.get('conversation_id',''), row.get('turn_index')))
# Full BEST dataset minus training
candidates = []
with open('datasets/wildfeedback/filtered_BEST.jsonl') as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            key = (row.get('conversation_id',''), row.get('turn_index'))
            if key not in train_ids:
                candidates.append(key)
held = random.sample(candidates, min(500, len(candidates)))
with open('$DATA/wildfeedback_held/held_out_ids.json', 'w') as f:
    json.dump(held, f)
print(f'  Saved {len(held)} held-out IDs ({len(candidates)} candidates, {len(train_ids)} training)')
" 2>/dev/null || echo "  WARNING: Could not compute held-out IDs"
fi

echo ""
echo "=== AlpacaEval (already exists) ==="
echo "  $(python3 -c "import json; print(len(json.load(open('alpaca_eval_data/gpt4_turbo_reference.json'))))" 2>/dev/null || echo 0) prompts"

echo ""
echo "Done! Check data/benchmark_data/ for all benchmark data."
ls -la "$DATA"/*/
