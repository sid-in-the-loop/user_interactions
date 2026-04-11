#!/bin/bash
# Reorg y* and filtered data into datasets/wildchat and datasets/wildfeedback.
# Run from repo root: bash scripts/data/reorg_ystar_dirs.sh

set -e
REPO=/home/ssmurali/user_interactions
cd "$REPO"

WC="$REPO/datasets/wildchat"
WF="$REPO/datasets/wildfeedback"
DATA="$REPO/data"
FD="$REPO/filtered_data"

echo "=== 1. Ensure wildfeedback dir and move interactions ==="
mkdir -p "$WF"
if [[ -f "$WC/wildfeedback_interactions.jsonl" ]]; then
  mv "$WC/wildfeedback_interactions.jsonl" "$WF/interactions.jsonl"
  echo "Moved wildfeedback_interactions.jsonl -> $WF/interactions.jsonl"
fi

echo "=== 2. Copy wildchat y* from data/ into datasets/wildchat/ (unified names) ==="
# Full
[[ -f "$DATA/ystar_thinking.jsonl" ]]    && cp "$DATA/ystar_thinking.jsonl"    "$WC/ystar_thinking_full.jsonl"
[[ -f "$DATA/ystar_nonthinking.jsonl" ]] && cp "$DATA/ystar_nonthinking.jsonl" "$WC/ystar_nonthinking_full.jsonl"
# Best
[[ -f "$DATA/ystar_best/ystar_thinking.jsonl" ]]    && cp "$DATA/ystar_best/ystar_thinking.jsonl"    "$WC/ystar_thinking_best.jsonl"
[[ -f "$DATA/ystar_best/ystar_nonthinking.jsonl" ]] && cp "$DATA/ystar_best/ystar_nonthinking.jsonl" "$WC/ystar_nonthinking_best.jsonl"
# Decent (only nonthinking exists so far)
[[ -f "$DATA/ystar_decent/ystar_thinking.jsonl" ]]    && cp "$DATA/ystar_decent/ystar_thinking.jsonl"    "$WC/ystar_thinking_decent.jsonl"
[[ -f "$DATA/ystar_decent/ystar_nonthinking.jsonl" ]] && cp "$DATA/ystar_decent/ystar_nonthinking.jsonl" "$WC/ystar_nonthinking_decent.jsonl"
# Noise
[[ -f "$DATA/ystar_noise/ystar_thinking.jsonl" ]]    && cp "$DATA/ystar_noise/ystar_thinking.jsonl"    "$WC/ystar_thinking_noise.jsonl"
[[ -f "$DATA/ystar_noise/ystar_nonthinking.jsonl" ]] && cp "$DATA/ystar_noise/ystar_nonthinking.jsonl" "$WC/ystar_nonthinking_noise.jsonl"
echo "Wildchat y* files in $WC:"
ls -la "$WC"/ystar_*.jsonl 2>/dev/null || true

echo "=== 3. Copy filtered_data (wildchat tiers) into datasets/wildchat/ ==="
[[ -f "$FD/filtered_BEST.jsonl" ]]   && cp "$FD/filtered_BEST.jsonl"   "$WC/filtered_best.jsonl"
[[ -f "$FD/filtered_DECENT.jsonl" ]] && cp "$FD/filtered_DECENT.jsonl"   "$WC/filtered_decent.jsonl"
[[ -f "$FD/filtered_NOISE.jsonl" ]]   && cp "$FD/filtered_NOISE.jsonl"   "$WC/filtered_noise.jsonl"
# Full: already in wildchat as filtered_tuples.jsonl
if [[ -f "$WC/filtered_tuples.jsonl" ]]; then
  echo "Wildchat full tuples: $WC/filtered_tuples.jsonl (already there)"
fi
echo "Wildchat filtered:"
ls -la "$WC"/filtered_*.jsonl 2>/dev/null || true

echo "=== Done. datasets/wildfeedback/ will get filtered_* after you run the filter on converted tuples. ==="
