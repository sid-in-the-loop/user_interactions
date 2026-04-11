#!/usr/bin/env python3
"""
Sample a fixed set of (conversation_id, turn_index) pairs from a JSONL file
and save them as a reusable IDs file for winrate_eval.py --ids-file.

Usage:
  python scripts/eval/sample_winrate_ids.py \
    --input datasets/wildfeedback/qwen3_8b/ystar_thinking_full.jsonl \
    --output eval_results/winrate_sample_500_seed42.json \
    --n 500 --seed 42
"""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Any ystar JSONL to sample IDs from")
    parser.add_argument("--output", required=True, help="Output JSON file with sampled IDs")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    ids = [{"conversation_id": r["conversation_id"], "turn_index": r.get("turn_index")} for r in sample]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"Saved {len(ids)} IDs → {args.output}")


if __name__ == "__main__":
    main()
