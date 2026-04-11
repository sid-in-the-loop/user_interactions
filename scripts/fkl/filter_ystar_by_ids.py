#!/usr/bin/env python3
"""Filter a y* JSONL to only rows whose (conversation_id, turn_index) appear in a reference JSONL.

Usage:
    python scripts/fkl/filter_ystar_by_ids.py \
        --ystar datasets/wildfeedback/qwen3_8b/ystar_thinking_full.jsonl \
        --ref   datasets/wildfeedback/filtered_BEST.jsonl \
        --out   datasets/wildfeedback/qwen3_8b/ystar_thinking_best.jsonl
"""
import argparse
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ystar", required=True, help="Full y* JSONL to filter")
    p.add_argument("--ref", required=True, help="Reference JSONL whose IDs define the subset")
    p.add_argument("--out", required=True, help="Output JSONL path")
    args = p.parse_args()

    ids = set()
    with open(args.ref) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ids.add((d["conversation_id"], d.get("turn_index", 0)))

    kept = 0
    with open(args.ystar) as fin, open(args.out, "w") as fout:
        for line in fin:
            if line.strip():
                d = json.loads(line)
                key = (d["conversation_id"], d.get("turn_index", 0))
                if key in ids:
                    fout.write(line)
                    kept += 1

    print(f"Kept {kept}/{len(ids)} rows → {args.out}")

if __name__ == "__main__":
    main()
