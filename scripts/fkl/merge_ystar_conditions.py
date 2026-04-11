#!/usr/bin/env python3
"""Merge 3 per-condition y* files into one combined file for winrate eval."""
import argparse, json
from pathlib import Path

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--prefix30",  required=True)
parser.add_argument("--noprefix",  required=True)
parser.add_argument("--full",      required=True)
parser.add_argument("--output",    required=True)
args = parser.parse_args()

p30    = {(d["conversation_id"], d["turn_index"]): d for d in load_jsonl(args.prefix30)}
nopre  = {(d["conversation_id"], d["turn_index"]): d for d in load_jsonl(args.noprefix)}
full   = {(d["conversation_id"], d["turn_index"]): d for d in load_jsonl(args.full)}

# Keep only samples present in all 3
keys = set(p30) & set(nopre) & set(full)
print(f"prefix30={len(p30)}  noprefix={len(nopre)}  full={len(full)}  common={len(keys)}")

merged = []
for k in sorted(keys):
    base = p30[k].copy()
    base["y_star_noprefix"] = nopre[k]["y_star_noprefix"]
    base["y_star_full"]     = full[k]["y_star_full"]
    merged.append(base)

save_jsonl(merged, args.output)
print(f"Saved {len(merged)} samples → {args.output}")
