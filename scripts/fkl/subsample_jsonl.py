#!/usr/bin/env python3
"""
Subsample a JSONL file by a fraction (default 0.5 = half). Uses fixed seed for reproducibility.
Usage:
  python scripts/fkl/subsample_jsonl.py datasets/wildchat/filtered_tuples.jsonl --frac 0.5 --seed 42
  # Writes to same dir with _subsampled suffix: filtered_tuples_subsampled.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input .jsonl path")
    parser.add_argument("--frac", type=float, default=0.5, help="Fraction to keep (0.5 = half)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: <stem>_subsampled.jsonl in same dir)")
    args = parser.parse_args()

    lines = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    n = len(lines)
    k = max(0, min(n, int(round(n * args.frac))))
    rng = __import__("random").Random(args.seed)
    indices = set(rng.sample(range(n), k))
    subsampled = [lines[i] for i in sorted(indices)]

    out = args.output
    if out is None:
        out = args.input.parent / f"{args.input.stem}_subsampled.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for line in subsampled:
            f.write(line + "\n")
    print(f"Wrote {len(subsampled)} lines (frac={args.frac}) to {out}")


if __name__ == "__main__":
    main()
