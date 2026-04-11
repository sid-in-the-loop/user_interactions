#!/usr/bin/env python3
"""
Step 3 — Aggregate FKL probe signal across checkpoints.

Reads *_signal.jsonl files (id, category, per_token_kl), computes mean/median
per-token KL by (checkpoint, category). SDPO claim: unrelated ≈ 0, corrective > 0.

Usage:
  python scripts/fkl/aggregate_probe_signal.py --results_dir results [--output results/probe_signal_summary.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Aggregate probe signal (Step 3).")
    parser.add_argument("--results_dir", type=Path, default=REPO_ROOT / "results", help="Dir containing *_signal.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Optional: write summary CSV here")
    parser.add_argument("--suffix", type=str, default="_signal.jsonl", help="Glob suffix for signal files")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all *_signal.jsonl except e.g. final_signal.jsonl if you want only named checkpoints
    pattern = f"*{args.suffix}"
    files = sorted(results_dir.glob(pattern))
    # Exclude generic "final_signal.jsonl" (often incomplete)
    files = [f for f in files if (f.stem.replace("_signal", "") or f.stem) != "final"]

    if not files:
        print(f"No files matching {pattern} in {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in files:
        ckpt_name = path.stem.replace("_signal", "") if path.stem.endswith("_signal") else path.stem
        by_cat = {}  # category -> list of (mean per sample over tokens)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cat = obj.get("category", "unknown")
                kls = obj.get("per_token_kl") or []
                if not kls:
                    continue
                mean_tokens = sum(kls) / len(kls)
                by_cat.setdefault(cat, []).append(mean_tokens)
        for cat, means in by_cat.items():
            n = len(means)
            mean_kl = sum(means) / n
            sorted_means = sorted(means)
            median_kl = sorted_means[n // 2] if n else 0.0
            variance = sum((x - mean_kl) ** 2 for x in means) / n if n else 0.0
            std_kl = variance ** 0.5
            rows.append({
                "checkpoint": ckpt_name,
                "category": cat,
                "mean_kl": mean_kl,
                "median_kl": median_kl,
                "std_kl": std_kl,
                "n_samples": n,
            })

    # Print table
    print("checkpoint\tcategory\tmean_kl\tmedian_kl\tstd_kl\tn_samples")
    for r in rows:
        print(f"{r['checkpoint']}\t{r['category']}\t{r['mean_kl']:.6f}\t{r['median_kl']:.6f}\t{r['std_kl']:.6f}\t{r['n_samples']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as out:
            out.write("checkpoint,category,mean_kl,median_kl,std_kl,n_samples\n")
            for r in rows:
                out.write(f"{r['checkpoint']},{r['category']},{r['mean_kl']:.6f},{r['median_kl']:.6f},{r['std_kl']:.6f},{r['n_samples']}\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
