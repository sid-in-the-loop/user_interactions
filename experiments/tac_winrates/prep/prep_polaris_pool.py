"""Seed the POLARIS filter pipeline with a raw pool.

Samples ~3000 problems from polaris-data-53K.jsonl into the unified schema
with y/o left empty (will be filled after filtering + critique generation).
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/u/ssredharan/user_interactions/datasets/polaris/polaris-data-53K.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/u/ssredharan/user_interactions/experiments/tac_winrates/data/polaris_pool.jsonl",
    )
    ap.add_argument("--n", type=int, default=0,
                    help="0 = no cap; use all available polaris rows.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} polaris rows")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out = []
    cap = args.n if args.n > 0 else len(rows) + 1
    for r in rows:
        if len(out) >= cap:
            break
        x = r.get("problem", "") or ""
        gt = r.get("answer", None)
        if not x.strip() or gt is None:
            continue
        out.append({
            "id": f"polaris_{len(out)}",
            "dataset": "polaris",
            "x": x,
            "y": None,
            "o": None,
            "ground_truth": str(gt),
            "eval_type": "verifier",
            "difficulty": r.get("difficulty", None),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
