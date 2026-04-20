"""Build a TAC-positive training set for hindsight distillation.

Definition of "TAC-positive" for an example at a given prefix level p:
  - POLARIS (verifier):  math_verify(y_star_p) correct AND math_verify(y_base) wrong
  - wildchat / webinstruct_cft (judge): y_star_p beats y_base in BOTH orderings
    (position-bias-corrected win)

Reads Phase 1 artifacts (no new generation, no API cost):
  data/{dataset}_unified.jsonl
  results/generations/{dataset}_generations.jsonl
  results/eval_{dataset}.csv

Emits one training row per winning (id, prefix). Training-ready schema:
  {id, dataset, x, y_star, prefix_used, ground_truth, source_y, source_o}

At test time the student sees only x → it must imitate y_star.
Multiple prefixes per id are kept (prefix-level acts as noise / augmentation).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ["wildchat", "webinstruct", "polaris"]
EVAL_TYPE = {
    "wildchat": "judge",
    "webinstruct": "judge",
    "polaris": "verifier",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_verdicts(csv_path):
    """Return {(id, comparison, prefix_pct): {order: verdict}}."""
    out = defaultdict(dict)
    if not Path(csv_path).exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["id"], row["comparison"], int(row["prefix_pct"]))
            out[key][row["order"]] = row["verdict"]
    return out


def judge_ystar_wins(verdicts_for_key):
    v1 = verdicts_for_key.get("ystar_first")   # A=ystar
    v2 = verdicts_for_key.get("ystar_second")  # B=ystar
    return v1 == "A" and v2 == "B"


def build_for_dataset(dataset, data_dir, gen_dir, eval_dir, prefix_levels):
    unified_path = Path(data_dir) / f"{dataset}_unified.jsonl"
    gens_path = Path(gen_dir) / f"{dataset}_generations.jsonl"
    eval_path = Path(eval_dir) / f"eval_{dataset}.csv"

    unified = {r["id"]: r for r in load_jsonl(unified_path)}
    gens = {r["id"]: r for r in load_jsonl(gens_path)}
    verdicts = load_verdicts(eval_path)

    rows = []
    counts = {p: 0 for p in prefix_levels}
    unique_ids = set()

    for rid, u in unified.items():
        if rid not in gens:
            continue
        g = gens[rid]
        for p in prefix_levels:
            ystar = g.get(f"y_star_{p}")
            ybase = g.get("y_base")
            if ystar is None or ybase is None:
                continue
            key = (rid, "y_star_vs_y_base", p)
            vfor = verdicts.get(key, {})
            if not vfor:
                continue
            win = judge_ystar_wins(vfor)  # works for both judge and verifier
            if not win:
                continue
            rows.append({
                "id": rid,
                "dataset": u["dataset"],
                "x": u["x"],
                "y_star": ystar,
                "prefix_used": p,
                "ground_truth": u.get("ground_truth"),
                "source_y": u.get("y"),
                "source_o": u.get("o"),
            })
            counts[p] += 1
            unique_ids.add(rid)

    return rows, counts, len(unique_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="experiments/tac_winrates/data")
    ap.add_argument("--gen_dir",
                    default="experiments/tac_winrates/results/generations")
    ap.add_argument("--eval_dir", default="experiments/tac_winrates/results")
    ap.add_argument("--out_dir",
                    default="experiments/tac_winrates/data/tac_positive")
    ap.add_argument("--prefixes", type=int, nargs="+", default=[70, 100],
                    help="Which prefix levels to consider. Phase 1 generated "
                         "{0,30,70,100}; the 'upper range' is {70,100}.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    print(f"Extracting TAC-positive pairs at prefix levels {args.prefixes}\n")
    for ds in DATASETS:
        rows, counts, unique = build_for_dataset(
            ds, args.data_dir, args.gen_dir, args.eval_dir, args.prefixes
        )
        out_path = out_dir / f"tac_positive_{ds}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_prefix = "  ".join(f"p{p}={counts[p]}" for p in args.prefixes)
        print(f"{ds:<12}  rows={len(rows):<5} unique_ids={unique:<5} "
              f"{per_prefix}  -> {out_path}")
        combined.extend(rows)

    comb_path = out_dir / "tac_positive_combined.jsonl"
    with open(comb_path, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ncombined      rows={len(combined)}  -> {comb_path}")


if __name__ == "__main__":
    main()
