"""Build training mixtures at target winrates.

For each (dataset, target_winrate) pair, sample examples so the sampled set has
(target_winrate)% y*-wins and (1 - target_winrate)% tie-or-loss. Shuffles the
order of win / non-win rows so the mixture is well-mixed.

Verdict sources:
  - POLARIS (verifier): y_star_0 vs y_base via math_verify. "win" iff y_star
    correct AND y_base wrong. Determined directly here from generations +
    ground_truth — does NOT need an eval CSV.
  - wildchat / webinstruct (judge): we read the existing eval CSV and use
    pos-bias-corrected verdict (ystar wins BOTH orderings).

Inputs (per dataset):
  data/{dataset}_unified.jsonl
  results/generations/{dataset}_generations.jsonl  (has y_base + y_star_0)
  results/eval_{dataset}.csv  (only required for judge datasets)

Outputs:
  data/mixtures/mix_{dataset}_teacher_xo_w{100,70,50,20}.jsonl

Each row:
  {id, dataset, x, y_star, y_base, verdict: "win"|"notwin",
   prefix_used: 0, ground_truth, source_o}
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.math_scorer import MathVerifyScorer


TARGETS = [100, 70, 50, 20]
JUDGE_DATASETS = {"wildchat", "webinstruct_cft"}


# ─── IO ──────────────────────────────────────────────────────────────────────

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_verdicts_csv(path):
    """Return {id: 'win'|'notwin'} using y_star_vs_y_base at prefix 0, with
    pos-bias correction (A-wins-first AND B-wins-second)."""
    rows = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["comparison"] != "y_star_vs_y_base":
                continue
            if int(row["prefix_pct"]) != 0:
                continue
            rows[row["id"]][row["order"]] = row["verdict"]
    out = {}
    for rid, d in rows.items():
        v1 = d.get("ystar_first")
        v2 = d.get("ystar_second")
        if v1 is None or v2 is None:
            continue
        ystar_wins = v1 == "A" and v2 == "B"
        out[rid] = "win" if ystar_wins else "notwin"
    return out


def verdicts_polaris(unified_by_id, gens_by_id):
    scorer = MathVerifyScorer()
    out = {}
    for rid, u in unified_by_id.items():
        if rid not in gens_by_id:
            continue
        g = gens_by_id[rid]
        yb = g.get("y_base")
        ys = g.get("y_star_0")
        gt = u.get("ground_truth")
        if yb is None or ys is None or gt is None:
            continue
        ystar_ok = scorer.score(ys, gt) >= 0.5
        ybase_ok = scorer.score(yb, gt) >= 0.5
        out[rid] = "win" if (ystar_ok and not ybase_ok) else "notwin"
    return out


# ─── Sampling ────────────────────────────────────────────────────────────────

def sample_mixture(wins, nonwins, target_pct, rng, max_size=None):
    """Return a list of selected ids with exactly target_pct% wins.

    Uses as many examples as possible given supply constraints, optionally
    capped by max_size. If target_pct=100 and we only have 300 wins, returns
    300 rows. If target_pct=20 and we have tons of nonwins but only 300 wins,
    returns 300 / 0.20 = 1500 rows (300 wins + 1200 nonwins).
    """
    p = target_pct / 100.0
    n_wins = len(wins)
    n_nonwins = len(nonwins)

    if p == 1.0:
        size = min(n_wins, max_size or n_wins)
        sampled_wins = rng.sample(wins, size)
        sampled_non = []
    elif p == 0.0:
        size = min(n_nonwins, max_size or n_nonwins)
        sampled_wins = []
        sampled_non = rng.sample(nonwins, size)
    else:
        # Given p, need w_required wins and n_required nonwins with
        # w/(w+n) = p. Maximize total subject to w <= n_wins and n <= n_nonwins.
        # w_total_if_wins_limit    = n_wins / p
        # w_total_if_nonwins_limit = n_nonwins / (1 - p)
        total_by_wins = n_wins / p if p > 0 else float("inf")
        total_by_non = n_nonwins / (1 - p) if p < 1 else float("inf")
        total = int(min(total_by_wins, total_by_non))
        if max_size is not None:
            total = min(total, max_size)
        n_required_wins = int(round(total * p))
        n_required_non = total - n_required_wins
        sampled_wins = rng.sample(wins, n_required_wins)
        sampled_non = rng.sample(nonwins, n_required_non)

    out = sampled_wins + sampled_non
    rng.shuffle(out)
    return out


# ─── Per-dataset builder ─────────────────────────────────────────────────────

def build_for_dataset(dataset, data_dir, gen_dir, eval_dir, out_dir,
                      max_mix_size, seed):
    unified_path = Path(data_dir) / f"{dataset}_unified.jsonl"
    gens_path = Path(gen_dir) / f"{dataset}_generations.jsonl"
    unified = {r["id"]: r for r in load_jsonl(unified_path)}
    gens = {r["id"]: r for r in load_jsonl(gens_path)}

    ds_label = unified[next(iter(unified))]["dataset"]
    if ds_label in JUDGE_DATASETS:
        eval_path = Path(eval_dir) / f"eval_{dataset}.csv"
        verdicts = load_verdicts_csv(eval_path)
    else:
        verdicts = verdicts_polaris(unified, gens)

    print(f"[{dataset}] verdicts: {len(verdicts)} total", flush=True)
    wins = [rid for rid, v in verdicts.items() if v == "win"]
    non = [rid for rid, v in verdicts.items() if v == "notwin"]
    print(f"[{dataset}]   wins={len(wins)}  notwins={len(non)}  "
          f"raw_winrate={len(wins) / max(1, len(verdicts)):.3f}", flush=True)

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for target in TARGETS:
        ids = sample_mixture(wins, non, target, rng, max_size=max_mix_size)
        rows = []
        for rid in ids:
            u = unified[rid]
            g = gens[rid]
            rows.append({
                "id": rid,
                "dataset": u["dataset"],
                "x": u["x"],
                "y_star": g.get("y_star_0"),
                "y_base": g.get("y_base"),
                "verdict": verdicts[rid],
                "prefix_used": 0,
                "ground_truth": u.get("ground_truth"),
                "source_o": u.get("o"),
            })
        fname = f"mix_{dataset}_teacher_xo_w{target}.jsonl"
        out_path = out_dir / fname
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_win = sum(1 for r in rows if r["verdict"] == "win")
        observed = n_win / max(1, len(rows))
        summary.append((dataset, target, len(rows), n_win, observed, str(out_path)))
        print(f"[{dataset}] w{target}: rows={len(rows):<5} wins={n_win:<5} "
              f"observed={observed:.3f}  -> {out_path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="experiments/tac_winrates/data")
    ap.add_argument("--gen_dir",
                    default="experiments/tac_winrates/results/generations")
    ap.add_argument("--eval_dir",
                    default="experiments/tac_winrates/results")
    ap.add_argument("--out_dir",
                    default="experiments/tac_winrates/data/mixtures")
    ap.add_argument("--datasets", nargs="+",
                    default=["wildchat", "webinstruct", "polaris"])
    ap.add_argument("--max_mix_size", type=int, default=0,
                    help="Cap on rows per mixture; 0 = no cap (use as many as "
                         "supply allows).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    max_cap = args.max_mix_size if args.max_mix_size > 0 else None

    all_summary = []
    for ds in args.datasets:
        summary = build_for_dataset(
            ds, args.data_dir, args.gen_dir, args.eval_dir,
            args.out_dir, max_cap, args.seed,
        )
        all_summary.extend(summary)

    # write a readable summary CSV
    summary_path = Path(args.out_dir) / "_mixtures_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "target_winrate_pct", "rows", "win_rows",
                    "observed_winrate", "file"])
        for row in all_summary:
            w.writerow(row)
    print(f"\nsummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
