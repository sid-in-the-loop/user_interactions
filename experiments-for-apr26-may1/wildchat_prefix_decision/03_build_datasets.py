"""WildChat dataset construction (Prompt 4).

Inputs:
  - data/judgments.jsonl   from Prompt 3
    (each row has verdicts for cond_xo / cond_xyo / cond_xyo_ystart, both judges)

For each of the 3 conditioning variants, build two splits based on the STUDENT
judge's resolved winner (both-orderings rule):
  teacher_wins_{cond}:   rows where student verdict winner == 'y_star'
  teacher_loses_{cond}:  rows where student verdict winner == 'y'

Each (wins, loses) pair is downsampled to equal size — smaller side keeps all,
larger side is sampled with a fixed seed. Downsampled ids are saved to
downsample_indices.json for reproducibility.

Each output row:
  {example_id, x, y, o, y_star, y_base,
   student_verdict, gpt4o_mini_verdict, agreement}
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CONDS = ("cond_xo", "cond_xyo", "cond_xyo_ystart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments",
        default="experiments-for-apr26-may1/wildchat_prefix_decision/data/judgments.jsonl")
    ap.add_argument("--out_dir",
        default="experiments-for-apr26-may1/wildchat_prefix_decision/data/wildchat")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.judgments) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} judgment rows from {args.judgments}")

    # Per-cond agreement counters (for printing)
    agreement = {c: {"both": 0, "agree": 0} for c in CONDS}
    raw_counts = {c: {"y_star": 0, "y": 0, "tie": 0, "missing": 0} for c in CONDS}

    # Build per-cond record lists, indexed by what student says
    # split_pool[cond]['wins']  = list of records where student.winner == 'y_star'
    # split_pool[cond]['loses'] = list of records where student.winner == 'y'
    split_pool = {c: {"wins": [], "loses": []} for c in CONDS}

    for d in rows:
        eid = d["example_id"]
        x, y, o = d["x"], d["y"], d["o"]
        y_base = d.get("y_base", "")
        verdicts = d.get("verdicts", {})

        for cond in CONDS:
            cv = verdicts.get(cond, {})
            s = cv.get("student", {}) or {}
            g = cv.get("gpt4o_mini", {}) or {}
            s_winner = s.get("winner")
            o_winner = g.get("winner")
            agree = bool(cv.get("agreement", False))

            # Counters
            if s_winner is None:
                raw_counts[cond]["missing"] += 1
            else:
                raw_counts[cond][s_winner] += 1
            if s_winner is not None and o_winner is not None:
                agreement[cond]["both"] += 1
                if s_winner == o_winner:
                    agreement[cond]["agree"] += 1

            # Need a non-empty y_star for this cond to include in any split
            ystar_field = f"y_star_{cond}"
            ystar = d.get(ystar_field, "")
            if not ystar:
                continue

            rec = {
                "example_id": eid,
                "x": x, "y": y, "o": o,
                "y_star": ystar,
                "y_base": y_base,
                "student_verdict":    s_winner,
                "gpt4o_mini_verdict": o_winner,
                "agreement":          agree,
            }
            if s_winner == "y_star":
                split_pool[cond]["wins"].append(rec)
            elif s_winner == "y":
                split_pool[cond]["loses"].append(rec)
            # ties / missing → not in either split

    # Print raw counts and agreement
    print("\n=== Raw student-verdict distribution per conditioning ===")
    for cond in CONDS:
        c = raw_counts[cond]
        total = sum(c.values())
        print(f"  {cond:<18}  y_star={c['y_star']}  y={c['y']}  tie={c['tie']}  "
              f"missing={c['missing']}  total={total}")

    print("\n=== Student vs gpt4o_mini agreement per conditioning ===")
    for cond in CONDS:
        a = agreement[cond]
        rate = a["agree"] / a["both"] if a["both"] else 0.0
        print(f"  {cond:<18}  {a['agree']}/{a['both']} = {rate:.4f}")

    # Downsample, write splits
    splits_info = {}
    print("\n=== Building splits ===")
    for cond in CONDS:
        wins  = split_pool[cond]["wins"]
        loses = split_pool[cond]["loses"]
        n = min(len(wins), len(loses))

        # Reproducible sample: sort by id then shuffle with seeded rng
        wins_sorted  = sorted(wins,  key=lambda r: r["example_id"])
        loses_sorted = sorted(loses, key=lambda r: r["example_id"])
        rng_w = random.Random(args.seed + 1)
        rng_l = random.Random(args.seed + 2)
        rng_w.shuffle(wins_sorted)
        rng_l.shuffle(loses_sorted)
        wins_take  = wins_sorted[:n]
        loses_take = loses_sorted[:n]
        # On-disk order: by id (deterministic)
        wins_take.sort(key=lambda r: r["example_id"])
        loses_take.sort(key=lambda r: r["example_id"])

        wins_path  = out_dir / f"teacher_wins_{cond}.jsonl"
        loses_path = out_dir / f"teacher_loses_{cond}.jsonl"
        with open(wins_path, "w") as f:
            for r in wins_take:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(loses_path, "w") as f:
            for r in loses_take:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  {cond:<18}  raw {len(wins)}/{len(loses)}  → {n} each")
        print(f"    → {wins_path}")
        print(f"    → {loses_path}")

        splits_info[cond] = {
            "raw_wins":      len(wins),
            "raw_losses":    len(loses),
            "downsampled_n": n,
            "wins_ids":      [r["example_id"] for r in wins_take],
            "loses_ids":     [r["example_id"] for r in loses_take],
        }

    # Save indices
    indices_path = out_dir / "downsample_indices.json"
    with open(indices_path, "w") as f:
        json.dump({"seed": args.seed, "splits": splits_info}, f, indent=2)
    print(f"\ndownsample indices: {indices_path}")

    # Save a stats.txt
    lines = []
    lines.append(f"input rows: {len(rows)}")
    lines.append("")
    lines.append("Raw student-verdict distribution:")
    for cond in CONDS:
        c = raw_counts[cond]
        lines.append(f"  {cond:<18}  y_star={c['y_star']}  y={c['y']}  tie={c['tie']}  "
                     f"missing={c['missing']}")
    lines.append("")
    lines.append("Student vs gpt4o_mini agreement:")
    for cond in CONDS:
        a = agreement[cond]
        rate = a["agree"] / a["both"] if a["both"] else 0.0
        lines.append(f"  {cond:<18}  {a['agree']}/{a['both']} = {rate:.4f}")
    lines.append("")
    lines.append("Splits (downsampled, equal size per pair):")
    for cond, info in splits_info.items():
        lines.append(f"  {cond:<18}  raw {info['raw_wins']}/{info['raw_losses']}  "
                     f"→ {info['downsampled_n']} each")
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n")
    print(f"stats: {out_dir / 'stats.txt'}")


if __name__ == "__main__":
    main()
