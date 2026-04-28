"""Aggregate generations + judge verdicts into the final
data/webinstruct_prefix_decision.jsonl, plus a printed/written summary table.

Both-orderings rule:
  y_star wins iff (order_AB verdict == 'A') AND (order_BA verdict == 'B')
  y_base wins iff (order_AB verdict == 'B') AND (order_BA verdict == 'A')
  else tie
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.stats import wilson


COMPARISONS = ("no_y_vs_base", "full_vs_base", "seeded_vs_base")
JUDGES = ("student", "gpt4o_mini")


def load_verdicts(path):
    """Return {(id, comparison): {'AB': verdict, 'BA': verdict}}."""
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["id"], row["comparison"])][row["order"]] = row["verdict"]
    return out


def resolve(verdicts_for_pair: dict) -> str:
    """Apply both-orderings rule to a {'AB': v, 'BA': v} dict.
    Returns 'y_star', 'y_base', or 'tie'."""
    ab = verdicts_for_pair.get("AB", "tie")
    ba = verdicts_for_pair.get("BA", "tie")
    if ab == "A" and ba == "B":
        return "y_star"
    if ab == "B" and ba == "A":
        return "y_base"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--student_csv", required=True)
    ap.add_argument("--openai_csv", required=True)
    ap.add_argument("--output", required=True,
                    help="merged jsonl: webinstruct_prefix_decision.jsonl")
    ap.add_argument("--summary", required=True,
                    help="text summary path")
    args = ap.parse_args()

    student = load_verdicts(args.student_csv)
    openai_ = load_verdicts(args.openai_csv)
    print(f"student verdict-pairs loaded: {len(student)}")
    print(f"openai  verdict-pairs loaded: {len(openai_)}")

    # Counters per (judge, comparison): wins/losses/ties.
    counts = {(j, c): {"y_star": 0, "y_base": 0, "tie": 0}
              for j in JUDGES for c in COMPARISONS}
    # Agreement: total resolved comparisons where both judges have verdicts.
    agreement = {c: {"both_present": 0, "agree": 0} for c in COMPARISONS}

    n_rows = 0
    out_f = open(args.output, "w", buffering=1)
    with open(args.gens) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rid = d["id"]
            n_rows += 1

            verdicts_out = {j: {} for j in JUDGES}
            for c in COMPARISONS:
                s_pair = student.get((rid, c), {})
                o_pair = openai_.get((rid, c), {})

                s_winner = resolve(s_pair) if s_pair else None
                o_winner = resolve(o_pair) if o_pair else None

                verdicts_out["student"][c] = {
                    "order_AB": s_pair.get("AB"),
                    "order_BA": s_pair.get("BA"),
                    "winner": s_winner,
                }
                verdicts_out["gpt4o_mini"][c] = {
                    "order_AB": o_pair.get("AB"),
                    "order_BA": o_pair.get("BA"),
                    "winner": o_winner,
                }

                if s_winner is not None:
                    counts[("student", c)][s_winner] += 1
                if o_winner is not None:
                    counts[("gpt4o_mini", c)][o_winner] += 1
                if s_winner is not None and o_winner is not None:
                    agreement[c]["both_present"] += 1
                    if s_winner == o_winner:
                        agreement[c]["agree"] += 1

            d["verdicts"] = verdicts_out
            out_f.write(json.dumps(d) + "\n")
    out_f.close()

    # Build summary
    lines = []
    lines.append(f"N rows: {n_rows}")
    lines.append("")
    lines.append("=== Per-judge winrates (y_star vs y_base, both-orderings rule) ===")
    lines.append(f"{'judge':<12} {'comparison':<16} {'wins':>6} {'losses':>7} {'ties':>5} "
                 f"{'N_decided':>10} {'winrate':>9} {'95% CI':>20}")
    summary_struct = {}
    for j in JUDGES:
        for c in COMPARISONS:
            cnt = counts[(j, c)]
            wins, losses, ties = cnt["y_star"], cnt["y_base"], cnt["tie"]
            n_decided = wins + losses
            p, lo, hi = wilson(wins, n_decided)
            ci = f"[{lo:.3f}, {hi:.3f}]"
            lines.append(f"{j:<12} {c:<16} {wins:>6} {losses:>7} {ties:>5} "
                         f"{n_decided:>10} {p:>9.3f} {ci:>20}")
            summary_struct.setdefault(j, {})[c] = {
                "wins": wins, "losses": losses, "ties": ties,
                "winrate": p, "ci_low": lo, "ci_high": hi,
            }
    lines.append("")
    lines.append("=== Agreement between student and gpt4o_mini ===")
    for c in COMPARISONS:
        a = agreement[c]
        rate = a["agree"] / a["both_present"] if a["both_present"] else 0.0
        lines.append(f"{c:<16}  agree {a['agree']}/{a['both_present']}  = {rate:.3f}")

    # Decision: pick variant with highest student winrate among (no_y, full, seeded)
    student_rates = {
        "no_y":   summary_struct.get("student", {}).get("no_y_vs_base",   {}).get("winrate", -1),
        "full":   summary_struct.get("student", {}).get("full_vs_base",   {}).get("winrate", -1),
        "seeded": summary_struct.get("student", {}).get("seeded_vs_base", {}).get("winrate", -1),
    }
    pick = max(student_rates, key=student_rates.get)
    lines.append("")
    lines.append("=== Decision (highest student winrate) ===")
    for k in ("no_y", "full", "seeded"):
        lines.append(f"{k:<7} winrate = {student_rates[k]:.3f}")
    lines.append(f"WINNER: {pick}")

    text = "\n".join(lines)
    print(text)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n")
    print(f"\nfinal jsonl: {args.output}\nsummary: {args.summary}")


if __name__ == "__main__":
    main()
