"""Aggregate raw CSVs into phase1_summary.csv + Plot A and Plot B.

Position-bias correction: per (id, comparison, prefix_pct) we see two orderings.
Count as a LEFT (ystar) win iff ystar wins both orderings; RIGHT win iff the
comparator wins both; otherwise TIE.

Plot A: y_star vs y_base, 3 curves (one per dataset).
Plot B: y_star vs y, 2 curves (wildchat + webinstruct).
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_ORDER = ["wildchat", "webinstruct_cft", "polaris"]
DATASET_LABEL = {
    "wildchat": "Wildchat",
    "webinstruct_cft": "WebInstruct-CFT",
    "polaris": "POLARIS",
}
PREFIX_PCTS = [0, 30, 70, 100]


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_csv_rows(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def pos_bias_corrected(raw_rows):
    """Return list of {id, dataset, comparison, prefix_pct, outcome in {left,right,tie}}."""
    by_key = defaultdict(dict)
    for r in raw_rows:
        key = (r["id"], r["dataset"], r["comparison"], int(r["prefix_pct"]))
        by_key[key][r["order"]] = r["verdict"]

    out = []
    for (rid, ds, cmp_, pct), verdicts in by_key.items():
        v1 = verdicts.get("ystar_first")   # A=ystar
        v2 = verdicts.get("ystar_second")  # B=ystar
        if v1 is None or v2 is None:
            continue
        ystar_won_first = v1 == "A"
        ystar_won_second = v2 == "B"
        ystar_lost_first = v1 == "B"
        ystar_lost_second = v2 == "A"
        if ystar_won_first and ystar_won_second:
            outcome = "left"     # ystar wins
        elif ystar_lost_first and ystar_lost_second:
            outcome = "right"    # comparator wins
        else:
            outcome = "tie"
        out.append({
            "id": rid, "dataset": ds, "comparison": cmp_,
            "prefix_pct": pct, "outcome": outcome,
        })
    return out


def summarise(corrected, out_csv):
    agg = defaultdict(lambda: {"left": 0, "right": 0, "tie": 0})
    for r in corrected:
        key = (r["dataset"], r["prefix_pct"], r["comparison"])
        agg[key][r["outcome"]] += 1

    rows = []
    for (ds, pct, cmp_), cnt in sorted(agg.items()):
        n = cnt["left"] + cnt["right"] + cnt["tie"]
        p = cnt["left"] / n if n else 0.0
        lo, hi = wilson(p, n)
        rows.append({
            "dataset": ds, "prefix_pct": pct, "comparison_type": cmp_,
            "winrate": round(p, 4),
            "wilson_lower": round(lo, 4),
            "wilson_upper": round(hi, 4),
            "n": n,
            "left_wins": cnt["left"],
            "right_wins": cnt["right"],
            "ties": cnt["tie"],
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "dataset", "prefix_pct", "comparison_type",
            "winrate", "wilson_lower", "wilson_upper",
            "n", "left_wins", "right_wins", "ties",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} summary rows -> {out_csv}", flush=True)
    return rows


def make_plot(summary_rows, comparison, out_png_pdf_base, title):
    plt.figure(figsize=(6.4, 4.6), dpi=300)
    plt.rcParams["font.family"] = "serif"
    any_data = False
    for ds in DATASET_ORDER:
        rows = [r for r in summary_rows
                if r["dataset"] == ds and r["comparison_type"] == comparison]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["prefix_pct"]))
        xs = [int(r["prefix_pct"]) for r in rows]
        ys = [float(r["winrate"]) for r in rows]
        los = [float(r["wilson_lower"]) for r in rows]
        his = [float(r["wilson_upper"]) for r in rows]
        label = DATASET_LABEL.get(ds, ds)
        line, = plt.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        plt.fill_between(xs, los, his, alpha=0.15, color=line.get_color())
        any_data = True

    if not any_data:
        print(f"skipping plot {comparison}: no data", flush=True)
        plt.close()
        return

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Prefix %")
    plt.ylabel("Win-rate")
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(PREFIX_PCTS)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png_pdf_base + ".png")
    plt.savefig(out_png_pdf_base + ".pdf")
    plt.close()
    print(f"wrote {out_png_pdf_base}.[png,pdf]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csvs", nargs="+", required=True,
                    help="One or more raw eval CSVs (e.g. one per dataset).")
    ap.add_argument("--summary_csv",
                    default="experiments/tac_winrates/results/phase1_summary.csv")
    ap.add_argument("--raw_combined_csv",
                    default="experiments/tac_winrates/results/phase1_raw.csv")
    ap.add_argument("--plot_dir",
                    default="experiments/tac_winrates/results")
    args = ap.parse_args()

    raw_rows = load_csv_rows(args.raw_csvs)
    print(f"loaded {len(raw_rows)} raw rows across {len(args.raw_csvs)} files",
          flush=True)

    # Write a combined raw csv per spec (phase1_raw.csv).
    Path(args.raw_combined_csv).parent.mkdir(parents=True, exist_ok=True)
    if raw_rows:
        with open(args.raw_combined_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            for r in raw_rows:
                w.writerow(r)
        print(f"wrote combined raw -> {args.raw_combined_csv}", flush=True)

    corrected = pos_bias_corrected(raw_rows)
    print(f"after pos-bias correction: {len(corrected)} unique comparisons", flush=True)

    summary = summarise(corrected, args.summary_csv)

    make_plot(summary, "y_star_vs_y_base",
              str(Path(args.plot_dir) / "plot_A_ystar_vs_ybase"),
              "y* vs y_base across prefix %")
    make_plot(summary, "y_star_vs_y",
              str(Path(args.plot_dir) / "plot_B_ystar_vs_y"),
              "y* vs y across prefix %")


if __name__ == "__main__":
    main()
