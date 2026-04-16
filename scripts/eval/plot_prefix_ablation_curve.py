#!/usr/bin/env python3
"""
Plot prefix ablation: win rate vs y_base at each prefix percentage.

Reads winrate_summary.txt from the ablation output dir.

Usage:
  python scripts/eval/plot_prefix_ablation_curve.py \
      --summary data/winrate_results/prefix_ablation_orig_gpt4omini/winrate_summary.txt \
      --output plots/prefix_ablation_curve.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":     "cm",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      8.5,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           150,
    "savefig.dpi":          900,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
})


def parse_summary(path: str) -> dict:
    """Parse winrate_summary.txt, return {prefix_pct: win_rate}."""
    rates = {}
    with open(path) as f:
        for line in f:
            parts = line.split("|")
            if len(parts) >= 5:
                name = parts[0].strip()
                m = re.match(r"p(\d+)\s+vs\s+ybase", name)
                if m:
                    pct = int(m.group(1))
                    wr_str = parts[-1].strip().rstrip("%")
                    try:
                        rates[pct] = float(wr_str)
                    except ValueError:
                        continue
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default="plots/prefix_ablation_curve.pdf")
    args = parser.parse_args()

    rates = parse_summary(args.summary)
    if not rates:
        print("No data found!")
        return

    pcts = sorted(rates.keys())
    wrs = [rates[p] for p in pcts]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Main line
    ax.plot(pcts, wrs, "o-", color="#2166AC", markersize=6, linewidth=2.0, zorder=5)

    # Value labels
    for p, w in zip(pcts, wrs):
        ax.annotate(f"{w:.1f}", (p, w), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#333")

    # 50% reference line
    ax.axhline(50, color="#999", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(95, 51, "50%", fontsize=7.5, color="#777", va="bottom", ha="right")

    # Axes
    ax.set_xlabel("Prefix percentage (%)")
    ax.set_ylabel("$y^*$ Win Rate vs $y_{base}$  (ties excluded)  %")
    ax.set_xticks(pcts)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, max(wrs) * 1.3)

    # Spine styling
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="#888")
    ax.set_axisbelow(True)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
