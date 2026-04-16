#!/usr/bin/env python3
"""
Plot win rates for Qwen3-8B: y* vs y (GPT-4) AND y* vs y_base.

Two groups of bars side by side:
  Left group:  prefix30/noprefix/full vs y (GPT-4 turbo)
  Right group: prefix30/noprefix/full vs y_base

Uses Gemini 2.5 Flash judge results.

Usage:
  python scripts/eval/plot_ybase_comparison.py \
      --vs-y-dir data/winrate_results/ystar_vs_y_qwen3_8b_gemini \
      --vs-ybase-dir data/winrate_results/ybase_vs_ystar_qwen3_8b_gemini \
      --output plots/qwen3_8b_ystar_vs_y_and_ybase.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "figure.dpi":        150,
})


def parse_winrate_summary(summary_path: Path) -> dict:
    """Parse winrate_summary.txt and extract win rates by comparison name."""
    rates = {}
    if not summary_path.exists():
        return rates
    for line in summary_path.read_text().splitlines():
        parts = line.split("|")
        if len(parts) >= 5:
            name = parts[0].strip()
            pct_str = parts[-1].strip().rstrip("%")
            try:
                rates[name] = float(pct_str)
            except ValueError:
                continue
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs-y-dir", required=True,
                        help="Dir with winrate_summary.txt for y* vs y")
    parser.add_argument("--vs-ybase-dir", required=True,
                        help="Dir with winrate_summary.txt for y* vs y_base")
    parser.add_argument("--output", default="plots/qwen3_8b_ystar_vs_y_and_ybase.png")
    parser.add_argument("--title", default="Qwen3-8B")
    args = parser.parse_args()

    vs_y_rates = parse_winrate_summary(Path(args.vs_y_dir) / "winrate_summary.txt")
    vs_ybase_rates = parse_winrate_summary(Path(args.vs_ybase_dir) / "winrate_summary.txt")

    print("vs y rates:", vs_y_rates)
    print("vs ybase rates:", vs_ybase_rates)

    # Map comparison names to values
    conditions = ["prefix30", "noprefix", "full"]
    colors = {"prefix30": "#4C72B0", "noprefix": "#C44E52", "full": "#55A868"}

    # Extract values
    vs_y_vals = []
    vs_ybase_vals = []
    for cond in conditions:
        # vs y: look for "prefix30 vs y" etc
        vy = vs_y_rates.get(f"{cond} vs y", 0)
        vs_y_vals.append(vy)
        # vs ybase: look for "prefix30 vs ybase" etc
        vyb = vs_ybase_rates.get(f"{cond} vs ybase", 0)
        vs_ybase_vals.append(vyb)

    # Plot: two groups ("vs y (GPT-4)" and "vs y_base"), 3 bars each
    n_conds = len(conditions)
    bar_w = 0.18
    group_gap = 0.15

    # Group centers
    group_centers = np.array([0, 1.2])
    group_labels = ["vs $y$ (GPT-4 turbo)", "vs $y_{base}$"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for gi, (center, vals) in enumerate(zip(group_centers, [vs_y_vals, vs_ybase_vals])):
        offsets = np.linspace(-(n_conds - 1) * bar_w / 2,
                               (n_conds - 1) * bar_w / 2,
                               n_conds)
        for ci, (cond, val, off) in enumerate(zip(conditions, vals, offsets)):
            x = center + off
            bar = ax.bar(x, val, width=bar_w * 0.85,
                         color=colors[cond], alpha=0.88, zorder=3,
                         edgecolor="white", linewidth=0.5)
            ax.text(x, val + 1.0, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#222")

    ax.axhline(50, color="#333", linewidth=0.9, linestyle="--", zorder=2, alpha=0.7)
    ax.text(group_centers[-1] + 0.5, 51, "50%",
            va="bottom", ha="right", fontsize=8, color="#555")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_xlim(group_centers[0] - 0.55, group_centers[-1] + 0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Win Rate  (ties excluded)  %", fontsize=11)
    # No title — clean for paper figures

    # Legend
    handles = [mpatches.Patch(color=colors[c], alpha=0.88, label=c) for c in conditions]
    ax.legend(handles=handles, loc="upper right", fontsize=9.5,
              frameon=True, framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True) if not out.parent.exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=900, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
