#!/usr/bin/env python3
"""
Combined winrate plot: 3 model groups × 4 comparisons per group.

Groups: Qwen3-4B, Qwen3-8B, OLMo-3-7B-SFT
Bars per group:
  blue  — prefix30 vs y
  green — full vs y
  red   — noprefix vs y
  grey  — prefix30 vs full

NeurIPS style: serif, no top/right spines, dashed grid, 900 DPI.

Usage:
  python scripts/eval/plot_combined_winrate.py \
      --results_dirs \
          data/winrate_results/prefix_ablation_qwen3_4b \
          data/winrate_results/prefix_ablation_qwen3_8b \
          data/winrate_results/prefix_ablation_olmo_7b \
      --labels "Qwen3-4B" "Qwen3-8B" "OLMo-3-7B SFT" \
      --output  data/winrate_results/combined_winrate.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# NeurIPS rcParams
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "figure.dpi":         150,
})

BAR_COLORS = {
    "prefix30_vs_y":   "#4C72B0",   # blue
    "full_vs_y":       "#55A868",   # green
    "noprefix_vs_y":   "#C44E52",   # red
    "prefix30_vs_full": "#8C8C8C",  # grey
}
BAR_ORDER   = ["prefix30_vs_y", "full_vs_y", "noprefix_vs_y", "prefix30_vs_full"]
BAR_LABELS  = ["prefix30 vs y", "full vs y", "noprefix vs y", "prefix30 vs full"]


def parse_summary(results_dir: Path) -> dict | None:
    """Parse winrate_prefix_summary.txt → {comparison: win_rate}."""
    summary_file = results_dir / "winrate_prefix_summary.txt"
    if not summary_file.exists():
        return None

    rates = {}
    key_map = {
        "prefix30 vs y":      "prefix30_vs_y",
        "noprefix vs y":      "noprefix_vs_y",
        "full vs y":          "full_vs_y",
        "prefix30 vs full":   "prefix30_vs_full",
    }
    for line in summary_file.read_text().splitlines():
        for label, key in key_map.items():
            if line.strip().startswith(label):
                parts = line.split()
                pct = parts[-1].rstrip("%")
                rates[key] = float(pct) / 100.0
                break
    return rates if len(rates) == 4 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="One results dir per model, in order")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display label per model (same order)")
    parser.add_argument("--output", default="data/winrate_results/combined_winrate.png")
    args = parser.parse_args()

    assert len(args.results_dirs) == len(args.labels), \
        "--results_dirs and --labels must have same length"

    n_models  = len(args.labels)
    n_bars    = len(BAR_ORDER)
    bar_w     = 0.17
    group_gap = 1.1          # center-to-center between groups
    group_xs  = np.arange(n_models) * group_gap

    # Bar offsets within each group: centered
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for mi, (rdir, label) in enumerate(zip(args.results_dirs, args.labels)):
        rates = parse_summary(Path(rdir))

        for bi, (key, offset) in enumerate(zip(BAR_ORDER, offsets)):
            x = group_xs[mi] + offset
            color = BAR_COLORS[key]

            if rates is None or key not in rates:
                # Pending — draw a hatched placeholder
                ax.bar(x, 0.5, width=bar_w * 0.88,
                       color="white", edgecolor=color, linewidth=1.2,
                       hatch="///", alpha=0.5, zorder=3)
                ax.text(x, 0.52, "pending", ha="center", va="bottom",
                        fontsize=6.5, color="#888", rotation=90)
            else:
                val = rates[key]
                bar = ax.bar(x, val, width=bar_w * 0.88,
                             color=color, alpha=0.88, zorder=3,
                             edgecolor="white", linewidth=0.5)
                ax.text(x, val + 0.012, f"{val*100:.1f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color="#222")

    # 50% reference line
    ax.axhline(0.5, color="#333", linewidth=0.9, linestyle="--", zorder=2, alpha=0.7)
    ax.text(group_xs[-1] + group_gap * 0.45, 0.502, "50%",
            va="bottom", ha="right", fontsize=8, color="#555")

    # x-axis labels
    ax.set_xticks(group_xs)
    ax.set_xticklabels(args.labels, fontsize=11)
    ax.set_xlim(group_xs[0] - group_gap * 0.55, group_xs[-1] + group_gap * 0.55)

    # y-axis
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_ylabel("Win Rate  (ties excluded)", fontsize=10)

    # Legend
    handles = [
        mpatches.Patch(color=BAR_COLORS[k], alpha=0.88, label=l)
        for k, l in zip(BAR_ORDER, BAR_LABELS)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5,
              frameon=True, framealpha=0.9, edgecolor="#ccc",
              ncol=2, columnspacing=1.0, handlelength=1.2)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=900, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
