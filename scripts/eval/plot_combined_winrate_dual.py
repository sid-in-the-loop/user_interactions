#!/usr/bin/env python3
"""
Combined winrate plot: 3 model groups × 4 comparisons × 2 judges.

Each comparison shows two bars:
  - dotted/hatched = GPT-4o-mini
  - solid          = Gemini 2.5 Flash

Usage:
  python scripts/eval/plot_combined_winrate_dual.py \
      --gpt_dirs  data/winrate_results/prefix_ablation_qwen3_4b \
                  data/winrate_results/prefix_ablation_qwen3_8b \
                  data/winrate_results/prefix_ablation_olmo_7b \
      --gem_dirs  data/winrate_results/gemini/qwen3_4b \
                  data/winrate_results/gemini/qwen3_8b \
                  data/winrate_results/gemini/olmo_7b \
      --labels "Qwen3-4B" "Qwen3-8B" "OLMo-3-7B SFT" \
      --output  data/winrate_results/combined_winrate_dual.png
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

BAR_COLORS = {
    "prefix30_vs_y":    "#4C72B0",
    "full_vs_y":        "#55A868",
    "noprefix_vs_y":    "#C44E52",
    "prefix30_vs_full": "#8C8C8C",
}
BAR_ORDER  = ["prefix30_vs_y", "full_vs_y", "noprefix_vs_y", "prefix30_vs_full"]
BAR_LABELS = ["prefix30 vs y", "full vs y", "noprefix vs y", "prefix30 vs full"]


def parse_old_summary(results_dir: Path) -> dict | None:
    """Read old-format winrate_prefix_summary.txt → {key: float 0-1}."""
    f = results_dir / "winrate_prefix_summary.txt"
    if not f.exists():
        return None
    key_map = {
        "prefix30 vs y":    "prefix30_vs_y",
        "noprefix vs y":    "noprefix_vs_y",
        "full vs y":        "full_vs_y",
        "prefix30 vs full": "prefix30_vs_full",
    }
    rates = {}
    for line in f.read_text().splitlines():
        for label, key in key_map.items():
            if line.strip().startswith(label):
                pct = line.split()[-1].rstrip("%")
                rates[key] = float(pct) / 100.0
                break
    return rates if rates else None


def parse_new_summary(results_dir: Path) -> dict | None:
    """Read new-format: one subdir per comparison, each with winrate_summary.txt."""
    subdir_map = {
        "prefix30_vs_y":    "prefix30_vs_y",
        "full_vs_y":        "full_vs_y",
        "noprefix_vs_y":    "noprefix_vs_y",
        "prefix30_vs_full": "prefix30_vs_full",
    }
    rates = {}
    for key, subdir in subdir_map.items():
        f = results_dir / subdir / "winrate_summary.txt"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            parts = line.split("|")
            if len(parts) >= 5:
                pct = parts[-1].strip().rstrip("%")
                try:
                    rates[key] = float(pct) / 100.0
                    break
                except ValueError:
                    continue
    return rates if rates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_dirs", nargs="+", required=True)
    parser.add_argument("--gem_dirs", nargs="+", required=True)
    parser.add_argument("--labels",   nargs="+", required=True)
    parser.add_argument("--output",   default="data/winrate_results/combined_winrate_dual.png")
    args = parser.parse_args()

    assert len(args.gpt_dirs) == len(args.gem_dirs) == len(args.labels)

    n_models  = len(args.labels)
    n_comps   = len(BAR_ORDER)
    # Each comparison = 2 sub-bars (gpt + gemini); groups of 4 comparisons
    pair_w    = 0.07   # width of each sub-bar
    pair_gap  = 0.01   # gap between the two sub-bars in a pair
    comp_gap  = 0.06   # gap between comparison groups within a model
    group_gap = 1.1    # gap between model groups

    # Compute x positions
    pair_span = 2 * pair_w + pair_gap
    comp_span = pair_span + comp_gap
    group_span = n_comps * comp_span - comp_gap

    group_centers = np.arange(n_models) * group_gap
    # Offsets of each comparison center from group center
    comp_offsets = np.linspace(-group_span / 2 + pair_span / 2,
                                group_span / 2 - pair_span / 2,
                                n_comps)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for mi, (gpt_dir, gem_dir, label) in enumerate(
            zip(args.gpt_dirs, args.gem_dirs, args.labels)):
        gpt_rates = parse_old_summary(Path(gpt_dir))
        gem_rates = parse_new_summary(Path(gem_dir))

        for ci, (key, offset) in enumerate(zip(BAR_ORDER, comp_offsets)):
            color = BAR_COLORS[key]
            cx = group_centers[mi] + offset

            gpt_x = cx - (pair_w + pair_gap) / 2
            gem_x = cx + (pair_w + pair_gap) / 2

            # GPT-4o-mini bar (dotted/hatched)
            gpt_val = gpt_rates.get(key) if gpt_rates else None
            if gpt_val is not None:
                ax.bar(gpt_x, gpt_val, width=pair_w,
                       color=color, alpha=0.45, zorder=3,
                       edgecolor=color, linewidth=1.0,
                       hatch="///", linestyle="--")
                ax.text(gpt_x, gpt_val + 0.01, f"{gpt_val*100:.1f}",
                        ha="center", va="bottom", fontsize=6.5, color="#444")

            # Gemini bar (solid)
            gem_val = gem_rates.get(key) if gem_rates else None
            if gem_val is not None:
                ax.bar(gem_x, gem_val, width=pair_w,
                       color=color, alpha=0.88, zorder=3,
                       edgecolor="white", linewidth=0.5)
                ax.text(gem_x, gem_val + 0.01, f"{gem_val*100:.1f}",
                        ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold", color="#222")

    ax.axhline(0.5, color="#333", linewidth=0.9, linestyle="--", zorder=2, alpha=0.7)
    ax.text(group_centers[-1] + group_gap * 0.48, 0.502, "50%",
            va="bottom", ha="right", fontsize=8, color="#555")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(args.labels, fontsize=12)
    ax.set_xlim(group_centers[0] - group_gap * 0.55,
                group_centers[-1] + group_gap * 0.55)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_ylabel("Win Rate  (ties excluded)", fontsize=11)

    # Legend: color patches + judge style
    color_handles = [
        mpatches.Patch(color=BAR_COLORS[k], alpha=0.88, label=l)
        for k, l in zip(BAR_ORDER, BAR_LABELS)
    ]
    judge_handles = [
        mpatches.Patch(facecolor="#aaa", hatch="///", alpha=0.5,
                       edgecolor="#aaa", label="GPT-4o-mini (dotted)"),
        mpatches.Patch(facecolor="#aaa", alpha=0.88, label="Gemini 2.5 Flash (solid)"),
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper right", fontsize=8.5,
                     frameon=True, framealpha=0.9, edgecolor="#ccc",
                     ncol=2, columnspacing=1.0, handlelength=1.2)
    ax.add_artist(leg1)
    ax.legend(handles=judge_handles, loc="upper left", fontsize=8.5,
              frameon=True, framealpha=0.9, edgecolor="#ccc",
              handlelength=1.5)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=900, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
