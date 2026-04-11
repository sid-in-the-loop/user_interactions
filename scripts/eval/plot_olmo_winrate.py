#!/usr/bin/env python3
"""
NeurIPS-style bar chart for OLMo win-rate results.

Reads winrate_olmo_results.jsonl (output of winrate_olmo.py) and produces a
three-bar horizontal bar chart showing wins/(wins+losses) with 95% Wilson CI
error bars for each of the three comparisons.

Style: serif font, no top/right spines, light dashed grid, 300 DPI.

Usage:
    python scripts/eval/plot_olmo_winrate.py \\
        --results  data/winrate_results/olmo_wf/winrate_olmo_results.jsonl \\
        --output   data/winrate_results/olmo_wf/winrate_olmo.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
    "mathtext.fontset":  "stix",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.35,
    "grid.linewidth":    0.7,
})

COMPARISONS = ["y_base vs y", "y* vs y", "y* vs y_base"]

# Axis labels for each comparison (shorter, for the chart)
LABELS = [
    "OLMo base\nvs GPT-4",
    "OLMo+hindsight\nvs GPT-4",
    "OLMo+hindsight\nvs OLMo base",
]


# ── Stats helpers ──────────────────────────────────────────────────────────────

def wilson_ci(wins: int, n_decisive: int, z: float = 1.96) -> tuple[float, float]:
    if n_decisive == 0:
        return 0.0, 0.0
    p = wins / n_decisive
    denom  = 1 + z * z / n_decisive
    center = (p + z * z / (2 * n_decisive)) / denom
    half   = z * math.sqrt(p * (1 - p) / n_decisive + z * z / (4 * n_decisive ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_stats(rows: list[dict]) -> dict:
    counts = {c: {"wins": 0, "losses": 0, "ties": 0} for c in COMPARISONS}
    for row in rows:
        for name in COMPARISONS:
            w = row["comparisons"].get(name, {}).get("winner", "tie")
            if w == "1":
                counts[name]["wins"]   += 1
            elif w == "2":
                counts[name]["losses"] += 1
            else:
                counts[name]["ties"]   += 1
    stats = {}
    for name in COMPARISONS:
        c = counts[name]
        w, l = c["wins"], c["losses"]
        nd = w + l
        wr = w / nd if nd else float("nan")
        lo, hi = wilson_ci(w, nd)
        stats[name] = {
            "winrate":  wr,
            "ci_lo":    lo,
            "ci_hi":    hi,
            "wins":     w,
            "losses":   l,
            "ties":     c["ties"],
            "decisive": nd,
        }
    return stats


# ── Plot ───────────────────────────────────────────────────────────────────────

# Colors: one distinct color per comparison (muted palette)
BAR_COLORS  = ["#5B8DB8", "#E07B54", "#6AAB6A"]   # blue, orange, green
EDGE_COLORS = ["#2E5F8A", "#B8502A", "#3D7A3D"]


def make_chart(stats: dict, n_total: int, output_path: str) -> None:
    winrates  = [stats[c]["winrate"]                         for c in COMPARISONS]
    err_lo    = [stats[c]["winrate"] - stats[c]["ci_lo"]     for c in COMPARISONS]
    err_hi    = [stats[c]["ci_hi"]   - stats[c]["winrate"]   for c in COMPARISONS]
    decisive  = [stats[c]["decisive"]                        for c in COMPARISONS]

    x = np.arange(len(COMPARISONS))
    bar_width = 0.52

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    bars = ax.bar(
        x,
        [wr * 100 for wr in winrates],
        bar_width,
        yerr=[[e * 100 for e in err_lo], [e * 100 for e in err_hi]],
        color=BAR_COLORS,
        edgecolor=EDGE_COLORS,
        linewidth=0.9,
        error_kw=dict(elinewidth=1.4, capsize=5, ecolor="black", capthick=1.2),
        zorder=3,
    )

    # 50% reference line (random)
    ax.axhline(50, color="gray", linewidth=0.9, linestyle="--", alpha=0.6, zorder=2)
    ax.text(len(COMPARISONS) - 0.1, 51.5, "50%", ha="right", va="bottom",
            fontsize=8, color="gray", fontstyle="italic")

    # Value labels on top of bars
    for bar, wr, nd, lo_e, hi_e in zip(bars, winrates, decisive, err_lo, err_hi):
        if math.isnan(wr):
            continue
        hi_y = (wr + hi_e) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            hi_y + 1.2,
            f"{wr * 100:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
        # n decisive below the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            1.5,
            f"n={nd}",
            ha="center", va="bottom",
            fontsize=7.5, color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9.5)
    ax.set_ylabel("Win rate  (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_tick_params(labelsize=9)

    ax.set_title(
        f"OLMo-3-7B-Instruct win rates on WildFeedback  "
        f"(N={n_total}, judge: GPT-4o-mini, 95% Wilson CI)",
        fontsize=10,
        pad=10,
    )

    # Remove top and right spines (rcParams already set, but be explicit)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)

    # Grid only on y-axis (behind bars)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.7)
    ax.xaxis.grid(False)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OLMo win-rate results (NeurIPS style).")
    parser.add_argument(
        "--results",
        default="data/winrate_results/olmo_wf/winrate_olmo_results.jsonl",
        help="Path to winrate_olmo_results.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/winrate_results/olmo_wf/winrate_olmo.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    rows  = load_results(args.results)
    stats = compute_stats(rows)

    # Print table to stdout as well
    print(f"\n{'Comparison':<22} {'WR%':>7}  {'95% CI':>17}  {'Wins':>5} {'Losses':>7} {'Ties':>5}")
    print("─" * 70)
    for name in COMPARISONS:
        s = stats[name]
        ci = f"[{s['ci_lo']*100:.1f}, {s['ci_hi']*100:.1f}]"
        print(
            f"{name:<22} {s['winrate']*100:>6.1f}%  {ci:>17}  "
            f"{s['wins']:>5} {s['losses']:>7} {s['ties']:>5}"
        )
    print()

    make_chart(stats, n_total=len(rows), output_path=args.output)


if __name__ == "__main__":
    main()
