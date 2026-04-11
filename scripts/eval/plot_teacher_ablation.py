#!/usr/bin/env python3
"""
Plot teacher-prompt ablation results for OLMo-3-7B-Instruct (NeurIPS style).

Two-panel figure:
  Left  — y*_{A,B,C,D} vs y_base   (primary metric; 50% = coin flip reference)
  Right — y*_{B,C,D}   vs y*_A     (relative to current best)

Usage:
    python scripts/eval/plot_teacher_ablation.py \\
        --results-dir data/winrate_results/olmo_teacher_ablation \\
        --output      data/winrate_results/olmo_teacher_ablation/teacher_ablation.png
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

PROMPT_LABELS = {
    "A": "A\n(paper xo)",
    "B": "B\n(explicit)",
    "C": "C\n(minimal)",
    "D": "D\n(Socratic)",
}

COLORS_VS_YBASE = {
    "A": "#2E86C1",
    "B": "#27AE60",
    "C": "#E07B54",
    "D": "#9B59B6",
}
COLORS_VS_A = {
    "B": "#27AE60",
    "C": "#E07B54",
    "D": "#9B59B6",
}


def wilson_ci(wins: int, losses: int) -> tuple[float, float, float]:
    n = wins + losses
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    z = 1.96
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, center - margin, center + margin


def load_winrate(path: Path, comp_name: str) -> tuple[int, int, int]:
    """Parse winrate_results.jsonl from winrate_eval.py pair mode."""
    wins = losses = ties = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        comp = r.get("comparisons", {}).get(comp_name, {})
        w = comp.get("winner", "")
        if w == "1":
            wins += 1
        elif w == "2":
            losses += 1
        elif w == "tie":
            ties += 1
    return wins, losses, ties


def spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    ax.set_axisbelow(True)


def bar_with_ci(ax, x, wr, lo, hi, color, label, width=0.55):
    ax.bar(x, wr * 100, width=width, color=color, alpha=0.82, label=label, zorder=3)
    ax.errorbar(
        x, wr * 100,
        yerr=[[wr * 100 - lo * 100], [hi * 100 - wr * 100]],
        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4,
    )
    ax.text(x, wr * 100 + (hi - wr) * 100 + 1.2, f"{wr*100:.1f}%",
            ha="center", va="bottom", fontsize=8.5, color="black")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/winrate_results/olmo_teacher_ablation")
    parser.add_argument("--output",      default="data/winrate_results/olmo_teacher_ablation/teacher_ablation.png")
    args = parser.parse_args()

    d = Path(args.results_dir)

    # ── Load all 7 comparisons ─────────────────────────────────────────────────
    vs_ybase: dict[str, tuple] = {}
    vs_A:     dict[str, tuple] = {}

    for prompt in ["A", "B", "C", "D"]:
        f = d / f"ystar_{prompt}_vs_ybase" / "winrate_results.jsonl"
        comp = f"y*_{prompt} vs y_base"
        w, l, t = load_winrate(f, comp)
        wr, lo, hi = wilson_ci(w, l)
        vs_ybase[prompt] = (wr, lo, hi, w, l, t)

    for prompt in ["B", "C", "D"]:
        f = d / f"ystar_{prompt}_vs_A" / "winrate_results.jsonl"
        comp = f"y*_{prompt} vs y*_A"
        w, l, t = load_winrate(f, comp)
        wr, lo, hi = wilson_ci(w, l)
        vs_A[prompt] = (wr, lo, hi, w, l, t)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   gridspec_kw={"width_ratios": [4, 3]})

    # Left panel — vs y_base
    prompts_left = ["A", "B", "C", "D"]
    xs = np.arange(len(prompts_left))
    for i, p in enumerate(prompts_left):
        wr, lo, hi, w, l, t = vs_ybase[p]
        bar_with_ci(ax1, xs[i], wr, lo, hi, COLORS_VS_YBASE[p],
                    label=f"{PROMPT_LABELS[p].split(chr(10))[0]} (W={w} L={l} T={t})")

    ax1.axhline(50, color="gray", linestyle="--", linewidth=0.9, alpha=0.6, label="50% (coin flip)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([PROMPT_LABELS[p] for p in prompts_left], fontsize=9)
    ax1.set_ylabel("Win rate, %", fontsize=10)
    ax1.set_title("$y^*$ vs $y_{\\rm base}$\n(higher = teacher prompt better than baseline)", fontsize=9, pad=8)
    ax1.set_ylim(0, 75)
    ax1.legend(fontsize=7.5, framealpha=0.4, loc="upper right")
    spine_style(ax1)

    # Right panel — vs y*_A
    prompts_right = ["B", "C", "D"]
    xs2 = np.arange(len(prompts_right))
    for i, p in enumerate(prompts_right):
        wr, lo, hi, w, l, t = vs_A[p]
        bar_with_ci(ax2, xs2[i], wr, lo, hi, COLORS_VS_A[p],
                    label=f"{PROMPT_LABELS[p].split(chr(10))[0]} (W={w} L={l} T={t})")

    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([PROMPT_LABELS[p] for p in prompts_right], fontsize=9)
    ax2.set_ylabel("Win rate vs Prompt A, %", fontsize=10)
    ax2.set_title("$y^*_{B,C,D}$ vs $y^*_A$\n(>50% means better than paper template)", fontsize=9, pad=8)
    ax2.set_ylim(0, 75)
    ax2.legend(fontsize=7.5, framealpha=0.4, loc="upper right")
    spine_style(ax2)

    fig.suptitle("OLMo-3-7B-Instruct teacher prompt ablation (WildFeedback, n≈500)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
