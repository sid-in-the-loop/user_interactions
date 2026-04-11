#!/usr/bin/env python3
"""
Plot AlpacaEval LC win rate learning curves for OLMo FKL variants T-2, T-4, T-5.

Reads per-step leaderboard.csv files written by alpaca_eval_daemon.py.
Produces:
  - One combined plot: LC winrate vs training step, NeurIPS style
  - Summary table: peak LC winrate and step for each variant

Usage:
    python scripts/eval/plot_olmo_fkl_curves.py \\
        --eval_dir /data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3/alpaca_eval \\
        --output   /data/group_data/cx_group/ssmurali/offpolicy/fkl/olmo3/fkl_curves.png
"""

from __future__ import annotations

import argparse
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

VARIANTS = {
    "T2": ("olmo_fkl_T2", "T-2 (soft FKL)",         "#2E86C1"),
    "T4": ("olmo_fkl_T4", "T-4 (contrastive FKL)",  "#27AE60"),
    "T5": ("olmo_fkl_T5", "T-5 (JSD β=0.5)",        "#E07B54"),
}


def parse_lc_winrate(leaderboard: Path, model_name: str) -> float | None:
    if not leaderboard.exists():
        return None
    lines = leaderboard.read_text().splitlines()
    if len(lines) < 2:
        return None
    header = [h.strip().strip('"') for h in lines[0].split(",")]
    try:
        lc_idx = header.index("length_controlled_winrate")
    except ValueError:
        try:
            lc_idx = header.index("win_rate")
        except ValueError:
            return None
    for line in lines[1:]:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if parts and parts[0] == model_name:
            try:
                return float(parts[lc_idx])
            except (ValueError, IndexError):
                pass
    return None


def load_curve(eval_dir: Path, run_name: str) -> list[tuple[int, float]]:
    """Scan all step-XXXXXX dirs for a run, return sorted (step, lc_winrate) pairs."""
    run_dir = eval_dir / run_name
    if not run_dir.exists():
        return []
    points = []
    for step_dir in sorted(run_dir.glob("step-*")):
        try:
            step = int(step_dir.name.replace("step-", ""))
        except ValueError:
            continue
        leaderboard = step_dir / "weighted_alpaca_eval_gpt4_turbo" / "leaderboard.csv"
        model_name  = f"{run_name}_step{step}"
        wr = parse_lc_winrate(leaderboard, model_name)
        if wr is not None and not math.isnan(wr):
            points.append((step, wr))
    return sorted(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True,
                        help="Root eval output directory (contains olmo_fkl_T2/, etc.)")
    parser.add_argument("--output",   default="fkl_curves.png")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    data: dict[str, list[tuple[int, float]]] = {}
    for key, (run_name, label, color) in VARIANTS.items():
        data[key] = load_curve(eval_dir, run_name)
        print(f"{label}: {len(data[key])} eval points")

    if not any(data.values()):
        print("No eval data found. Run training and daemon first.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for key, (run_name, label, color) in VARIANTS.items():
        pts = data[key]
        if not pts:
            continue
        steps = [p[0] for p in pts]
        wrs   = [p[1] * 100 for p in pts]
        ax.plot(steps, wrs, marker="o", markersize=4, linewidth=1.5,
                color=color, label=label, zorder=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    ax.set_axisbelow(True)

    ax.set_xlabel("Training step",     fontsize=10)
    ax.set_ylabel("AlpacaEval LC Win Rate, %", fontsize=10)
    ax.set_title("OLMo-3-7B FKL variants — AlpacaEval LC Win Rate (GPT-4o-mini judge)\n"
                 "WildFeedback, Prompt C teacher ($\\it{Note: o}$)", fontsize=9, pad=8)
    ax.legend(fontsize=8.5, framealpha=0.4, loc="best")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # Summary table
    print()
    print("=" * 62)
    print("  OLMo FKL — Peak AlpacaEval LC Win Rate")
    print("=" * 62)
    print(f"  {'Variant':<28}  {'Peak WR':>8}  {'At step':>8}")
    print("  " + "─" * 46)
    for key, (run_name, label, color) in VARIANTS.items():
        pts = data[key]
        if not pts:
            print(f"  {label:<28}  {'—':>8}  {'—':>8}")
            continue
        best_step, best_wr = max(pts, key=lambda p: p[1])
        print(f"  {label:<28}  {best_wr*100:>7.1f}%  {best_step:>8}")
    print("=" * 62)


if __name__ == "__main__":
    main()
