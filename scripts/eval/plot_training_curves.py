#!/usr/bin/env python3
"""
2×2 grid: rows = condition (prefix30, full), cols = dataset (wfbest, wffull)
Each subplot: 3 lines (SFT, FKL, JSD) + horizontal dashed line for base model.
x = optimizer step, y = lc_win_rate (or win_rate).

Usage:
    python scripts/eval/plot_training_curves.py
    python scripts/eval/plot_training_curves.py --metric win_rate --out plots/win_rate.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.linestyle":  "--",
    "grid.alpha":      0.4,
})

RESULTS_ROOT = Path("eval_results/alpaca")

METHOD_STYLE = {
    "sft": dict(color="#2166ac", marker="o", label="SFT"),
    "fkl": dict(color="#d6604d", marker="s", label="FKL"),
    "jsd": dict(color="#4dac26", marker="^", label="JSD"),
}

SUBPLOT_TITLES = {
    ("p30",  "wfbest"): "prefix30 · WF-BEST",
    ("p30",  "wffull"): "prefix30 · WF-FULL",
    ("full", "wfbest"): "full prefix · WF-BEST",
    ("full", "wffull"): "full prefix · WF-FULL",
}


def load_scores(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    pts = []
    for score_file in run_dir.rglob("scores.json"):
        d = json.loads(score_file.read_text())
        step = d.get("step")
        val  = d.get(metric)
        if step is not None and val is not None and step != 999999:
            pts.append((step, val))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="lc_win_rate", choices=["lc_win_rate", "win_rate"])
    ap.add_argument("--out",    default="plots/training_curves.pdf")
    ap.add_argument("--results", default=str(RESULTS_ROOT))
    args = ap.parse_args()

    results_root = Path(args.results)

    # Base model score (horizontal line)
    base_score = None
    base_path  = results_root / "base" / "scores.json"
    if base_path.exists():
        base_score = json.loads(base_path.read_text()).get(args.metric)

    conditions = ["p30", "full"]
    datasets   = ["wfbest", "wffull"]
    methods    = ["sft", "fkl", "jsd"]

    # Collect all values to auto-scale y-axis
    all_vals = []
    all_data = {}
    for cond in conditions:
        for ds in datasets:
            for method in methods:
                run_name = f"qwen3_8b_{method}_{cond}_{ds}"
                pts = load_scores(results_root / run_name, args.metric)
                all_data[(cond, ds, method)] = pts
                all_vals += [v for _, v in pts]
    if base_score is not None:
        all_vals.append(base_score)
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        pad = (vmax - vmin) * 0.15
        ylim = (vmin - pad, vmax + pad)
    else:
        ylim = (75, 90)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False, sharey=True)
    fig.suptitle(f"Qwen3-8B AlpacaEval · {args.metric.replace('_', ' ')}", fontsize=13)

    for row, cond in enumerate(conditions):
        for col, ds in enumerate(datasets):
            ax = axes[row][col]
            ax.set_title(SUBPLOT_TITLES[(cond, ds)], fontsize=10)

            plotted_any = False
            for method in methods:
                pts = all_data[(cond, ds, method)]
                if not pts:
                    continue
                steps, vals = zip(*pts)
                style = METHOD_STYLE[method]
                ax.plot(steps, vals, marker=style["marker"], color=style["color"],
                        label=style["label"], linewidth=1.8, markersize=5)
                for x, y in zip(steps, vals):
                    ax.annotate(f"{y:.1f}", xy=(x, y),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom",
                                fontsize=6.5, color=style["color"])
                plotted_any = True

            if base_score is not None:
                ax.axhline(base_score, color="gray", linestyle="--",
                           linewidth=1.2, label=f"Base ({base_score:.1f})")

            ax.set_xlabel("Optimizer step")
            ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(args.metric.replace("_", " ") + " (%)")

            if plotted_any or base_score is not None:
                ax.legend(fontsize=9)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")

    # Also print a quick table
    print(f"\n{'Run':<40} {'Steps':>6}  {args.metric}")
    print("-" * 60)
    for cond in conditions:
        for ds in datasets:
            for method in methods:
                run_name = f"qwen3_8b_{method}_{cond}_{ds}"
                pts = load_scores(results_root / run_name, args.metric)
                if pts:
                    best_step, best_val = max(pts, key=lambda x: x[1])
                    print(f"  {run_name:<38} {best_step:>6}   {best_val:.1f}%")
    if base_score:
        print(f"\n  {'base':<38}           {base_score:.1f}%")


if __name__ == "__main__":
    main()
