#!/usr/bin/env python3
"""
Plot AlpacaEval (Gemini judge) win_rate and lc_win_rate over training steps.

Layout: 2 rows (win_rate / lc_win_rate) × 4 cols (p30_wfbest, p30_wffull,
        full_wfbest, full_wffull).
Lines : SFT (blue), FKL (red), JSD (green).
Base  : gray dashed horizontal reference.

Usage:
    python scripts/eval/plot_alpaca_gemini_ckpts.py \
        --results_root eval_results/alpaca_gemini \
        --output docs/figures/alpaca_gemini_ckpts.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "grid.color":         "#888888",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth":    1.5,
})

BLUE  = "#2166ac"
RED   = "#d6604d"
GREEN = "#4dac26"
GRAY  = "#878787"

OBJ_STYLE = {
    "sft": dict(color=BLUE,  label="SFT",  marker="o"),
    "fkl": dict(color=RED,   label="FKL",  marker="s"),
    "jsd": dict(color=GREEN, label="JSD",  marker="^"),
}

COLS = [
    ("p30", "wfbest"),
    ("p30", "wffull"),
    ("full", "wfbest"),
    ("full", "wffull"),
]
COL_TITLES = [
    "prefix30 · wfbest",
    "prefix30 · wffull",
    "full · wfbest",
    "full · wffull",
]


def load_run(results_root: Path, run_name: str) -> list[dict]:
    """Return sorted list of {step, win_rate, lc_win_rate} for one run."""
    run_dir = results_root / run_name
    if not run_dir.exists():
        return []
    points = []
    for step_dir in run_dir.iterdir():
        sc = step_dir / "scores.json"
        if not sc.exists():
            continue
        d = json.loads(sc.read_text())
        points.append(d)
    # sort by step; treat 999999 (final) as last
    points.sort(key=lambda d: d["step"])
    return points


def get_xs(points: list[dict], final_x: float) -> list[float]:
    """Map step numbers to x coords; replace 999999 with final_x."""
    return [final_x if p["step"] == 999999 else p["step"] for p in points]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="eval_results/alpaca_gemini")
    parser.add_argument("--output", default="docs/figures/alpaca_gemini_ckpts.png")
    args = parser.parse_args()

    root = Path(args.results_root)

    # Load base model scores
    base_sc = root / "base" / "scores.json"
    base_wr = base_lc = None
    if base_sc.exists():
        bd = json.loads(base_sc.read_text())
        base_wr = bd["win_rate"]
        base_lc = bd["lc_win_rate"]

    # Build data dict: data[(prefix, dataset)][objective] = [(step, wr, lc), ...]
    data = {}
    for prefix, dataset in COLS:
        data[(prefix, dataset)] = {}
        for obj in ("sft", "fkl", "jsd"):
            run_name = f"qwen3_8b_{obj}_{prefix}_{dataset}"
            pts = load_run(root, run_name)
            data[(prefix, dataset)][obj] = pts

    # Determine x range (max real step across all runs)
    all_real_steps = []
    for cond in data.values():
        for pts in cond.values():
            all_real_steps += [p["step"] for p in pts if p["step"] != 999999]
    max_step = max(all_real_steps) if all_real_steps else 200
    final_x  = max_step * 1.12   # position for "final" checkpoint

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharey="row")
    fig.subplots_adjust(hspace=0.38, wspace=0.22)

    metrics = [
        ("win_rate",    "Win Rate (%) — Gemini judge",    (55, 72)),
        ("lc_win_rate", "LC Win Rate (%) — Gemini judge", (35, 48)),
    ]

    for row, (metric_key, ylabel, ylim) in enumerate(metrics):
        for col, ((prefix, dataset), col_title) in enumerate(zip(COLS, COL_TITLES)):
            ax = axes[row][col]

            # Base horizontal reference
            base_val = base_wr if metric_key == "win_rate" else base_lc
            if base_val is not None:
                ax.axhline(base_val, color=GRAY, linewidth=1.0,
                           linestyle="--", zorder=2, alpha=0.8,
                           label=f"Base ({base_val:.1f}%)")

            for obj, style in OBJ_STYLE.items():
                pts = data[(prefix, dataset)][obj]
                if not pts:
                    continue
                xs = get_xs(pts, final_x)
                ys = [p[metric_key] for p in pts]
                ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                        markersize=4, linewidth=1.4, label=style["label"], zorder=3)

            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(4))

            # X-axis: real steps + "final" tick
            real_ticks = sorted(set(
                p["step"] for cond_data in data.values()
                for obj_pts in cond_data.values()
                for p in obj_pts
                if p["step"] != 999999
            ))
            tick_positions = list(real_ticks) + [final_x]
            tick_labels    = [str(t) for t in real_ticks] + ["final"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
            ax.set_xlim(-5, final_x + max_step * 0.04)

            if row == 0:
                ax.set_title(col_title, fontsize=9, pad=5)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=4)
            if row == 1:
                ax.set_xlabel("Training step", labelpad=3, fontsize=8)

    # Single legend below
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=9,
               handlelength=1.8, handletextpad=0.5, columnspacing=1.2)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
