#!/usr/bin/env python3
"""
Figure 1: Win Rate + Response Length over training steps — RKL vs JSD.

Dual y-axes: win rate (left), avg response length (right).
Shows RKL collapse (WR rises then falls, length explodes) vs JSD stability.

Usage:
  python scripts/eval/plot_rkl_vs_jsd.py --output plots/rkl_vs_jsd.pdf
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── NeurIPS style ────────────────────────────────────────────────────────────

def set_neurips_style():
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
        "axes.grid":            False,
        "figure.dpi":           150,
        "savefig.dpi":          900,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
    })


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jsd(base_dir: str) -> list[dict]:
    """Load JSD checkpoint data from eval_results/alpaca/ structure."""
    rows = []
    base = Path(base_dir)
    for step_dir in sorted(base.iterdir()):
        if not step_dir.is_dir():
            continue
        scores_f = step_dir / "scores.json"
        outputs_f = step_dir / "model_outputs.json"
        if not scores_f.exists() or not outputs_f.exists():
            continue
        scores = json.loads(scores_f.read_text())
        outputs = json.loads(outputs_f.read_text())
        avg_len = sum(len(o["output"]) for o in outputs) / len(outputs)
        rows.append({
            "step": scores["step"],
            "win_rate": scores["win_rate"],
            "lc_win_rate": scores["lc_win_rate"],
            "avg_len": avg_len,
        })
    return sorted(rows, key=lambda r: r["step"])


def load_rkl(base_dir: str) -> list[dict]:
    """Load RKL checkpoint data from dist=sample-sdpo structure."""
    rows = []
    base = Path(base_dir)
    for ckpt_dir in sorted(base.iterdir()):
        if not ckpt_dir.is_dir() or "samples" not in ckpt_dir.name:
            continue
        csv_f = ckpt_dir / "weighted_alpaca_eval_gpt-4o-mini-2024-07-18" / "leaderboard.csv"
        outputs_f = ckpt_dir / "model_outputs.json"
        if not csv_f.exists() or not outputs_f.exists():
            continue
        with open(csv_f) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        outputs = json.loads(outputs_f.read_text())
        avg_len = sum(len(o["output"]) for o in outputs) / len(outputs)
        # Extract step number from dirname
        name = ckpt_dir.name  # e.g. checkpoint-350-44544samples
        step = int(name.split("-")[1])
        rows.append({
            "step": step,
            "win_rate": float(row["win_rate"]),
            "lc_win_rate": float(row["length_controlled_winrate"]),
            "avg_len": avg_len,
        })
    return sorted(rows, key=lambda r: r["step"])


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_dual_axes(jsd_data, rkl_data, output_path, metric="lc_win_rate"):
    set_neurips_style()

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_alpha(0.4)
    ax2.spines["top"].set_visible(False)

    # Use checkpoint index as x-axis (both have ~10 evenly-spaced ckpts)
    n_jsd = len(jsd_data)
    n_rkl = len(rkl_data)
    jsd_x = np.arange(1, n_jsd + 1)
    rkl_x = np.arange(1, n_rkl + 1)

    jsd_wr = np.array([d[metric] for d in jsd_data])
    rkl_wr = np.array([d[metric] for d in rkl_data])
    jsd_len = np.array([d["avg_len"] for d in jsd_data])
    rkl_len = np.array([d["avg_len"] for d in rkl_data])

    # Colors
    c_jsd = "#2166AC"   # blue
    c_rkl = "#B2182B"   # red

    # Win rate lines (left axis, solid)
    l1, = ax1.plot(jsd_x, jsd_wr, "o-", color=c_jsd, markersize=5, linewidth=2.0,
                   zorder=5, label="JSD — win rate")
    l2, = ax1.plot(rkl_x, rkl_wr, "s-", color=c_rkl, markersize=5, linewidth=2.0,
                   zorder=5, label="RKL — win rate")

    # Length lines (right axis, dashed)
    l3, = ax2.plot(jsd_x, jsd_len, "o--", color=c_jsd, markersize=3, linewidth=1.0,
                   alpha=0.45, zorder=4, label="JSD — avg length")
    l4, = ax2.plot(rkl_x, rkl_len, "s--", color=c_rkl, markersize=3, linewidth=1.0,
                   alpha=0.45, zorder=4, label="RKL — avg length")

    # Axes labels
    ax1.set_xlabel("Checkpoint (evenly spaced over training)")
    metric_label = "LC Win Rate (%)" if metric == "lc_win_rate" else "Win Rate (%)"
    ax1.set_ylabel(metric_label)
    ax2.set_ylabel("Avg response length (chars)", color="#777")
    ax2.tick_params(axis="y", colors="#888")

    # x ticks
    max_n = max(n_jsd, n_rkl)
    ax1.set_xticks(range(1, max_n + 1))
    ax1.set_xlim(0.5, max_n + 0.5)

    # y limits
    ax1.set_ylim(30, 102)

    # 50% reference
    ax1.axhline(50, color="#999", linewidth=0.7, linestyle=":", alpha=0.5)

    # Spine styling
    ax1.spines["left"].set_alpha(0.4)
    ax1.spines["bottom"].set_alpha(0.4)

    # Grid
    ax1.yaxis.grid(True, linestyle="--", alpha=0.2, color="#888")
    ax1.set_axisbelow(True)

    # Legend
    ax1.legend([l1, l2, l3, l4],
               ["JSD — win rate", "RKL — win rate",
                "JSD — avg length", "RKL — avg length"],
               loc="lower left", frameon=True, framealpha=0.9,
               edgecolor="#ccc", ncol=2, columnspacing=1.0,
               handlelength=2.2, fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsd-dir",
                        default="eval_results/alpaca/qwen3_8b_jsd_p30_wfbest")
    parser.add_argument("--rkl-dir",
                        default="dist=sample-sdpo-logpscale-filtered")
    parser.add_argument("--output", default="plots/rkl_vs_jsd.pdf")
    parser.add_argument("--metric", default="lc_win_rate",
                        choices=["win_rate", "lc_win_rate"])
    args = parser.parse_args()

    jsd_data = load_jsd(args.jsd_dir)
    rkl_data = load_rkl(args.rkl_dir)

    print(f"JSD: {len(jsd_data)} checkpoints")
    print(f"RKL: {len(rkl_data)} checkpoints")

    plot_dual_axes(jsd_data, rkl_data, args.output, args.metric)

    # Also generate raw WR version
    if args.metric == "lc_win_rate":
        raw_output = args.output.replace(".pdf", "_raw_wr.pdf").replace(".png", "_raw_wr.png")
        plot_dual_axes(jsd_data, rkl_data, raw_output, "win_rate")


if __name__ == "__main__":
    main()
