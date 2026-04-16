#!/usr/bin/env python3
"""
Phase 3: Aggregate all scores and produce training curves.

Walks eval_results/*/step-*/**/scores.json, collects into CSV + plots.

Usage:
  python scripts/eval/aggregate_scores.py --results_root eval_results
"""

import argparse
import json
import re
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_neurips_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def step_number(step_name):
    if step_name == "final":
        return 999999
    m = re.search(r"\d+", step_name)
    return int(m.group()) if m else 0


def get_primary_metric(benchmark_name, scores):
    """Extract the primary metric for a benchmark from its scores.json."""
    if benchmark_name == "alpaca_eval":
        return scores.get("win_rate_no_ties", scores.get("win_rate", 0))
    elif benchmark_name == "arena_hard":
        return scores.get("win_rate_no_ties", scores.get("win_rate", 0))
    elif benchmark_name == "mt_bench":
        return scores.get("overall_score", 0)
    elif benchmark_name == "math500":
        return scores.get("accuracy", 0)
    elif benchmark_name == "reasoning_gym":
        return scores.get("overall_accuracy", 0)
    elif benchmark_name == "wildfeedback_held":
        return scores.get("win_rate_no_ties", scores.get("win_rate", 0))
    elif benchmark_name == "writingbench":
        return scores.get("overall_score", 0)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="eval_results")
    parser.add_argument("--output", default="eval_results/aggregate.csv")
    parser.add_argument("--plots_dir", default="plots/benchmark_curves")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = []

    # Collect all scores
    for scores_json in sorted(root.rglob("scores.json")):
        parts = scores_json.relative_to(root).parts
        if len(parts) < 3:
            continue
        method = parts[0]
        step_name = parts[1]
        benchmark = parts[2]

        with open(scores_json) as f:
            scores = json.load(f)

        metric = get_primary_metric(benchmark, scores)
        step = step_number(step_name)

        rows.append({
            "method": method,
            "step_name": step_name,
            "step": step,
            "benchmark": benchmark,
            "metric": metric,
            "scores_path": str(scores_json),
        })

    if not rows:
        print("No scores found!")
        return

    # Save CSV
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "step_name", "step", "benchmark", "metric"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ["method", "step_name", "step", "benchmark", "metric"]})
    print(f"Saved {len(rows)} rows to {csv_path}")

    # Print summary table
    benchmarks = sorted(set(r["benchmark"] for r in rows))
    methods = sorted(set(r["method"] for r in rows))

    print(f"\n{'Method':<25}", end="")
    for b in benchmarks:
        print(f" {b:<15}", end="")
    print()
    print("-" * (25 + 15 * len(benchmarks)))

    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        # Get best score per benchmark
        print(f"{method:<25}", end="")
        for b in benchmarks:
            b_rows = [r for r in method_rows if r["benchmark"] == b]
            if b_rows:
                best = max(r["metric"] for r in b_rows)
                print(f" {best:<15.1f}", end="")
            else:
                print(f" {'—':<15}", end="")
        print()

    # Plot training curves
    set_neurips_style()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for benchmark in benchmarks:
        fig, ax = plt.subplots(figsize=(6, 4))

        for method, color in zip(methods, colors):
            b_rows = sorted(
                [r for r in rows if r["method"] == method and r["benchmark"] == benchmark],
                key=lambda r: r["step"]
            )
            if not b_rows:
                continue
            steps = [r["step"] for r in b_rows]
            metrics = [r["metric"] for r in b_rows]
            label = method.replace("_p30", "")
            ax.plot(steps, metrics, "o-", color=color, markersize=3, linewidth=1.5, label=label)

        ax.set_xlabel("Training step")
        ax.set_ylabel(f"{benchmark} metric")
        ax.legend(loc="best", ncol=2)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path = plots_dir / f"{benchmark}_curves.pdf"
        fig.savefig(out_path)
        fig.savefig(out_path.with_suffix(".png"))
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\nAll plots in {plots_dir}/")


if __name__ == "__main__":
    main()
