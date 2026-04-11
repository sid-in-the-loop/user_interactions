#!/usr/bin/env python3
"""
Plot probe signal summary: mean KL by checkpoint and category (unrelated vs corrective).
Usage: python scripts/fkl/plot_probe_signal.py [--input results/probe_signal_summary.csv] [--output results/probe_signal_plot.png]
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "probe_signal_summary.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "probe_signal_plot.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy", file=__import__("sys").stderr)
        raise SystemExit(1)

    rows = []
    with open(args.input) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Order checkpoints for display
    order = ["baseline_sft_fp32", "baseline_v1_s500", "baseline_v1", "baseline_v2_s500", "baseline_v2_s1000", "baseline_v2"]
    seen = {r["checkpoint"] for r in rows}
    labels = [c for c in order if c in seen]
    if len(labels) < len(seen):
        labels = sorted(seen)

    by_ckpt = {}
    for r in rows:
        ckpt = r["checkpoint"]
        cat = r["category"]
        by_ckpt.setdefault(ckpt, {})[cat] = {
            "mean": float(r["mean_kl"]),
            "std": float(r["std_kl"]),
        }

    x = np.arange(len(labels))
    w = 0.35
    unrel_means = [by_ckpt[c]["unrelated"]["mean"] if "unrelated" in by_ckpt.get(c, {}) else 0 for c in labels]
    unrel_stds = [by_ckpt[c]["unrelated"]["std"] if "unrelated" in by_ckpt.get(c, {}) else 0 for c in labels]
    corr_means = [by_ckpt[c]["corrective"]["mean"] if "corrective" in by_ckpt.get(c, {}) else 0 for c in labels]
    corr_stds = [by_ckpt[c]["corrective"]["std"] if "corrective" in by_ckpt.get(c, {}) else 0 for c in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, unrel_means, w, yerr=unrel_stds, label="unrelated", capsize=3, color="tab:orange", alpha=0.9)
    ax.bar(x + w / 2, corr_means, w, yerr=corr_stds, label="corrective", capsize=3, color="tab:blue", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean per-token KL")
    ax.set_title("FKL probe signal: KL(π(·|x,o,y_{<i}) || π(·|x,y_{<i})) by category")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
