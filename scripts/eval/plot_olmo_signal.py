#!/usr/bin/env python3
"""
Plot OLMo feedback signal diagnostic results (NeurIPS style).

Three plots:
  1. Histogram of per-sample mean delta by tier
  2. Mean delta vs sequence position (token index), one curve per tier
  3. Scatter of per-sample mean delta colored by tier

Usage:
    python scripts/eval/plot_olmo_signal.py \\
        --input  data/olmo_signal/olmo_signal.jsonl \\
        --output_dir data/olmo_signal
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

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

TIER_ORDER  = ["BEST", "DECENT", "NOISE", "SWITCH", "BAD", "UNCATEGORIZED", "UNKNOWN"]
TIER_COLORS = {
    "BEST":          "#2E86C1",
    "DECENT":        "#27AE60",
    "NOISE":         "#E07B54",
    "SWITCH":        "#9B59B6",
    "BAD":           "#C0392B",
    "UNCATEGORIZED": "#7F8C8D",
    "UNKNOWN":       "#BDC3C7",
}


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    ax.set_axisbelow(True)


# ── Plot 1: histogram by tier ──────────────────────────────────────────────────

def plot_histogram(records: list[dict], out_path: str) -> None:
    by_tier = defaultdict(list)
    for r in records:
        if not math.isnan(r["mean_delta"]) and r["n_tokens"] > 0:
            by_tier[r["tier"]].append(r["mean_delta"])

    active = [t for t in TIER_ORDER if by_tier[t]]
    fig, ax = plt.subplots(figsize=(7, 4))

    for tier in active:
        vals = by_tier[tier]
        ax.hist(
            vals,
            bins=40,
            alpha=0.55,
            color=TIER_COLORS.get(tier, "#888888"),
            label=f"{tier} (n={len(vals)})",
            density=True,
            linewidth=0.5,
            edgecolor="white",
        )

    ax.set_xlabel("Per-sample mean $\\Delta$ (nats)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "Distribution of per-sample KL shift by feedback tier\n"
        r"$\Delta = \mathbb{E}_n[\mathrm{KL}(p(\cdot|x,y,o,y^{\rm base}_{<n})\,\|\,p(\cdot|x,y^{\rm base}_{<n}))]$",
        fontsize=9, pad=8,
    )
    ax.legend(fontsize=8, framealpha=0.4)
    spine_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Plot 2: delta vs sequence position ─────────────────────────────────────────

def plot_position(records: list[dict], out_path: str, max_pos: int = 256) -> None:
    """
    For each tier, average per_token_kl[n] across samples that have at least n+1 tokens.
    """
    by_tier: dict[str, list[list[float]]] = defaultdict(list)
    for r in records:
        if r["per_token_kl"] and r["n_tokens"] > 0:
            by_tier[r["tier"]].append(r["per_token_kl"])

    active = [t for t in TIER_ORDER if by_tier[t]]
    fig, ax = plt.subplots(figsize=(8, 4))

    for tier in active:
        seqs = by_tier[tier]
        max_len = min(max_pos, max(len(s) for s in seqs))
        means, stderrs, positions = [], [], []
        for pos in range(max_len):
            vals = [s[pos] for s in seqs if len(s) > pos]
            if len(vals) < 3:
                continue
            mu = np.mean(vals)
            se = np.std(vals, ddof=1) / math.sqrt(len(vals))
            means.append(mu)
            stderrs.append(se)
            positions.append(pos)

        if not positions:
            continue
        positions = np.array(positions)
        means     = np.array(means)
        stderrs   = np.array(stderrs)
        color = TIER_COLORS.get(tier, "#888888")
        ax.plot(positions, means, color=color, linewidth=1.4, label=tier)
        ax.fill_between(positions, means - stderrs, means + stderrs,
                        color=color, alpha=0.15)

    ax.set_xlabel("Token position in $y^{\\rm base}$", fontsize=10)
    ax.set_ylabel("Mean KL shift (nats)", fontsize=10)
    ax.set_title(
        "KL shift vs. sequence position by feedback tier",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=8, framealpha=0.4)
    spine_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Plot 3: scatter per-sample delta colored by tier ──────────────────────────

def plot_scatter(records: list[dict], out_path: str) -> None:
    active_tiers = [t for t in TIER_ORDER
                    if any(r["tier"] == t for r in records if not math.isnan(r["mean_delta"]))]

    fig, ax = plt.subplots(figsize=(8, 4))

    for tier in active_tiers:
        recs = [r for r in records if r["tier"] == tier and not math.isnan(r["mean_delta"])]
        if not recs:
            continue
        xs = list(range(len(recs)))  # sample index within tier (for spread)
        ys = [r["mean_delta"] for r in recs]
        # jitter x by tier offset for visual separation
        tier_offset = active_tiers.index(tier)
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(xs))
        ax.scatter(
            [tier_offset + j for j in jitter],
            ys,
            color=TIER_COLORS.get(tier, "#888888"),
            alpha=0.5,
            s=14,
            label=f"{tier} (n={len(recs)})",
            linewidths=0,
        )
        # tier mean line
        mu = np.mean(ys)
        ax.plot([tier_offset - 0.4, tier_offset + 0.4], [mu, mu],
                color=TIER_COLORS.get(tier, "#888888"), linewidth=2.0)

    ax.set_xticks(range(len(active_tiers)))
    ax.set_xticklabels(active_tiers, fontsize=9)
    ax.set_ylabel("Per-sample mean $\\Delta$ (nats)", fontsize=10)
    ax.set_title("Per-sample KL shift by feedback tier", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.4, loc="upper right")
    spine_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OLMo feedback signal diagnostic.")
    parser.add_argument("--input",      default="data/olmo_signal/olmo_signal.jsonl")
    parser.add_argument("--output_dir", default="data/olmo_signal")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)
    print(f"Loaded {len(records)} records")

    plot_histogram(records, str(out_dir / "signal_histogram.png"))
    plot_position( records, str(out_dir / "signal_position.png"))
    plot_scatter(  records, str(out_dir / "signal_scatter.png"))

    print("Done.")


if __name__ == "__main__":
    main()
