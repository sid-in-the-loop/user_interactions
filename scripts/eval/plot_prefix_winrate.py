"""
Plot win rate results for the y* prefix ablation (3 conditions) and save the
curated dataset.

Reads:
  - winrate_prefix_results.jsonl  (from winrate_prefix_eval.py)
  - ystar_prefix_best.jsonl       (full tuples, for filtering)

Produces (NeurIPS style, 900 DPI):
  - prefix_winrate_vs_y.png        — win rate of each condition vs y
  - prefix_winrate_h2h.png         — head-to-head win rates between conditions
  - prefix_winrate_scatter.png     — per-sample avg_score scatter (prefix30 vs noprefix)
  - ystar_prefix30_wins.jsonl      — samples where prefix30 beats y
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# NeurIPS style
# ─────────────────────────────────────────────────────────────────────────────

def set_neurips_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "grid.color":        "#888888",
        "axes.axisbelow":    True,
        "figure.dpi":        900,
        "savefig.dpi":       900,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
    })


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def win_rate(results: list[dict], key: str) -> tuple[float, int, int, int]:
    w = sum(1 for r in results if r.get(key, {}).get("outcome") == "win")
    l = sum(1 for r in results if r.get(key, {}).get("outcome") == "loss")
    t = sum(1 for r in results if r.get(key, {}).get("outcome") == "tie")
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    return wr, w, l, t


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: win rate of each condition vs y
# ─────────────────────────────────────────────────────────────────────────────

def plot_vs_y(results: list[dict], output_path: str, model_name: str = "") -> None:
    conditions = [
        ("prefix30_vs_y", "prefix30"),
        ("noprefix_vs_y", "noprefix"),
        ("full_vs_y",     "full"),
    ]
    # Filter to comparisons that exist in the data
    conditions = [(k, l) for k, l in conditions if any(k in r for r in results)]
    if not conditions:
        return

    labels = [l for _, l in conditions]
    rates  = []
    for key, _ in conditions:
        wr, w, l, t = win_rate(results, key)
        rates.append(wr)

    colors = ["#1565C0", "#78909C", "#1E88E5"][:len(conditions)]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(x, rates, width=0.5, color=colors[:len(conditions)], alpha=0.87)
    ax.axhline(50, color="#E53935", lw=0.9, linestyle="--", label="50% (chance)")

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win rate vs $y$  [wins / (wins + losses)]  (%)")
    title = f"Win Rate vs GPT-4 Baseline  (ties excluded)"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title, pad=8)
    ax.legend(fontsize=8, framealpha=0.6)
    ax.set_ylim(0, max(rates) * 1.25 if rates else 100)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved vs-y bar chart → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: head-to-head between conditions
# ─────────────────────────────────────────────────────────────────────────────

def plot_h2h(results: list[dict], output_path: str, model_name: str = "") -> None:
    h2h_comps = [
        ("prefix30_vs_noprefix", "prefix30\nvs noprefix"),
        ("prefix30_vs_full",     "prefix30\nvs full"),
        ("noprefix_vs_full",     "noprefix\nvs full"),
    ]
    h2h_comps = [(k, l) for k, l in h2h_comps if any(k in r for r in results)]
    if not h2h_comps:
        print("No head-to-head comparisons found, skipping h2h plot")
        return

    rates = []
    for key, _ in h2h_comps:
        wr, *_ = win_rate(results, key)
        rates.append(wr)

    labels = [l for _, l in h2h_comps]
    colors = ["#2E7D32", "#6A1B9A", "#E65100"][:len(h2h_comps)]
    x = np.arange(len(h2h_comps))

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(x, rates, width=0.5, color=colors, alpha=0.87)
    ax.axhline(50, color="#E53935", lw=0.9, linestyle="--", label="50% (chance)")

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win rate of left condition  (%)")
    title = "Head-to-Head Win Rates"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title, pad=8)
    ax.legend(fontsize=8, framealpha=0.6)
    ax.set_ylim(0, max(rates) * 1.25 if rates else 100)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved h2h bar chart → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: scatter — per-sample scores (prefix30 vs noprefix)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(results: list[dict], output_path: str) -> None:
    key_p  = "prefix30_vs_y"
    key_np = "noprefix_vs_y"
    if not any(key_p in r for r in results) or not any(key_np in r for r in results):
        print("Skipping scatter — missing prefix30_vs_y or noprefix_vs_y")
        return

    scores_p  = [r[key_p]["avg_score"]  for r in results if key_p  in r and key_np in r]
    scores_np = [r[key_np]["avg_score"] for r in results if key_p  in r and key_np in r]

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.015, 0.015, size=len(scores_p))
    xv = np.array(scores_p)  + jitter
    yv = np.array(scores_np) + jitter

    colors = ["#1976D2" if sp > snp else "#E53935" if sp < snp else "#78909C"
              for sp, snp in zip(scores_p, scores_np)]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="#BDBDBD", lw=1.0, zorder=0)
    ax.scatter(xv, yv, c=colors, s=12, alpha=0.55, linewidths=0)

    ax.set_xlabel("avg score: prefix30 vs $y$")
    ax.set_ylabel("avg score: noprefix vs $y$")
    ax.set_title("Per-sample Scores\n(blue=prefix30 better, red=noprefix better)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1976D2", markersize=7, label="prefix30 better"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#E53935", markersize=7, label="noprefix better"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#78909C", markersize=7, label="equal"),
    ], loc="upper left", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Curated dataset
# ─────────────────────────────────────────────────────────────────────────────

def save_prefix30_wins(results: list[dict], tuples: list[dict], output_path: str) -> None:
    win_ids = {
        (r["conversation_id"], r.get("turn_index"))
        for r in results
        if r.get("prefix30_vs_y", {}).get("outcome") == "win"
    }
    filtered = [
        t for t in tuples
        if (t.get("conversation_id"), t.get("turn_index")) in win_ids
    ]
    save_jsonl(filtered, output_path)
    n = len(tuples)
    print(f"Saved {len(filtered)} prefix30-win samples ({len(filtered)/max(n,1)*100:.1f}% of total) → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    default="data/winrate_results/prefix_ablation/winrate_prefix_results.jsonl")
    parser.add_argument("--tuples",     default="datasets/wildfeedback/ystar_prefix_best.jsonl")
    parser.add_argument("--output_dir", default="data/winrate_results/prefix_ablation")
    parser.add_argument("--model_name", default="", help="Model label for plot titles")
    args = parser.parse_args()

    set_neurips_style()

    results = load_jsonl(args.results)
    print(f"Loaded {len(results)} result rows from {args.results}")

    tuples = load_jsonl(args.tuples) if Path(args.tuples).exists() else []
    if not tuples:
        print(f"Warning: could not load tuples from {args.tuples}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_vs_y(results,   str(out / "prefix_winrate_vs_y.png"),   args.model_name)
    plot_h2h(results,    str(out / "prefix_winrate_h2h.png"),     args.model_name)
    plot_scatter(results, str(out / "prefix_winrate_scatter.png"))

    if tuples:
        save_prefix30_wins(results, tuples, str(out / "ystar_prefix30_wins.jsonl"))

    # Print summary table
    n = len(results)
    all_comps = [
        ("prefix30_vs_y",       "prefix30 vs y"),
        ("noprefix_vs_y",       "noprefix  vs y"),
        ("full_vs_y",           "full      vs y"),
        ("prefix30_vs_noprefix","prefix30 vs noprefix"),
        ("prefix30_vs_full",    "prefix30 vs full"),
        ("noprefix_vs_full",    "noprefix vs full"),
    ]
    present = [(k, l) for k, l in all_comps if any(k in r for r in results)]
    print()
    print(f"{'Comparison':<28} {'Win':>6} {'Tie':>6} {'Loss':>6}  {'WinRate':>8}")
    print("-" * 58)
    for key, label in present:
        wr, w, l, t = win_rate(results, key)
        print(f"{label:<28} {w:>6} {t:>6} {l:>6}  {wr:>7.1f}%")


if __name__ == "__main__":
    main()
