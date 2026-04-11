#!/usr/bin/env python3
"""
Training curves for LoRA ablation: prefix30 vs full × SFT/FKL/JSD.
2 rows (lc_win_rate / win_rate) × 2 cols (full / prefix30).
3 lines per plot (SFT=blue, FKL=red, JSD=green) + base dashed.

Usage:
    python scripts/eval/plot_lora_ablation_curves.py
    python scripts/eval/plot_lora_ablation_curves.py --metric win_rate --out plots/lora_ablation_win_rate.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.linestyle":  "--",
    "grid.alpha":      0.4,
})

RESULTS_ROOT = Path("eval_results/alpaca_lora_ablation")

METHOD_STYLE = {
    "sft": dict(color="#2166ac", marker="o", label="SFT"),
    "fkl": dict(color="#d6604d", marker="s", label="FKL"),
    "jsd": dict(color="#4dac26", marker="^", label="JSD"),
}

SUBPLOT_TITLES = {
    "full":     "full · dataset",
    "prefix30": "prefix30 · dataset",
}


def load_scores(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    pts = []
    for score_file in run_dir.rglob("scores.json"):
        d = json.loads(score_file.read_text())
        step = d.get("step")
        val  = d.get(metric)
        if step is not None and val is not None:
            pts.append((step, val))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_ROOT))
    ap.add_argument("--out_lc",  default="plots/lora_ablation_lc_win_rate.pdf")
    ap.add_argument("--out_wr",  default="plots/lora_ablation_win_rate.pdf")
    args = ap.parse_args()

    root = Path(args.results)

    prefixes = ["full", "prefix30"]
    methods  = ["sft", "fkl", "jsd"]
    metrics  = [
        ("lc_win_rate", "lc win rate (%)", args.out_lc),
        ("win_rate",    "win rate (%)",    args.out_wr),
    ]

    # Load base
    base_scores = {}
    base_path = root / "base" / "scores.json"
    if base_path.exists():
        bd = json.loads(base_path.read_text())
        base_scores["lc_win_rate"] = bd.get("lc_win_rate")
        base_scores["win_rate"]    = bd.get("win_rate")

    # Collect all data
    all_data = {}
    for prefix in prefixes:
        for method in methods:
            run_name = f"{prefix}_{method}"
            pts = load_scores(root / run_name, "lc_win_rate")
            all_data[(prefix, method, "lc_win_rate")] = pts
            pts2 = load_scores(root / run_name, "win_rate")
            all_data[(prefix, method, "win_rate")] = pts2

    for metric_key, ylabel, out_path in metrics:
        # Auto y-limits
        all_vals = []
        for prefix in prefixes:
            for method in methods:
                all_vals += [v for _, v in all_data[(prefix, method, metric_key)]]
        bv = base_scores.get(metric_key)
        if bv: all_vals.append(bv)
        vmin, vmax = min(all_vals), max(all_vals)
        pad = (vmax - vmin) * 0.18
        ylim = (vmin - pad, vmax + pad)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        fig.suptitle(f"Qwen3-8B LoRA Ablation · {ylabel}", fontsize=13)

        for col, prefix in enumerate(prefixes):
            ax = axes[col]
            ax.set_title(SUBPLOT_TITLES[prefix], fontsize=10)

            # Find max real step for this subplot to position "final"
            all_real = [s for m in methods for s, _ in all_data[(prefix, m, metric_key)] if s != 999999]
            final_x  = max(all_real) * 1.15 if all_real else 120

            for method in methods:
                pts = all_data[(prefix, method, metric_key)]
                if not pts:
                    continue
                steps_raw, vals = zip(*pts)
                steps = [final_x if s == 999999 else s for s in steps_raw]
                style = METHOD_STYLE[method]
                ax.plot(steps, vals, marker=style["marker"], color=style["color"],
                        label=style["label"], linewidth=1.8, markersize=5)
                for x, y in zip(steps, vals):
                    ax.annotate(f"{y:.1f}", xy=(x, y),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom",
                                fontsize=6.5, color=style["color"])

            if bv is not None:
                ax.axhline(bv, color="gray", linestyle="--",
                           linewidth=1.2, label=f"Base ({bv:.1f})")

            ax.set_xlabel("Optimizer step")
            ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)

        plt.tight_layout()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close()


if __name__ == "__main__":
    main()
