#!/usr/bin/env python3
"""
Correlate per-sample KL delta with y* vs y_base winrate outcome.

Joins:
    data/olmo_signal/olmo_signal.jsonl          (mean_delta, tier)
    data/winrate_results/olmo_wf_500/winrate_olmo_results.jsonl  (y* vs y_base winner)

Drops ties. Outcome: 1 = y* won, 0 = y_base won.

Computes:
    - Point-biserial correlation (delta vs outcome)
    - Winrate by delta quartile (Q1..Q4)
    - Winrate by tier, split into delta above/below median within tier
    - Logistic regression P(y* wins | delta)

Outputs:
    correlation_summary.txt
    plot_scatter_logistic.png   -- scatter + logistic curve
    plot_quartile_winrate.png   -- winrate per quartile bar chart

Usage:
    python scripts/eval/correlate_signal_winrate.py \\
        --signal    data/olmo_signal/olmo_signal.jsonl \\
        --winrate   data/winrate_results/olmo_wf_500/winrate_olmo_results.jsonl \\
        --output_dir data/olmo_signal
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _pointbiserialr(outcomes: list[int], deltas: list[float]) -> tuple[float, float]:
    """Point-biserial correlation without scipy."""
    n = len(outcomes)
    y = np.array(outcomes, dtype=float)
    x = np.array(deltas, dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    # two-tailed t-test
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
    # approximate p-value via normal for large n, else t-dist
    import math as _math
    p = 2 * (1 - _math.erf(abs(t) / _math.sqrt(2)) / 1)
    # use scipy if available, else normal approx
    try:
        from scipy import stats as _stats
        p = float(_stats.t.sf(abs(t), df=n - 2) * 2)
    except ImportError:
        p = float(2 * (1 - 0.5 * (1 + _math.erf(abs(t) / _math.sqrt(2)))))
    return r, p

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.35,
    "grid.linewidth":   0.7,
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
Q_COLORS = ["#AED6F1", "#85C1E9", "#2E86C1", "#1A5276"]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def logistic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def fit_logistic(deltas: list[float], outcomes: list[int]) -> tuple[float, float]:
    """Simple MLE logistic regression via gradient descent (no scipy)."""
    X = np.array(deltas, dtype=float)
    y = np.array(outcomes, dtype=float)
    # standardize for stability
    mu, sd = X.mean(), X.std() + 1e-8
    Xs = (X - mu) / sd
    a, b = 0.0, 0.0
    lr = 0.1
    for _ in range(2000):
        p = logistic(Xs, a, b)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        grad_a = np.mean(p - y)
        grad_b = np.mean((p - y) * Xs)
        a -= lr * grad_a
        b -= lr * grad_b
    # convert b back to original scale
    b_orig = b / sd
    a_orig = a - b_orig * mu
    return float(a_orig), float(b_orig)


def spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    ax.set_axisbelow(True)


# ── Plot 1: scatter + logistic ─────────────────────────────────────────────────

def plot_scatter_logistic(
    deltas: list[float],
    outcomes: list[int],
    tiers: list[str],
    a: float, b: float,
    out_path: str,
) -> None:
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.04, 0.04, len(outcomes))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    active_tiers = [t for t in TIER_ORDER if t in set(tiers)]
    for tier in active_tiers:
        idx = [i for i, t in enumerate(tiers) if t == tier]
        ax.scatter(
            [deltas[i] for i in idx],
            [outcomes[i] + jitter[i] for i in idx],
            color=TIER_COLORS.get(tier, "#888"),
            alpha=0.45, s=16, linewidths=0, label=tier,
        )

    # logistic curve
    x_range = np.linspace(min(deltas) * 0.95, max(deltas) * 1.05, 300)
    ax.plot(x_range, logistic(x_range, a, b),
            color="black", linewidth=2.0, label="Logistic fit", zorder=5)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Per-sample mean $\\Delta$ (nats)", fontsize=10)
    ax.set_ylabel("Outcome  (1 = y* wins, 0 = y_base wins)", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["y_base wins (0)", "y* wins (1)"], fontsize=9)
    ax.set_title(
        "KL shift vs. y* win outcome with logistic regression fit",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=8, framealpha=0.4, loc="center right")
    spine_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Plot 2: winrate per quartile ───────────────────────────────────────────────

def plot_quartile_winrate(
    quartile_stats: list[dict],
    out_path: str,
) -> None:
    labels    = [f"Q{q['quartile']}\n(δ∈[{q['delta_lo']:.2f},{q['delta_hi']:.2f}])"
                 for q in quartile_stats]
    winrates  = [q["winrate"] * 100 for q in quartile_stats]
    err_lo    = [(q["winrate"] - q["ci_lo"]) * 100 for q in quartile_stats]
    err_hi    = [(q["ci_hi"] - q["winrate"]) * 100 for q in quartile_stats]
    ns        = [q["n_decisive"] for q in quartile_stats]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        x, winrates, width=0.55,
        color=Q_COLORS[:len(labels)],
        edgecolor=[c for c in Q_COLORS[:len(labels)]],
        linewidth=0.8,
        yerr=[err_lo, err_hi],
        error_kw=dict(elinewidth=1.3, capsize=5, ecolor="black", capthick=1.1),
        zorder=3,
    )
    ax.axhline(50, color="gray", linewidth=0.9, linestyle="--", alpha=0.6, zorder=2)
    ax.text(len(labels) - 0.6, 51.5, "50%", ha="right", va="bottom",
            fontsize=8, color="gray", fontstyle="italic")

    for bar, wr, n in zip(bars, winrates, ns):
        hi_y = bar.get_height() + bar.get_bbox().y1 - bar.get_bbox().y0 + 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(err_hi) + 1.5,
                f"{wr:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("y* win rate vs. y_base (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("y* win rate by KL-shift quartile  (95% Wilson CI)", fontsize=10, pad=8)
    spine_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal",
        default="data/olmo_signal/olmo_signal.jsonl")
    parser.add_argument("--winrate",
        default="data/winrate_results/olmo_wf_500/winrate_olmo_results.jsonl")
    parser.add_argument("--output_dir",
        default="data/olmo_signal")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & join ────────────────────────────────────────────────────────────
    signal_records = {
        (r["conversation_id"], r["turn_index"]): r
        for r in load_jsonl(args.signal)
    }
    winrate_records = load_jsonl(args.winrate)

    joined = []
    for wr in winrate_records:
        key = (wr["conversation_id"], wr["turn_index"])
        if key not in signal_records:
            continue
        comp = wr["comparisons"].get("y* vs y_base", {})
        winner = comp.get("winner", "tie")
        if winner == "tie":
            continue
        outcome = 1 if winner == "1" else 0  # 1=y* won, 0=y_base won
        sig = signal_records[key]
        if math.isnan(sig.get("mean_delta", float("nan"))):
            continue
        joined.append({
            "key":        key,
            "tier":       sig["tier"],
            "mean_delta": sig["mean_delta"],
            "outcome":    outcome,
            "o_text":     sig.get("o_text", ""),
        })

    print(f"Joined {len(joined)} decisive samples (ties dropped)", flush=True)
    if not joined:
        print("No overlap — check file paths.", flush=True)
        return

    deltas   = [r["mean_delta"] for r in joined]
    outcomes = [r["outcome"]    for r in joined]
    tiers    = [r["tier"]       for r in joined]

    # ── Point-biserial correlation ─────────────────────────────────────────────
    r_pb, p_val = _pointbiserialr(outcomes, deltas)

    # ── Quartile breakdown ─────────────────────────────────────────────────────
    quartile_edges = np.quantile(deltas, [0, 0.25, 0.5, 0.75, 1.0])
    quartile_stats = []
    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        # last quartile: inclusive upper bound
        if q < 3:
            subset = [r for r in joined if lo <= r["mean_delta"] < hi]
        else:
            subset = [r for r in joined if lo <= r["mean_delta"] <= hi]
        wins = sum(r["outcome"] for r in subset)
        n    = len(subset)
        wr   = wins / n if n else float("nan")
        ci_lo, ci_hi = wilson_ci(wins, n)
        quartile_stats.append({
            "quartile":   q + 1,
            "n":          n,
            "n_decisive": n,
            "wins":       wins,
            "winrate":    wr,
            "ci_lo":      ci_lo,
            "ci_hi":      ci_hi,
            "delta_lo":   lo,
            "delta_hi":   hi,
            "mean_delta": np.mean([r["mean_delta"] for r in subset]) if subset else float("nan"),
        })

    # ── Tier breakdown ─────────────────────────────────────────────────────────
    tier_stats = {}
    for tier in TIER_ORDER:
        subset = [r for r in joined if r["tier"] == tier]
        if not subset:
            continue
        med = np.median([r["mean_delta"] for r in subset])
        hi_sub  = [r for r in subset if r["mean_delta"] >= med]
        lo_sub  = [r for r in subset if r["mean_delta"] <  med]
        def wr_stats(s):
            w = sum(r["outcome"] for r in s)
            n = len(s)
            wr = w / n if n else float("nan")
            lo, hi = wilson_ci(w, n)
            return {"n": n, "wins": w, "winrate": wr, "ci_lo": lo, "ci_hi": hi}
        tier_stats[tier] = {
            "all":    wr_stats(subset),
            "hi_delta": wr_stats(hi_sub),
            "lo_delta": wr_stats(lo_sub),
            "median_delta": med,
        }

    # ── Logistic fit ───────────────────────────────────────────────────────────
    a, b = fit_logistic(deltas, outcomes)

    # ── Summary text ───────────────────────────────────────────────────────────
    lines = [
        "OLMo Signal–Winrate Correlation",
        f"N decisive samples: {len(joined)}",
        "",
        f"Point-biserial r(delta, outcome) = {r_pb:+.4f}  (p = {p_val:.4f})",
        f"Logistic fit: logit P(y* wins) = {a:+.4f} + {b:+.4f} * delta",
        "",
        "── Quartile breakdown (y* vs y_base winrate) ──────────────────────────",
        f"{'Quartile':<10} {'N':>5} {'Wins':>6} {'WR%':>7} {'95% CI':>18} {'Mean delta':>12}",
        "─" * 64,
    ]
    for q in quartile_stats:
        ci = f"[{q['ci_lo']*100:.1f}, {q['ci_hi']*100:.1f}]"
        lines.append(
            f"Q{q['quartile']} (δ∈[{q['delta_lo']:.3f},{q['delta_hi']:.3f}])"
            f"  {q['n']:>5} {q['wins']:>6} {q['winrate']*100:>6.1f}%  {ci:>18}  {q['mean_delta']:>10.4f}"
        )

    lines += [
        "",
        "── Tier breakdown (all / hi-delta half / lo-delta half) ───────────────",
        f"{'Tier':<16} {'N':>5} {'WR% all':>9} {'WR% hi-δ':>10} {'WR% lo-δ':>10}  median-δ",
        "─" * 64,
    ]
    for tier in TIER_ORDER:
        if tier not in tier_stats:
            continue
        ts = tier_stats[tier]
        lines.append(
            f"{tier:<16} {ts['all']['n']:>5} "
            f"{ts['all']['winrate']*100:>8.1f}% "
            f"{ts['hi_delta']['winrate']*100:>9.1f}% "
            f"{ts['lo_delta']['winrate']*100:>9.1f}%  "
            f"{ts['median_delta']:.4f}"
        )

    summary_text = "\n".join(lines)
    print(summary_text)
    summary_path = out_dir / "correlation_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_scatter_logistic(
        deltas, outcomes, tiers, a, b,
        str(out_dir / "plot_scatter_logistic.png"),
    )
    plot_quartile_winrate(
        quartile_stats,
        str(out_dir / "plot_quartile_winrate.png"),
    )


if __name__ == "__main__":
    main()
