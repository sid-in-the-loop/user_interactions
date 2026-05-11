"""Meeting-ready figs: 8 selected runs (cond_xyo_ystart), split into mode-covering
and mode-seeking panels. Three figures: KL × task, forgetting, alpaca trajectory.

Templates match template-figs/fig-5.png aesthetic:
  - linear x (auto-fit), radial blue gradient
  - numbered ckpt points (1-6) along trajectories
  - EMA(0.99) smoothing
  - serif typography, italic captions, anchor labels
"""
import json, os, re, sys, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

for p in list(sys.path):
    if "user_interactions" in p: sys.path.remove(p)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ─── style ───────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE_CMAP = mpl.colormaps["Blues"]

# ─── selected runs ───────────────────────────────────────────────────────────
SELECTED = {
    # rid: (obj, direction, label, color)
    "WC-9":  ("sft",     "wins",  "wins + sft",     "#5b8def"),
    "WC-10": ("fkl",     "wins",  "wins + fkl",     "#0b3d91"),
    "WC-11": ("sft",     "loses", "loses + sft",    "#c0c0c0"),
    "WC-12": ("fkl",     "loses", "loses + fkl",    "#d62728"),
    "WC-17": ("sdpo",    "wins",  "wins + sdpo",    "#5b8def"),
    "WC-18": ("sdpo",    "loses", "loses + sdpo",   "#c0c0c0"),
    "WC-23": ("pc_sdpo", "wins",  "wins + pc_sdpo", "#0b3d91"),
    "WC-24": ("pc_sdpo", "loses", "loses + pc_sdpo","#d62728"),
}
# Line style by direction
LSTYLE = {"wins": "-", "loses": "--"}
PANELS = {
    "mode_covering": ["WC-9","WC-10","WC-11","WC-12"],          # sft, fkl
    "mode_seeking":  ["WC-17","WC-18","WC-23","WC-24"],         # sdpo, pc_sdpo
}

CKPTS = ["step-30","step-60","step-90","step-120","step-150","final"]
CKPT_X = [30, 60, 90, 120, 150, 180]
EPOCH_LABEL = [1, 2, 3, 4, 5, 6]

# ─── data load ───────────────────────────────────────────────────────────────
ROOT = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/demo2teacher"
JUDGE = f"{ROOT}/_alpaca_judge_clean/summary.json"
BASE_AMC = 25.0; BASE_AIME = 3.3; BASE_ALPACA = 0.638  # post-strip baseline
OUT = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/_plots/meeting"
Path(OUT).mkdir(parents=True, exist_ok=True)

# alpaca winrate
alp = defaultdict(dict)
j = json.load(open(JUDGE))
for k, info in j.items():
    if "__" in k:
        rid, ck = k.split("__", 1)
        alp[rid][ck] = info.get("winrate", 0)

# math (avg over 3 seeds)
def parse_math(path):
    if not os.path.exists(path): return {}
    out = {}
    for line in open(path):
        m = re.search(r"(amc23|aime24).*Final Accuracy:\s*([0-9.]+)", line)
        if m: out[m.group(1)] = float(m.group(2))
    return out

amc = defaultdict(dict)
aim = defaultdict(dict)
for rid in SELECTED:
    for ck in CKPTS:
        a = []; i = []
        for s in [0,1,2]:
            r = parse_math(f"{ROOT}/{rid}/{ck}/math_s{s}/summary.txt")
            if "amc23" in r: a.append(r["amc23"])
            if "aime24" in r: i.append(r["aime24"])
        if a: amc[rid][ck] = np.mean(a)
        if i: aim[rid][ck] = np.mean(i)

# KL
kl = defaultdict(dict)
for rid in SELECTED:
    p = f"{ROOT}/{rid}/kl_vs_base.json"
    if os.path.exists(p):
        d = json.load(open(p))
        for ck, info in d.get("results", {}).items():
            kl[rid][ck] = info.get("kl_vs_base")

# ─── helpers ─────────────────────────────────────────────────────────────────
def ema(y, beta=0.99):
    y = np.asarray(y, dtype=float)
    if y.size == 0: return y
    out = np.empty_like(y); s = 0.0
    for t, v in enumerate(y):
        s = beta * s + (1 - beta) * v
        out[t] = s / (1 - beta**(t+1))
    return out

def add_radial_gradient(ax, xlim, ylim, base_x=0.0, base_y=None, levels=10):
    """Radial blue contours centered at base policy (origin in KL space).
    Lighter shading = farther from base. Visualizes KL distance directly."""
    x = np.linspace(xlim[0], xlim[1], 240)
    y = np.linspace(ylim[0], ylim[1], 240)
    X, Y = np.meshgrid(x, y)
    cx, cy = base_x, base_y if base_y is not None else (ylim[0]+ylim[1])/2
    # KL-aware radial decay: x distance scaled by xlim, y by ylim, both normalized
    rad = np.sqrt(((X-cx)/(xlim[1]-xlim[0]))**2 + ((Y-cy)/(ylim[1]-ylim[0]))**2)
    Z = np.exp(-2.5 * rad)
    ax.contourf(X, Y, Z, levels=levels, cmap=BLUE_CMAP, alpha=0.42, zorder=0)

def add_anchor(ax, x, y, label, color="#1a1a1a", italic=True):
    ax.scatter([x], [y], s=80, color=color, edgecolor="white", linewidth=1.0, zorder=6)
    txtkw = dict(fontsize=9.5, color=color, fontweight="bold")
    ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", **txtkw, zorder=7)

# ─── plot funcs ──────────────────────────────────────────────────────────────
def fig_kl_task(panel_name, run_ids, title_extra="", out_path=None):
    """KL × task perf — task = alpaca winrate (chat-side metric)."""
    trajs = []
    for rid in run_ids:
        if rid not in SELECTED: continue
        obj, direction, label, color = SELECTED[rid]
        xs, ys = [], []
        for ck in CKPTS:
            xv = kl.get(rid, {}).get(ck)
            yv = alp.get(rid, {}).get(ck)
            if xv is not None and yv is not None:
                xs.append(xv); ys.append(yv)
        if xs:
            trajs.append((label, direction, color, xs, ys))

    if not trajs:
        print(f"  ! no data for {panel_name}"); return

    all_x = [x for _,_,_,xs,_ in trajs for x in xs]
    all_y = [y for _,_,_,_,ys in trajs for y in ys]
    xlim = (0, max(all_x) * 1.15)
    # Force tight y-range around 0.63-0.69 so trajectory differences are visible
    ymin = min(all_y + [BASE_ALPACA]) - 0.005
    ymax = max(all_y + [BASE_ALPACA]) + 0.010
    ylim = (ymin, ymax)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # gradient centered at base policy (origin in KL-from-base space)
    add_radial_gradient(ax, xlim, ylim, base_x=0.0, base_y=BASE_ALPACA)

    # base policy reference
    ax.axhline(BASE_ALPACA, color="#1a1a1a", linestyle=":", linewidth=1.0, alpha=0.55, zorder=2)
    add_anchor(ax, 0.0, BASE_ALPACA, "Base Policy", color="#1a1a1a")

    for label, direction, color, xs, ys in trajs:
        # Prepend base policy as the (KL=0, winrate=base) anchor — every traj starts here
        xs_full = [0.0] + list(xs)
        ys_full = [BASE_ALPACA] + list(ys)
        ys_s = ema(ys_full) if len(ys_full) > 2 else np.asarray(ys_full)
        ls = LSTYLE[direction]
        # ONLY the smoothed line — no faint duplicate
        ax.plot(xs_full, ys_s, color=color, linestyle=ls, marker="o", markersize=5,
                linewidth=1.8, alpha=0.95, label=label, zorder=4)
        # number ckpt points 1-6 (skip base which is index 0)
        for i, (x, y) in enumerate(zip(xs_full[1:], ys_s[1:]), start=1):
            ax.annotate(str(i), (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=color, alpha=0.85)

    ax.set_xlabel(r"KL($\pi_{\mathrm{ckpt}} \,\|\, \pi_{\mathrm{base}}$)")
    ax.set_ylabel("Alpaca winrate (vs gpt-4-1106-preview)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title(f"KL × winrate trajectory — {panel_name.replace('_', ' ')}{title_extra}",
                 fontsize=10.5, style="italic", pad=10)
    out_path = out_path or f"{OUT}/fig5_kl_task__{panel_name}.png"
    fig.savefig(out_path); plt.close(fig)
    print(f"  ✓ {out_path}")

def fig_forgetting(panel_name, run_ids, out_path=None):
    """Math forgetting on amc23 (avg over 3 seeds, EMA smoothed)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # gradient bg, anchored at (0, base) so trajectories drop AWAY from base
    xlim = (0, 200); ylim = (BASE_AMC * 0.5, BASE_AMC * 1.10)
    add_radial_gradient(ax, xlim, ylim, base_x=0, base_y=BASE_AMC)

    plotted = False
    for rid in run_ids:
        if rid not in SELECTED: continue
        obj, direction, label, color = SELECTED[rid]
        xs_full = [0]; ys_full = [BASE_AMC]   # start every traj at base
        for ck, x in zip(CKPTS, CKPT_X):
            v = amc.get(rid, {}).get(ck)
            if v is not None:
                xs_full.append(x); ys_full.append(v)
        if len(xs_full) <= 1: continue
        ys_s = ema(np.asarray(ys_full)) if len(ys_full) > 2 else np.asarray(ys_full)
        ls = LSTYLE[direction]
        ax.plot(xs_full, ys_s, color=color, linestyle=ls, marker="o", markersize=5,
                linewidth=1.8, alpha=0.95, label=label, zorder=4)
        plotted = True

    # base reference
    ax.axhline(BASE_AMC, color="#1a1a1a", linestyle=":", linewidth=1.4, alpha=0.85, zorder=2)
    ax.text(xlim[1]*0.99, BASE_AMC, f" base = {BASE_AMC:.1f}%",
            fontsize=9.5, va="center", ha="right", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    ax.set_xlabel("Training step")
    ax.set_ylabel("amc23 accuracy (%)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if plotted: ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title(f"Math forgetting on amc23 — {panel_name.replace('_',' ')}",
                 fontsize=10.5, style="italic", pad=10)
    out_path = out_path or f"{OUT}/fig6_forgetting__{panel_name}.png"
    fig.savefig(out_path); plt.close(fig)
    print(f"  ✓ {out_path}")

def fig_alpaca_traj(panel_name, run_ids, out_path=None):
    """Alpaca winrate over training steps (vs gpt-4-1106-preview)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xlim = (0, 200)

    plotted = False; all_y = [BASE_ALPACA]
    plot_data = []
    for rid in run_ids:
        if rid not in SELECTED: continue
        obj, direction, label, color = SELECTED[rid]
        xs_full = [0]; ys_full = [BASE_ALPACA]
        for ck, x in zip(CKPTS, CKPT_X):
            v = alp.get(rid, {}).get(ck)
            if v is not None:
                xs_full.append(x); ys_full.append(v)
        if len(xs_full) > 1:
            all_y.extend(ys_full)
            plot_data.append((label, direction, color, xs_full, ys_full))
            plotted = True

    ymin = min(all_y) - 0.005
    ymax = max(all_y) + 0.010
    ylim = (ymin, ymax)
    for label, direction, color, xs_full, ys_full in plot_data:
        ys_s = ema(np.asarray(ys_full)) if len(ys_full) > 2 else np.asarray(ys_full)
        ls = LSTYLE[direction]
        ax.plot(xs_full, ys_s, color=color, linestyle=ls, marker="o", markersize=5,
                linewidth=1.8, alpha=0.95, label=label, zorder=4)
    # redraw lines on top of gradient (added after gradient)
    # gradient was added too late; fine — patches add zorder=0 underneath

    ax.axhline(BASE_ALPACA, color="#1a1a1a", linestyle=":", linewidth=1.4, alpha=0.85, zorder=2)
    ax.text(xlim[1]*0.99, BASE_ALPACA, f" base = {BASE_ALPACA:.3f}",
            fontsize=9.5, va="center", ha="right", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    ax.set_xlabel("Training step")
    ax.set_ylabel("Alpaca winrate (vs gpt-4-1106-preview)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if plotted: ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title(f"Alpaca winrate over training — {panel_name.replace('_',' ')}",
                 fontsize=10.5, style="italic", pad=10)
    out_path = out_path or f"{OUT}/fig7_alpaca__{panel_name}.png"
    fig.savefig(out_path); plt.close(fig)
    print(f"  ✓ {out_path}")

# ─── main ────────────────────────────────────────────────────────────────────
print(f"Output dir: {OUT}\n")
for panel, rids in PANELS.items():
    print(f"── panel: {panel} ──")
    fig_kl_task(panel, rids)
    fig_forgetting(panel, rids)
    fig_alpaca_traj(panel, rids)
    print()

print(f"\n✓ done — 6 figs in {OUT}")
