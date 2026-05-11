"""Pareto-style trajectory: x = amc23 acc, y = alpaca winrate.
Each run = one trajectory of 6 ckpts (smoothed). Starts at base policy.
Two panels: mode_covering, mode_seeking.

Reading the plot:
  - upper-right = best (math preserved + alpaca gained)
  - lower-left  = worst (forgetting + winrate loss)
  - all trajectories START at base policy (32.5%, 0.638)
"""
import json, os, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

for p in list(sys.path):
    if "user_interactions" in p: sys.path.remove(p)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE_CMAP = mpl.colormaps["Blues"]

ROOT = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/demo2teacher"
JUDGE = f"{ROOT}/_alpaca_judge_clean/summary.json"
BASE_AMC = 25.0; BASE_ALPACA = 0.638
OUT = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results/_plots/meeting"
Path(OUT).mkdir(parents=True, exist_ok=True)

# 8 selected runs (cond_xyo_ystart only)
SELECTED = {
    "WC-9":  ("sft",     "wins",  "wins + sft",     "#5b8def"),
    "WC-10": ("fkl",     "wins",  "wins + fkl",     "#0b3d91"),
    "WC-11": ("sft",     "loses", "loses + sft",    "#999999"),
    "WC-12": ("fkl",     "loses", "loses + fkl",    "#d62728"),
    "WC-17": ("sdpo",    "wins",  "wins + sdpo",    "#5b8def"),
    "WC-18": ("sdpo",    "loses", "loses + sdpo",   "#999999"),
    "WC-23": ("pc_sdpo", "wins",  "wins + pc_sdpo", "#0b3d91"),
    "WC-24": ("pc_sdpo", "loses", "loses + pc_sdpo","#d62728"),
}
LSTYLE = {"wins": "-", "loses": "--"}
PANELS = {
    "mode_covering": ["WC-9","WC-10","WC-11","WC-12"],
    "mode_seeking":  ["WC-17","WC-18","WC-23","WC-24"],
}
CKPTS = ["step-30","step-60","step-90","step-120","step-150","final"]

# data
alp = defaultdict(dict)
for k, info in json.load(open(JUDGE)).items():
    if "__" in k:
        rid, ck = k.split("__", 1)
        alp[rid][ck] = info.get("winrate", 0)

def parse_math(p):
    if not os.path.exists(p): return {}
    out = {}
    for line in open(p):
        m = re.search(r"amc23.*Final Accuracy:\s*([0-9.]+)", line)
        if m: out["amc23"] = float(m.group(1))
    return out

amc = defaultdict(dict)
for rid in SELECTED:
    for ck in CKPTS:
        accs = []
        for s in [0,1,2]:
            r = parse_math(f"{ROOT}/{rid}/{ck}/math_s{s}/summary.txt")
            if "amc23" in r: accs.append(r["amc23"])
        if accs: amc[rid][ck] = np.mean(accs)

def ema(y, beta=0.99):
    y = np.asarray(y, dtype=float)
    if y.size == 0: return y
    out = np.empty_like(y); s = 0.0
    for t, v in enumerate(y):
        s = beta*s + (1-beta)*v
        out[t] = s / (1 - beta**(t+1))
    return out


def panel(panel_name, run_ids):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    trajs = []
    for rid in run_ids:
        if rid not in SELECTED: continue
        obj, direction, label, color = SELECTED[rid]
        xs = [BASE_AMC]; ys = [BASE_ALPACA]   # start at base
        for ck in CKPTS:
            xv = amc.get(rid, {}).get(ck)
            yv = alp.get(rid, {}).get(ck)
            if xv is not None and yv is not None:
                xs.append(xv); ys.append(yv)
        if len(xs) > 1:
            trajs.append((label, direction, color, xs, ys))

    if not trajs:
        print(f"  ! no data for {panel_name}"); plt.close(fig); return

    # plot range — clean, anchored on base
    all_x = [x for _,_,_,xs,_ in trajs for x in xs]
    all_y = [y for _,_,_,_,ys in trajs for y in ys]
    ypad = 0.005
    xlim = (20, 26)
    ylim = (min(all_y)-ypad, max(all_y)+ypad+0.005)

    # base policy reference lines + anchor
    ax.axvline(BASE_AMC, color="#1a1a1a", linestyle=":", linewidth=0.9, alpha=0.5, zorder=2)
    ax.axhline(BASE_ALPACA, color="#1a1a1a", linestyle=":", linewidth=0.9, alpha=0.5, zorder=2)
    ax.scatter([BASE_AMC], [BASE_ALPACA], s=120, color="#1a1a1a",
               edgecolor="white", linewidth=1.4, zorder=10, marker="*")
    ax.annotate("Base Policy", (BASE_AMC, BASE_ALPACA),
                xytext=(8, -14), textcoords="offset points",
                fontsize=10, color="#1a1a1a", fontweight="bold")

    # trajectories
    for label, direction, color, xs, ys in trajs:
        ys_s = ema(ys)
        xs_s = ema(xs)
        ls = LSTYLE[direction]
        ax.plot(xs_s, ys_s, color=color, linestyle=ls, marker="o", markersize=5,
                linewidth=1.9, alpha=0.95, label=label, zorder=4)
        # number ckpt points 1-6 (skip base which is index 0)
        for i, (x, y) in enumerate(zip(xs_s[1:], ys_s[1:]), start=1):
            ax.annotate(str(i), (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=color, alpha=0.9)

    # quadrant guide text (subtle, in corners)
    ax.text(xlim[1]-0.3, ylim[1]-0.002, "preserves math\n+ gains chat", fontsize=8, style="italic",
            color="#0b3d91", ha="right", va="top", alpha=0.75)
    ax.text(xlim[0]+0.3, ylim[0]+0.005, "forgets math\n+ loses chat", fontsize=8, style="italic",
            color="#7f1d1d", ha="left", va="bottom", alpha=0.55)

    ax.set_xlabel("amc23 accuracy (%)")
    ax.set_ylabel("Alpaca winrate (vs gpt-4-1106-preview)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title(f"Math vs chat trade-off — {panel_name.replace('_',' ')}\n"
                 f"trajectories start at Base Policy, walk through 6 ckpts (1=earliest, 6=final)",
                 fontsize=10.5, style="italic", pad=10)
    out_path = f"{OUT}/pareto_amc_alpaca__{panel_name}.png"
    fig.savefig(out_path); plt.close(fig)
    print(f"  ✓ {out_path}")


for p, rids in PANELS.items():
    print(f"── panel: {p} ──")
    panel(p, rids)
print(f"\n✓ done — {OUT}")
