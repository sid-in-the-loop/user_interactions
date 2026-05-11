"""Publication-style figs matching the template-figs/ aesthetic.

Modes:
  --dummy            fabricate plausible data, plot to _dummy_plots/
  (default)          read real eval data, plot to _plots/

Figures:
  fig1_conditioning_ladder  — KL × winrate, conditioning anchors + demonstrator
  fig2_judge_agreement      — (gpt4o vs student preference) density + outliers
  fig3_displacement_arrows  — adv mean × variance, arrow per family
  fig4_kl_adv_bubble        — KL × adv with bubble = adv variance
  fig5_kl_task_trajectory   — KL × task perf, numbered ckpt points
  fig6_forgetting_curves    — math acc vs training step (forgetting)
"""
import argparse, json, os, re, sys, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

# strip user_interactions from sys.path so we don't import the local 'wandb' folder
for p in list(sys.path):
    if "user_interactions" in p: sys.path.remove(p)

# ─── styling ─────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

BLUE_CMAP = mpl.colormaps["Blues"]
OBJ_COLOR = {"sft": "#1f77b4", "fkl": "#2ca02c", "sdpo": "#d62728", "pc_sdpo": "#9467bd"}

# ─── EMA smoothing (wandb-style) ─────────────────────────────────────────────
def ema(values, beta=0.99):
    """Exponential moving average. Wandb-style: y_t = β·y_{t-1} + (1-β)·x_t,
    bias-corrected. Pass-through if values is empty."""
    values = np.asarray(values, dtype=float)
    if values.size == 0: return values
    out = np.empty_like(values)
    s = 0.0
    for t, v in enumerate(values):
        s = beta * s + (1 - beta) * v
        # bias-correct so early values aren't pulled to 0
        out[t] = s / (1 - beta ** (t + 1))
    return out

# ─── styling helpers ─────────────────────────────────────────────────────────
def add_blue_gradient(ax, xrange, yrange, levels=12, alpha_max=0.55):
    """Soft sequential-blue background contour to mimic template aesthetic."""
    x = np.linspace(xrange[0], xrange[1], 200)
    y = np.linspace(yrange[0], yrange[1], 200)
    X, Y = np.meshgrid(x, y)
    # density "anchored" near upper-right (positive region), gentle radial decay
    cx, cy = xrange[1] - 0.05*(xrange[1]-xrange[0]), yrange[1] - 0.05*(yrange[1]-yrange[0])
    rad = np.sqrt(((X-cx)/(xrange[1]-xrange[0]))**2 + ((Y-cy)/(yrange[1]-yrange[0]))**2)
    Z = np.exp(-2.0 * rad)
    ax.contourf(X, Y, Z, levels=levels, cmap=BLUE_CMAP, alpha=alpha_max, zorder=0)

def add_anchor(ax, x, y, label, kind="point", color="#1f3b66", italic=True):
    """Anchor labels in the style of fig-1 ('Base Policy', 'Demonstrator (y)')."""
    if kind == "point":
        ax.scatter([x], [y], s=80, color=color, edgecolor="white", linewidth=1.2, zorder=5)
    txtkw = dict(fontsize=9, color=color)
    if italic: txtkw["style"] = "italic"
    ax.annotate(label, (x, y), xytext=(8, 8), textcoords="offset points", **txtkw, zorder=6)

def bezier_arrow(ax, p_from, p_to, color="#3a5f8f", lw=1.4, rad=0.25, alpha=0.85):
    """Curved arrow between two points (template fig-1, fig-3 style)."""
    arr = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=12,
                          color=color, linewidth=lw, alpha=alpha,
                          connectionstyle=f"arc3,rad={rad}", zorder=4)
    ax.add_patch(arr)

# ─── DUMMY data generator ─────────────────────────────────────────────────────
def dummy_data():
    """Plausible numbers for each fig. Used to preview the styling."""
    rng = np.random.default_rng(0)
    d = {}

    # fig-1 anchors
    d["fig1"] = {
        "base":         (0.0, 0.50),       # base policy region
        "demo":         (1.20, 0.78),      # demonstrator (y)
        "teachers": [
            ("Teacher (x, o)",                          0.85, 0.69),
            ("Teacher (x, y_prefix, o)",                0.55, 0.74),
            ("Teacher (x, y_prefix, o, y*[:5])",        0.30, 0.79),
        ],
    }

    # fig-2 judge agreement: ~5000 rows, two correlated [0,1] scores
    n = 5000
    s = rng.beta(2.5, 2.5, n)
    g = np.clip(s + rng.normal(0, 0.18, n), 0, 1)
    correct = (rng.random(n) < 0.55)
    d["fig2"] = {"student": s, "gpt4o": g, "correct": correct}

    # fig-3 arrows in (adv_mean, adv_var) space
    arrows = [
        # (label, from_x, from_y, to_x, to_y, color)
        ("WC cond_xyo wins",          0.10, 0.35, 1.05, 0.85, "#0b3d91"),
        ("WC cond_xo wins",           0.10, 0.30, 0.65, 0.45, "#5b8def"),
        ("WC cond_xyo_ystart wins",   0.10, 0.32, 1.10, 0.75, "#1d5fc4"),
        ("WI cond_xyo wins",          0.10, 0.25, 0.55, 0.40, "#7fb3d5"),
        ("teacher_loses (all)",      -0.05, 0.20, 0.10, 0.20, "gray"),
        ("SDPO original",            -0.30, 0.55, -0.15, 0.50, "#d62728"),
    ]
    d["fig3"] = arrows

    # fig-4 bubbles
    bubbles = [
        # (label, kl, adv_mean, adv_var, color)
        ("cond_xyo wins (WC)",          0.55, 0.85, 0.42, "#0b3d91"),
        ("cond_xyo_ystart wins (WC)",   0.40, 0.82, 0.40, "#1d5fc4"),
        ("cond_xo wins (WC)",           0.85, 0.55, 0.32, "#5b8def"),
        ("cond_xyo wins (WI)",          0.65, 0.55, 0.22, "#7fb3d5"),
        ("cond_xyo loses",              0.95, -0.05, 0.35, "#bdbdbd"),
        ("teacher_loses (all)",         1.05, -0.15, 0.30, "#888888"),
        ("SDPO original",              -0.05, -0.30, 0.25, "#d62728"),
    ]
    d["fig4"] = bubbles

    # fig-5 trajectories: (label, x_seq=KL, y_seq=task_perf, color)
    e = np.arange(1, 7)
    trajs = [
        ("teacher_wins + FKL",  np.linspace(0.05, 0.40, 6),  np.linspace(0.46, 0.74, 6) + rng.normal(0,0.01,6),  "#0b3d91"),
        ("teacher_wins + SFT",  np.linspace(0.05, 0.95, 6),  np.linspace(0.46, 0.66, 6) + rng.normal(0,0.012,6), "#5b8def"),
        ("teacher_loses + FKL", np.linspace(0.05, 0.85, 6),  np.linspace(0.46, 0.46, 6) + rng.normal(0,0.008,6), "#d62728"),
        ("teacher_loses + SFT", np.linspace(0.05, 1.55, 6),  np.linspace(0.46, 0.42, 6) + rng.normal(0,0.012,6), "#a0a0a0"),
    ]
    d["fig5"] = {"trajs": trajs, "epochs": e, "teacher": (0.45, 0.79), "base": (0.05, 0.46)}

    # fig-6 forgetting: 24 runs × 6 ckpts on amc23, vs base
    runs = []
    for i in range(24):
        obj = ["sft","fkl","sdpo","pc_sdpo"][i % 4]
        cond_idx = (i // 4) % 3
        wins = (i % 2 == 0)
        steps = np.array([30, 60, 90, 120, 150, 180])
        # different decay shapes per objective
        if obj == "sft":
            y = 32.5 + rng.normal(0,1.5) - np.linspace(0,8,6) + rng.normal(0,1.0,6)
        elif obj == "fkl":
            y = 32.5 + rng.normal(0,1.5) - np.linspace(0,12,6) + rng.normal(0,1.2,6)
        elif obj == "sdpo":
            y = 32.5 + rng.normal(0,1.5) - np.linspace(0,18,6) + rng.normal(0,1.5,6)
        else:
            y = 32.5 + rng.normal(0,1.5) - np.linspace(0,15,6) + rng.normal(0,1.5,6)
        runs.append({"obj": obj, "cond_idx": cond_idx, "wins": wins, "x": steps, "y": y})
    d["fig6"] = {"runs": runs, "base": 32.5}
    return d

# ─── REAL data loader ────────────────────────────────────────────────────────
def load_real_data(eval_root):
    """Walk eval_results dirs and return data structures matching dummy_data()."""
    BASE_AMC = 32.5  # Qwen3-4B baseline on amc23
    BASE_AIME = 3.3
    real = {"fig6": {"runs": [], "base": BASE_AMC, "base_aime": BASE_AIME}}

    DATASETS = ["wins_cond_xo","loses_cond_xo","wins_cond_xyo",
                "loses_cond_xyo","wins_cond_xyo_ystart","loses_cond_xyo_ystart"]
    meta = {}
    for i, ds in enumerate(DATASETS):
        meta[f"WC-{i*2 + 1}"]  = ("sft", ds, ds.startswith("wins"))
        meta[f"WC-{i*2 + 2}"]  = ("fkl", ds, ds.startswith("wins"))
        meta[f"WC-{13 + i}"]    = ("sdpo", ds, ds.startswith("wins"))
        meta[f"WC-{19 + i}"]    = ("pc_sdpo", ds, ds.startswith("wins"))

    # math accuracy per (rid, ckpt, dataset)
    math_acc = defaultdict(dict)
    for sp in glob.glob(f"{eval_root}/demo2teacher/WC-*/*/math/summary.txt"):
        parts = sp.split(os.sep); rid, ck = parts[-4], parts[-3]
        for line in open(sp):
            m = re.search(r"(amc23|aime24)\s+\S+\s+\S+\s+Final Accuracy:\s*([0-9.]+)", line)
            if m:
                math_acc[(rid, ck)][m.group(1)] = float(m.group(2))

    # Aggregate per run for fig6
    CKPT_ORDER = ["step-30","step-60","step-90","step-120","step-150","final"]
    CKPT_X = [30, 60, 90, 120, 150, 180]
    for rid, (obj, ds, wins) in meta.items():
        ys = []; xs = []
        for ck, x in zip(CKPT_ORDER, CKPT_X):
            v = math_acc.get((rid, ck), {}).get("amc23")
            if v is not None: ys.append(v); xs.append(x)
        if not ys: continue
        cond_idx = 0 if "cond_xo" in ds and "xyo" not in ds else (2 if "ystart" in ds else 1)
        real["fig6"]["runs"].append({
            "obj": obj, "cond_idx": cond_idx, "wins": wins,
            "x": np.array(xs), "y": np.array(ys),
        })

    # Judge data for fig2 — read from dataset prefix-decision archive
    # (already-judged 5700 rows with student_verdict + gpt4o_mini_verdict)
    real["fig2"] = None
    try:
        ds_path = "/home/ssmurali/demo2teacher_data/wildchat/teacher_wins_cond_xo.jsonl"
        if os.path.exists(ds_path):
            scores_s, scores_g, agree = [], [], []
            with open(ds_path) as f:
                for line in f:
                    r = json.loads(line)
                    s = 1.0 if r.get("student_verdict") == "y_star" else 0.0
                    g = 1.0 if r.get("gpt4o_mini_verdict") == "y_star" else 0.0
                    # add small noise so a 0/1 scatter looks continuous
                    scores_s.append(s + np.random.uniform(-0.05, 0.05))
                    scores_g.append(g + np.random.uniform(-0.05, 0.05))
                    agree.append(r.get("agreement", False))
            real["fig2"] = {"student": np.array(scores_s),
                            "gpt4o":   np.array(scores_g),
                            "correct": np.array(agree)}
    except Exception as e:
        print(f"[fig2] real data unavailable ({e}); will use dummy")

    # Fig 5 — needs KL(π_ckpt ∥ π_base) per ckpt + math acc per ckpt
    kl = {}
    for p in glob.glob(f"{eval_root}/demo2teacher/WC-*/kl_vs_base.json"):
        rid = os.path.basename(os.path.dirname(p))
        try:
            obj = json.load(open(p))
            for ck, info in obj.get("results", {}).items():
                kl[(rid, ck)] = info.get("kl_vs_base")
        except Exception: pass

    # math acc per (rid, ckpt) — average over seeds if seeded dirs exist
    CKPT_ORDER = ["step-30","step-60","step-90","step-120","step-150","final"]
    EPOCH_X = list(range(1, 7))
    math_amc_by_ckpt = defaultdict(dict)
    for sp in glob.glob(f"{eval_root}/demo2teacher/WC-*/*/math*/summary.txt"):
        parts = sp.split(os.sep)
        # parts[-3] could be "math" or "math_s0" — and ckpt is parts[-3]'s parent
        rid = parts[-4]; ck = parts[-3]
        # need to handle math_s0/summary.txt: parts[-1]=summary.txt, parts[-2]=math_sX, parts[-3]=ckpt
        # OR math/amc23/test_*.jsonl: parts[-1]=summary.txt sits at .../math/summary.txt so parts[-2]=math, parts[-3]=ckpt
        try:
            txt = open(sp).read()
            for line in txt.splitlines():
                m = re.search(r"amc23.*Final Accuracy:\s*([0-9.]+)", line)
                if m:
                    math_amc_by_ckpt[(rid, ck)].setdefault("amc23", []).append(float(m.group(1)))
        except Exception: pass

    fig5_trajs = []
    cmap = {"sft":"#1f77b4","fkl":"#2ca02c","sdpo":"#d62728","pc_sdpo":"#9467bd"}
    # meta values are (obj, ds, wins) — unpack accordingly
    for rid, val in meta.items():
        obj = val[0]; ds = val[1]
        xs = []; ys = []
        for ck in CKPT_ORDER:
            klv = kl.get((rid, ck))
            # search any math_s* result for this ckpt
            ckp_keys = [(rid, c) for c in [ck, "math", "math_s0"]]
            mathv = None
            for k in math_amc_by_ckpt:
                if k[0] == rid and k[1] == ck:
                    accs = math_amc_by_ckpt[k].get("amc23", [])
                    if accs: mathv = sum(accs)/len(accs); break
            # fallback: ckpt itself, parent dir is ckpt name not the math_sX
            if mathv is None:
                # check if any /math_sX dir for this ckpt landed
                for sp in glob.glob(f"{eval_root}/demo2teacher/{rid}/{ck}/math*/summary.txt"):
                    txt = open(sp).read()
                    m = re.search(r"amc23.*Final Accuracy:\s*([0-9.]+)", txt)
                    if m:
                        mathv = float(m.group(1)); break
            if klv is not None and mathv is not None:
                xs.append(klv); ys.append(mathv/100.0)  # normalize % to 0-1
        if xs:
            wins = ds.startswith("wins")
            label = f"{obj}+{ds.split('_',1)[1]}"
            color = cmap.get(obj, "#888")
            fig5_trajs.append((label, xs, ys, color))

    real["fig5"] = {"trajs": fig5_trajs, "epochs": EPOCH_X[:6], "teacher": None, "base": None} if fig5_trajs else None

    real["fig1"] = None
    real["fig3"] = None
    real["fig4"] = None
    return real

# ─── plot functions (apply to either real or dummy) ─────────────────────────
def fig1_conditioning_ladder(d, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    add_blue_gradient(ax, (-0.05, 1.4), (0.45, 0.85))

    add_anchor(ax, *d["base"], "Base Policy", color="#1a1a1a")
    add_anchor(ax, *d["demo"], "Demonstrator (y)", color="#0b3d91")

    teacher_pts = [(t[1], t[2]) for t in d["teachers"]]
    for (lab, x, y) in d["teachers"]:
        ax.scatter([x], [y], s=85, color="#1d5fc4", edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(lab, (x, y), xytext=(-12, 12), textcoords="offset points",
                    fontsize=8.5, color="#1a1a1a", ha="right")
    # curved arrows demonstrator → teacher_xo → teacher_xyo → teacher_xyo_ystart
    chain = [d["demo"]] + teacher_pts
    for a, b in zip(chain[:-1], chain[1:]):
        bezier_arrow(ax, a, b, color="#1a1a1a", lw=1.0, rad=-0.35, alpha=0.7)

    ax.set_xlabel(r"KL($\pi_{\mathrm{teacher}} \,\|\, \pi_{\mathrm{base}}$)")
    ax.set_ylabel(r"Winrate of $y^*$ over $y_{\mathrm{base}}$")
    ax.set_xlim(-0.05, 1.4); ax.set_ylim(0.45, 0.85)
    ax.set_title("Stronger conditioning moves teacher closer to base while maintaining/improving winrate.",
                 fontsize=10, style="italic", pad=12, loc="center")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def fig2_judge_agreement(d, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    s = d["student"]; g = d["gpt4o"]; correct = d["correct"]
    # 2D hist density
    h, xe, ye = np.histogram2d(g, s, bins=30, range=[[0,1],[0,1]])
    ax.imshow(h.T, origin="lower", extent=[0,1,0,1], aspect="auto",
              cmap=BLUE_CMAP, norm=mpl.colors.LogNorm(vmin=1, vmax=h.max()),
              alpha=0.85, zorder=0)
    ax.scatter(g[correct], s[correct], marker="^", s=18, color="#d62728",
               edgecolor="white", linewidth=0.4, alpha=0.8, label="correct", zorder=3)
    ax.scatter(g[~correct], s[~correct], marker="^", s=18, color="#0b3d91",
               edgecolor="white", linewidth=0.4, alpha=0.8, label="incorrect", zorder=3)
    ax.plot([0,1],[0,1], "--", color="#444", lw=0.8, alpha=0.6)
    ax.text(0.02, 0.97, "Student prefers y*,\nGPT-4o-mini prefers y", fontsize=8.5,
            style="italic", va="top", color="#1a1a1a")
    ax.text(0.98, 0.03, "GPT-4o-mini prefers y*,\nStudent prefers y", fontsize=8.5,
            style="italic", va="bottom", ha="right", color="#1a1a1a")
    ax.text(0.98, 0.97, "Both prefer y*", fontsize=8.5, style="italic", va="top", ha="right", color="#1a1a1a")
    ax.text(0.02, 0.03, "Both prefer y", fontsize=8.5, style="italic", va="bottom", color="#1a1a1a")
    ax.set_xlabel(r"GPT-4o-mini preference score for $y^*$")
    ax.set_ylabel(r"Student preference score for $y^*$")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def fig3_displacement_arrows(d, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    add_blue_gradient(ax, (-0.4, 1.45), (0.0, 1.0))
    for (lab, x0, y0, x1, y1, c) in d:
        bezier_arrow(ax, (x0, y0), (x1, y1), color=c, lw=2.0, rad=0.0, alpha=0.95)
        ax.annotate(lab, (x1, y1), xytext=(8, 4), textcoords="offset points", fontsize=8.5)
    ax.axvline(0, color="#444", linestyle=":", lw=0.6, alpha=0.6)
    ax.text(0.01, 0.05, "zero mean", fontsize=8, style="italic", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Advantage Mean")
    ax.set_ylabel("Advantage Variance")
    ax.set_xlim(-0.4, 1.45); ax.set_ylim(0, 1.0)
    ax.set_title("Displacement arrows — total signal absorbed across training",
                 fontsize=10, style="italic", pad=10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def fig4_kl_adv_bubble(d, out):
    # Auto-detect range from data so small KL values (0.001-0.15) plot well
    kls = [x[1] for x in d if x[1] and x[1] > 0]
    advs = [x[2] for x in d]
    if kls:
        xlim = (max(min(kls)*0.5, 1e-4), max(kls)*1.5)
    else:
        xlim = (1e-4, 2.0)
    if advs:
        ypad = (max(advs)-min(advs))*0.2 if max(advs)>min(advs) else 0.5
        ylim = (min(advs)-ypad, max(advs)+ypad)
    else:
        ylim = (-0.4, 1.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xscale("log")
    # log-aware gradient
    x_grad = np.geomspace(xlim[0], xlim[1], 200)
    y_grad = np.linspace(ylim[0], ylim[1], 200)
    Xg, Yg = np.meshgrid(x_grad, y_grad)
    cx_log = np.log10(xlim[0]) + 0.1*(np.log10(xlim[1])-np.log10(xlim[0]))
    Zg = np.exp(-2.0*((np.log10(Xg)-cx_log)**2 + ((Yg-ylim[1]+0.05)/(ylim[1]-ylim[0]))**2))
    ax.contourf(Xg, Yg, Zg, levels=12, cmap=BLUE_CMAP, alpha=0.55, zorder=0)

    for (lab, kl, adv, var, c) in d:
        if kl is None or kl <= 0: continue
        ax.scatter([kl], [adv], s=400 * var + 30, color=c, alpha=0.78,
                   edgecolor="white", linewidth=1.0, zorder=4)
        ax.annotate(lab, (kl, adv), xytext=(8, 6), textcoords="offset points", fontsize=8.5)
    ax.axhline(0, color="#444", linestyle=":", lw=0.6, alpha=0.6)
    ax.set_xlabel(r"KL($\pi_{\mathrm{teacher}} \,\|\, \pi_{\mathrm{base}}$)  [log scale]")
    ax.set_ylabel("Advantage Mean at init")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Teacher quality in KL–advantage space",
                 fontsize=10, style="italic", pad=10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def fig5_kl_task_trajectory(d, out):
    # Auto-detect plot range from real data — handles small KL values (0.001-0.14)
    all_x = []; all_y = []
    for lab, xs, ys, color in d["trajs"]:
        all_x.extend([x for x in xs if x and x > 0])
        all_y.extend(ys)
    if all_x:
        xlim = (max(min(all_x)*0.5, 1e-4), max(all_x)*1.5)
    else:
        xlim = (1e-4, 2.0)
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        ypad = (ymax-ymin)*0.15 if ymax>ymin else 0.05
        ylim = (ymin-ypad, ymax+ypad)
    else:
        ylim = (0.40, 0.80)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xscale("log")
    # gradient on log axis: just shade RHS
    x_grad = np.geomspace(xlim[0], xlim[1], 200)
    y_grad = np.linspace(ylim[0], ylim[1], 200)
    Xg, Yg = np.meshgrid(x_grad, y_grad)
    Zg = np.exp(-2.0*((np.log10(Xg)-np.log10(xlim[1])+0.05)**2 + ((Yg-ylim[1]+0.02)/(ylim[1]-ylim[0]))**2))
    ax.contourf(Xg, Yg, Zg, levels=12, cmap=BLUE_CMAP, alpha=0.55, zorder=0)

    if d.get("teacher"): add_anchor(ax, *d["teacher"], "Teacher", color="#0b3d91")
    if d.get("base"):    add_anchor(ax, max(xlim[0], 1e-3), d["base"][1] if isinstance(d["base"], tuple) else ylim[0]+0.02, "Base Policy", color="#1a1a1a")

    for lab, xs, ys, color in d["trajs"]:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        ys_smooth = ema(ys, beta=0.99) if len(ys) > 2 else ys
        # raw faint
        ax.plot(xs, ys, color=color, marker="o", markersize=4, linewidth=0.5, alpha=0.35, zorder=3)
        # EMA-smoothed bold
        ax.plot(xs, ys_smooth, color=color, marker="o", markersize=5, linewidth=1.7, alpha=0.95, label=lab, zorder=4)
        for (x, y, e) in zip(xs, ys_smooth, d["epochs"]):
            if x > 0:
                ax.annotate(str(e), (x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=color)
    ax.set_xlabel(r"KL($\pi_{\mathrm{ckpt}} \,\|\, \pi_{\mathrm{base}}$)  [log scale]")
    ax.set_ylabel("Task Performance")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_title("KL grows ~exponentially through training; performance traces a path away from base.",
                 fontsize=10, style="italic", pad=10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def fig6_forgetting(d, out):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    base = d["base"]
    obj_color = OBJ_COLOR
    for run in d["runs"]:
        ls = "-" if run["wins"] else "--"
        alpha = [0.45, 0.75, 1.0][run["cond_idx"]]
        ys = np.asarray(run["y"], dtype=float)
        ys_smooth = ema(ys, beta=0.99) if len(ys) > 2 else ys
        # raw faint
        ax.plot(run["x"], ys, color=obj_color[run["obj"]], linestyle=ls,
                alpha=alpha*0.35, linewidth=0.5, marker="o", markersize=2.5, zorder=3)
        # EMA-smoothed bold
        ax.plot(run["x"], ys_smooth, color=obj_color[run["obj"]], linestyle=ls,
                alpha=alpha, linewidth=1.6, marker="o", markersize=3.2, zorder=4)
    ax.axhline(base, color="#1a1a1a", linestyle=":", lw=1.2, alpha=0.85)
    ax.text(ax.get_xlim()[1], base, f"  base = {base:.1f}%", fontsize=9, va="center", style="italic")
    # legend
    handles = [plt.Line2D([],[], color=obj_color[o], lw=2, label=o) for o in ["sft","fkl","sdpo","pc_sdpo"]]
    handles += [plt.Line2D([],[], color="black", lw=1.5, label="wins (solid)"),
                plt.Line2D([],[], color="black", lw=1.5, linestyle="--", label="loses (dashed)")]
    ax.legend(handles=handles, loc="lower left", ncol=2, framealpha=0.9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("amc23 accuracy (%)")
    ax.set_xlim(0, 200)
    ax.set_title("Forgetting on amc23 while training on WildChat — 24 runs × 6 ckpts",
                 fontsize=10, style="italic", pad=10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

# ─── orchestrator ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    EVAL_ROOT = "/home/ssmurali/user_interactions/CFT-Eric-Zhu/eval_results"
    if args.dummy:
        out_dir = args.out_dir or f"{EVAL_ROOT}/_dummy_plots"
        d = dummy_data()
        print(f"[DUMMY MODE] writing previews to {out_dir}")
    else:
        out_dir = args.out_dir or f"{EVAL_ROOT}/_plots"
        d_real = load_real_data(EVAL_ROOT)
        d_dummy = dummy_data()
        # Use real where available, dummy where not (yet)
        d = {}
        for k in ["fig1","fig2","fig3","fig4","fig5","fig6"]:
            d[k] = d_real.get(k) if d_real.get(k) is not None else d_dummy[k]
        print(f"[REAL MODE] writing to {out_dir} (figs missing real data fall back to dummy)")
        for k in ["fig1","fig2","fig3","fig4","fig5","fig6"]:
            tag = "REAL" if d_real.get(k) is not None else "dummy"
            print(f"  {k:<6} {tag}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig1_conditioning_ladder(d["fig1"],   f"{out_dir}/fig1_conditioning_ladder.png");  print(f"  ✓ fig1")
    fig2_judge_agreement(    d["fig2"],   f"{out_dir}/fig2_judge_agreement.png");      print(f"  ✓ fig2")
    fig3_displacement_arrows(d["fig3"],   f"{out_dir}/fig3_displacement_arrows.png");  print(f"  ✓ fig3")
    fig4_kl_adv_bubble(      d["fig4"],   f"{out_dir}/fig4_kl_adv_bubble.png");        print(f"  ✓ fig4")
    fig5_kl_task_trajectory( d["fig5"],   f"{out_dir}/fig5_kl_task_trajectory.png");   print(f"  ✓ fig5")
    fig6_forgetting(         d["fig6"],   f"{out_dir}/fig6_forgetting.png");           print(f"  ✓ fig6")
    print(f"\n✓ done — {out_dir}/")

if __name__ == "__main__":
    main()
