"""Pull wandb KL/loss/adv per run, join with math+alpaca eval results, plot.

Run from anywhere (it strips user_interactions from sys.path so the local 'wandb'
folder doesn't shadow the package). Outputs to:
  CFT-Eric-Zhu/eval_results/demo2teacher/_plots/
"""
import os, sys, re, json, glob
from collections import defaultdict
from pathlib import Path

# strip the local wandb dir from sys.path
for p in list(sys.path):
    if "user_interactions" in p:
        sys.path.remove(p)

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO = "/home/ssmurali/user_interactions"
EVAL_ROOT = f"{REPO}/CFT-Eric-Zhu/eval_results/demo2teacher"
JUDGE_PATH = f"{EVAL_ROOT}/_alpaca_judge/summary.json"
BASE_PATH  = f"{REPO}/CFT-Eric-Zhu/eval_results/baseline_qwen25math7b/math/summary.txt"
PLOTS = f"{EVAL_ROOT}/_plots"
Path(PLOTS).mkdir(parents=True, exist_ok=True)

# ─── eval-side run_id mapping (matches dirs on disk) ─────────────────────────
DATASETS = ["wins_cond_xo", "loses_cond_xo", "wins_cond_xyo", "loses_cond_xyo",
            "wins_cond_xyo_ystart", "loses_cond_xyo_ystart"]
# disk_run_id -> (obj, ds)
disk_meta = {}
for i, ds in enumerate(DATASETS):
    disk_meta[f"WI-{i*2 + 1}"]  = ("sft", ds)
    disk_meta[f"WI-{i*2 + 2}"]  = ("fkl", ds)
    disk_meta[f"WI-{13 + i}"]    = ("sdpo", ds)
    disk_meta[f"WI-{19 + i}"]    = ("pc_sdpo", ds)

# ─── load eval results ───────────────────────────────────────────────────────
m = re.search(r"Final Accuracy:\s*([0-9.]+)", open(BASE_PATH).read())
BASE_MATH = float(m.group(1))
print(f"Base math-500: {BASE_MATH}%")

math_acc = defaultdict(dict)
for sp in glob.glob(f"{EVAL_ROOT}/*/*/math/summary.txt"):
    parts = sp.split(os.sep); rid, ck = parts[-4], parts[-3]
    m = re.search(r"Final Accuracy:\s*([0-9.]+)", open(sp).read())
    if m: math_acc[rid][ck] = float(m.group(1))

alpa = defaultdict(dict)
judge = json.load(open(JUDGE_PATH))
for tag, info in judge.items():
    if "__" not in tag: continue
    rid, ck = tag.split("__", 1)
    alpa[rid][ck] = info.get("winrate", 0)

# ─── pull wandb ──────────────────────────────────────────────────────────────
print("Pulling wandb runs ...")
api = wandb.Api()
runs = list(api.runs("demonstrator-to-teacher", per_page=200))
print(f"  found {len(runs)} runs")

# wandb run name format: webinstruct__qwen25-math-7b__{wins|loses}__{cond}__{obj}__{run_id_local}
# tags carry: [run_id_local, conditioning, model, objective, family, win/loss]
# Map wandb run name → (obj, ds) via tags
def parse_wandb_run(r):
    tags = set(r.tags)
    obj = next((o for o in ["sft","fkl","sdpo","pc_sdpo"] if o in tags), None)
    cond = next((c for c in ["cond_xo","cond_xyo_ystart","cond_xyo"] if c in tags), None)
    direction = "wins" if "wins" in tags else ("loses" if "loses" in tags else None)
    if not (obj and cond and direction): return None, None
    ds = f"{direction}_{cond}"
    return obj, ds

# For each (obj, ds), there should be one wandb run. Look up its history.
wandb_data = {}  # (obj, ds) -> dataframe with step, kl/T_S, kl/S_T, loss, advantage/mean
for r in runs:
    obj, ds = parse_wandb_run(r)
    if obj is None: continue
    if r.state != "finished": continue
    key = (obj, ds)
    if key in wandb_data: continue  # take first finished
    try:
        h = r.history(samples=2000, pandas=True)  # pull all keys
    except Exception as e:
        print(f"  ! {r.name}: {e}")
        continue
    if h.empty: continue
    # Unify loss column: SFT logs loss/sft, FKL logs loss/fkl, SDPO/PC-SDPO log loss
    if "loss" not in h.columns:
        for cand in ("loss/sft", "loss/fkl", "loss/pc_sdpo", "loss/sdpo"):
            if cand in h.columns:
                h["loss"] = h[cand]; break
    wandb_data[key] = h
print(f"  pulled history for {len(wandb_data)} (obj,ds) configs")

# ─── helper: map ckpt_name → opt_step ────────────────────────────────────────
# Trained 5 evenly-spaced ckpts over total_steps; with total_steps=455 (5 epochs):
# step-91 = epoch1, step-182 = epoch2, step-273 = epoch3, step-364 = epoch4, final = epoch5
CKPT_STEP = {"step-91": 91, "step-182": 182, "step-273": 273, "step-364": 364, "final": 455}
EVAL_CKPTS = ["step-91", "step-182", "final"]

def kl_at(df, target_step, col):
    """Return KL value at step closest to target_step."""
    if df is None or df.empty or col not in df.columns: return None
    df2 = df.dropna(subset=[col])
    if df2.empty: return None
    idx = (df2["step"] - target_step).abs().idxmin()
    return float(df2.loc[idx, col])

# ─── build the master joined table ───────────────────────────────────────────
rows = []
for rid, m_row in math_acc.items():
    obj, ds = disk_meta.get(rid, (None, None))
    if obj is None: continue
    h = wandb_data.get((obj, ds))
    for ck in EVAL_CKPTS:
        if ck not in m_row: continue
        step = CKPT_STEP[ck]
        rows.append({
            "run_id": rid, "obj": obj, "ds": ds,
            "ckpt": ck, "step": step,
            "math": m_row.get(ck),
            "alpaca": alpa[rid].get(ck),
            "kl_TS": kl_at(h, step, "kl/T_S"),
            "kl_ST": kl_at(h, step, "kl/S_T"),
            "loss": kl_at(h, step, "loss"),
        })
df = pd.DataFrame(rows)
df.to_csv(f"{PLOTS}/master.csv", index=False)
print(f"\nMaster table written: {PLOTS}/master.csv  ({len(df)} rows)")
print(df.head(8).to_string())

# ─── plotting ────────────────────────────────────────────────────────────────
OBJ_COLOR = {"sft": "#1f77b4", "fkl": "#2ca02c", "sdpo": "#d62728", "pc_sdpo": "#9467bd"}
OBJ_ORDER = ["sft","fkl","sdpo","pc_sdpo"]

# (1) KL trajectory full training curves: 12 panels (4 obj × 3 cond)
def plot_kl_trajectories():
    # SFT has no KL — exclude it. Only fkl, sdpo, pc_sdpo.
    objs_with_kl = ["fkl","sdpo","pc_sdpo"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for i, obj in enumerate(objs_with_kl):
        for j, cond in enumerate(["cond_xo","cond_xyo","cond_xyo_ystart"]):
            ax = axes[i, j]
            plotted = False
            for direction, lstyle in [("wins","-"), ("loses","--")]:
                key = (obj, f"{direction}_{cond}")
                h = wandb_data.get(key)
                if h is None: continue
                if "kl/T_S" not in h.columns or "kl/S_T" not in h.columns: continue
                hd = h.dropna(subset=["kl/T_S","kl/S_T","step"])
                if hd.empty: continue
                ax.plot(hd["step"], hd["kl/T_S"], color=OBJ_COLOR[obj], linestyle=lstyle, alpha=0.85, label=f"{direction} KL(T||S)")
                ax.plot(hd["step"], hd["kl/S_T"], color=OBJ_COLOR[obj], linestyle=lstyle, alpha=0.45, label=f"{direction} KL(S||T)")
                plotted = True
            ax.set_title(f"{obj} | {cond}", fontsize=10)
            if j==0: ax.set_ylabel("KL (nats)")
            if i==len(objs_with_kl)-1: ax.set_xlabel("opt step")
            ax.grid(alpha=0.3)
            if plotted and i==0 and j==0: ax.legend(fontsize=7)
            ax.set_yscale("symlog", linthresh=1)
    fig.suptitle("KL trajectories during training. SFT excluded (no teacher distribution to KL against).", fontsize=11)
    fig.tight_layout()
    out = f"{PLOTS}/01_kl_trajectories.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {out}")

# (2) Math accuracy vs KL(T||S) scatter — one point per (run, ckpt)
def plot_kl_vs_acc():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ycol, title in [(axes[0], "math", f"Math-500 acc vs KL(T||S)  (base={BASE_MATH}%)"),
                            (axes[1], "alpaca", "Alpaca winrate vs KL(T||S)")]:
        for obj in OBJ_ORDER:
            sub = df[df.obj == obj]
            ax.scatter(sub["kl_TS"], sub[ycol], color=OBJ_COLOR[obj], label=obj, alpha=0.75, s=50, edgecolors="white", linewidths=0.5)
        if ycol == "math": ax.axhline(BASE_MATH, color="gray", linestyle="--", alpha=0.6, label=f"base ({BASE_MATH}%)")
        ax.set_xlabel("KL(p_teacher || p_student) at ckpt")
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xscale("symlog", linthresh=1)
    fig.suptitle("Quality vs divergence — does the student that stays close to teacher do better?", fontsize=11)
    fig.tight_layout()
    out = f"{PLOTS}/02_kl_vs_quality.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {out}")

# (3) Trajectory by objective: math + alpaca over ckpts (3 ckpts, mean over conds)
def plot_obj_trajectories():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = [1, 2, 3]
    xticklab = ["epoch1", "epoch2", "final"]
    for ax, ycol, ttl, ref in [(axes[0], "math", "Math-500", BASE_MATH), (axes[1], "alpaca", "Alpaca winrate", None)]:
        for obj in OBJ_ORDER:
            means, stds = [], []
            for ck in EVAL_CKPTS:
                vals = df[(df.obj==obj) & (df.ckpt==ck)][ycol].dropna().values
                if len(vals) > 0:
                    means.append(vals.mean()); stds.append(vals.std())
                else:
                    means.append(np.nan); stds.append(0)
            means, stds = np.array(means), np.array(stds)
            ax.errorbar(x, means, yerr=stds, color=OBJ_COLOR[obj], label=obj, marker="o", capsize=3, linewidth=2)
        if ref is not None: ax.axhline(ref, color="gray", linestyle="--", alpha=0.6, label=f"base ({ref}%)")
        ax.set_xticks(x); ax.set_xticklabels(xticklab)
        ax.set_ylabel(ycol); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Quality trajectory by objective (mean ± sd over conditionings)", fontsize=11)
    fig.tight_layout()
    out = f"{PLOTS}/03_obj_trajectories.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {out}")

# (4) Pareto: math vs alpaca, all 72 points
def plot_pareto():
    fig, ax = plt.subplots(figsize=(8, 6))
    for obj in OBJ_ORDER:
        sub = df[df.obj == obj]
        ax.scatter(sub["math"], sub["alpaca"], color=OBJ_COLOR[obj], label=obj, alpha=0.75, s=60, edgecolors="white", linewidths=0.5)
    ax.axvline(BASE_MATH, color="gray", linestyle="--", alpha=0.6, label=f"base math ({BASE_MATH}%)")
    ax.set_xlabel("math-500 accuracy (%)")
    ax.set_ylabel("alpaca winrate (vs text-davinci-003)")
    ax.set_title("Math vs Alpaca tradeoff — every (run, ckpt) point colored by objective")
    ax.legend(); ax.grid(alpha=0.3)
    out = f"{PLOTS}/04_math_vs_alpaca_pareto.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {out}")

# (5) KL final value (mean per obj × cond) — bar
def plot_kl_final_bars():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    conds = ["cond_xo","cond_xyo","cond_xyo_ystart"]
    width = 0.20
    x = np.arange(len(conds))
    for ax, kl_col, ttl in [(axes[0], "kl_TS", "KL(p_T || p_S) at final ckpt"),
                            (axes[1], "kl_ST", "KL(p_S || p_T) at final ckpt")]:
        for k, obj in enumerate(OBJ_ORDER):
            means = []
            for cond in conds:
                vals = df[(df.obj==obj) & (df.ckpt=="final") & (df.ds.str.contains(cond, regex=False))]
                # special case: cond_xo matches cond_xyo too — need exact filter
                if cond == "cond_xo":
                    vals = df[(df.obj==obj) & (df.ckpt=="final") & df.ds.str.endswith("cond_xo")]
                elif cond == "cond_xyo":
                    vals = df[(df.obj==obj) & (df.ckpt=="final") & df.ds.str.endswith("cond_xyo")]
                else:
                    vals = df[(df.obj==obj) & (df.ckpt=="final") & df.ds.str.endswith("cond_xyo_ystart")]
                means.append(vals[kl_col].mean())
            ax.bar(x + k*width - 1.5*width, means, width, color=OBJ_COLOR[obj], label=obj)
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15)
        ax.set_ylabel("KL (nats)"); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        ax.set_yscale("symlog", linthresh=1)
    fig.suptitle("Distribution divergence at end of training", fontsize=11)
    fig.tight_layout()
    out = f"{PLOTS}/05_kl_final_bars.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {out}")

# (6) Loss curves
def plot_loss():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, obj in zip(axes.flat, OBJ_ORDER):
        any_plotted = False
        for direction, lstyle in [("wins","-"), ("loses","--")]:
            for cond, alpha in [("cond_xo",0.5),("cond_xyo",0.8),("cond_xyo_ystart",1.0)]:
                key = (obj, f"{direction}_{cond}")
                h = wandb_data.get(key)
                if h is None or "loss" not in h.columns: continue
                hd = h.dropna(subset=["loss","step"])
                if hd.empty: continue
                ax.plot(hd["step"], hd["loss"], linestyle=lstyle, color=OBJ_COLOR[obj], alpha=alpha,
                        label=f"{direction[0]}-{cond}")
                any_plotted = True
        ax.set_title(obj); ax.set_ylabel("loss"); ax.set_xlabel("opt step")
        if any_plotted: ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Training loss by objective (solid=wins, dashed=loses)", fontsize=12)
    fig.tight_layout()
    out = f"{PLOTS}/06_loss_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {out}")

print("\nGenerating plots ...")
plot_kl_trajectories()
plot_kl_vs_acc()
plot_obj_trajectories()
plot_pareto()
plot_kl_final_bars()
plot_loss()
print(f"\n✓ All plots in {PLOTS}/")
