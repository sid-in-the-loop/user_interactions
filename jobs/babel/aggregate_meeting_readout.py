"""Aggregate everything into master.csv + per-model readout.txt for the meeting.

Reads:
  - math_s0/s1/s2/summary.txt for amc23 + aime24
  - alpaca/outputs.jsonl + _alpaca_judge_final/summary.json (or recompute from .judge.jsonl)
  - kl_vs_base.json
  - _wandb_pulled/training_data.json
  - baseline_qwen3_4b/math/summary.txt

Writes:
  - _plots/master.csv (one row per ckpt)
  - _plots/per_model_readout.txt (one block per run, ranked by amc23 retention)
"""
import json, os, re, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = "/home/ssmurali/user_interactions"
RES  = f"{REPO}/CFT-Eric-Zhu/eval_results"
ROOT = f"{RES}/demo2teacher"
PLOTS = f"{RES}/_plots"
Path(PLOTS).mkdir(parents=True, exist_ok=True)

# ─── baselines ───────────────────────────────────────────────────────────────
BASE_AMC, BASE_AIME = 32.5, 3.3
bp = f"{RES}/baseline_qwen3_4b/math/summary.txt"
if os.path.exists(bp):
    txt = open(bp).read()
    m = re.search(r"amc23.*?Final Accuracy:\s*([0-9.]+)", txt)
    if m: BASE_AMC = float(m.group(1))
    m = re.search(r"aime24.*?Final Accuracy:\s*([0-9.]+)", txt)
    if m: BASE_AIME = float(m.group(1))

# ─── meta ────────────────────────────────────────────────────────────────────
DATASETS = ["wins_cond_xo","loses_cond_xo","wins_cond_xyo","loses_cond_xyo",
            "wins_cond_xyo_ystart","loses_cond_xyo_ystart"]
meta = {}
for i, ds in enumerate(DATASETS):
    meta[f"WC-{i*2+1}"] = ("sft", ds);  meta[f"WC-{i*2+2}"] = ("fkl", ds)
    meta[f"WC-{13+i}"]  = ("sdpo", ds); meta[f"WC-{19+i}"]  = ("pc_sdpo", ds)

CKPTS = ["step-30","step-60","step-90","step-120","step-150","final"]
CKPT_X = [30, 60, 90, 120, 150, 180]

# ─── load eval data ──────────────────────────────────────────────────────────
def parse_math_summary(path):
    """Returns dict of {dataset: acc}. summary.txt may have multiple datasets."""
    if not os.path.exists(path): return {}
    out = {}
    for line in open(path):
        m = re.search(r"(amc23|aime24)\s+\S+\s+\S+\s+Final Accuracy:\s*([0-9.]+)", line)
        if m: out[m.group(1)] = float(m.group(2))
    return out

# math (3 seeds): per (run, ckpt, dataset, seed) -> acc
math_per_seed = defaultdict(lambda: defaultdict(dict))  # math_per_seed[(rid,ck)][ds] = [s0,s1,s2]
for rid in meta:
    for ck in CKPTS:
        per_ds = defaultdict(list)
        for s in [0,1,2]:
            path = f"{ROOT}/{rid}/{ck}/math_s{s}/summary.txt"
            res = parse_math_summary(path)
            for ds in ("amc23","aime24"):
                if ds in res: per_ds[ds].append(res[ds])
        # also fall back to old single-seed math dir if seeded missing
        if not per_ds:
            res = parse_math_summary(f"{ROOT}/{rid}/{ck}/math/summary.txt")
            for ds in ("amc23","aime24"):
                if ds in res: per_ds[ds].append(res[ds])
        math_per_seed[(rid,ck)] = dict(per_ds)

# alpaca winrate (re-compute from .judge.jsonl files)
def load_judge_dir(d):
    out = {}
    for p in glob.glob(f"{d}/*.judge.jsonl"):
        name = os.path.basename(p).replace(".judge.jsonl","")
        if "__" not in name: continue
        rid, ck = name.split("__", 1)
        rows = [json.loads(l) for l in open(p)]
        valid = [r for r in rows if r.get("winner") in ("A","B","tie")]
        if not valid: continue
        wins = sum(1 for r in valid if r["winner"]=="A")
        ties = sum(1 for r in valid if r["winner"]=="tie")
        out[(rid, ck)] = (wins + 0.5*ties) / len(valid)
    return out

alpa = load_judge_dir(f"{ROOT}/_alpaca_judge_final")
if not alpa:
    alpa = load_judge_dir(f"{ROOT}/_alpaca_judge_v2")
print(f"  loaded {len(alpa)} alpaca winrates")

# KL vs base
kl = {}
for p in glob.glob(f"{ROOT}/*/kl_vs_base.json"):
    rid = os.path.basename(os.path.dirname(p))
    d = json.load(open(p))
    for ck, info in d.get("results", {}).items():
        kl[(rid, ck)] = info.get("kl_vs_base")

# wandb training data
wb = {}
wb_path = f"{RES}/_wandb_pulled/training_data.json"
if os.path.exists(wb_path):
    wb = json.load(open(wb_path))

# ─── master.csv ──────────────────────────────────────────────────────────────
import csv
master_path = f"{PLOTS}/master.csv"
with open(master_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id","obj","dataset","wins","ckpt","step",
                "amc23_mean","amc23_std","amc23_seeds",
                "aime24_mean","aime24_std","aime24_seeds",
                "alpaca_winrate","kl_vs_base"])
    for rid, (obj, ds) in meta.items():
        wins = ds.startswith("wins")
        for ck, x in zip(CKPTS, CKPT_X):
            mr = math_per_seed.get((rid, ck), {})
            amc = mr.get("amc23", []); aim = mr.get("aime24", [])
            row = [rid, obj, ds, int(wins), ck, x,
                   f"{np.mean(amc):.2f}" if amc else "",
                   f"{np.std(amc):.2f}"  if amc else "",
                   ",".join(f"{v:.1f}" for v in amc) if amc else "",
                   f"{np.mean(aim):.2f}" if aim else "",
                   f"{np.std(aim):.2f}"  if aim else "",
                   ",".join(f"{v:.1f}" for v in aim) if aim else "",
                   f"{alpa.get((rid,ck), '')}",
                   f"{kl.get((rid,ck), '')}"]
            w.writerow(row)
print(f"  master.csv: {master_path}")

# ─── per-model readout ──────────────────────────────────────────────────────
read_path = f"{PLOTS}/per_model_readout.txt"
with open(read_path, "w") as f:
    f.write(f"WC eval readout — {len(meta)} runs × 6 ckpts\n")
    f.write(f"baseline Qwen3-4B:  amc23={BASE_AMC}%  aime24={BASE_AIME}%  alpaca = pending re-judge\n")
    f.write("="*80 + "\n\n")

    # Rank runs by amc23 best-ckpt mean (highest retention = top)
    rid_rank = []
    for rid in meta:
        amc_best = 0
        for ck in CKPTS:
            v = math_per_seed.get((rid, ck), {}).get("amc23", [])
            if v: amc_best = max(amc_best, np.mean(v))
        rid_rank.append((rid, amc_best))
    rid_rank.sort(key=lambda x: -x[1])

    for rid, _ in rid_rank:
        obj, ds = meta[rid]
        f.write(f"{rid}  ({obj}, {ds})\n")
        # Math trajectory
        for db, base in [("amc23", BASE_AMC), ("aime24", BASE_AIME)]:
            line = f"  {db:<8} (base={base:.1f}%):  "
            for ck in CKPTS:
                v = math_per_seed.get((rid, ck), {}).get(db, [])
                if v:
                    m = np.mean(v); s = np.std(v) if len(v)>1 else 0
                    line += f"{m:.1f}±{s:.1f}  "
                else:
                    line += "  --   "
            f.write(line.rstrip() + "\n")
        # Alpaca trajectory
        line = f"  alpaca:                "
        for ck in CKPTS:
            v = alpa.get((rid, ck))
            line += f"{v:.3f}        " if v is not None else "  --          "
        f.write(line.rstrip() + "\n")
        # KL trajectory
        line = f"  kl_vs_base:            "
        for ck in CKPTS:
            v = kl.get((rid, ck))
            line += f"{v:+.3f}        " if v is not None else "  --          "
        f.write(line.rstrip() + "\n")
        f.write("\n")

    # Cross-tabulation by objective (final ckpt mean)
    f.write("="*80 + "\n")
    f.write("MEANS BY OBJECTIVE (final ckpt, across 6 conditionings)\n")
    f.write("="*80 + "\n")
    f.write(f"{'obj':<10}{'amc23':>10}{'aime24':>10}{'alpaca':>10}{'kl':>10}\n")
    for obj in ["sft","fkl","sdpo","pc_sdpo"]:
        amcs = []; aims = []; alps = []; kls = []
        for rid, (o, ds) in meta.items():
            if o != obj: continue
            v = math_per_seed.get((rid, "final"), {}).get("amc23", [])
            if v: amcs.append(np.mean(v))
            v = math_per_seed.get((rid, "final"), {}).get("aime24", [])
            if v: aims.append(np.mean(v))
            v = alpa.get((rid, "final"))
            if v is not None: alps.append(v)
            v = kl.get((rid, "final"))
            if v is not None: kls.append(v)
        f.write(f"{obj:<10}"
                f"{(sum(amcs)/len(amcs) if amcs else 0):>9.1f}%"
                f"{(sum(aims)/len(aims) if aims else 0):>9.1f}%"
                f"{(sum(alps)/len(alps) if alps else 0):>10.3f}"
                f"{(sum(kls)/len(kls) if kls else 0):>10.3f}\n")

print(f"  per_model_readout.txt: {read_path}")
print(f"\n── PREVIEW (first 60 lines) ──")
print("".join(open(read_path).readlines()[:60]))
